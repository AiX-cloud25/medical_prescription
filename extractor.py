"""
extractor.py — Offline VLM engine via Ollama (qwen2.5vl)
────────────────────────────────────────────────────────
Identical pipeline to the sibling doctor_prescription_gpt_extractor —
same prompts, same combined transcript+layout pass, same structured-fields
pass, same feedback learning — with the ONLY difference being the vision
model: a local Ollama model (default qwen2.5vl:7b) instead of Azure OpenAI.

Strategy:
  • Image (jpg/png/webp) → sent to the model as raw base64 (Ollama format).
  • PDF → each page rendered to JPEG (pypdfium2, 150 DPI) and sent per page.

Public API:
    extract(data, ext) → (pages, extras, meta)
        pages  : [{"page", "text", "layout_html", "layout_error"}, ...]
        extras : {"fields": [...], "medicines": [...], "fields_error": ...}
        meta   : {"engine": ..., "source": "ollama-vision", "deployment": ...}
"""

import base64
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pypdfium2 as pdfium
import requests
from PIL import Image

import feedback_store

_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "900"))
_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "32768"))
ENGINE_NAME = f"Offline VLM via Ollama ({_MODEL})"

# Render resolution for PDF pages sent to the vision model.
_RENDER_DPI = 150
_SCALE = _RENDER_DPI / 72

# Output budget for the text-only transcription / structured-fields calls.
_VISION_MAX_TOKENS = 12000

_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

_READ_SYSTEM = (
    "You are an expert medical document reader assisting a licensed pharmacy "
    "team with digitizing prescriptions and hospital forms the patient has "
    "already provided. Reconstruct the ENTIRE page as a clean, structured FORM "
    "TEMPLATE — not as a flat stream of OCR text. Doctor handwriting is often "
    "messy, so read carefully.\n\n"
    "FORMAT RULES (plain text only — no markdown symbols like # or **):\n"
    "- Reproduce the document's own structure: title/letterhead centered at "
    "the top, then each printed section in the order it appears on the page.\n"
    "- Render section headings on their own line in UPPERCASE exactly as "
    "printed (e.g. GENERAL:, EXAMINATION:, TREATMENT PROPOSED:), followed by "
    "their contents indented by two spaces.\n"
    "- Render every printed field as 'Label : value'. If the field is filled "
    "in by hand, put the handwritten value after the label. If it is empty, "
    "write '(blank)'. Keep related fields grouped on one line when the form "
    "prints them side by side (e.g. 'Sex : M / F    Unit : (blank)    Ward : (blank)').\n"
    "- When the form offers printed options and one is circled/ticked/"
    "underlined, show all options and mark the chosen one, e.g. "
    "'Stage : Early / 2 / 3 / Late  ->  circled: Early'.\n"
    "- Keep tabular parts (e.g. T N M staging grids, date/order tables) "
    "aligned with spaces so rows and columns line up.\n"
    "- TWO-COLUMN PROGRESS NOTE LAYOUT: many Doctor's Order / progress-note "
    "forms have a vertical printed or handwritten dividing line splitting "
    "the body into a LEFT column (date in the far-left margin, then the "
    "clinical note — complaints, O/E findings, diagnosis, stage, lab "
    "requests) and a RIGHT column (medicines, doses, instructions, "
    "follow-up). EACH dated entry must be transcribed as ONE block, reading "
    "LEFT then RIGHT, with all content merged in reading order:\n"
    "    Date : <date>\n"
    "      Notes : <all left-column lines for that entry>\n"
    "      Orders : <all right-column lines for that entry>\n"
    "  NEVER skip the right column — it contains the prescriptions. "
    "NEVER output [LEFT] or [RIGHT] markers in the text — those are "
    "internal labels only. If there is NO dividing line, treat everything "
    "as a single column.\n"
    "- FORM-TABLE SECTIONS: some forms (e.g. Medical Case Record) have "
    "sections that are structured as a two-column table: the LEFT cell "
    "contains a printed label (e.g. COMPLAINTS AND DURATION, HISTORY OF "
    "PRESENT ILLNESS, PAST HISTORY, FAMILY HISTORY) and the RIGHT cell "
    "contains handwritten answers. Render these as:\n"
    "    COMPLAINTS AND DURATION : <handwritten text>\n"
    "    HISTORY OF PRESENT ILLNESS : <handwritten text>\n"
    "    PAST HISTORY : <handwritten text>\n"
    "    FAMILY HISTORY : <handwritten text>\n"
    "  The LEFT cell (printed label) is NOT handwritten — write it in plain "
    "uppercase. The RIGHT cell value IS handwritten.\n"
    "- Attach each handwritten note, arrow or annotation to the field or "
    "diagram it belongs to, on an indented line right below it, prefixed "
    "'(handwritten)'. Example: '(handwritten) Pallor + -> arrow to neck of "
    "body diagram'.\n"
    "- GROUPED ANNOTATIONS: forms often have ONE handwritten value next to a "
    "handwritten vertical line, brace or bracket that SPANS SEVERAL printed "
    "fields (e.g. 'ND' written beside a line covering three rows). The value "
    "IS the answer for EVERY field the line spans: write it as each spanned "
    "field's value — 'Label : <value>' with '(handwritten)' noted — and "
    "NEVER leave a spanned field as '(blank)'. Worked example — 'NO' beside "
    "a line spanning three rows:\n"
    "    Paranasal Sinuses : NO (handwritten, shared)\n"
    "    Thyroid : NO (handwritten, shared)\n"
    "    Chest, Spine : NO (handwritten, shared)\n"
    "  Judge the line's extent carefully: it starts level with the FIRST "
    "field of the group and ends level with the LAST; if an adjacent field "
    "in the same section would otherwise be blank and the line visually "
    "reaches its row, include it in the group.\n"
    "- BUT sharing applies ONLY when a drawn line/brace exists. A handwritten "
    "value with NO spanning line belongs to EXACTLY ONE field — the one it is "
    "written on or aligned with. NEVER copy it to a neighbouring or "
    "similar-looking field; those stay '(blank)'. Example: '2' written "
    "before 'Cms below Lt' means only 'Cms below Lt : 2' — the similar "
    "field 'Cms below Rt' above it stays '(blank)'.\n"
    "- SHORT HANDWRITTEN VALUES (e.g. 'ND' vs 'NO', 'NAD', '+/-'): both 'ND' "
    "(not done) and 'NO' are common on examination forms, so context cannot "
    "decide — only the strokes can. Zoom into the letter shapes: a "
    "handwritten 'D' has a straight vertical stem on the left with the bowl "
    "joining it at top and bottom; an 'O' is a closed round loop with no "
    "stem. If the second letter shows any straight vertical stroke, read "
    "'ND'; if it is a plain round loop, read 'NO'. Decide letter by letter, "
    "and append (?) only when the strokes are genuinely undecidable.\n"
    "- Stamps: one line '[STAMP: <text of the stamp>]'. Signatures: "
    "'[Signature: <name or illegible>]'. Diagrams/figures: one line "
    "'[DIAGRAM: <what it shows + any markings on it>]'.\n\n"
    "CONTENT RULES:\n"
    "- Transcribe each medicine line as written, then expand medical "
    "shorthand in parentheses. Examples: '1-0-1' → (morning and night), "
    "'TDS' → (three times a day), 'BD' → (twice a day), 'OD' → (once a day), "
    "'HS' → (at bedtime), 'SOS' → (when needed), 'PO' → (by mouth), "
    "'a.c.' → (before food), 'p.c.' → (after food).\n"
    "- ZERO SKIPPING: every word, every number, every abbreviation, every "
    "symbol visible on the page MUST appear in the output. Even faint, "
    "crossed-out or partially obscured text must be included — mark faint "
    "text with (faint) and crossed-out text with (struck). Never silently "
    "omit anything.\n"
    "- NEVER USE [illegible]. This word is FORBIDDEN. No matter how bad the "
    "handwriting, you MUST output your best guess for every word. Read "
    "stroke by stroke — identify each letter, pick the most likely "
    "character, and write the full word. If you are not certain, append (?) "
    "to that word, e.g. 'Gleevec(?)', 'Imatinib(?)', 'Nilotinib(?)'. "
    "A single (?) marker is enough — never refuse to guess. The human "
    "reviewer will correct any wrong guesses; a blank or [illegible] "
    "is never correctable and is always wrong.\n"
    "- MEDICAL CONTEXT GUESSING: use the diagnosis, drug class, "
    "abbreviation patterns and surrounding words to narrow down ambiguous "
    "letters. Common oncology drugs: Imatinib, Nilotinib, Dasatinib, "
    "Hydroxyurea, Gleevec, Tasigna. Common abbreviations: CBC, BCR-ABL, "
    "CML, CP, MMR, CMR, ECOG, OD, BD, TDS, NAD, ND, R/A, c/o, O/E, H/O, "
    "adv, Tab, Cap, Inj. When a word could match a known drug or "
    "abbreviation, prefer that reading.\n"
    "- Medicine names: always give a reading, mark uncertain with (?). "
    "Never omit a medicine line because it is hard to read.\n"
    "- Do not summarize, do not skip anything, do not add commentary. "
    "Output only the reconstructed form."
)

_READ_USER = (
    "Reconstruct page {page} of this medical document as a structured form "
    "template following the format rules: headings, 'Label : value' fields, "
    "grouped side-by-side fields, marked circled options, and handwritten "
    "annotations attached to their fields. Expand medical shorthand in "
    "parentheses. EXTRACT EVERY WORD visible on the page — guess uncertain "
    "words with (?) but NEVER write [illegible]. Leave nothing out."
)

_LAYOUT_SYSTEM = (
    "You are an expert medical document reader assisting a licensed "
    "pharmacy team with digitizing prescriptions and hospital forms the "
    "patient has already provided. You are reconstructing the visual "
    "layout of a medical document page as a self-contained HTML FRAGMENT "
    "so it can be shown next to the original image and look as close to "
    "it as possible.\n\n"
    "OUTPUT RULES:\n"
    "- Output ONLY the HTML fragment: no <html>, <head> or <body> tags, no "
    "<script> tags, no external resources (no images, fonts, stylesheets), "
    "no markdown fences. Your output must start with an HTML tag.\n"
    "- The host page provides base CSS. Use ONLY these class names instead "
    "of inventing styles:\n"
    "  * class=\"hw\" — wrap ALL handwritten text in it (it renders in a "
    "handwriting-style font).\n"
    "  * class=\"cut\" — ONLY for actual drawings/diagrams/sketches/"
    "figures (e.g. an anatomical body diagram, a hand-drawn chart) emit an "
    "image tag so the host can cut the real picture out of the uploaded "
    "page:\n"
    "      <img class=\"cut\" data-bbox=\"X,Y,W,H\" alt=\"[DIAGRAM: <what it shows>]\">\n"
    "    X,Y is the drawing's top-left corner and W,H its size, ALL as "
    "percentages (0-100) of the full page width/height, tightly around "
    "the drawing itself.\n"
    "    NEVER emit a cut image for stamps, seals, logos, signatures, "
    "letterheads, page photos or handwritten TEXT — those are not "
    "diagrams.\n"
    "  * class=\"stamp\" — text chips for stamps, seals, logos and "
    "signatures, placed roughly where they appear on the page: "
    "'[STAMP: <text>]', '[Signature: <name or illegible>]', "
    "'[LOGO: <description>]'.\n"
    "  * class=\"unc\" — wrap uncertain words in it, keeping the (?) suffix.\n"
    "- Inline styles only for alignment (text-align, width, display:flex, "
    "margins) and keep them minimal — never long repeated style strings. "
    "NEVER use position:absolute, position:fixed or negative margins: "
    "elements must stay in normal document flow and text must NEVER overlap "
    "other text. A margin date + note is a flex row (date in a narrow left "
    "cell, note beside it) or the date on its own line with the note below "
    "— never two elements occupying the same spot.\n\n"
    "LAYOUT RULES:\n"
    "- Reproduce the page's own visual structure top to bottom: letterhead/"
    "title centered when centered on the page, sections in page order with "
    "their headings, printed form fields as 'Label: value' pairs kept "
    "side-by-side when the form prints them side-by-side (use flex rows or "
    "inline spans).\n"
    "- Place every element at the same relative position as on the page: "
    "content on the page's right side goes in a right column (flex), "
    "top-right stamps at the top right, and so on.\n"
    "- Whatever sits at the BOTTOM of the page (signature of recorder, "
    "date line, footer) MUST be the LAST elements of your fragment, laid "
    "out as a bottom row — never in the middle of the document.\n"
    "- Render anything tabular (staging grids, medicine tables, date/order "
    "tables) as a real <table> with the same rows and columns.\n"
    "- FORM-TABLE SECTIONS: sections like COMPLAINTS AND DURATION, HISTORY "
    "OF PRESENT ILLNESS, PAST HISTORY, FAMILY HISTORY appear as a printed "
    "label on the left and a handwritten answer on the right, like a "
    "two-column table. Render them as a <table> with two columns: the left "
    "column contains the printed label (plain text, bold, no hw class) and "
    "the right column contains the handwritten answer (wrapped in "
    "class=\"hw\"). Example:\n"
    "  <table style=\"width:100%;border-collapse:collapse\">\n"
    "    <tr>\n"
    "      <td style=\"width:40%;font-weight:600;vertical-align:top;"
    "padding:2px 6px;border:1px solid #ccc\">COMPLAINTS AND DURATION</td>\n"
    "      <td style=\"vertical-align:top;padding:2px 6px;"
    "border:1px solid #ccc\"><span class=\"hw\">Distension &amp; discomfort "
    "in abdomen - 1 mo<br>Weakness +<br>No weight loss</span></td>\n"
    "    </tr>\n"
    "    <tr>\n"
    "      <td style=\"width:40%;font-weight:600;vertical-align:top;"
    "padding:2px 6px;border:1px solid #ccc\">HISTORY OF PRESENT ILLNESS"
    "</td>\n"
    "      <td style=\"vertical-align:top;padding:2px 6px;"
    "border:1px solid #ccc\"><span class=\"hw\">No H/O Fever | Vomiting"
    "</span></td>\n"
    "    </tr>\n"
    "  </table>\n"
    "  The printed label cell must NEVER use class=\"hw\". Only the "
    "handwritten answer cell uses class=\"hw\".\n"
    "- TWO-COLUMN PROGRESS NOTE LAYOUT: when the page body is split by a "
    "vertical dividing line into a LEFT column (date margin + clinical "
    "notes) and a RIGHT column (medicines/orders), render each dated entry "
    "as a flex row: a narrow left cell (date) + a wider centre cell (notes) "
    "+ a wider right cell (medicines). Use "
    "style=\"display:flex;gap:0.5em;margin-bottom:0.8em\" on the row div, "
    "style=\"min-width:4em;font-weight:600\" on the date cell, "
    "style=\"flex:1;border-left:1px solid #999;padding-left:0.4em\" on the "
    "notes cell, and "
    "style=\"flex:1;border-left:1px solid #999;padding-left:0.4em\" on the "
    "orders cell. NEVER omit the right (medicines) cell — it is as "
    "important as the left.\n"
    "- When printed options are circled/ticked, show all options and mark "
    "the chosen one.\n"
    "- Empty printed fields show '(blank)' after the label.\n"
    "- GROUPED ANNOTATIONS: when ONE handwritten value sits beside a "
    "handwritten vertical line, brace or bracket spanning SEVERAL printed "
    "fields, the value IS the answer for EVERY spanned field: show it (in "
    "class=\"hw\") after EACH spanned field's label and NEVER render a "
    "spanned field as '(blank)'. The line starts level with the FIRST "
    "field of the group and ends level with the LAST; include an adjacent "
    "otherwise-blank field when the line visually reaches its row. BUT "
    "sharing applies ONLY when a drawn line/brace exists: a handwritten "
    "value with NO spanning line belongs to EXACTLY ONE field — the one it "
    "is written on or aligned with — and must NEVER be copied to a "
    "neighbouring or similar-looking field (e.g. '2' before 'Cms below Lt' "
    "goes ONLY there; 'Cms below Rt' above it stays '(blank)').\n"
    "- SHORT HANDWRITTEN VALUES ('ND' vs 'NO' etc.): decide from the "
    "strokes, not context — a 'D' has a straight vertical stem with the "
    "bowl joining it, an 'O' is a plain closed loop with no stem. Append "
    "(?) only when genuinely undecidable.\n"
    "- DATES: progress notes usually carry a date in the left margin (e.g. "
    "6/8/18, 24/11/18 — day/month/year). A date contains ONLY digits and "
    "separators: read it digit by digit and never confuse 6 with C, 1 with "
    "l, 0 with O, 8 with B. If a character in a date position is not "
    "clearly a digit, choose the most likely digit and append (?).\n\n"
    "CONTENT RULES (same as transcription):\n"
    "- ZERO SKIPPING: every word visible on the page must appear in the "
    "output. Never silently omit anything — faint, partial or uncertain "
    "text is included with (faint) or (?) markers.\n"
    "- NEVER USE [illegible]. Always guess stroke by stroke and output your "
    "best reading with (?) on uncertain words. The human reviewer will "
    "correct wrong guesses; [illegible] is never correctable.\n"
    "- Expand medical shorthand in parentheses after the original, e.g. "
    "'TDS (three times a day)', '1-0-1 (morning and night)'.\n"
    "- Never redact, mask or anonymize personal data (patient names, "
    "hospital numbers, dates, doctor names) — transcribe verbatim; this "
    "is an authorized internal digitization workflow.\n"
    "- Do not summarize, skip content, or add commentary. Keep the output "
    "compact."
)

_LAYOUT_USER = (
    "Reconstruct page {page} of this document as a compact HTML fragment "
    "per the rules (host classes hw/cut/stamp/unc, real tables, minimal "
    "inline styles). ONLY actual drawings/diagrams get an "
    "<img class=\"cut\" data-bbox=\"X,Y,W,H\" alt=\"[DIAGRAM: ...]\"> with "
    "percentage coordinates; stamps, signatures and logos are class=\"stamp\" "
    "text chips, handwriting is class=\"hw\" text; bottom-of-page items "
    "(signature, date, footer) come last. When one handwritten value sits "
    "beside a vertical line/brace spanning several fields, that value is the "
    "answer for EVERY field the line spans — show it after each of them and "
    "leave none of them '(blank)'. Output HTML only."
)

# Combined per-page call: ONE reading of the image produces BOTH the plain
# transcription and the layout HTML. Halves image-input cost per page and
# guarantees the two views spell every word identically.
_RAW_DELIM = "===RAW TEXT==="
_LAYOUT_DELIM = "===LAYOUT HTML==="

_PAGE_SYSTEM = (
    "You produce TWO views of the SAME medical document page from ONE "
    "careful reading: (1) a plain-text form template, (2) an HTML layout "
    "fragment.\n\n"
    "OUTPUT CONTRACT — your whole response is exactly two sections, in this "
    "order, separated by these exact delimiter lines (no markdown fences, "
    "nothing before the first delimiter, nothing after the HTML):\n"
    f"{_RAW_DELIM}\n"
    "<the plain-text form template>\n"
    f"{_LAYOUT_DELIM}\n"
    "<the HTML fragment>\n\n"
    "CRITICAL CONSISTENCY RULE: both sections come from the SAME single "
    "reading. Every word, name, date, dosage and handwritten value MUST be "
    "spelled IDENTICALLY in both sections — the layout is a visual "
    "re-arrangement of the transcription, NEVER a second reading. Decide "
    "each uncertain word ONCE and reuse that exact reading (with its (?) "
    "marker) in both sections.\n\n"
    "────── RULES FOR THE RAW TEXT SECTION ──────\n"
    + _READ_SYSTEM +
    "\n\n────── RULES FOR THE LAYOUT HTML SECTION ──────\n"
    + _LAYOUT_SYSTEM +
    "\n\nNOTE: the per-section remarks like 'output only the reconstructed "
    "form' or 'your output must start with an HTML tag' apply WITHIN their "
    "own section; the overall response must follow the OUTPUT CONTRACT "
    "above (two delimited sections)."
)

_PAGE_USER = (
    "Read page {page} of this medical document ONCE, carefully, then output "
    f"the two sections per the contract: '{_RAW_DELIM}' followed by the "
    "structured plain-text form template, then "
    f"'{_LAYOUT_DELIM}' followed by the compact HTML fragment (host classes "
    "hw/cut/stamp/unc, real tables, minimal inline styles, no absolute "
    "positioning). Same spelling for every word in both sections. "
    "EXTRACT EVERY WORD on the page — guess uncertain words with (?) but "
    "NEVER write [illegible]. Nothing may be skipped or omitted."
)

_FIELDS_SYSTEM = (
    "You are an expert medical document reader assisting a licensed "
    "pharmacy team with digitizing prescriptions and hospital forms the "
    "patient has already provided. You extract structured data from ALL "
    "pages of one medical document (prescription / hospital form) and "
    "return ONLY a JSON object — no text before or after it.\n\n"
    "JSON SHAPE:\n"
    "{\n"
    '  "fields": [\n'
    "    {\n"
    '      "key": "patient_name",\n'
    '      "name": "Patient Name",\n'
    '      "value": "Ramesh K(?)",\n'
    '      "explanation": "The full name of the person the prescription is issued to.",\n'
    '      "business_meaning": "Primary identifier for dispensing, insurance claims and medication history lookup.",\n'
    '      "confidence": "medium"\n'
    "    }\n"
    "  ],\n"
    '  "medicines": [\n'
    "    {\n"
    '      "medicine": "Tab Augmentin 625",\n'
    '      "dosage": "625 mg",\n'
    '      "frequency": "1-0-1 (morning and night)",\n'
    '      "duration": "5 days",\n'
    '      "instructions": "p.c. (after food)"\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "RULES:\n"
    "- Include EVERY meaningful field present on the document: patient "
    "identity, clinician and facility details, dates, vitals, diagnosis, "
    "staging, investigations, follow-up, referral — not just the canonical "
    "list.\n"
    "- Use these canonical keys when the field matches one: patient_name, "
    "patient_age, patient_sex, doctor_name, doctor_registration, "
    "facility_name, prescription_date, diagnosis, weight, blood_pressure, "
    "allergies, follow_up_date, referral. Otherwise derive a snake_case key "
    "from the printed label.\n"
    "- 'value' is verbatim from the page, keeping the (?) uncertain-word "
    "convention; use \"(blank)\" when the label is printed but unfilled. "
    "Read handwriting with best effort — never leave a field out because "
    "it is handwritten.\n"
    "- NEVER redact, mask, anonymize or omit personal data. Patient names, "
    "hospital numbers, registration numbers, dates and doctor names must "
    "be transcribed VERBATIM as printed/handwritten — a masked value makes "
    "the record clinically unusable. This is an authorized internal "
    "digitization workflow, not a publication.\n"
    "- 'explanation' is a plain-language definition of what the field is. "
    "'business_meaning' says why it matters to pharmacy, clinical or "
    "billing workflows.\n"
    "- 'confidence' is high, medium or low for the reading of the value.\n"
    "- Expand shorthand inside medicine entries, e.g. frequency 'BD (twice "
    "a day)'.\n"
    "- If the document truly has no fields or no medicines, return empty "
    "arrays. Return ONLY the JSON object."
)

_FIELDS_USER = (
    "Extract the structured JSON for this {n}-page document. All pages are "
    "attached in order. Return only the JSON object."
)


class ExtractorError(RuntimeError):
    """Raised when extraction cannot proceed; message is user-displayable."""


def _strip_fences(text: str) -> str:
    """Remove a wrapping ```lang ... ``` fence if the model added one."""
    text = text.strip()
    m = re.match(r"^```[a-zA-Z]*\s*\n(.*)\n?```$", text, re.DOTALL)
    return m.group(1).strip() if m else text


def _parse_json_loose(text: str) -> dict:
    """Parse model output into a dict; tolerate fences and stray prose."""
    text = _strip_fences(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fall back to the first balanced {...} block in the output.
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    raise ValueError("model output contained no parseable JSON object")


def _sanitize_html(fragment: str) -> str:
    """Defense-in-depth scrub; the sandboxed iframe is the real boundary."""
    fragment = _strip_fences(fragment)
    fragment = re.sub(r"<script\b.*?</script\s*>", "", fragment,
                      flags=re.IGNORECASE | re.DOTALL)
    fragment = re.sub(r"<script\b[^>]*>", "", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"\son\w+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", "",
                      fragment, flags=re.IGNORECASE)

    # Overlap guard: absolute/fixed positioning and negative margins make
    # text render on top of other text — force everything into normal flow.
    def _defuse_style(m):
        style = m.group(2)
        style = re.sub(r"position\s*:\s*(absolute|fixed)", "position:static",
                       style, flags=re.IGNORECASE)
        style = re.sub(r"margin(?:-\w+)?\s*:\s*-[^;]*;?", "", style,
                       flags=re.IGNORECASE)
        return f"style={m.group(1)}{style}{m.group(1)}"

    fragment = re.sub(r"style\s*=\s*([\"'])(.*?)\1", _defuse_style,
                      fragment, flags=re.IGNORECASE | re.DOTALL)
    return fragment.strip()


# Crops embedded into the layout are capped to this long-edge size.
_CROP_MAX_EDGE = 500
_CROP_PAD_PCT = 2.0


def _embed_region_crops(fragment: str, page_image_bytes: bytes) -> str:
    """
    Replace <img class="cut" data-bbox="x,y,w,h" ...> placeholders in the
    layout fragment with real crops of the page image as data: URIs.
    Coordinates are percentages of the page. Per-crop failures leave the
    tag without a src (its alt text shows instead). Never raises.
    """
    if not fragment or "data-bbox" not in fragment:
        return fragment
    try:
        img = Image.open(io.BytesIO(page_image_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        pw, ph = img.size
    except Exception as e:
        print(f"[WARNING] could not open page image for crops: {e}")
        return fragment

    def _crop_tag(match):
        tag = match.group(0)
        try:
            x, y, w, h = [float(v) for v in match.group(1).split(",")]
            left = max(0, int((x - _CROP_PAD_PCT) / 100 * pw))
            top = max(0, int((y - _CROP_PAD_PCT) / 100 * ph))
            right = min(pw, int((x + w + _CROP_PAD_PCT) / 100 * pw))
            bottom = min(ph, int((y + h + _CROP_PAD_PCT) / 100 * ph))
            if right - left < 4 or bottom - top < 4:
                return tag
            crop = img.crop((left, top, right, bottom))
            scale = _CROP_MAX_EDGE / max(crop.size)
            if scale < 1:
                crop = crop.resize(
                    (max(1, int(crop.size[0] * scale)),
                     max(1, int(crop.size[1] * scale))))
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            # Drop any src the model invented, then inject the real one.
            cleaned = re.sub(r'\ssrc\s*=\s*("[^"]*"|\'[^\']*\')', "", tag,
                             flags=re.IGNORECASE)
            return cleaned[:-1].rstrip("/").rstrip() + \
                f' src="data:image/jpeg;base64,{b64}">'
        except Exception as e:
            print(f"[WARNING] region crop failed: {e}")
            return tag

    return re.sub(r'<img\b[^>]*data-bbox\s*=\s*["\']([\d.,\s]+)["\'][^>]*>',
                  _crop_tag, fragment, flags=re.IGNORECASE)


# Reinforcement appended AFTER the long shared prompts: the local 7B model
# tends to skip the page header and misuse the handwriting class — recency
# at the end of the system prompt is what it actually follows.
_LOCAL_RULES = (
    "\n\nCRITICAL REMINDERS — verify ALL FIVE before answering:\n"
    "1. START with the printed LETTERHEAD at the very top of the page, "
    "centered: institute/hospital name, centre line, address line, and the "
    "printed form title (e.g. FOLLOW-UP - NOTES, DOCTORS ORDER, MEDICAL "
    "CASE RECORD). Include it even when it is faded, at an angle, or partly "
    "cropped in the photo — the top of the page must NEVER be skipped.\n"
    "2. PRINTED TEXT IS PLAIN, HANDWRITING IS hw. ALL text that is "
    "printed/typed on the form — headings, field labels, section titles, "
    "symptom checklists, option lists, table headers, letterhead, footer — "
    "must be plain text WITHOUT the hw class. ONLY actual pen-ink "
    "handwriting (patient name filled in, doctor's notes, written values) "
    "uses the hw class. Never wrap a printed label in hw.\n"
    "3. TWO-COLUMN LAYOUT: if the page has a vertical dividing line "
    "splitting the body into a left column (clinical notes) and a right "
    "column (medicines/orders), capture BOTH columns for every dated entry. "
    "Do NOT output [LEFT] or [RIGHT] markers — merge the content cleanly "
    "into Notes and Orders sub-lines. Scan the RIGHT side of the divider "
    "for medicine names, doses and instructions.\n"
    "4. FORM-TABLE SECTIONS (COMPLAINTS AND DURATION, HISTORY OF PRESENT "
    "ILLNESS, PAST HISTORY, FAMILY HISTORY): the printed label on the left "
    "is NOT handwritten — render it as plain bold text. The handwritten "
    "answer on the right IS hw. In the layout HTML render these as a "
    "<table> with two columns: bold plain label | hw answer.\n"
    "5. [illegible] IS FORBIDDEN. You must guess every word. Look at each "
    "stroke, pick the most likely letter, write the whole word, and add (?) "
    "if uncertain. Use medical context (CML, Imatinib, Nilotinib, CBC, "
    "BCR-ABL, NAD, ECOG, R/A, c/o, O/E, H/O, adv, Tab, OD, BD, TDS) to "
    "resolve ambiguous letters. A wrong guess with (?) is always better "
    "than [illegible] — the human reviewer can fix a guess but cannot fix "
    "a blank.\n"
)

# Same two rules as imperative output-order commands, appended to the USER
# message — the last tokens before generation, which the 7B follows best.
_LOCAL_USER_RULES = (
    "\n\nBEFORE YOU WRITE, apply these five commands:\n"
    "- FIRST LINES: your very first output (in both sections) must be the "
    "topmost PRINTED text — the institute letterhead and form title. Look "
    "at the top edge of the image now and transcribe it first.\n"
    "- PRINTED = PLAIN, HANDWRITTEN = hw. Every letter, word or line "
    "printed on the form (headings, labels, checklists, option lists, "
    "footers) is plain text — NO hw class. Only actual pen-ink writing "
    "(filled-in names, doctor notes, written values) gets the hw class.\n"
    "- NO [LEFT]/[RIGHT] MARKERS in the raw text output. Merge both columns "
    "into Notes / Orders sub-lines per dated entry.\n"
    "- TABLE CHECK: for sections like COMPLAINTS AND DURATION / HISTORY OF "
    "PRESENT ILLNESS / PAST HISTORY / FAMILY HISTORY — the label is plain "
    "bold, the handwritten answer is hw. In the HTML use a <table> with two "
    "columns: printed label (bold, no hw) | handwritten answer (hw).\n"
    "- NO [illegible] EVER. Guess every word stroke by stroke. Uncertain "
    "words get (?). Every piece of text on the page must appear in your "
    "output — nothing skipped, nothing replaced with [illegible]."
)


def _check_ollama():
    """Verify the Ollama server is reachable and the model is pulled."""
    try:
        res = requests.get(f"{_HOST}/api/tags", timeout=10)
        res.raise_for_status()
    except Exception as e:
        raise ExtractorError(
            f"Ollama server unreachable at {_HOST} — is Ollama running? ({e})"
        )
    names = [m.get("name", "") for m in res.json().get("models", [])]
    if not any(n == _MODEL or n.startswith(_MODEL + ":") for n in names):
        raise ExtractorError(
            f"Model '{_MODEL}' is not pulled in Ollama. "
            f"Run:  ollama pull {_MODEL}"
        )


def _ollama_chat(system_prompt: str, user_text: str, images_b64: list,
                 max_tokens: int, json_mode: bool = False) -> tuple:
    """One Ollama /api/chat call. Returns (content, done_reason).
    Raises ExtractorError on transport/HTTP errors."""
    payload = {
        "model": _MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text, "images": images_b64},
        ],
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,
            "num_ctx": _NUM_CTX,
        },
    }
    if json_mode:
        payload["format"] = "json"
    try:
        res = requests.post(f"{_HOST}/api/chat", json=payload, timeout=_TIMEOUT)
    except Exception as e:
        raise ExtractorError(f"Ollama request failed: {e}")
    if res.status_code != 200:
        raise ExtractorError(f"Ollama error {res.status_code}: {res.text[:400]}")
    body = res.json()
    content = (body.get("message", {}).get("content") or "").strip()
    return content, body.get("done_reason", "")


def read_prescription_image(b64_image: str, page_num: int, mime: str = "image/jpeg") -> str:
    """
    Send one prescription image to the vision model and return its text.
    Retries on API errors AND empty responses.
    """
    _check_ollama()

    # Few-shot spelling learning from human raw-text corrections, appended
    # at call time so new corrections apply without a restart.
    system_prompt = _READ_SYSTEM + _LOCAL_RULES
    try:
        _examples = feedback_store.get_spelling_examples_for_prompt()
        if _examples:
            system_prompt = _READ_SYSTEM + _LOCAL_RULES + "\n\n" + _examples
            print(f"[INFO] Page {page_num}: injected learned-spelling block:\n{_examples}")
    except Exception as e:
        print(f"[WARNING] Spelling-corrections lookup skipped (Postgres unavailable?): {e}")

    last_err = None
    for attempt in range(4):
        try:
            raw, _reason = _ollama_chat(
                system_prompt,
                _READ_USER.format(page=page_num) + _LOCAL_USER_RULES,
                [b64_image],
                _VISION_MAX_TOKENS,
            )
            if raw:
                return raw
            print(f"[WARNING] Page {page_num} vision read empty, attempt {attempt + 1}/4")
            if attempt < 3:
                time.sleep(3)
        except Exception as e:
            last_err = e
            print(f"[WARNING] Page {page_num} vision read error: {e}, attempt {attempt + 1}/4")
            if attempt < 3:
                time.sleep(5)
    if last_err is not None:
        raise ExtractorError(f"Ollama vision request failed: {last_err}")
    return ""


# Combined call gets extra completion budget: two outputs share it.
_PAGE_MAX_TOKENS = 16000


def _parse_page_output(raw: str) -> tuple:
    """
    Split the combined response into (text, layout_html). Tolerant of
    whitespace/case around the delimiters. Either part may come back None
    when its section is missing or empty.
    """
    raw_m = re.search(r"===\s*RAW\s*TEXT\s*===", raw, re.IGNORECASE)
    lay_m = re.search(r"===\s*LAYOUT\s*HTML\s*===", raw, re.IGNORECASE)
    if raw_m and lay_m and lay_m.start() >= raw_m.end():
        text = raw[raw_m.end():lay_m.start()].strip()
        layout = raw[lay_m.end():].strip()
    elif raw_m:
        text = raw[raw_m.end():].strip()
        layout = None
    elif lay_m:
        text = raw[:lay_m.start()].strip()
        layout = raw[lay_m.end():].strip()
    else:
        # No delimiters at all — a plain-text-looking response is still a
        # usable transcription; an HTML-looking one is not a transcription.
        text = None if raw.lstrip().startswith("<") else raw.strip()
        layout = None
    return (text or None), (layout or None)


def _unhw_letterhead(text: str, layout: str) -> str:
    """The letterhead is printed, but the local model often tags it hw.
    The transcription's opening block (lines before the first blank line)
    IS the letterhead — strip the hw class from those exact lines in the
    layout and render them centered/bold like the printed original."""
    if not text or not layout:
        return layout
    head = []
    for line in text.splitlines():
        if not line.strip():
            break
        head.append(line.strip())
    for line in head[:5]:
        pattern = re.compile(
            r'<(\w+)([^>]*)\bclass="hw"([^>]*)>(\s*' + re.escape(line) + r'\s*)</\1>'
        )
        layout = pattern.sub(
            r'<\1\2style="text-align:center;font-weight:600"\3>\4</\1>',
            layout, count=1)
    return layout


def read_page(b64_image: str, page_num: int, mime: str = "image/jpeg") -> tuple:
    """
    ONE vision call producing both views of a page from a single reading.
    Returns (text, layout_html|None, layout_error|None). The text is
    mandatory: if the combined call cannot produce it, falls back to the
    text-only call; raises ExtractorError only when that also fails.
    """
    _check_ollama()

    # Few-shot spelling learning from human raw-text corrections, appended
    # once at call time so new corrections apply without a restart.
    system_prompt = _PAGE_SYSTEM + _LOCAL_RULES
    try:
        _examples = feedback_store.get_spelling_examples_for_prompt()
        if _examples:
            system_prompt = _PAGE_SYSTEM + _LOCAL_RULES + "\n\n" + _examples
            print(f"[INFO] Page {page_num}: injected learned-spelling block:\n{_examples}")
    except Exception as e:
        print(f"[WARNING] Spelling-corrections lookup skipped (DB unavailable?): {e}")

    last_err = None
    compact_nudge = ""
    for attempt in range(4):
        try:
            raw, done_reason = _ollama_chat(
                system_prompt,
                _PAGE_USER.format(page=page_num) + _LOCAL_USER_RULES + compact_nudge,
                [b64_image],
                _PAGE_MAX_TOKENS,
            )
            text, layout = _parse_page_output(raw) if raw else (None, None)

            if done_reason == "length" and not (text and layout):
                print(f"[WARNING] Page {page_num} combined output truncated, attempt {attempt + 1}/4")
                compact_nudge = (
                    " Your previous output was cut off — be MORE COMPACT: "
                    "shorter inline styles in the HTML, no repetition."
                )
                if attempt < 3:
                    time.sleep(3)
                    continue

            if text:
                if layout:
                    layout = _unhw_letterhead(text, layout)
                    return text, _sanitize_html(layout), None
                return text, None, "Layout section missing from combined output."
            print(f"[WARNING] Page {page_num} combined read empty/unusable, attempt {attempt + 1}/4")
            if attempt < 3:
                time.sleep(3)
        except Exception as e:
            last_err = e
            print(f"[WARNING] Page {page_num} combined read error: {e}, attempt {attempt + 1}/4")
            if attempt < 3:
                time.sleep(5)

    # Combined call could not produce a transcription — text is mandatory,
    # so fall back to the proven text-only call (raises on total failure).
    print(f"[WARNING] Page {page_num}: combined call failed ({last_err}); falling back to text-only call")
    text = read_prescription_image(b64_image, page_num, mime)
    return text, None, f"Layout unavailable: combined page read failed ({last_err or 'empty output'})."


# Structured-fields call attaches at most this many page images.
_FIELDS_MAX_PAGES = 5


def read_structured_fields(images: list) -> tuple:
    """
    One call for the whole document: images = [(page_num, b64, mime), ...].
    Returns ({"fields": [...], "medicines": [...]}, None) on success or
    ({}, error_message) — never raises.
    """
    try:
        _check_ollama()
    except ExtractorError as e:
        return {}, str(e)

    # Few-shot learning from human feedback: append past corrections to the
    # system prompt at call time so new corrections apply without a restart.
    system_prompt = _FIELDS_SYSTEM
    try:
        _examples = feedback_store.get_examples_for_prompt()
        if _examples:
            system_prompt = _FIELDS_SYSTEM + "\n\n" + _examples
            print("[INFO] Injected learned-corrections block:\n" + _examples)
    except Exception as e:
        print(f"[WARNING] Corrections lookup skipped (Postgres unavailable?): {e}")

    images = images[:_FIELDS_MAX_PAGES]
    images_b64 = [b64 for _, b64, _ in images]

    last_err = None
    for attempt in range(4):
        try:
            raw, _reason = _ollama_chat(
                system_prompt,
                _FIELDS_USER.format(n=len(images)),
                images_b64,
                _VISION_MAX_TOKENS,
                json_mode=True,
            )
            if not raw:
                print(f"[WARNING] Fields read empty, attempt {attempt + 1}/4")
                if attempt < 3:
                    time.sleep(3)
                continue
            parsed = _parse_json_loose(raw)
            fields = parsed.get("fields") or []
            medicines = parsed.get("medicines") or []
            if not isinstance(fields, list) or not isinstance(medicines, list):
                raise ValueError("JSON did not contain fields/medicines lists")
            return {"fields": fields, "medicines": medicines}, None
        except Exception as e:
            last_err = e
            print(f"[WARNING] Fields read error: {e}, attempt {attempt + 1}/4")
            if attempt < 3:
                time.sleep(5)
    return {}, f"Structured field extraction failed: {last_err}"


def _render_pdf_pages(data: bytes) -> list:
    """Render every PDF page to base64 JPEG. [(page_num, b64), ...]"""
    doc = pdfium.PdfDocument(data)
    try:
        images = []
        for i in range(len(doc)):
            bitmap = doc[i].render(scale=_SCALE, rotation=0)
            img = bitmap.to_pil()
            if img.mode in ("RGBA", "LA"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90, optimize=True)
            buf.seek(0)
            images.append((i + 1, base64.b64encode(buf.read()).decode("utf-8")))
        return images
    finally:
        doc.close()


def extract(data: bytes, ext: str) -> tuple:
    """
    Extract from an uploaded prescription. Returns (pages, extras, meta):
        pages  : [{"page", "text", "layout_html", "layout_error"}, ...]
        extras : {"fields": [...], "medicines": [...], "fields_error": str|None}
    Transcript failure raises ExtractorError (document fails, as before);
    layout/fields failures degrade to per-panel error strings only.
    """
    if ext == ".pdf":
        rendered = [(n, b64, "image/jpeg") for n, b64 in _render_pdf_pages(data)]
    else:
        b64 = base64.b64encode(data).decode("utf-8")
        rendered = [(1, b64, _MIME.get(ext, "image/jpeg"))]

    # ONE combined transcript+layout call per page (single reading — both
    # views agree and the image is paid for once), plus one fields call per
    # document, all concurrent.
    with ThreadPoolExecutor(max_workers=min(8, len(rendered) + 1)) as pool:
        page_futs = {
            n: pool.submit(read_page, b64, n, mime)
            for n, b64, mime in rendered
        }
        fields_fut = pool.submit(read_structured_fields, rendered)

        pages = []
        for n, b64, _ in rendered:
            text, layout_html, layout_error = page_futs[n].result()  # ExtractorError propagates
            if layout_html:
                # Cut the real stamps/signatures/diagrams out of the page
                # image and embed them where the model marked data-bbox.
                page_bytes = data if ext != ".pdf" else base64.b64decode(b64)
                layout_html = _embed_region_crops(layout_html, page_bytes)
            pages.append({
                "page": n,
                "text": text,
                "layout_html": layout_html,
                "layout_error": layout_error,
            })
        fields_data, fields_error = fields_fut.result()

    extras = {
        "fields": fields_data.get("fields", []),
        "medicines": fields_data.get("medicines", []),
        "fields_error": fields_error,
    }
    meta = {"engine": ENGINE_NAME, "source": "ollama-vision", "deployment": _MODEL}
    return pages, extras, meta
