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

_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip().rstrip("/")
_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b").strip()
_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "900"))
_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "32768"))
ENGINE_NAME = f"Offline VLM via Ollama ({_MODEL})"

# Render resolution for PDF pages sent to the vision model.
# 300 DPI gives much better quality for handwritten medical documents.
_RENDER_DPI = 300
_SCALE = _RENDER_DPI / 72

# Output budget for the text-only transcription / structured-fields calls.
_VISION_MAX_TOKENS = 20000

_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

_READ_SYSTEM = """You are an expert medical-document digitization assistant.
Your task is to faithfully transcribe ONE document page exactly as it appears.

CORE RULES

1. EXTRACT EVERYTHING
- Read the entire page from top-to-bottom and left-to-right.
- Extract all visible printed text, handwriting, numbers, symbols,
  annotations, stamps, signatures and markings.
- Do not skip any visible content.

2. NO HALLUCINATION
- Output only content physically visible on the page.
- Do not invent diagnoses, medications, names, dates or findings.
- Do not use information from previous pages or future pages.

3. HANDWRITING
- Never output [illegible].
- Always provide your best reading.
- If uncertain, append (?) to the uncertain word.
  Example: lymphadenopathy(?)

--------------------------------------
PAGE TYPE — CHOOSE THE RIGHT FORMAT
--------------------------------------

Decide the page type FIRST, then format the whole page accordingly.

A. TABULAR REPORT PAGE
   (lab / haematology / biochemistry / investigation results printed
   as a grid of rows and columns)
   - Output the report header (facility, patient details, dates)
     first as Label : Value lines.
   - Output the results as a pipe-separated table (rule 6).

B. PRESCRIPTION / CLINIC NOTE PAGE
   (mostly handwritten; no printed results grid)
   - NEVER force this content into a table.
   - Output the letterhead / header first (if present).
   - Then patient details as Label : Value lines (if present).
   - Then every prescription item or note on its own line, top to
     bottom, exactly in the order written on the page.

C. TWO-COLUMN FORM PAGE
   (printed checklist / form on one half of the page, handwritten
   clinical notes on the other half)
   - Extract BOTH halves completely — the printed half AND every
     handwritten line on the other half.
   - Output each handwritten section under its own printed heading
     (e.g. COMPLAINTS AND DURATION, HISTORY OF PRESENT ILLNESS,
     PAST HISTORY, GENERAL EXAMINATION), keeping every line.
   - Handwriting placed to the RIGHT of a heading on the same row
     (e.g. a value written beside "P.B.") belongs to that row —
     extract it too.

If you are NOT CERTAIN the page shows a printed results grid, treat it
as a note page (B) — do not create a table.

--------------------------------------
OUTPUT STRUCTURE
--------------------------------------

4. FORM FIELDS
Convert structured fields into:
  Label : Value

Examples:
  Name : Kavitha
  Age : 38
  Hospital No. : 10452/25

Preserve slashes, punctuation and formatting exactly.
(The names and numbers above are format examples only — never copy
them into your output; always read the actual values from the page.)

5. SECTION HEADINGS
Preserve printed section headings in UPPERCASE.
Example:
  GENERAL EXAMINATION
  PAST HISTORY
  FAMILY HISTORY

6. TABLES
Reproduce tables using pipe-separated columns.
Example:
  Test Name | Result | Reference Range
  Hb        | 12.8   | 11.0 - 14.0

Maintain row order and column alignment.

7. NARRATIVE TEXT
For paragraphs, clinic notes and histories:
- Preserve wording exactly.
- Output each note or sentence on its own line.

8. MULTIPLE GROUPS ON THE SAME ROW
Handwritten pages often contain SEVERAL separate groups of writing on
the SAME horizontal row — e.g. one group on the left, a gap, then a
second group written on the right (side-by-side columns of drugs,
doses, vitals or notes).
- Scan every row across its FULL width, from the left edge to the
  right edge, before moving down to the next row.
- After transcribing the left group, look at the middle and the right
  of the SAME row and transcribe any further writing found there.
- Never stop at the first group; never drop writing on the right side
  of a row.
- Output side-by-side groups either on one line separated by " | ",
  or as consecutive lines in left-to-right order.

9. FINAL COMPLETENESS SWEEP
Before finishing, look over the page once more for handwriting you
have not yet transcribed:
- page margins and corners
- the right half of every row
- the space between printed sections
- below the last printed section
- consecutive handwritten lines under one heading — count them on the
  page and make sure the SAME number of lines appears in your output
  (do not skip alternate lines).
Add anything found, in its correct position.

--------------------------------------
CIRCLE-IF-POSITIVE CHECKLISTS
--------------------------------------

IMPORTANT: Only apply these rules when the page contains a section
explicitly labelled "(Circle If Positive)" with printed symptom/condition
lists. Do NOT add these sections to pages that do not have this label
(clinic prescriptions, haematology reports, follow-up notes, etc.).

When present, under headings such as:
  GENERAL
  G.I. TRACT
  ENT (INCLUDING ORAL CAVITY)
  BREAST
  G.U. TRACT
  MUSCULO-SKELETAL SYSTEM
  PAST HISTORY
  FAMILY HISTORY

Rules:
- Extract ALL printed symptoms/items.
- Determine whether each item was selected by the clinician.
- Selection may appear as: circle, tick/check mark, underline,
  cross mark, highlight, or obvious handwritten selection.
- Mark (Circled) ONLY when a drawn pen mark clearly encloses,
  ticks, or underlines that EXACT item.
- If a mark is ambiguous, faint, touches several items, or you are
  not sure — do NOT mark any item. A false (Circled) is worse than
  a missed one.
- Never mark an item merely because a neighbouring item is marked,
  and never infer selection from the diagnosis or other pages.
- Commas, print artifacts, shadows and paper folds are NOT selection
  marks.
- For selected items append: (Circled)
- For unselected items: output without any marker.

Example:
  GENERAL :
  Fatigue (Circled), Weight loss, Chills, Unexplained fever
  FAMILY HISTORY :
  Cancer, Tuberculosis (Circled), Diabetes

--------------------------------------
SPECIAL SYMBOLS
--------------------------------------

  Circled Plus  -> (+)
  Circled Minus -> (-)
  Circled Left  -> (L)
  Circled Right -> (R)

Example:
  Pallor : (+)
  Icterus : (-)
  Nodes : (+) ALN

Never convert these symbols into @ or other characters.

--------------------------------------
TNM / STAGE GRIDS
--------------------------------------

Output only marked values.
Examples:
  STAGE : T2, N0, M0
If empty:
  STAGE : (blank)

--------------------------------------
DOCUMENT NUMBERS
--------------------------------------

Preserve all identifiers exactly.
  10452/25 must remain 10452/25
Never merge digits or remove slashes.

--------------------------------------
STAMPS / SIGNATURES / DIAGRAMS / IMAGES
--------------------------------------

  Stamps:       [STAMP: text]
  Signatures:   [SIGNATURE: text]
  Diagrams:     [DIAGRAM: description and markings]
  Photos/Logos: [IMAGE: short description]

Every picture printed or pasted on the page (photograph, hospital logo,
graphic seal, barcode/QR code, scanned image) must be recorded with an
[IMAGE: ...] marker at its position in the reading order.
"""

_READ_USER = (
    "Extract the COMPLETE content of page {page} from this single image.\n\n"
    "Read the entire page from top-to-bottom and left-to-right. "
    "Capture every visible printed item, handwritten note, number, date, "
    "symbol, form field, table cell, stamp, annotation, and marking.\n\n"
    "Formatting rules:\n"
    "- Form fields → Label : Value\n"
    "- Section headings → Preserve exactly as shown\n"
    "- Tabular report pages (lab/investigation results printed as a grid) → "
    "pipe-separated table\n"
    "- Prescription / clinic-note pages → header first, then patient details "
    "as Label : Value, then each item on its own line — NEVER as a table\n"
    "- Paragraphs and notes → Keep each statement on its own line\n"
    "- Rows with several handwritten groups side-by-side → scan the FULL row "
    "width and transcribe every group, left to right\n\n"
    "Circle-If-Positive checklists (ONLY when the label is printed on this page):\n"
    "- Extract ALL printed symptoms/items under each checklist section.\n"
    "- Detect clinician selections including circles, ticks, checks, "
    "underlines, highlights, crosses, or other clear selection marks.\n"
    "- Append '(Circled)' to selected items.\n"
    "- Leave unselected items unchanged.\n"
    "- Do not infer selections without visual evidence.\n"
    "- If a mark is ambiguous or you are unsure, leave the item "
    "UNMARKED — a false '(Circled)' is worse than a missed one.\n"
    "- Do NOT add checklist sections to pages that do not have this label.\n\n"
    "Handwriting:\n"
    "- Never output [illegible].\n"
    "- Always provide your best reading.\n"
    "- Mark uncertain words with '(?)'.\n\n"
    "Accuracy rules:\n"
    "- Preserve dates, registration numbers, hospital numbers, and identifiers exactly.\n"
    "- Do not merge numbers or remove slashes.\n"
    "- Do not add information that is not visible.\n"
    "- Do not use information from other pages.\n"
    "- Output only what appears on this page."
)

_LAYOUT_SYSTEM = """You are an expert medical document layout reconstruction assistant.
Your task is to convert the provided medical document page into a
self-contained HTML fragment that visually resembles the original page.

OUTPUT RULES
- Output ONLY valid HTML.
- Do not output markdown.
- Do not output explanations.
- Start directly with an HTML element.
- Do not use <html>, <head>, <body>, <script>, or external resources.

AVAILABLE CLASSES

  hw     Handwritten content.
  stamp  Text-only stamps, seals, and signatures.
  unc    Uncertain text.
  cut    ANY pictorial region: photos, logos, printed diagrams,
         hand-drawn sketches, graphic seals, barcodes, QR codes.

Examples:
  <span class="hw">Left breast lump x 6 months</span>
  <span class="stamp">[STAMP: KIDWAI MEMORIAL INSTITUTE]</span>
  <span class="unc">lymphadenopathy(?)</span>
  <img class="cut"
       data-bbox="10,20,30,25"
       alt="[DIAGRAM: breast lesion map]">

------------------------------------
LAYOUT FIDELITY
------------------------------------

Preserve the page structure as closely as possible.
- Center centered content.
- Preserve single-column and multi-column layouts.
- Keep section order unchanged.
- Preserve relative grouping of nearby items.
- Keep footer elements at the bottom.
- Preserve visual hierarchy.

------------------------------------
DOCUMENT STRUCTURE
------------------------------------

Headers          Use heading tags.
Labels and values Use label:value pairs.
Sections         Use section containers.
Tables           Use real HTML tables.
Checklist blocks Use lists or paragraph blocks.
Narrative notes  Use paragraphs.

Use a real HTML <table> ONLY when the page itself shows a genuine
printed grid of rows and columns (lab / investigation reports).
Prescription pages and handwritten clinic notes must NOT be converted
into tables — render them as a header, label:value lines, and
line-by-line notes in the order written on the page.

------------------------------------
HANDWRITING
------------------------------------

Wrap handwritten content in:
  <span class="hw">...</span>

Do not convert handwriting into printed text.

UNCERTAIN WORDS: every word that carries the (?) marker in the raw
text MUST be wrapped in its own <span class="unc">word(?)</span> in
the HTML — wrap ONLY the uncertain word(s), never the whole line,
and keep the (?) inside the span.

------------------------------------
CHECKLISTS
------------------------------------

For '(Circle If Positive)' sections (only when present on the page):
- Preserve the section heading.
- Include all checklist items present in the extracted text.
- If an item is selected, append '(Circled)'.
- If not selected, leave unchanged.

Example:
  <div>
    <strong>GENERAL:</strong>
    Fatigue (Circled), Weight loss, Chills
  </div>

------------------------------------
SPECIAL CONTENT
------------------------------------

  Stamps:     <span class="stamp">[STAMP: text]</span>
  Signatures: <span class="stamp">[SIGNATURE: name]</span>
  Pictures:   <img class="cut" data-bbox="X,Y,W,H" alt="[IMAGE: description]">

EVERY picture on the page (photo, logo, printed diagram, hand-drawn
sketch, graphic seal, barcode, QR code) MUST be output as exactly ONE
img.cut tag, placed at its position in the reading order.
- data-bbox is REQUIRED. X,Y = top-left corner, W,H = width,height —
  all four are PERCENTAGES (0-100) of the full page.
  Example: a hospital logo in the top-left tenth of the page
  -> data-bbox="2,1,12,8".
- Never redraw a picture as text or ASCII art. Never omit data-bbox.
- Text-only rubber stamps stay <span class="stamp">; use img.cut when
  the stamp/seal is a graphic image.

------------------------------------
ACCURACY RULES
------------------------------------

- Never omit visible content.
- Never hallucinate content.
- Preserve dates and hospital numbers exactly.
- Preserve slashes (10452/25 remains 10452/25).
- Use (?) for uncertain words.
- Never use [illegible].
"""

_LAYOUT_USER = (
    "Reconstruct page {page} as a compact HTML fragment that closely matches "
    "the visual structure of the original document.\n\n"
    "Requirements:\n"
    "- Output valid HTML only.\n"
    "- Preserve the original reading order and page layout.\n"
    "- Maintain headers, sections, columns, tables, form fields, checklists, "
    "stamps, signatures, diagrams, and notes in their approximate positions.\n"
    "- Use real HTML tables for tabular content.\n"
    "- Use minimal inline styles only when required for alignment.\n"
    "- Keep all content in normal document flow; never overlap elements.\n\n"
    "Available classes:\n"
    "- class=\"hw\" for handwritten text.\n"
    "- class=\"stamp\" for stamps, seals, logos, and signatures.\n"
    "- class=\"unc\" for uncertain text containing '(?)' — wrap each "
    "uncertain word in its own <span class=\"unc\">.\n"
    "- class=\"cut\" for every pictorial region: photos, logos, diagrams, "
    "sketches, charts, figures, graphic seals, barcodes, QR codes.\n\n"
    "For every picture (photo, logo, diagram, seal, barcode):\n"
    "<img class=\"cut\" data-bbox=\"X,Y,W,H\" alt=\"[IMAGE: description]\">\n"
    "data-bbox is REQUIRED — X,Y,W,H are percentage coordinates (0-100) "
    "relative to the page, placed where the picture sits in reading order.\n\n"
    "Checklist sections:\n"
    "- Preserve all checklist items present in the extracted content.\n"
    "- Append '(Circled)' to selected items.\n"
    "- Preserve the original section heading.\n\n"
    "Grouped handwritten annotations:\n"
    "- If a handwritten value is connected by a visible brace, bracket, or "
    "vertical line spanning multiple fields, repeat that value for every "
    "field covered by the span.\n"
    "- Otherwise assign handwritten values only to their directly associated field.\n\n"
    "Bottom-of-page content such as signatures, dates, approvals, and "
    "footers must appear last in the HTML fragment.\n\n"
    "Return only the HTML fragment."
)

# Combined per-page call delimiters
_RAW_DELIM = "===RAW TEXT==="
_LAYOUT_DELIM = "===LAYOUT HTML==="

_PAGE_SYSTEM = (
    "You perform ONE careful visual reading of a single medical document page "
    "and produce TWO synchronized representations of that page:\n\n"
    "1. RAW TEXT VIEW\n"
    "   A complete transcription preserving all visible content.\n\n"
    "2. HTML LAYOUT VIEW\n"
    "   A visual reconstruction of the SAME content as an HTML fragment.\n\n"
    "============================================================\n"
    "OUTPUT CONTRACT\n"
    "============================================================\n"
    "Your entire response MUST contain exactly TWO sections in the "
    "following order:\n\n"
    f"{_RAW_DELIM}\n"
    "<raw text content>\n\n"
    f"{_LAYOUT_DELIM}\n"
    "<html fragment>\n\n"
    "Do not output anything before the first delimiter.\n"
    "Do not output anything after the HTML fragment.\n"
    "Do not use markdown code fences.\n\n"
    "============================================================\n"
    "CRITICAL CONSISTENCY RULE\n"
    "============================================================\n"
    "Perform only ONE reading of the page.\n\n"
    "The HTML section MUST be generated from the exact content extracted "
    "for the RAW section.\n\n"
    "Every word, name, date, number, dosage, identifier, symbol, "
    "hospital number, handwritten value, uncertain reading '(?)', "
    "and circled selection MUST appear identically in both sections.\n\n"
    "Never perform a second OCR pass for the HTML section.\n"
    "Never reinterpret uncertain text differently between sections.\n"
    "Never add, remove, expand, summarize, normalize, or correct content "
    "between the two sections.\n\n"
    "The RAW section defines the content.\n"
    "The HTML section defines only the visual structure.\n\n"
    "============================================================\n"
    "RAW TEXT SECTION RULES\n"
    "============================================================\n\n"
    + _READ_SYSTEM +
    "\n\n"
    "============================================================\n"
    "HTML LAYOUT SECTION RULES\n"
    "============================================================\n\n"
    + _LAYOUT_SYSTEM +
    "\n\n"
    "The RAW rules apply only inside the RAW section.\n"
    "The LAYOUT rules apply only inside the HTML section.\n"
    "The global OUTPUT CONTRACT and CONSISTENCY RULE apply to both sections."
)

_PAGE_USER = (
    "You are viewing exactly ONE document image: page {page}.\n\n"
    "Perform ONE careful visual reading of this page only.\n"
    "Do not use information from previous pages, later pages, memory, "
    "or previous extraction results.\n\n"
    "After completing that single reading, produce the two required "
    "sections exactly as defined in the output contract:\n\n"
    f"1. {_RAW_DELIM}\n"
    "- Complete raw transcription of the page.\n"
    "- Include every visible printed item, handwritten note, number, "
    "date, symbol, annotation, stamp, signature, checklist item, and label.\n\n"
    f"2. {_LAYOUT_DELIM}\n"
    "- HTML reconstruction of the SAME content.\n"
    "- Preserve the page structure, grouping, tables, columns, sections, "
    "checklists, form fields, and relative layout.\n\n"
    "Consistency requirements:\n"
    "- Read each word only once.\n"
    "- Use exactly the same spelling, punctuation, numbers, dates, "
    "identifiers, and uncertainty markers in both sections.\n"
    "- Never reinterpret or re-read text while generating the HTML.\n"
    "- The HTML section must be a visual representation of the raw section, "
    "not a second extraction.\n\n"
    "Accuracy requirements:\n"
    "- Extract every visible item.\n"
    "- Do not omit content.\n"
    "- Do not invent content.\n"
    "- Do not summarize.\n"
    "- Never output [illegible].\n"
    "- If uncertain, provide your best reading and append '(?)'.\n\n"
    "Formatting requirements:\n"
    "- Tabular report pages (printed grid of results) → table format in both "
    "sections (pipe table in RAW, <table> in HTML).\n"
    "- Prescription / clinic-note pages → header first, then details, then "
    "line-by-line items — NEVER forced into a table in either section.\n"
    "- Scan each handwritten row across its FULL width — transcribe every "
    "group on the row (left, middle, right), not just the first group.\n\n"
    "Return only the two contracted sections."
)

_FIELDS_SYSTEM = """You are an expert medical document information extraction assistant.
Your task is to extract structured data from all pages of a medical document
(prescription, hospital form, investigation report, discharge summary, or
clinical record) and return ONLY a valid JSON object.

OUTPUT FORMAT
{
  "fields": [
    {
      "key": "patient_name",
      "name": "Patient Name",
      "value": "Ramesh K(?)",
      "explanation": "Full name of the patient.",
      "business_meaning": "Primary identifier for clinical, dispensing and billing workflows.",
      "confidence": "medium"
    }
  ],
  "medicines": [
    {
      "medicine": "Tab Augmentin 625",
      "dosage": "625 mg",
      "frequency": "1-0-1 (morning and night)",
      "duration": "5 days",
      "instructions": "After food"
    }
  ]
}

RULES

1. OUTPUT JSON ONLY
- Return a single valid JSON object.
- No markdown. No explanations outside JSON. No comments.

2. EXTRACT ALL MEANINGFUL FIELDS
Include all clinically or administratively relevant fields, including:
- Patient details, hospital numbers, registration numbers, case numbers
- Admission details, clinician details, facility details, dates
- Diagnoses, complaints, history, investigations, findings, vitals
- Measurements, staging information, procedures
- Follow-up information, referrals, allergies, treatment plans
Do not restrict extraction to predefined fields.

3. FIELD KEYS
Use these canonical keys whenever applicable:
  patient_name, patient_age, patient_sex, patient_id
  hospital_number, registration_number
  doctor_name, doctor_registration
  facility_name, prescription_date
  diagnosis, weight, height, blood_pressure
  pulse, temperature, allergies, follow_up_date, referral
For other fields: create a snake_case key from the printed label.
Examples: "family_history", "chief_complaints", "stage", "hospital_no"

4. FIELD VALUES
- Preserve values exactly as written.
- Preserve dates, slashes, punctuation and identifiers.
- Preserve uncertainty markers "(?)" .
- Never normalize or rewrite values.
- If a printed field exists but is not filled: "value": "(blank)"

5. HANDWRITING
- Extract handwritten values with the same priority as printed values.
- Never omit a field because it is handwritten.
- Never use "[illegible]". Use "(?)" when uncertain.

6. CONFIDENCE
Allowed values: "high", "medium", "low"
Confidence refers to reading certainty of the extracted value.

7. MEDICINES
Extract every medication mentioned.
Each medicine object may contain:
  {"medicine": "", "dosage": "", "frequency": "", "duration": "", "instructions": ""}
Preserve prescription wording. Expand common shorthand when possible:
  BD -> twice daily,  TDS -> three times daily
  OD -> once daily,   HS -> at bedtime
Keep the original term visible.

8. PERSONAL DATA
Never redact, anonymize, mask, or remove patient names, hospital numbers,
registration numbers, dates, or doctor names. Transcribe exactly as written.

9. EMPTY DOCUMENTS
If no fields are found: {"fields": [], "medicines": []}

Return ONLY the JSON object.
"""

_FIELDS_USER = (
    "Extract structured data from this {n}-page medical document.\n\n"
    "Each image represents a different page and includes its page number.\n\n"
    "Process ALL pages together and return a SINGLE JSON object containing "
    "all extracted fields and medicines.\n\n"
    "Rules:\n"
    "- Extract information from every page.\n"
    "- Preserve the exact value as it appears on the page where it is found.\n"
    "- Do not combine, merge, infer, or synthesize values across pages.\n"
    "- Do not copy a value from one page to another.\n"
    "- If the same field appears on multiple pages, include the most complete "
    "visible value and preserve it exactly as written.\n"
    "- Extract handwritten and printed content equally.\n"
    "- Preserve dates, identifiers, hospital numbers, registration numbers, "
    "and uncertainty markers '(?)' exactly.\n"
    "- Include all medicines mentioned anywhere in the document.\n"
    "- Include all clinically and administratively meaningful fields.\n"
    "- Use '(blank)' only when a printed field exists but is unfilled.\n"
    "- Never use '[illegible]'; use your best reading with '(?)' when uncertain.\n"
    "- Return only the JSON object and nothing else."
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
    # Models sometimes wrap the fragment in <html>/<head>/<body> despite the
    # prompt — drop the wrappers, keep the content.
    fragment = re.sub(r"</?(?:html|head|body)\b[^>]*>", "", fragment,
                      flags=re.IGNORECASE)
    fragment = re.sub(r"<script\b.*?</script\s*>", "", fragment,
                      flags=re.IGNORECASE | re.DOTALL)
    fragment = re.sub(r"<script\b[^>]*>", "", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"\son\w+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", "",
                      fragment, flags=re.IGNORECASE)

    # Strip interactive form elements — the model sometimes outputs
    # <input>, <select>, <textarea> etc. which look wrong in a read-only view.
    # Replace <input ... value="X"> with just the value as a span.
    def _replace_input(m):
        tag = m.group(0)
        val_m = re.search(r'\bvalue\s*=\s*"([^"]*)"', tag, re.IGNORECASE)
        val = val_m.group(1) if val_m else ""
        placeholder_m = re.search(r'\bplaceholder\s*=\s*"([^"]*)"', tag, re.IGNORECASE)
        placeholder = placeholder_m.group(1) if placeholder_m else ""
        display = val or placeholder or ""
        return f'<span>{display}</span>' if display else ""

    fragment = re.sub(r"<input\b[^>]*/?>", _replace_input,
                      fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"<select\b.*?</select\s*>", "", fragment,
                      flags=re.IGNORECASE | re.DOTALL)
    fragment = re.sub(r"<textarea\b.*?</textarea\s*>", "", fragment,
                      flags=re.IGNORECASE | re.DOTALL)
    fragment = re.sub(r"<form\b[^>]*>|</form\s*>", "", fragment,
                      flags=re.IGNORECASE)
    fragment = re.sub(r"<button\b.*?</button\s*>", "", fragment,
                      flags=re.IGNORECASE | re.DOTALL)

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


def _wrap_uncertain(html: str) -> str:
    """
    Guarantee that every (?)-marked word in the layout HTML is wrapped in
    <span class="unc"> so the UI can highlight it — even when the model
    forgot the class. Walks tag/text pieces, skips text already inside an
    unc span, and never touches tag attributes. Never raises.
    """
    if not html or "(?)" not in html:
        return html
    try:
        parts = re.split(r"(<[^>]*>)", html)
        depth = 0  # <span> nesting depth while inside an unc span
        out = []
        for part in parts:
            if part.startswith("<"):
                low = part.lower()
                if low.startswith("<span"):
                    if depth:
                        depth += 1
                    elif re.search(r'class\s*=\s*["\'][^"\']*\bunc\b', low):
                        depth = 1
                elif low.startswith("</span") and depth:
                    depth -= 1
            elif depth == 0 and "(?)" in part:
                part = re.sub(r"(\S+\(\?\))",
                              r'<span class="unc">\1</span>', part)
            out.append(part)
        return "".join(out)
    except Exception as e:
        print(f"[WARNING] uncertain-word wrapping failed: {e}")
        return html


# Values that exist ONLY as format examples inside the prompts above.
# If one shows up in the output, the model echoed the prompt instead of
# reading the page — remove the echo so it can never leak into a document.
_PROMPT_ECHO_VALUES = ("Kavitha", "10452/25")
_PROMPT_ECHO_LINES = {
    "name : kavitha",
    "age : 38",
    "hospital no. : 10452/25",
    "hb        | 12.8   | 11.0 - 14.0",
    "test name | result | reference range",
}


def _clean_checklist_sections(text: str) -> str:
    """
    Conservative post-pass on the raw transcription:
      • drop lines that are verbatim echoes of prompt format examples
        (fabricated names/numbers that cannot come from a real page);
      • collapse runaway repetition (the same line emitted 3+ times in a
        row is a decoding loop, not page content).
    Never raises; returns the text otherwise unchanged.
    """
    if not text:
        return text
    out = []
    prev, run = None, 0
    for line in text.splitlines():
        key = " ".join(line.split()).lower()
        if key in _PROMPT_ECHO_LINES:
            continue
        if key and key == prev:
            run += 1
            if run >= 3:
                continue
        else:
            prev, run = key, 1
        out.append(line)
    return "\n".join(out)


def _clean_checklist_html(html: str) -> str:
    """Remove prompt-example echoes from the layout HTML. Never raises."""
    if not html:
        return html
    for val in _PROMPT_ECHO_VALUES:
        html = re.sub(
            r"(?<![A-Za-z0-9])" + re.escape(val) + r"(?![A-Za-z0-9])",
            "", html)
    return html


# Crops embedded into the layout are capped to this long-edge size.
_CROP_MAX_EDGE = int(os.getenv("CROP_MAX_EDGE", "500"))
_CROP_PAD_PCT = 2.0


def _normalize_bbox(x, y, w, h, pw, ph):
    """
    Normalize a model-emitted bbox to clamped page percentages.
    Qwen's grounding training sometimes emits absolute pixel coordinates
    despite the prompt asking for percentages — detect (any value > 100)
    and convert using the real page size. Returns (x, y, w, h) in percent
    or None when the box is degenerate or covers ~the whole page.
    """
    if max(x, y, w, h) > 100:
        x, y, w, h = x / pw * 100, y / ph * 100, w / pw * 100, h / ph * 100
    x = min(max(x, 0.0), 100.0)
    y = min(max(y, 0.0), 100.0)
    w = min(max(w, 0.0), 100.0 - x)
    h = min(max(h, 0.0), 100.0 - y)
    if w < 0.5 or h < 0.5 or w * h > 95 * 95:
        return None
    return x, y, w, h


def _bbox_iou(a, b):
    """Intersection-over-union of two (x, y, w, h) percent boxes."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


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

    injected = []  # percent boxes already embedded on this page

    def _crop_tag(match):
        tag = match.group(0)
        try:
            x, y, w, h = [float(v) for v in match.group(1).split(",")]
            box = _normalize_bbox(x, y, w, h, pw, ph)
            if box is None:
                return tag
            x, y, w, h = box
            # Duplicate tag for the same region (model repeats a logo).
            if any(_bbox_iou(box, prev) > 0.8 for prev in injected):
                return tag
            left = max(0, int((x - _CROP_PAD_PCT) / 100 * pw))
            top = max(0, int((y - _CROP_PAD_PCT) / 100 * ph))
            right = min(pw, int((x + w + _CROP_PAD_PCT) / 100 * pw))
            bottom = min(ph, int((y + h + _CROP_PAD_PCT) / 100 * ph))
            if right - left < 4 or bottom - top < 4:
                return tag
            injected.append(box)
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


# Reinforcement appended AFTER the main system prompt.
# These are the last instructions the model sees — highest recency weight.
_LOCAL_RULES = (
    "\n\nFINAL REMINDERS — APPLY THESE RULES TO EVERY PAGE\n\n"

    "1. SINGLE-PAGE EXTRACTION\n"
    "Extract only the content that is physically visible on the current page/image.\n"
    "Do not use information from previous pages, subsequent pages, memory, or assumptions.\n\n"

    "2. COMPLETE EXTRACTION\n"
    "Read the entire page from top to bottom and left to right.\n"
    "Extract every visible text element, handwritten note, printed text, number, symbol, "
    "label, heading, stamp, and annotation.\n"
    "Do not omit any visible content.\n"
    "A large vertical gap between handwritten lines NEVER means the page is "
    "finished — scan all the way to the bottom edge; lines separated by wide "
    "blank space are still part of the page and must be transcribed.\n\n"

    "3. \"CIRCLE IF POSITIVE\" CHECKLISTS\n"
    "When a section is labeled \"(Circle If Positive)\" and contains printed symptom or "
    "condition lists under headings such as GENERAL, G.I. TRACT, ENT, BREAST, G.U. TRACT, "
    "MUSCULO-SKELETAL SYSTEM, PAST HISTORY, FAMILY HISTORY, or similar sections:\n"
    "- Extract ALL printed items exactly as written.\n"
    "- Examine each item for clinician markings.\n"
    "- A marked item may be circled, ticked, checked, underlined, crossed, highlighted, "
    "or otherwise clearly selected.\n"
    "- For every selected item, append (Circled) immediately after the item.\n"
    "- For unselected items, output the item without any marker.\n"
    "- Only use (Circled) when there is clear visual evidence of selection.\n"
    "- Never infer selection from surrounding text or diagnoses.\n"
    "- When in doubt, leave the item UNMARKED — a false (Circled) is worse "
    "than a missed one. Never mark an item because its neighbour is marked.\n"
    "Example:\n"
    "  GENERAL : Fatigue (Circled), Weight loss, Chills, Unexplained fever\n"
    "  FAMILY HISTORY : Cancer, Tuberculosis (Circled), Diabetes\n\n"

    "4. CIRCLED SYMBOLS\n"
    "Preserve circled symbols exactly as they appear:\n"
    "  (+) = Circled Plus    (-) = Circled Minus\n"
    "  (L) = Circled L       (R) = Circled R\n"
    "Never replace these symbols with @, *, or any other character.\n\n"

    "5. NUMBERS AND IDENTIFIERS\n"
    "Preserve all numbers, slashes, dates, registration numbers, hospital numbers, "
    "and identifiers exactly as written.\n"
    "Example: 10452/25 must remain 10452/25\n"
    "Never merge, split, reformat, or normalize numeric values.\n\n"

    "6. UNCERTAIN TEXT\n"
    "Do not use [illegible], [unreadable], or blanks.\n"
    "If a word is difficult to read, provide your best interpretation and append (?) "
    "to the uncertain word or phrase.\n"
    "Example: lymphadenopathy(?)\n"
    "In the HTML section, wrap every (?) word in its own "
    "<span class=\"unc\">word(?)</span> — only the uncertain word, "
    "never the whole line.\n\n"

    "7. NO HALLUCINATION\n"
    "Output only information that is physically visible on the page.\n"
    "Do not invent, infer, expand abbreviations, add medical interpretations, "
    "or generate missing content.\n\n"

    "8. PRESERVE ORIGINAL STRUCTURE\n"
    "Maintain section headings, labels, field names, ordering, and document hierarchy "
    "exactly as they appear on the page whenever possible.\n\n"

    "9. HANDWRITTEN CONTENT\n"
    "Extract handwritten text with the same importance as printed text.\n"
    "Include all handwritten annotations, corrections, markings, numeric values, "
    "dates, and comments visible on the page.\n\n"

    "10. FULL-ROW SCANNING\n"
    "Handwritten pages often contain a SECOND group of writing after a gap on "
    "the SAME row (side-by-side groups or columns). Scan every row across its "
    "full width — left edge to right edge — and transcribe every group on the "
    "row, left to right. Never drop the group on the right.\n\n"

    "11. PAGE-TYPE FORMAT\n"
    "Format tabular report pages (printed grids of lab/investigation results) "
    "as tables. Format prescription and clinic-note pages as header + patient "
    "details + line-by-line items, never as a table.\n"
)

# Commands appended to the USER message — processed last before generation.
_LOCAL_USER_RULES = (
    "\n\nBefore writing your output, verify:\n"
    "- You are extracting ONLY from this single page image — no other pages.\n"
    "- Every visible item is included: top to bottom, left to right.\n"
    "- Circle-if-Positive sections: ALL items listed, selected ones marked (Circled).\n"
    "- Circled symbols: (+) (-) (L) (R) — never @.\n"
    "- All numbers and slashes preserved exactly (e.g. 10452/25).\n"
    "- No [illegible] — use best guess with (?) for uncertain text.\n"
    "- No hallucination — only what is physically visible on this page.\n"
    "- Handwritten text extracted with same priority as printed text.\n"
    "- Every row scanned across its FULL width — no group on the middle or "
    "right side of a row is missed.\n"
    "- Consecutive handwritten lines under a heading: ALL transcribed — "
    "count the lines on the page and match that count in your output, even "
    "when wide blank space separates them.\n"
    "- Every picture (photo, logo, diagram, seal, barcode) has one "
    "<img class=\"cut\" data-bbox=\"X,Y,W,H\"> tag in the HTML section, at "
    "its reading-order position, with percentage coordinates.\n"
    "- Every (?) word is wrapped in <span class=\"unc\"> in the HTML section.\n"
    "- (Circled) used ONLY with clear visual evidence; when unsure, leave "
    "the item unmarked.\n"
    "- Tabular reports formatted as tables; prescriptions line-by-line, "
    "never forced into a table.\n"
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


# ── Orientation auto-correction ──────────────────────────────────────────
# Ward photos are often captured sideways or upside-down; a rotated page
# destroys the VLM's reading order (garbled text, missed handwriting,
# hallucinated tables). Each page is uprighted BEFORE extraction using one
# tiny vision call on a downscaled copy.
_ORIENT_MAX_EDGE = 768
# A single-image YES/NO check is the only orientation question small VLMs
# answer reliably (4-way "how many degrees" and 2-image comparisons were
# tested and found position-biased / constant-answer).
_UPRIGHT_SYSTEM = (
    "You check one photo of a document. Answer EXACTLY YES or NO: "
    "YES only if the text is upright and reads normally left-to-right; "
    "NO if the text is sideways or upside-down."
)
_UPRIGHT_USER = (
    "Is the document text in this image upright and readable "
    "left-to-right? Answer YES or NO only."
)


def _is_upright(img: Image.Image):
    """Ask the vision model whether the page reads upright.
    Returns True / False, or None on any failure — never raises."""
    try:
        small = img.copy()
        small.thumbnail((_ORIENT_MAX_EDGE, _ORIENT_MAX_EDGE))
        if small.mode not in ("RGB", "L"):
            small = small.convert("RGB")
        buf = io.BytesIO()
        small.save(buf, format="JPEG", quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        raw, _ = _ollama_chat(_UPRIGHT_SYSTEM, _UPRIGHT_USER, [b64], 5)
        ans = (raw or "").strip().upper()
        if ans.startswith("YES"):
            return True
        if ans.startswith("NO"):
            return False
        return None
    except Exception as e:
        print(f"[WARNING] Orientation check failed: {e}")
        return None


def _upright(img: Image.Image, page_num: int) -> Image.Image:
    """
    Rotate the page image so its text is upright. Never raises.

    Portrait pages are assumed upright and only flipped when the model
    positively confirms BOTH that the original is not upright AND that
    the 180° flip is — so a flaky answer can never break a good page.
    Landscape pages are almost always sideways phone photos: try 90° CW
    then 90° CCW and keep the first candidate the model confirms; if
    neither is confirmed the original is kept (true landscape document).
    """
    try:
        w, h = img.size
        if h >= w:
            if _is_upright(img) is False:
                flipped = img.rotate(180)
                if _is_upright(flipped) is True:
                    print(f"[INFO] Page {page_num}: upside-down photo — rotated 180°")
                    return flipped
            return img
        for deg in (90, 270):
            cand = img.rotate(-deg, expand=True)  # negative = clockwise
            if _is_upright(cand) is True:
                print(f"[INFO] Page {page_num}: sideways photo — "
                      f"rotated {deg}° clockwise")
                return cand
        print(f"[INFO] Page {page_num}: landscape page kept as-is "
              f"(no rotation confirmed)")
        return img
    except Exception as e:
        print(f"[WARNING] Page {page_num}: orientation correction skipped: {e}")
        return img


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


def _raw_text_to_html(raw_text: str) -> str:
    """
    Pure-Python guaranteed fallback: convert raw plain text to readable HTML.
    Never raises. Handles letterhead, UPPERCASE headings, Label : value lines,
    two-column Notes/Orders blocks, and plain indented content.
    """
    if not raw_text:
        return '<div style="color:#888;font-style:italic">No content extracted.</div>'

    import html as _html  # stdlib — always available

    lines = raw_text.splitlines()
    parts = [
        '<div style="font-family:Georgia,serif;font-size:13px;'
        'line-height:1.7;color:#1f2937;padding:4px">'
    ]
    in_letterhead = True
    i = 0
    while i < len(lines):
        raw_line = lines[i]
        s = raw_line.strip()
        i += 1

        if not s:
            in_letterhead = False
            parts.append('<div style="height:0.35em"></div>')
            continue

        escaped = _html.escape(s)

        # Letterhead — first non-blank lines before first blank line
        if in_letterhead:
            parts.append(
                f'<div style="text-align:center;font-weight:700;'
                f'font-size:14px;margin-bottom:2px">{escaped}</div>'
            )
            continue

        # ALL-CAPS section heading ending with colon (e.g. "EXAMINATION :")
        if s == s.upper() and (s.endswith(":") or s.endswith(": ")) and len(s) < 80:
            parts.append(
                f'<div style="font-weight:700;margin-top:0.6em;'
                f'border-bottom:1px solid #d1d5db;padding-bottom:2px">{escaped}</div>'
            )
            continue

        # Notes : <value>  or  Orders : <value>  (two-column progress note)
        if s.startswith(("Notes :", "Orders :")):
            label, _, val = s.partition(" : ")
            e_label = _html.escape(label)
            e_val = _html.escape(val) if val else ""
            hw = f'<span style="font-family:\'Segoe Print\',cursive;color:#1d4ed8">{e_val}</span>' if e_val else ""
            parts.append(
                f'<div style="display:flex;gap:0.4em;margin-left:1em">'
                f'<span style="min-width:5em;font-weight:600;color:#374151">{e_label} :</span>'
                f'{hw}</div>'
            )
            continue

        # Label : value
        if " : " in s:
            label, _, val = s.partition(" : ")
            e_label = _html.escape(label)
            e_val = _html.escape(val) if val else "(blank)"
            parts.append(
                f'<div style="display:flex;gap:0.5em;margin:1px 0">'
                f'<span style="min-width:14em;font-weight:600">{e_label} :</span>'
                f'<span>{e_val}</span></div>'
            )
            continue

        # Date line (starts with "Date :" pattern already handled above,
        # but bare dates like "18/11/18" at start of line)
        indent = len(raw_line) - len(raw_line.lstrip())
        margin = f"margin-left:{min(indent, 8) * 0.5}em" if indent else ""
        style = f' style="{margin}"' if margin else ""
        parts.append(f'<div{style}>{escaped}</div>')

    parts.append('</div>')
    return "\n".join(parts)


# Slim system prompt for the dedicated layout-only call — much shorter than
# the full _LAYOUT_SYSTEM so the model has headroom to produce HTML output.
_LAYOUT_SLIM_SYSTEM = (
    "You produce an HTML layout fragment of a medical document page. "
    "Output ONLY the HTML — no markdown fences, no explanation, start directly with an HTML tag.\n\n"
    "RULES:\n"
    "- class=\"hw\" = ONLY for actual pen-ink handwriting (filled values, doctor notes). "
    "Printed form text (headings, labels, symptom lists, footers) is NEVER hw — plain text only.\n"
    "- class=\"unc\" = uncertain words with (?) suffix.\n"
    "- class=\"stamp\" = stamps, seals, signatures: '[STAMP: text]'.\n"
    "- class=\"cut\" = every pictorial region (photo, logo, diagram, sketch, "
    "graphic seal, barcode, QR code): "
    "<img class=\"cut\" data-bbox=\"X,Y,W,H\" alt=\"[IMAGE: description]\"> — "
    "data-bbox REQUIRED, X,Y,W,H as percentages (0-100) of the page, tag "
    "placed at the picture's reading-order position.\n"
    "- NO position:absolute, NO negative margins.\n"
    "- Mirror the image structure: single-column page → single column HTML. "
    "Two-column page → flex row. Printed symptom list → <p> not <table>. "
    "Form with label+handwritten answer → <table> two columns. "
    "Lab/investigation report with a printed results grid → real <table>. "
    "Handwritten prescription / clinic note → line-by-line <div>s in writing "
    "order, NEVER a fabricated table.\n"
    "- Letterhead centered at top. Bottom items (date, signature) last.\n"
    "- Describe annotations: e.g. <span class=\"hw\">[circled: Early]</span> or "
    "<span class=\"hw\">[arrow → Lymph nodes, Pallor +]</span>.\n"
    "- Output compact HTML only."
)


def _generate_layout(b64_image: str, page_num: int, raw_text: str) -> tuple:
    """
    Dedicated layout-only call used when the combined call produced text
    but missed the HTML section. Uses the already-extracted raw_text as
    context so the layout can be consistent with it.
    Returns (layout_html, error_str|None).
    Falls back to a Python-generated HTML if the model keeps failing.
    """
    layout_user = f"""Convert page {page_num} into a structured HTML layout.

Requirements:
1. Preserve the visual structure of the page.
2. Keep section headings, tables, columns, form fields and lists.
3. Use the image to distinguish:
   - Printed text: normal HTML elements.
   - Handwritten text: wrap in <span class="hw">...</span>
4. Preserve the original reading order.
5. Represent checklists faithfully.
6. For '(Circle If Positive)' sections:
   - Include every checklist item.
   - If an item is visibly selected, append '(Circled)'.
7. Do not invent content.
8. Maintain spacing and grouping wherever possible.
9. Output valid HTML only.

TRANSCRIBED TEXT:
{raw_text}

Return only the HTML fragment.
Start with an HTML tag."""
    last_err = None
    for attempt in range(3):
        try:
            raw, _reason = _ollama_chat(
                _LAYOUT_SLIM_SYSTEM,
                layout_user,
                [b64_image],
                6000,
            )
            if raw:
                # Strip any prose the model put before the HTML
                html_start = raw.find("<")
                if html_start != -1:
                    html = raw[html_start:]
                    print(f"[INFO] Page {page_num}: layout-only call succeeded "
                          f"(attempt {attempt + 1})")
                    return _sanitize_html(html), None
            print(f"[WARNING] Page {page_num} layout-only call returned no HTML, "
                  f"attempt {attempt + 1}/3. Raw: {repr((raw or '')[:120])}")
            if attempt < 2:
                time.sleep(3)
        except Exception as e:
            last_err = e
            print(f"[WARNING] Page {page_num} layout-only call error: {e}, "
                  f"attempt {attempt + 1}/3")
            if attempt < 2:
                time.sleep(5)

    # Model failed all attempts — generate basic HTML from raw text in Python.
    print(f"[INFO] Page {page_num}: model layout failed ({last_err or 'no HTML returned'}), "
          f"using Python-generated fallback layout.")
    fallback_html = _raw_text_to_html(raw_text)
    return fallback_html, None


def read_page(
    b64_image: str,
    page_num: int,
    mime: str = "image/jpeg",
) -> tuple[str, str | None, str | None]:
    """
    Extracts a single medical-record page.

    Performs a single vision-model read to produce:
    - Complete OCR transcription
    - Structured HTML layout
    - Detection of handwritten annotations
    - Preservation of form structure
    - Recognition of '(Circle If Positive)' checklists

    Returns:
        (extracted_text, layout_html, layout_error)
        The extracted text is mandatory.
        A text-only fallback is automatically used when the
        combined extraction cannot produce a reliable transcript.

    Raises:
        ExtractorError:
            When a valid transcription cannot be obtained.
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
                    text = _clean_checklist_sections(text)
                    layout = _clean_checklist_html(layout)
                    return text, _sanitize_html(layout), None
                # Combined call gave text but no layout section.
                print(f"[INFO] Page {page_num}: combined output had no layout "
                      f"section — running dedicated layout call.")
                text = _clean_checklist_sections(text)
                layout_html, _layout_err = _generate_layout(b64_image, page_num, text)
                if not layout_html:
                    layout_html = _raw_text_to_html(text)
                layout_html = _unhw_letterhead(text, layout_html)
                layout_html = _clean_checklist_html(layout_html)
                return text, _sanitize_html(layout_html), None
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
    text = _clean_checklist_sections(text)
    layout_html = _raw_text_to_html(text)
    layout_html = _clean_checklist_html(layout_html)
    return text, _sanitize_html(layout_html), None


# Structured-fields calls attach at most this many page images per call;
# longer documents are processed in chunks and the results merged, so no
# page is ever silently dropped.
_FIELDS_MAX_PAGES = 5


def _fields_single_call(system_prompt: str, images: list) -> tuple:
    """One structured-fields call over a chunk of pages, with retries.
    Returns ({"fields": [...], "medicines": [...]}, None) or ({}, error)."""
    images_b64 = [b64 for _, b64, _ in images]
    # Build a page-labeling preamble so the model knows which image = which page
    page_labels = ", ".join(f"image {i+1} = page {n}" for i, (n, _, _) in enumerate(images))
    fields_user = _FIELDS_USER.format(n=len(images)) + f"\n\nPage mapping: {page_labels}."

    last_err = None
    for attempt in range(4):
        try:
            raw, _reason = _ollama_chat(
                system_prompt,
                fields_user,
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


def read_structured_fields(images: list) -> tuple:
    """
    Structured extraction for the whole document:
    images = [(page_num, b64, mime), ...].
    Documents longer than _FIELDS_MAX_PAGES are processed in page chunks
    and the chunk results merged (fields deduped by key, medicines by name)
    so every page contributes. Returns ({"fields": [...], "medicines": [...]},
    error|None) — never raises.
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

    all_fields, all_medicines, errors = [], [], []
    for start in range(0, len(images), _FIELDS_MAX_PAGES):
        chunk = images[start:start + _FIELDS_MAX_PAGES]
        data, err = _fields_single_call(system_prompt, chunk)
        if err:
            errors.append(err)
            continue
        all_fields.extend(f for f in data["fields"] if isinstance(f, dict))
        all_medicines.extend(m for m in data["medicines"] if isinstance(m, dict))

    if not all_fields and not all_medicines and errors:
        return {}, "; ".join(errors)

    # Merge across chunks: first occurrence of a field key wins, except a
    # '(blank)' value is upgraded when a later chunk found a real value.
    merged_fields, by_key = [], {}
    for f in all_fields:
        k = str(f.get("key") or f.get("name") or "").strip().lower()
        if not k:
            merged_fields.append(f)
            continue
        if k in by_key:
            old = by_key[k]
            oldv = str(old.get("value") or "").strip()
            newv = str(f.get("value") or "").strip()
            if oldv in ("", "(blank)") and newv not in ("", "(blank)"):
                old["value"] = newv
            continue
        by_key[k] = f
        merged_fields.append(f)

    merged_meds, seen_med = [], set()
    for m in all_medicines:
        sig = re.sub(r"[^a-z0-9]+", "", str(m.get("medicine") or "").lower())
        if sig and sig in seen_med:
            continue
        seen_med.add(sig)
        merged_meds.append(m)

    return ({"fields": merged_fields, "medicines": merged_meds},
            "; ".join(errors) if errors else None)


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
            img = _upright(img, i + 1)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90, optimize=True)
            buf.seek(0)
            images.append((i + 1, base64.b64encode(buf.read()).decode("utf-8")))
        return images
    finally:
        doc.close()


# ── Missed-line verification pass ────────────────────────────────────
# Second look at pages that contain handwriting: the model receives the
# page image plus the numbered transcription and reports lines that are
# completely missing, which are merged back into text and layout.
_VERIFY = os.getenv("VERIFY_MISSED_LINES", "1").strip().lower() in (
    "1", "true", "yes", "on")

# Gap between anchor words in layout HTML: tags, punctuation, whitespace.
# (Same pattern as backend._ANCHOR_GAP.)
_ANCHOR_GAP = r"(?:<[^>]*>|[^A-Za-z0-9<])*"

_VERIFY_SYSTEM = (
    "You are a transcription completeness checker for medical documents. "
    "You receive ONE page image and a NUMBERED transcription of that page. "
    "Compare them and report ONLY lines of visible text (printed or "
    "handwritten) that are COMPLETELY MISSING from the transcription. "
    "Do NOT report spelling differences, re-wordings, formatting changes, "
    "or lines that are already present. "
    "Pay special attention to: handwritten lines separated by large "
    "vertical gaps, the right half of every row, margins and corners, and "
    "content below the last printed section. "
    "Transcribe each missing line exactly as written, appending (?) to any "
    "uncertain word. "
    'Output ONLY JSON: {"missing": [{"after_line": <int>, "text": "<line>"}]} '
    "where after_line is the transcription line number that the missing "
    "line appears BELOW on the page (0 = above the first line). "
    'If nothing is missing output {"missing": []}.'
)

_VERIFY_USER = (
    "Transcription of page {page} (numbered):\n\n{numbered}\n\n"
    "Compare against the page image and return the JSON now."
)


def _page_has_handwriting(text: str, layout_html: str) -> bool:
    """Heuristic: the page contains handwriting or uncertain words."""
    if layout_html and re.search(r'class\s*=\s*["\'][^"\']*\bhw\b',
                                 layout_html):
        return True
    return "(?)" in (text or "")


def _insert_line_into_layout(layout_html: str, anchor_line: str,
                             cand: str) -> str:
    """
    Insert a recovered line into the layout HTML right after the block
    containing the anchor line (the transcript line directly above it on
    the page). Falls back to appending at the end of the fragment.
    """
    escaped = (cand.replace("&", "&amp;")
                   .replace("<", "&lt;").replace(">", "&gt;"))
    new_block = f'<div><span class="hw">{escaped}</span></div>'
    tokens = re.findall(r"[A-Za-z0-9]+", anchor_line or "")[-6:]
    block_close = re.compile(
        r"</div\s*>|</p\s*>|</li\s*>|</tr\s*>|</h[1-6]\s*>|<br\s*/?>",
        re.IGNORECASE)
    for k in range(len(tokens)):
        anchor_rx = re.compile(
            _ANCHOR_GAP.join(re.escape(w) for w in tokens[k:]),
            re.IGNORECASE)
        m = anchor_rx.search(layout_html)
        if not m:
            continue
        bm = block_close.search(layout_html, m.end())
        pos = bm.end() if bm else m.end()
        return layout_html[:pos] + new_block + layout_html[pos:]
    return layout_html + new_block


def _verify_completeness(b64_image: str, page_num: int, text: str,
                         layout_html: str) -> tuple:
    """
    One extra vision call that recovers lines the first read skipped.
    Returns (text, layout_html) — unchanged on any failure; never raises.
    """
    try:
        lines = text.splitlines()
        numbered = "\n".join(f"{i + 1}: {ln}" for i, ln in enumerate(lines))
        raw, _ = _ollama_chat(
            _VERIFY_SYSTEM,
            _VERIFY_USER.format(page=page_num, numbered=numbered),
            [b64_image],
            3000,
            json_mode=True,
        )
        missing = (_parse_json_loose(raw) if raw else {}).get("missing") or []
        if not isinstance(missing, list):
            return text, layout_html

        line_token_sets = [
            set(re.findall(r"[a-z0-9]+", ln.lower())) for ln in lines]
        accepted = []
        for item in missing[:20]:
            if not isinstance(item, dict):
                continue
            cand = str(item.get("text") or "").strip()
            if len(cand) < 2:
                continue
            cand_toks = set(re.findall(r"[a-z0-9]+", cand.lower()))
            if not cand_toks:
                continue
            # Fuzzy duplicate: most of the candidate already sits in one
            # existing line — the model re-reported, not recovered.
            if any(toks and len(cand_toks & toks) >= 0.7 * len(cand_toks)
                   for toks in line_token_sets):
                continue
            try:
                after = int(item.get("after_line", len(lines)))
            except (TypeError, ValueError):
                after = len(lines)
            accepted.append((min(max(after, 0), len(lines)), cand))

        # Insert bottom-up so earlier indices stay valid.
        for after, cand in sorted(accepted, key=lambda t: t[0], reverse=True):
            anchor_line = lines[after - 1] if after >= 1 else ""
            if layout_html:
                layout_html = _insert_line_into_layout(
                    layout_html, anchor_line, cand)
            lines.insert(after, cand)
        if accepted:
            text = "\n".join(lines)
        print(f"[INFO] Page {page_num}: verify pass recovered "
              f"{len(accepted)} line(s)")
        return text, layout_html
    except Exception as e:
        print(f"[WARNING] Page {page_num}: verify pass skipped ({e})")
        return text, layout_html


def _read_page_full(b64_image: str, page_num: int, mime: str) -> tuple:
    """read_page + optional missed-line verification + unc wrapping."""
    text, layout_html, layout_error = read_page(b64_image, page_num, mime)
    if _VERIFY and text and _page_has_handwriting(text, layout_html):
        text, layout_html = _verify_completeness(
            b64_image, page_num, text, layout_html)
    if layout_html:
        layout_html = _wrap_uncertain(layout_html)
    return text, layout_html, layout_error


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
        # Open, upright and re-encode direct image uploads too, so a
        # sideways phone photo is corrected before extraction.
        try:
            img = Image.open(io.BytesIO(data))
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img = _upright(img, 1)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            rendered = [(1, b64, "image/jpeg")]
        except Exception as e:
            print(f"[WARNING] Image preprocessing failed ({e}); using original bytes")
            b64 = base64.b64encode(data).decode("utf-8")
            rendered = [(1, b64, _MIME.get(ext, "image/jpeg"))]

    # ONE combined transcript+layout call per page (single reading — both
    # views agree and the image is paid for once), plus one fields call per
    # document, all concurrent.
    with ThreadPoolExecutor(max_workers=min(8, len(rendered) + 1)) as pool:
        page_futs = {
            n: pool.submit(_read_page_full, b64, n, mime)
            for n, b64, mime in rendered
        }
        fields_fut = pool.submit(read_structured_fields, rendered)

        pages = []
        for n, b64, _ in rendered:
            text, layout_html, layout_error = page_futs[n].result()  # ExtractorError propagates
            if layout_html:
                # Cut the real stamps/signatures/diagrams out of the page
                # image and embed them where the model marked data-bbox.
                # b64 is the (possibly rotation-corrected) image the model
                # actually saw, so its bbox coordinates match this image.
                layout_html = _embed_region_crops(layout_html, base64.b64decode(b64))
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

