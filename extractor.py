"""
extractor.py — Offline VLM engine via Ollama (qwen2.5-VL)
─────────────────────────────────────────────────────────
Multi-pass extraction pipeline for medical prescription pages (see
README "Extraction pipeline" for the full per-page order):

  render 300 DPI → upright → combined raw+layout read (retry nudges,
  CJK-garbage gate) → sparse-page rotation rescue → verify pass
  (missed lines) → header recovery (top strip) → diagram detection +
  crop verification + merge → deterministic repairs (C/o shorthand,
  checklist container) → regional-script translation → (?) uncertainty
  wrapping → region crop embedding.

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

# Bumped on every behavioral change; printed at import so the server log
# proves which build is actually running (deployments happen by git pull
# on a remote box — a stale checkout is otherwise invisible).
EXTRACTOR_BUILD = "2026-08-14-r13"
print(f"[INFO] extractor build {EXTRACTOR_BUILD} — model={_MODEL}, "
      f"host={_HOST}")

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

--------------------------------------
TOP OF PAGE
--------------------------------------

Start your transcription at the VERY TOP EDGE of the page. Content
printed near the top border is real page content, not background:
URLs, UHID / ID numbers, barcode digits, room numbers, token numbers,
visit dates, and page titles (e.g. OUT PATIENT RECORD). The photo may
show clutter around the page (table surface, other papers) — ignore
the clutter, but never skip text that is ON the page, however close
to its edge. Before finishing, look at the top 15% of the page once
more and confirm every line there appears in your output.

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
LANGUAGE
--------------------------------------

If any text on the page is written in a non-English / regional script
(Kannada, Hindi, Tamil, Telugu, or any other Indian regional language),
translate its MEANING into natural English — do not leave it in the
original script and do not transliterate it phonetically.
  Example: ಕಫ ಔಷಧಿ ದಿನಕ್ಕೆ 3 ಬಾರಿ -> Cough medicine 3 times a day

EXCEPTION — proper nouns: patient names, doctor names, place names, and
facility names written in a regional script must be TRANSLITERATED into
Roman letters (their phonetic English spelling), never translated as if
they were ordinary words.
  Example: ಕವಿತಾ -> Kavitha   (not a literal word-for-word translation)

If the translation itself is uncertain, append (?) to the translated
word or phrase, the same as any other uncertain reading.

Your entire output must contain ONLY English/Roman letters, digits and
punctuation. Never output Kannada, Devanagari/Hindi, Tamil, Telugu or
any other script anywhere — translate (or transliterate names) instead.
Never convert one Indian script into another (e.g. Kannada text must
never come out as Hindi/Devanagari). Never output Chinese, Japanese or
Korean characters — these are decoding errors, not page content.

--------------------------------------
PAGE TYPE — CHOOSE THE RIGHT FORMAT
--------------------------------------

Decide the page type FIRST, then format the whole page accordingly.

A. TABULAR REPORT PAGE
   (lab / haematology / biochemistry / investigation results printed
   as a grid of rows and columns)
   - Output the report header (facility, patient details, dates)
     first as Label : Value lines. The letterhead, report title, and
     the printed patient-details box at the TOP of the page are part
     of the page — NEVER skip them; transcribe them before the
     results table.
   - Output the results as a pipe-separated table (rule 6): ONE row per
     test with its Result and Reference Range in separate columns.
   - NEVER output lab results as one line per value, as Label : Value
     pairs, or as running text — a printed results grid is ALWAYS a
     table, in BOTH the raw text and the HTML layout.

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
A printed label with no filled value is STILL output, as
  Label : (blank)
Never skip a label because it is empty (e.g. Clinical History,
Examination Findings, Investigation, Diagnosis on an unfilled form).

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
For printed lab/haematology/biochemistry reports: one pipe row per test,
keeping Result and Reference Range in their own columns. Never flatten a
results grid into one line per value or Label : Value pairs.

Some reports print ONE reference-range block that visually spans SEVERAL
test rows (the range column looks merged down the page). Split it back
out: repeat the applicable range on EVERY row it covers. Example:
  Test Name | Result | Reference Range
  Haemoglobin | 12.8 | 11.0 - 14.0 gms%
  RBC Count   | 4.7  | 3.6 - 4.6 x10^6/ul
  Hematocrit  | 39.1 | 35 - 55 %
(Each row still gets its OWN reference range, even when the source page
prints the ranges as one grouped block to the right of several rows.)

Only the ACTUAL results grid (rows with a test name, a result, and a
range) is formatted this way. The facility name, report title, and
patient-details box printed ABOVE the grid are NOT part of the table —
keep them as plain lines / Label : Value lines before the pipe rows,
never inside them.
Never add a caption, title, or label for the table itself (e.g. never
output "Table 1" or "Table:") — output the rows directly, nothing else.

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
MEDICAL SHORTHAND
--------------------------------------

Recognize common clinical shorthand and transcribe it exactly:
  C/o = Complains of    H/o = History of     K/c/o = Known case of
  O/E = On examination  R/o = Rule out       b/l = bilateral
  F/H = Family history  P/H = Past history   S/p = Status post
  LMP, EDD, D/x, R/x, ECOG

Handwritten "C/o" at the start of a complaint line is often misread as
"y/o", "yo" or "40" — when a line begins a complaint, prefer C/o.
ECOG is a performance status (e.g. ECOG-2) — never transcribe it as
"ECG-2" when written beside a general-examination note.
A small hand-drawn triangle (Δ / delta) is clinical shorthand — output
it as text on its line (use "Δ"); it is never a diagram or image.

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
STAMPS / SIGNATURES / DIAGRAMS
--------------------------------------

  Stamps:     [STAMP: text]
  Signatures: [SIGNATURE: text]
  Hand-drawn clinical drawings (anatomy sketches, lesion maps,
  marked-up body outlines): [DIAGRAM: description and markings]

Logos, letterhead emblems, medical symbols (caduceus), barcodes,
QR codes, watermarks, photographs and decorative graphics: IGNORE the
graphic entirely — no marker for it. The letterhead TEXT itself is
still transcribed as normal text.
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
  cut    Hand-drawn clinical drawings ONLY: anatomy sketches,
         lesion maps, marked-up drawings with pen annotations.

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
When the page IS a printed lab/haematology/biochemistry report, the
results MUST be one real <table> with one <tr> per test and separate
<td> cells for Test Name, Result, and Reference Range — never a stack
of single-line divs. If the printed Reference Range column visually
spans several test rows as one merged block, still give EVERY <tr>
its own <td> with the applicable range repeated — never one giant
range cell spanning multiple rows, never a stack of un-tabled divs.
The facility name, report title, and patient-details box printed
ABOVE the grid go BEFORE the <table> as plain divs/label:value pairs
— never as rows inside it, and never all merged into one column.
Never output a caption/heading for the table (e.g. never "Table 1").
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
- Wrap the ENTIRE printed checklist (the '(Circle If Positive)' label and
  all its printed headings, e.g. GENERAL through FAMILY HISTORY) in ONE
  container: <div class="checklist"> ... </div> — this draws a border
  around the checklist so it is visually separate.
- Handwritten clinical sections (COMPLAINTS AND DURATION, HISTORY OF
  PRESENT ILLNESS, PAST HISTORY notes, GENERAL EXAMINATION, etc.) are
  NEVER part of the checklist — put them OUTSIDE the checklist div. When
  the page shows the checklist on one half and handwritten notes on the
  other half, use a two-column flex row: checklist div in one column,
  handwritten sections in the other.

Example:
  <div style="display:flex;gap:12px">
    <div class="checklist" style="flex:1">
      <div>(Circle If Positive)</div>
      <div><strong>GENERAL:</strong>
      Fatigue (Circled), Weight loss, Chills</div>
      ...
    </div>
    <div style="flex:1">
      <div><strong>COMPLAINTS AND DURATION</strong></div>
      <div><span class="hw">C/o &lt;complaint&gt; x &lt;duration&gt;</span></div>
      ...
    </div>
  </div>
(The example content is placeholder only — always read the actual page.)

------------------------------------
SPECIAL CONTENT
------------------------------------

  Stamps:     <span class="stamp">[STAMP: text]</span>
  Signatures: <span class="stamp">[SIGNATURE: name]</span>
  Drawings:   <img class="cut" data-bbox="X,Y,W,H" alt="[DIAGRAM: description]">

Hand-drawn clinical drawings (anatomy sketch, lesion map, marked-up
body outline) MUST each be output as exactly ONE img.cut tag, placed
at the drawing's reading-order position.
- data-bbox is REQUIRED. X,Y = top-left corner, W,H = width,height —
  all four are PERCENTAGES (0-100) of the full page. Cover the WHOLE
  drawing including its pen annotations.
- Never redraw a drawing as text or ASCII art. Never omit data-bbox.
- NEVER emit img.cut for: logos, letterhead emblems, medical symbols
  (caduceus), barcodes, QR codes, watermarks, photographs, decorative
  graphics — omit those entirely.
- Text-only rubber stamps stay <span class="stamp">.

------------------------------------
ACCURACY RULES
------------------------------------

- Never omit visible content.
- Never hallucinate content.
- Preserve dates and hospital numbers exactly.
- Preserve slashes (10452/25 remains 10452/25).
- Use (?) for uncertain words.
- Never use [illegible].
- Never reproduce ink colors — output plain unstyled text regardless of
  pen color (no color or background styles); highlighting is added by
  the system.
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
    "- class=\"cut\" ONLY for hand-drawn clinical drawings (anatomy "
    "sketches, lesion maps, marked-up outlines) — never logos, emblems, "
    "medical symbols, barcodes, QR codes or decorative graphics.\n\n"
    "For every hand-drawn clinical drawing:\n"
    "<img class=\"cut\" data-bbox=\"X,Y,W,H\" alt=\"[DIAGRAM: description]\">\n"
    "data-bbox is REQUIRED — X,Y,W,H are percentage coordinates (0-100) "
    "relative to the page, placed where the drawing sits in reading order.\n\n"
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

10. FACILITY NAME PER PAGE
A multi-page document may combine pages from DIFFERENT facilities (a
hospital's own pages plus referred/external lab or imaging reports).
facility_name reflects the letterhead actually printed on the page(s)
that field's value came from — never assume every page shares the
facility name seen on page 1. If a page's letterhead is genuinely
partially covered or cut off, transcribe only the visible fragment
rather than completing it into a full name from another page.

11. LANGUAGE
Narrative/free-text field values (e.g. chief_complaints, diagnosis,
family_history, advice, instructions) written in a regional script are
translated into English. Proper-noun fields (patient_name, doctor_name,
facility_name) are transliterated into Roman letters instead — never
translated as ordinary words. This does not relax rule 8: names are
still transcribed in full, just in Roman script.

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

    # Models sometimes mimic pen ink with <font> tags — drop the tags,
    # keep the content (all color comes from our CSS classes instead).
    fragment = re.sub(r"</?font\b[^>]*>", "", fragment, flags=re.IGNORECASE)

    # Overlap guard: absolute/fixed positioning and negative margins make
    # text render on top of other text — force everything into normal
    # flow. Color guard: strip color/background declarations so red-ink
    # handwriting never comes out red — in this UI red means "flagged
    # word", and everything else must be plain black.
    def _defuse_style(m):
        style = m.group(2)
        style = re.sub(r"position\s*:\s*(absolute|fixed)", "position:static",
                       style, flags=re.IGNORECASE)
        style = re.sub(r"margin(?:-\w+)?\s*:\s*-[^;]*;?", "", style,
                       flags=re.IGNORECASE)
        style = re.sub(r"(?:background-)?color\s*:\s*[^;]*;?", "", style,
                       flags=re.IGNORECASE)
        style = re.sub(r"background\s*:\s*[^;]*;?", "", style,
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

    The "(?)" marker itself stays in the HTML (nested in its own
    <span class="qhide">) rather than being deleted — it's still the
    internal signal this function relies on, and the raw-text
    correction/diff pipeline indirectly depends on it being present —
    but that inner span is CSS-hidden (buildLayoutDoc's .qhide rule), so
    no "?" is ever visible on screen, in the PDF, or in the Word export.
    (Not named "unc-*": the frontend's uncCount legend regex matches the
    whole word "unc" via \bunc\b, which "-" boundaries don't block — a
    name sharing that word would silently double the displayed count.)
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
                part = re.sub(
                    r"(\S+)(\(\?\))",
                    r'<span class="unc">\1<span class="qhide">\2</span></span>',
                    part)
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
# Padding (page-percent) added around a bbox before cropping — generous
# so slightly-tight model boxes still capture the whole drawing.
_CROP_PAD_PCT = float(os.getenv("CROP_PAD_PCT", "4.0"))

# Dedicated diagram-detection pass (second grounding call per page with
# diagram signals). 0 disables and falls back to inline layout bboxes.
_DETECT_DIAGRAMS = os.getenv("DETECT_DIAGRAMS", "1").strip().lower() in (
    "1", "true", "yes", "on")
_DIAGRAM_MAX_PER_PAGE = 6
# Loose match threshold: the layout call's bbox may be badly off, so a
# modest overlap is enough to say "same drawing" and adopt the
# detector's box instead.
_DIAGRAM_MATCH_IOU = 0.30
# Descriptions/alts that mean "not a clinical drawing" — never cropped.
_LOGO_ALT_RX = re.compile(
    r"logo|emblem|caduceus|symbol|barcode|qr\b|watermark|letterhead|"
    r"photo|seal|crest|icon", re.IGNORECASE)
# A genuine anatomy drawing's own description uses this vocabulary (the
# detector prompt asks it to describe what the drawing shows). A
# reference mark — a circled digit/letter with an arrow to a word — has
# none of it, so requiring a hit here rejects that whole failure class
# deterministically, on top of the prompt wording and the crop verifier.
# Also covers printed diagnostic-instrument graphs (scattergrams, ECG
# traces) — a different crop-worthy category with its own vocabulary.
_ANATOMY_RX = re.compile(
    r"breast|chest|abdomen|limb|organ|lesion|tumou?r|mass|outline|"
    r"sketch|anatomy|body|nodule|cyst|face|facial|cheek|neck|skin|"
    r"swelling|growth|nodal|node|site|region|scar|"
    r"scattergram|graph|plot|histogram|waveform|trace|analyzer|"
    r"scatter\s*plot", re.IGNORECASE)


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


def _bbox_ios(a, b):
    """Intersection over the SMALLER box's area — a box nested inside a
    bigger one scores ~1.0 here even when the IoU is low."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    smaller = min(a[2] * a[3], b[2] * b[3])
    return inter / smaller if smaller > 0 else 0.0


def _same_region(a, b) -> bool:
    """Two boxes describe the same drawing: solid overlap OR nesting."""
    return _bbox_iou(a, b) > 0.5 or _bbox_ios(a, b) > 0.55


def _union_box(a, b):
    """Covering rect of two (x, y, w, h) boxes."""
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    r = max(a[0] + a[2], b[0] + b[2])
    btm = max(a[1] + a[3], b[1] + b[3])
    return (x, y, r - x, btm - y)


def _crop_style_for_box(x, y, w, h) -> str:
    """
    Inline style that mirrors the drawing's position on the original page:
    right-half drawings float right beside the text, left-half float left,
    wide/centered ones render as a centered block.
    """
    cx = x + w / 2
    if w >= 55:
        return "display:block;margin:8px auto;"
    if cx >= 55:
        return "float:right;margin:4px 0 8px 12px;max-width:48%;"
    if cx <= 45:
        return "float:left;margin:4px 12px 8px 0;max-width:48%;"
    return "display:block;margin:8px auto;"


def _tag_alt(tag: str) -> str:
    m = re.search(r'\balt\s*=\s*("[^"]*"|\'[^\']*\')', tag, re.IGNORECASE)
    return m.group(1)[1:-1] if m else ""


def _embed_region_crops(fragment: str, page_image_bytes: bytes) -> str:
    """
    Replace <img class="cut" data-bbox="x,y,w,h" ...> placeholders in the
    layout fragment with real crops of the page image as data: URIs.
    Coordinates are percentages of the page. Tags describing logos or
    similar non-clinical graphics are deleted outright. Per-crop failures
    leave the tag src-less only when its alt is a [DIAGRAM marker (shown
    as a dashed placeholder); otherwise the tag is removed. Never raises.
    """
    if not fragment:
        return fragment
    cut_rx = re.compile(
        r'<img\b[^>]*data-bbox\s*=\s*["\']([\d.,\s]+)["\'][^>]*>',
        re.IGNORECASE)
    if "data-bbox" not in fragment:
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

    def _keep_or_drop(tag):
        """Failed/rejected tag: keep dashed placeholder only for diagrams."""
        return tag if _tag_alt(tag).lstrip().upper().startswith("[DIAGRAM") \
            else ""

    def _crop_tag(match):
        tag = match.group(0)
        try:
            # Logos/emblems/barcodes are never embedded, whatever the bbox.
            if _LOGO_ALT_RX.search(_tag_alt(tag)):
                return ""
            x, y, w, h = [float(v) for v in match.group(1).split(",")]
            box = _normalize_bbox(x, y, w, h, pw, ph)
            if box is None:
                return _keep_or_drop(tag)
            x, y, w, h = box
            # Duplicate tag for the same region (model repeats a drawing,
            # or emits a nested box of an already-embedded one).
            if any(_same_region(box, prev) for prev in injected):
                return ""
            left = max(0, int((x - _CROP_PAD_PCT) / 100 * pw))
            top = max(0, int((y - _CROP_PAD_PCT) / 100 * ph))
            right = min(pw, int((x + w + _CROP_PAD_PCT) / 100 * pw))
            bottom = min(ph, int((y + h + _CROP_PAD_PCT) / 100 * ph))
            if right - left < 4 or bottom - top < 4:
                return _keep_or_drop(tag)
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
            # Drop model-invented src/style, then inject the real crop and
            # a position style mirroring the original page placement.
            cleaned = re.sub(r'\s(?:src|style)\s*=\s*("[^"]*"|\'[^\']*\')',
                             "", tag, flags=re.IGNORECASE)
            return cleaned[:-1].rstrip("/").rstrip() + \
                f' style="{_crop_style_for_box(x, y, w, h)}"' + \
                f' src="data:image/jpeg;base64,{b64}">'
        except Exception as e:
            print(f"[WARNING] region crop failed: {e}")
            return _keep_or_drop(tag)

    return cut_rx.sub(_crop_tag, fragment)


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
    "blank space are still part of the page and must be transcribed.\n"
    "A line consisting of just one or two words, or written with wide "
    "internal spacing between its own words, is NOT blank — extract it "
    "exactly like any other line. Never skip a line because it looks "
    "sparse or unusually spaced; unusual spacing is a handwriting habit, "
    "not an empty row.\n\n"

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
    "exactly as they appear on the page whenever possible.\n"
    "Never invent a section heading that is not printed on the page, and never "
    "move content under a different heading than the one it is written "
    "under. If a numbered list (e.g. under INVESTIGATION) has an item "
    "written in a slightly different hand or with a connecting arrow/line "
    "at the end, it is still the NEXT numbered item of that SAME list — "
    "keep it there. Only create a DIAGNOSIS or PROPOSED section when that "
    "exact printed label exists on the page, and only place under it "
    "content that is physically written at that label's own position. A "
    "printed field with nothing written under it stays (blank) — do not "
    "fill it with content that belongs to a different field.\n"
    "Transcribe each section ONCE. Never repeat a heading or its "
    "checklist/items a second time later in your output, even if the "
    "wording differs slightly the second time.\n\n"

    "9. HANDWRITTEN CONTENT\n"
    "Extract handwritten text with the same importance as printed text.\n"
    "Include all handwritten annotations, corrections, markings, numeric values, "
    "dates, and comments visible on the page.\n\n"

    "10. FULL-ROW SCANNING\n"
    "Handwritten pages often contain a SECOND group of writing after a gap on "
    "the SAME row (side-by-side groups or columns). Scan every row across its "
    "full width — left edge to right edge — and transcribe every group on the "
    "row, left to right. Never drop the group on the right.\n"
    "A lab-instrument printout often has a numeric results table followed, "
    "BELOW it, by several short label-only columns side by side (e.g. "
    "'WBC IP Message', 'RBC IP Message', 'PLT IP Message' each with a "
    "list of flag names under it). Extract EVERY one of these columns, "
    "not just the ones that catch the eye first — the LEFTMOST column is "
    "the one most often missed.\n\n"

    "11. PAGE-TYPE FORMAT\n"
    "Format tabular report pages (printed grids of lab/investigation results) "
    "as tables. Format prescription and clinic-note pages as header + patient "
    "details + line-by-line items, never as a table.\n\n"

    "12. LANGUAGE — TRANSLATE TO ENGLISH\n"
    "Any text in Kannada or another regional script is translated into "
    "natural English, not transliterated and not left in the original "
    "script. Exception: patient/doctor/place names are transliterated into "
    "Roman letters, never translated as ordinary words. Uncertain "
    "translations still get (?). Your output must contain ONLY "
    "English/Roman characters — never any Kannada, Devanagari, Tamil or "
    "other script, never one Indian script converted into another, and "
    "never Chinese/Japanese/Korean characters (those are decoding "
    "errors, not page content).\n\n"

    "13. MARKED SELECTIONS ANYWHERE ON THE PAGE\n"
    "A handwritten pen mark selects ONE option in many places on these "
    "forms, not only inside a \"(Circle If Positive)\" symptom list. The "
    "mark may be a circle, an asterisk/star, a tick or check mark, an "
    "underline, or a cross — all of these mean the same thing: SELECTED. "
    "This includes:\n"
    "- A field listing several printed options (e.g. Build : Normal, "
    "Obese, Asthenic) where one option has a mark next to or through it "
    "— mark ONLY that option, using (Circled) regardless of the mark's "
    "shape: Build : Normal, Obese, Asthenic(Circled)\n"
    "- A scored/graded list (e.g. a Performance Status scale 0-4, each "
    "on its own line) where one line's leading number is marked — mark "
    "ONLY that number: 0(Circled) - Asymptomatic fully ambulatory\n"
    "Wrap only the specific marked word or number in (Circled) — same "
    "tag for every mark shape — never the whole line or field. Only mark "
    "it when a drawn pen mark clearly applies to that exact word/number "
    "— when in doubt, leave it unmarked.\n"
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
    "- Short (one or two word) lines and lines with wide internal spacing "
    "are extracted, not skipped as if blank.\n"
    "- Hand-drawn clinical drawings (and ONLY those — no logos, emblems, "
    "barcodes, QR codes) each have one <img class=\"cut\" "
    "data-bbox=\"X,Y,W,H\"> tag in the HTML section, at their "
    "reading-order position, with percentage coordinates. Small "
    "shorthand symbols — triangle/Δ, arrows, ticks, circled numbers — "
    "are TEXT, never img.cut.\n"
    "- Every (?) word is wrapped in <span class=\"unc\"> in the HTML section.\n"
    "- (Circled) used ONLY with clear visual evidence; when unsure, leave "
    "the item unmarked.\n"
    "- Tabular reports formatted as tables; prescriptions line-by-line, "
    "never forced into a table.\n"
    "- Any Kannada/regional-script text is translated to English (names are "
    "transliterated, not translated).\n"
    "- Complaint lines start with C/o (Complains of) — never transcribed as "
    "\"y/o\" or \"40\".\n"
    "- Your FIRST output line is the topmost visible text on the page — "
    "IDs, UHID numbers, tokens, room numbers, URLs at the top edge "
    "included.\n"
    "- Printed labels with empty values are listed as Label : (blank), "
    "never skipped.\n"
    "- Any marked word/option/number anywhere on the page — circled, "
    "starred, ticked, underlined or crossed, not just inside a (Circle "
    "If Positive) list — is marked (Circled), only that word, never the "
    "whole line.\n"
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
    "- class=\"cut\" = hand-drawn clinical drawings ONLY (anatomy sketch, "
    "lesion map, marked-up outline): "
    "<img class=\"cut\" data-bbox=\"X,Y,W,H\" alt=\"[DIAGRAM: description]\"> — "
    "data-bbox REQUIRED, X,Y,W,H as percentages (0-100) of the page, tag "
    "placed at the drawing's reading-order position. NEVER img.cut for "
    "logos, emblems, medical symbols, barcodes, QR codes, watermarks.\n"
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
    "- Use the TRANSCRIBED TEXT provided as the content — it is already in "
    "English; do not copy any regional-script text from the image itself.\n"
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
) -> tuple[str, str | None, str | None, set]:
    """
    Extracts a single medical-record page.

    Performs a single vision-model read to produce:
    - Complete OCR transcription
    - Structured HTML layout
    - Detection of handwritten annotations
    - Preservation of form structure
    - Recognition of '(Circle If Positive)' checklists

    Returns:
        (extracted_text, layout_html, layout_error, quality_flags)
        The extracted text is mandatory. quality_flags records anomalies
        seen during the read (e.g. "cjk" = a CJK-garbage output occurred
        on some attempt) so the caller can force extra recovery passes.
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

    quality_flags = set()
    last_err = None
    compact_nudge = ""
    cjk_nudge = ""
    indic_nudge = ""
    for attempt in range(4):
        try:
            raw, done_reason = _ollama_chat(
                system_prompt,
                _PAGE_USER.format(page=page_num) + _LOCAL_USER_RULES
                + compact_nudge + cjk_nudge + indic_nudge,
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

            # CJK characters = decode glitch, never real page content on
            # these documents. At temperature 0 a bare retry reproduces
            # the same output, so nudge the prompt to change the sample.
            if text and _cjk_garbage(text + (layout or "")):
                quality_flags.add("cjk")
                print(f"[WARNING] Page {page_num}: output contains "
                      f"Chinese/Japanese garbage, attempt {attempt + 1}/4")
                cjk_nudge = (
                    " Your previous output contained Chinese/Japanese "
                    "characters — that was a decoding error. Re-read the "
                    "page and output English/Roman characters ONLY."
                )
                if attempt < 3:
                    time.sleep(3)
                    continue
                # Last attempt still garbled — fall through to the
                # text-only fallback below for an independent chance.
                text, layout = None, None

            # Regional-script (Kannada/Hindi/Tamil/...) leftovers mean the
            # LANGUAGE rule was ignored — nudge for a retry with full page
            # context (much more reliable than the isolated-fragment
            # cleanup pass that runs later as a backstop).
            if text and _has_indic(text + (layout or "")):
                quality_flags.add("indic")
                print(f"[WARNING] Page {page_num}: output still contains "
                      f"regional-script text, attempt {attempt + 1}/4")
                indic_nudge = (
                    " Your previous output still contained Kannada/Hindi/"
                    "Tamil/Telugu or other regional-script characters — "
                    "translate them into English now (transliterate names "
                    "into Roman letters instead); output English/Roman "
                    "characters ONLY."
                )
                if attempt < 3:
                    time.sleep(3)
                    continue
                # Last attempt still has it — keep the text (unlike CJK,
                # this may be genuine content); the residual-script
                # translate pass gets a final chance at cleanup.

            if text:
                if layout:
                    layout = _unhw_letterhead(text, layout)
                    text = _clean_checklist_sections(text)
                    layout = _clean_checklist_html(layout)
                    return text, _sanitize_html(layout), None, quality_flags
                # Combined call gave text but no layout section.
                print(f"[INFO] Page {page_num}: combined output had no layout "
                      f"section — running dedicated layout call.")
                text = _clean_checklist_sections(text)
                layout_html, _layout_err = _generate_layout(b64_image, page_num, text)
                if not layout_html:
                    layout_html = _raw_text_to_html(text)
                layout_html = _unhw_letterhead(text, layout_html)
                layout_html = _clean_checklist_html(layout_html)
                return text, _sanitize_html(layout_html), None, quality_flags
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
    if _cjk_garbage(text):
        # Something beats nothing — keep it, but leave the flag set so
        # the translate and verify passes still run on this page.
        quality_flags.add("cjk")
    text = _clean_checklist_sections(text)
    layout_html = _raw_text_to_html(text)
    layout_html = _clean_checklist_html(layout_html)
    return text, _sanitize_html(layout_html), None, quality_flags


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

# Shared guard for every focused single-purpose call (verify/header/
# audit) — these don't get the full _READ_SYSTEM prompt, so without
# this they are the passes most prone to bleeding a letterhead/facility
# name seen on an EARLIER page of a mixed document bundle into a LATER
# page from a different facility (e.g. an external lab report pasted
# into a hospital case file, with only a partial sticker/barcode
# visible at its top).
_NO_CROSS_PAGE_RULE = (
    "This may be one page from a bundle of documents from DIFFERENT "
    "facilities — never assume this page shares the same hospital, "
    "letterhead, or facility name as any other page you may have seen. "
    "Transcribe ONLY characters actually visible on THIS page image. If "
    "a logo, sticker, or letterhead is partially covered, folded, or cut "
    "off, transcribe only the visible fragment (or mark it (?) if "
    "unclear) — never complete it into a full name from memory."
)

_VERIFY_SYSTEM = (
    "You are a transcription completeness checker for medical documents. "
    "You receive ONE page image and a NUMBERED transcription of that page. "
    "Compare them and report ONLY lines of visible text (printed or "
    "handwritten) that are COMPLETELY MISSING from the transcription. "
    "Do NOT report spelling differences, re-wordings, formatting changes, "
    "or lines that are already present. "
    "Pay special attention to: the printed letterhead, report title and "
    "patient-details box at the VERY TOP of the page, handwritten lines "
    "separated by large vertical gaps, the right half of every row, "
    "margins and corners, and content below the last printed section. "
    "Transcribe each missing line exactly as written, appending (?) to any "
    "uncertain word. Any recovered line written in a regional script "
    "(Kannada, Hindi, Tamil, etc.) must be translated into English (names "
    "transliterated instead), the same as the rest of the transcription. "
    + _NO_CROSS_PAGE_RULE + " "
    'Output ONLY JSON: {"missing": [{"after_line": <int>, "text": "<line>"}]} '
    "where after_line is the transcription line number that the missing "
    "line appears BELOW on the page (0 = above the first line). "
    'If nothing is missing output {"missing": []}.'
)

_VERIFY_USER = (
    "Transcription of page {page} (numbered):\n\n{numbered}\n\n"
    "Compare against the page image and return the JSON now."
)


# ── Residual non-Latin script translation ────────────────────────────
# Safety net for the LANGUAGE prompt rules: if any non-Latin text still
# made it into the output, translate/clean the leftover fragments with
# one text-only model call. Covers Indic scripts (U+0900-U+0D7F:
# Devanagari through Malayalam, incl. Kannada/Hindi/Tamil/Telugu) plus
# CJK (Chinese/Japanese/Korean — those are Qwen decode glitches, not
# page content, and get deleted rather than translated).
_NONLATIN_CHARS = (
    "ऀ-ൿ"                     # Indic: U+0900-U+0D7F
    "　-〿぀-ヿㇰ-ㇿ"   # CJK punct + kana
    "㐀-䶿一-鿿豈-﫿"   # CJK ideographs
    "･-ﾟ가-힯"                # halfwidth kana + hangul
)
_NONLATIN_RUN_RX = re.compile(
    rf"[{_NONLATIN_CHARS}]+(?:[^\S\n]+[{_NONLATIN_CHARS}]+)*")

# CJK-only detector for the garbage gate in read_page.
_CJK_RX = re.compile(
    r"[　-〿぀-ヿㇰ-ㇿ㐀-䶿"
    r"一-鿿豈-﫿･-ﾟ가-힯]")


def _cjk_garbage(text: str) -> bool:
    """True when the text carries enough CJK to indicate a decode glitch."""
    return len(_CJK_RX.findall(text or "")) >= 3


_INDIC_RX = re.compile(r"[ऀ-ൿ]")


def _has_indic(text: str) -> bool:
    """True when the text still carries any Indic-script character."""
    return bool(_INDIC_RX.search(text or ""))


# ── Optional offline MT library ──────────────────────────────────────
# A dedicated translation model (trained specifically for kn/hi/ta/te/ml
# -> en) is far more reliable than asking a general vision-language
# model to translate an isolated fragment with no page context — this
# is used automatically when installed, with the existing model-based
# call as the fallback (nothing changes if it isn't installed).
#
# Setup (optional):
#   pip install argostranslate
#   python -c "import argostranslate.package as p; p.update_package_index(); \
#     pkgs = p.get_available_packages(); \
#     [p.install_from_path(x.download()) for x in pkgs \
#      if x.from_code in ('kn','hi','ta','te','ml') and x.to_code == 'en']"
try:
    import argostranslate.translate as _argos_translate
    _ARGOS_AVAILABLE = True
    print("[INFO] argostranslate detected — will be used for regional-"
          "script fragments when a matching language package is installed")
except ImportError:
    _argos_translate = None
    _ARGOS_AVAILABLE = False

# Unicode block -> ISO 639-1 code, for picking the right argos-translate
# language pair per fragment.
_SCRIPT_LANG_RANGES = (
    (0x0C80, 0x0D00, "kn"),  # Kannada
    (0x0900, 0x0980, "hi"),  # Devanagari (Hindi/Marathi)
    (0x0B80, 0x0C00, "ta"),  # Tamil
    (0x0C00, 0x0C80, "te"),  # Telugu
    (0x0D00, 0x0D80, "ml"),  # Malayalam
)


def _guess_indic_lang(fragment: str):
    for ch in fragment:
        cp = ord(ch)
        for lo, hi, code in _SCRIPT_LANG_RANGES:
            if lo <= cp < hi:
                return code
    return None


def _argos_translate_fragment(fragment: str):
    """
    Try the offline argos-translate library for one fragment. Returns
    translated text, or None when unavailable/no matching language
    package/translation failed — caller falls back to the model call.
    Never raises.
    """
    if not _ARGOS_AVAILABLE:
        return None
    lang = _guess_indic_lang(fragment)
    if not lang:
        return None
    try:
        out = (_argos_translate.translate(fragment, lang, "en") or "").strip()
        if out and out != fragment and not _NONLATIN_RUN_RX.search(out):
            return out
    except Exception as e:
        print(f"[WARNING] argos-translate {lang}->en failed ({e})")
    return None


_TRANSLATE_SYSTEM = (
    "You translate fragments of regional-script or Chinese/Japanese text "
    "found in a medical document into English. Translate the MEANING into "
    "natural English; if a fragment is a person/place/facility name, "
    "transliterate it into Roman letters instead of translating. Append "
    "(?) to uncertain words. If a fragment is meaningless OCR noise with "
    'no recoverable meaning, output "" for it. Output ONLY JSON: '
    '{"translations": [{"original": "<fragment exactly as given>", '
    '"english": "<English text>"}]} — one entry per fragment, same order.'
)


def _translate_residual_scripts(page_num: int, text: str,
                                layout_html: str) -> tuple:
    """
    Replace any remaining regional-script fragments in text/layout with
    English via one text-only call. Zero cost for English-only pages.
    Returns (text, layout_html) — unchanged on any failure; never raises.
    """
    try:
        combined = (text or "") + "\n" + (layout_html or "")
        frags = []
        for m in _NONLATIN_RUN_RX.finditer(combined):
            frag = m.group(0).strip()
            if frag and frag not in frags:
                frags.append(frag)
            if len(frags) >= 20:
                break
        if not frags:
            return text, layout_html

        # Prefer the dedicated offline MT library per fragment (better
        # quality, no model call needed); only the leftovers go to Ollama.
        applied = 0
        remaining = []
        for f in frags:
            eng = _argos_translate_fragment(f)
            if eng is None:
                remaining.append(f)
                continue
            replaced = False
            if text and f in text:
                text = text.replace(f, eng)
                replaced = True
            if layout_html and f in layout_html:
                layout_html = layout_html.replace(f, eng)
                replaced = True
            if replaced:
                applied += 1
        if len(remaining) < len(frags):
            print(f"[INFO] Page {page_num}: argos-translate handled "
                  f"{len(frags) - len(remaining)}/{len(frags)} fragment(s)")
        if not remaining:
            print(f"[INFO] Page {page_num}: translated/cleaned "
                  f"{applied}/{len(frags)} residual non-Latin fragment(s)")
            return text, layout_html

        user = ("Fragments:\n"
                + "\n".join(f"{i + 1}. {f}" for i, f in enumerate(remaining))
                + "\n\nReturn the JSON now.")
        raw, _ = _ollama_chat(_TRANSLATE_SYSTEM, user, [], 1500,
                              json_mode=True)
        entries = (_parse_json_loose(raw) if raw else {}).get(
            "translations") or []
        for e in entries:
            if not isinstance(e, dict):
                continue
            orig = str(e.get("original") or "").strip()
            eng = str(e.get("english") or "").strip()
            # eng == "" means the fragment is OCR noise — delete it.
            if not orig or _NONLATIN_RUN_RX.search(eng):
                continue
            replaced = False
            if text and orig in text:
                text = text.replace(orig, eng)
                replaced = True
            if layout_html and orig in layout_html:
                layout_html = layout_html.replace(orig, eng)
                replaced = True
            if replaced:
                applied += 1
        print(f"[INFO] Page {page_num}: translated/cleaned "
              f"{applied}/{len(frags)} residual non-Latin fragment(s)")
        return text, layout_html
    except Exception as e:
        print(f"[WARNING] Page {page_num}: residual-script translation "
              f"skipped ({e})")
        return text, layout_html


def _page_has_handwriting(text: str, layout_html: str) -> bool:
    """Heuristic: the page contains handwriting or uncertain words."""
    if layout_html and re.search(r'class\s*=\s*["\'][^"\']*\bhw\b',
                                 layout_html):
        return True
    return "(?)" in (text or "")


_HW_SPAN_RX = re.compile(
    r'<span[^>]*class\s*=\s*["\'][^"\']*\bhw\b[^"\']*["\'][^>]*>(.*?)</span>',
    re.DOTALL | re.IGNORECASE)
_INNER_TAG_RX = re.compile(r"<[^>]+>")


def _handwriting_fraction(text: str, layout_html: str) -> float:
    """
    Share (0.0-1.0) of the page's transcribed content that is
    handwriting, estimated as (characters inside <span class="hw">
    tags) / (total transcript characters). A page that's almost
    entirely printed with a trace of handwriting (e.g. one signed
    name on a lab printout) scores low even though it "has"
    handwriting in the binary sense of _page_has_handwriting.
    """
    if not text or not layout_html:
        return 0.0
    hw_chars = sum(len(_INNER_TAG_RX.sub("", m))
                   for m in _HW_SPAN_RX.findall(layout_html))
    total_chars = len(text)
    return hw_chars / total_chars if total_chars else 0.0


_SECTION_FIRST_RX = re.compile(
    r"^(?:COMPLETE\s+)?(?:HAEMOGRAM|HAEMATOLOGY|BIOCHEMISTRY|"
    r"DIFFERENTIAL\s+COUNT|.*\bREPORT)\s*$", re.IGNORECASE)


def _header_suspect(text: str) -> bool:
    """
    A printed report whose results start almost immediately — the
    letterhead / report title / patient-details box at the top was
    probably skipped. Used to force the verify pass on such pages.
    Suspect when a pipe-table row sits in the first 5 non-blank lines,
    or the page OPENS with a results-section heading with no
    Label : Value line before it.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if any(ln.count("|") >= 2 for ln in lines[:5]):
        return True
    if lines and _SECTION_FIRST_RX.match(lines[0]) \
            and not any(" : " in ln for ln in lines[:4]):
        return True
    return False


# ── Targeted header recovery ─────────────────────────────────────────
# Read JUST the top strip of the page image and prepend whatever lines
# the main read missed. Focused single-region transcription is far more
# reliable than asking the model to notice an omission on a full page.
# HEADER_CHECK: "1"/"always" (default) = run on every page — guarantees
# top-of-page content every run; "suspect" = only when _header_suspect
# fires (legacy, lower latency); "0" = off.
_HEADER_CHECK = os.getenv("HEADER_CHECK", "1").strip().lower()
if _HEADER_CHECK in ("1", "true", "yes", "on"):
    _HEADER_CHECK = "always"
elif _HEADER_CHECK not in ("suspect", "always"):
    _HEADER_CHECK = "off"
_HEADER_CROP_FRAC = 0.30
_HEADER_SYSTEM = (
    "You transcribe the TOP region of a medical document page. It "
    "typically contains the letterhead / facility name, a report title, "
    "and a printed patient-details box. Output EVERY visible text line, "
    "top to bottom — facility lines as written, patient details as "
    "Label : Value lines. Plain text only, no markdown, no commentary. "
    "English/Roman characters only (translate or transliterate any "
    "regional-script text). " + _NO_CROSS_PAGE_RULE
)


def _alpha_tokens(s: str) -> set:
    """Letters-only tokens (len>=2) — digit-agnostic, so two independent
    misreads of the same handwritten value (different digits, same
    labels/units) are still recognized as the same field."""
    return set(re.findall(r"[a-z]{2,}", (s or "").lower()))


def _recover_header(b64_image: str, page_num: int, text: str,
                    layout_html: str) -> tuple:
    """
    Transcribe the top strip of the page and insert any lines missing
    from the transcript (and layout). Insertion tracks the position of
    the LAST existing line the crop's own reply matched, rather than
    always prepending at the very top — a page whose main read already
    got the letterhead right (position 0) but under-transcribed the
    patient-details row just below it must have that row inserted AFTER
    the letterhead, not in front of it. When nothing in the crop matches
    anything existing (a genuinely header-less page), this naturally
    falls back to inserting everything at position 0, same as before.
    Never raises.
    """
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64_image)))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        pw, ph = img.size
        crop = img.crop((0, 0, pw, max(60, int(ph * _HEADER_CROP_FRAC))))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=85)
        hb64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        raw, _ = _ollama_chat(
            _HEADER_SYSTEM, "Transcribe this header region now.",
            [hb64], 1200)
        if not raw or _cjk_garbage(raw):
            return text, layout_html
        lines = (text or "").splitlines()
        search_n = min(len(lines), 20)
        existing = [set(re.findall(r"[a-z0-9]+", ln.lower()))
                    for ln in lines[:search_n]]
        existing_alpha = [_alpha_tokens(ln) for ln in lines[:search_n]]

        insert_at = 0
        groups = []          # [(position_in_original_lines, [new_line, ...])]
        current_pos = 0
        current_group = []
        added = 0
        for ln in raw.splitlines():
            ln = ln.strip()
            toks = set(re.findall(r"[a-z0-9]+", ln.lower()))
            if len(ln) < 3 or not toks:
                continue
            cand_alpha = _alpha_tokens(ln)
            match_idx = None
            # Containment ratio is relative to the CANDIDATE's own token
            # count, not the existing line's — an existing line can
            # legitimately be a longer combined line (e.g. one vitals
            # line covering BP + SpO2 + PR) that several independent,
            # narrower candidates each partially re-read; each candidate
            # must still be recognized as covered by it.
            for i in range(search_n):
                if toks and existing[i] and len(toks & existing[i]) >= 0.6 * len(toks):
                    match_idx = i
                    break
                # Digit-agnostic pass: catches the same field re-read with
                # different (misread) digits, e.g. "BP 130/80" vs "BP!-
                # 120/80" — same labels/units, different numbers — so it
                # isn't inserted as a spurious near-duplicate line.
                if (cand_alpha and existing_alpha[i]
                        and len(cand_alpha & existing_alpha[i])
                        >= 0.5 * len(cand_alpha)):
                    match_idx = i
                    break
            if match_idx is not None:
                if current_group:
                    groups.append((current_pos, current_group))
                    current_group = []
                insert_at = match_idx + 1
                current_pos = insert_at
                continue
            current_group.append(ln)
            added += 1
            if added >= 12:
                break
        if current_group:
            groups.append((current_pos, current_group))
        if not groups:
            print(f"[INFO] Page {page_num}: header check — nothing missing")
            return text, layout_html

        esc = (lambda s: s.replace("&", "&amp;")
               .replace("<", "&lt;").replace(">", "&gt;"))
        total_added = sum(len(g[1]) for g in groups)
        # Insert bottom-up (by position, descending) so earlier positions
        # stay valid as later ones are applied.
        for pos, grp_lines in sorted(groups, key=lambda g: g[0], reverse=True):
            lines[pos:pos] = grp_lines
            if layout_html:
                block = "<div>" + "<br>".join(esc(l) for l in grp_lines) + "</div>"
                if pos == 0:
                    layout_html = block + layout_html
                else:
                    anchor_line = lines[pos - 1] if pos - 1 < len(lines) else ""
                    layout_html = _insert_html_after_anchor(
                        layout_html, anchor_line, block)
        text = "\n".join(lines)
        print(f"[INFO] Page {page_num}: recovered {total_added} header "
              f"line(s) from top-strip read")
        return text, layout_html
    except Exception as e:
        print(f"[WARNING] Page {page_num}: header recovery skipped ({e})")
        return text, layout_html


def _insert_html_after_anchor(layout_html: str, anchor_line: str,
                              new_block: str) -> str:
    """
    Insert a pre-built HTML block right after the block containing the
    anchor line (a transcript line). Falls back to appending at the end
    of the fragment when the anchor cannot be located.
    """
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


def _find_anchor_span(layout_html: str, line: str, start: int = 0):
    """
    Locate `line`'s tokens inside layout_html (searching from `start`
    onward — pass a prior match's end when locating a SECOND anchor, so
    a repeated line elsewhere in the document isn't matched instead),
    trying progressively shorter trailing-token suffixes (same strategy
    as _insert_html_after_anchor). Returns (start, end) of the match or
    None.
    """
    tokens = re.findall(r"[A-Za-z0-9]+", line or "")[-6:]
    for k in range(len(tokens)):
        anchor_rx = re.compile(
            _ANCHOR_GAP.join(re.escape(w) for w in tokens[k:]),
            re.IGNORECASE)
        m = anchor_rx.search(layout_html, start)
        if m:
            return m.start(), m.end()
    return None


# ── Deterministic lab-table builder ──────────────────────────────────
# Guaranteed fallback for tabular reports: the model has repeatedly
# struggled with reference-range columns that visually merge across
# several test rows. Rather than keep tuning the prompt, detect a clear
# pipe-table block in the (already-extracted) text and, when the
# model's own layout has no real <table> for it, build one in code.
_TABLE_MIN_ROWS = 3


def _looks_like_table_block(lines: list) -> list:
    """Contiguous runs of >=2 lines each containing >=2 '|' characters,
    with at least _TABLE_MIN_ROWS rows. Returns [(start, end), ...]
    (end exclusive, indices into `lines`)."""
    blocks = []
    start = None
    for i, ln in enumerate(lines):
        if ln.count("|") >= 2:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= _TABLE_MIN_ROWS:
                blocks.append((start, i))
            start = None
    if start is not None and len(lines) - start >= _TABLE_MIN_ROWS:
        blocks.append((start, len(lines)))
    return blocks


def _build_table_html(rows: list) -> str:
    """
    Build a <table> from pipe-separated text rows. Column count is
    taken from the header (first) row; short rows are padded, long
    rows have their overflow merged into the last cell.
    """
    esc = (lambda s: s.replace("&", "&amp;").replace("<", "&lt;")
           .replace(">", "&gt;").strip())
    split_rows = [[c.strip() for c in r.split("|")] for r in rows]
    n = len(split_rows[0]) if split_rows else 0
    n = max(n, 1)

    def _pad(cells):
        if len(cells) < n:
            return cells + [""] * (n - len(cells))
        if len(cells) > n:
            return cells[:n - 1] + [" ".join(cells[n - 1:])]
        return cells

    out = ["<table>"]
    for ri, cells in enumerate(split_rows):
        cells = _pad(cells)
        tag = "th" if ri == 0 else "td"
        cells_html = "".join(f"<{tag}>{esc(c)}</{tag}>" for c in cells)
        out.append(f"<tr>{cells_html}</tr>")
    out.append("</table>")
    return "".join(out)


# A model-emitted "Table 1" / "Table:" caption line — pure LLM habit,
# never real page content on these documents.
_TABLE_CAPTION_RX = re.compile(r"(?im)^[ \t]*table[ \t]*\d*[ \t]*:?[ \t]*$")
_TABLE_CAPTION_HTML_RX = re.compile(
    r'(?is)<(div|p|h[1-6]|span|strong|b)\b[^>]*>\s*table[ \t]*\d*[ \t]*:?'
    r'\s*</\1\s*>')


def _strip_table_captions(text: str, layout_html: str) -> tuple:
    """Remove a hallucinated 'Table 1' / 'Table:' caption line/element.
    Never raises."""
    try:
        if text:
            text = "\n".join(ln for ln in text.splitlines()
                             if not _TABLE_CAPTION_RX.match(ln))
        if layout_html:
            layout_html = _TABLE_CAPTION_HTML_RX.sub("", layout_html)
        return text, layout_html
    except Exception:
        return text, layout_html


def _enforce_lab_tables(page_num: int, text: str, layout_html: str) -> tuple:
    """
    Guaranteed fallback: when the extracted text clearly contains a
    pipe-table block but the layout has no real <table> for it — or has
    one that is degenerate (far fewer cells than rows, e.g. the model
    merged everything into one column) — build a correct one
    deterministically and splice it in (anchored on the block's
    first/last line, else appended). A properly multi-column model
    table is left untouched. Never raises.
    """
    if not text or not layout_html:
        return text, layout_html
    try:
        lines = text.splitlines()
        blocks = _looks_like_table_block(lines)
        if not blocks:
            return text, layout_html

        low = layout_html.lower()
        if "<table" in low:
            total_rows = sum(end - start for start, end in blocks)
            cell_count = low.count("<td") + low.count("<th")
            if cell_count >= 2 * total_rows:
                return text, layout_html  # genuinely multi-column — leave it
            print(f"[INFO] Page {page_num}: replacing a degenerate table "
                  f"({cell_count} cells for {total_rows} row(s))")
            # Documents in this pipeline have at most one results grid
            # per page — drop it and rebuild below rather than risk
            # splicing into malformed nested tags.
            layout_html = re.sub(r"<table\b.*?</table\s*>", "", layout_html,
                                 flags=re.IGNORECASE | re.DOTALL, count=1)

        for start, end in blocks:
            table_html = _build_table_html(lines[start:end])
            span_start = _find_anchor_span(layout_html, lines[start])
            span_end = _find_anchor_span(layout_html, lines[end - 1])
            if span_start and span_end and span_end[1] >= span_start[0]:
                layout_html = (layout_html[:span_start[0]] + table_html
                               + layout_html[span_end[1]:])
            else:
                layout_html = layout_html + table_html
            print(f"[INFO] Page {page_num}: enforced deterministic table "
                  f"for {end - start} row(s)")
        return text, layout_html
    except Exception as e:
        print(f"[WARNING] Page {page_num}: table enforcement skipped ({e})")
        return text, layout_html


def _insert_line_into_layout(layout_html: str, anchor_line: str,
                             cand: str) -> str:
    """
    Insert a recovered text line into the layout HTML right after the
    block containing the anchor line (the transcript line directly above
    it on the page).
    """
    escaped = (cand.replace("&", "&amp;")
                   .replace("<", "&lt;").replace(">", "&gt;"))
    return _insert_html_after_anchor(
        layout_html, anchor_line,
        f'<div><span class="hw">{escaped}</span></div>')


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


# ── Dedicated diagram-detection pass ─────────────────────────────────
# The combined page call is unreliable at localizing hand-drawn clinical
# drawings (misses them, or crops logos). This focused grounding call is
# the single source of truth: it confirms/re-boxes the layout's img.cut
# tags, deletes unconfirmed ones, and inserts tags for missed drawings
# anchored beside their own handwritten annotations.
_DIAGRAM_SYSTEM = (
    "You locate two kinds of clinically meaningful GRAPHICS on ONE "
    "medical document page:\n"
    "(1) Hand-drawn clinical DRAWINGS — a pen-drawn FIGURE built from "
    "NON-LETTER SHAPES: an anatomy outline (breast, chest, abdomen, "
    "limb, organ), a lesion/tumour map, circles or ovals drawn to "
    "represent organs or masses (e.g. a pair of breast circles with "
    "dots, hatching or arrows), a marked-up body outline, a freehand "
    "clinical illustration — including small or faint ones. Handwritten "
    "labels inside or beside such a figure are part of it. A circle "
    "drawn AROUND an existing word or number is NOT a drawing.\n"
    "(2) Printed DIAGNOSTIC GRAPHS from a lab or imaging instrument "
    "printout — e.g. a CBC analyzer scattergram (WDF, WDF-CBC, RBC, PLT "
    "plots), an ECG waveform trace, or a similar printed chart showing "
    "raw diagnostic data/curves. These carry real clinical information "
    "and MUST be captured, unlike decorative or purely informational "
    "printed graphics.\n"
    "NEVER report:\n"
    "- handwritten words, sentences, numbers or lab values — even messy, "
    "slanted, crossed-out or hard-to-read handwriting is TEXT, not a "
    "drawing\n"
    "- small shorthand symbols: a triangle/delta, arrows, ticks/check "
    "marks, circled words or numbers, brackets, underlines, plus/minus "
    "signs\n"
    "- a small circle drawn around a single digit or letter, with an "
    "arrow pointing to a word or phrase (e.g. a circled '1' or circled "
    "'R' used as a footnote/reference marker) — this is NOT a drawing, "
    "even though it has an arrow and a label like a real anatomy figure "
    "would\n"
    "- hospital/clinic logos, letterhead emblems, medical symbols "
    "(caduceus, cross), barcodes, QR codes, stamps, signatures, "
    "watermarks, photographs, and PURELY DECORATIVE printed graphics "
    "(icons, borders) that carry no diagnostic data.\n"
    "If a region contains only letters, digits and punctuation, it is NOT "
    "a drawing or graph.\n"
    'Output ONLY JSON: {"diagrams": [{"bbox": [x, y, w, h], '
    '"description": "<what the drawing/graph shows>", '
    '"labels": "<handwritten words, or the graph/axis name, near it>"}]}. '
    "bbox: x,y = top-left corner, w,h = size, all PERCENTAGES (0-100) of "
    "the page. Cover the WHOLE drawing/graph including its labels. "
    'Nothing found -> {"diagrams": []}.'
)

_DIAGRAM_USER = ("Find every hand-drawn clinical drawing AND every "
                 "printed diagnostic graph/scattergram on this page, "
                 "and return the JSON now.")

# Second opinion on each candidate box: crop it and ask what it contains.
# Classifying a small crop is far easier than grounding, so this reliably
# rejects handwriting/symbol regions the detector mistook for drawings.
_DIAGRAM_VERIFY_SYSTEM = (
    "You classify ONE cropped region of a medical document page. "
    "Answer with exactly one category:\n"
    "drawing — the crop contains EITHER a hand-drawn clinical figure "
    "(an anatomy outline — breast, chest, abdomen, limb, organ — a "
    "lesion or tumour map, circles/ovals drawn to represent organs or "
    "masses, a marked-up body outline, or any pen figure made of "
    "non-letter shapes, however rough) OR a printed diagnostic graph "
    "from a lab/imaging instrument (a scattergram, plot, histogram, or "
    "waveform/ECG trace showing real diagnostic data). Handwritten "
    "labels or axis text around or inside it do not change the answer.\n"
    "text — the crop contains only handwriting or printed text: words, "
    "sentences, numbers, lab values — even messy, slanted or "
    "crossed-out.\n"
    "symbol — the crop contains only a small shorthand mark: a "
    "triangle/delta, arrow, tick, circled word or number, bracket, "
    "underline, or plus/minus sign. A small circle around a single "
    "digit or letter with an arrow to a word (a footnote/reference "
    "marker, e.g. circled '1' or circled 'R') is a symbol, not a "
    "drawing, even though it has an arrow and a label.\n"
    "Distinguish carefully: a circle or loop drawn AROUND an existing "
    "word or number (to select or highlight it) is text/symbol — but "
    "circles or ovals drawn as SHAPES representing anatomy (e.g. one or "
    "two breast circles with dots, hatching, arrows or labels pointing "
    "at them) are a drawing. If the circle would be meaningless without "
    "the word inside it, it is text; if it depicts a body part or "
    "lesion, it is a drawing. If you are still not sure, answer text.\n"
    'Output ONLY JSON: {"verdict": "drawing" | "text" | "symbol"}.'
)
_DIAGRAM_VERIFY_USER = "Classify this cropped region and return the JSON now."
_VERIFY_CROP_PAD_PCT = 2.0
_VERIFY_CROP_MAX_EDGE = 512


def _verify_diagram_crop(img, box, page_num: int):
    """
    Crop `box` (percent) out of the page image and ask the model what it
    contains. Returns "drawing" / "text" / "symbol", or None on any
    failure (caller fails open). Never raises.
    """
    try:
        pw, ph = img.size
        x, y, w, h = box
        left = max(0, int((x - _VERIFY_CROP_PAD_PCT) / 100 * pw))
        top = max(0, int((y - _VERIFY_CROP_PAD_PCT) / 100 * ph))
        right = min(pw, int((x + w + _VERIFY_CROP_PAD_PCT) / 100 * pw))
        bottom = min(ph, int((y + h + _VERIFY_CROP_PAD_PCT) / 100 * ph))
        if right - left < 4 or bottom - top < 4:
            return None
        crop = img.crop((left, top, right, bottom))
        scale = _VERIFY_CROP_MAX_EDGE / max(crop.size)
        if scale < 1:
            crop = crop.resize((max(1, int(crop.size[0] * scale)),
                                max(1, int(crop.size[1] * scale))))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        raw, _ = _ollama_chat(_DIAGRAM_VERIFY_SYSTEM, _DIAGRAM_VERIFY_USER,
                              [b64], 120, json_mode=True)
        verdict = str((_parse_json_loose(raw) if raw else {})
                      .get("verdict") or "").strip().lower()
        return verdict if verdict in ("drawing", "text", "symbol") else None
    except Exception as e:
        print(f"[WARNING] Page {page_num}: crop verification error ({e})")
        return None


def _esc_attr(s: str) -> str:
    """Make a model-supplied description safe inside an alt attribute."""
    return re.sub(r'["<>\[\]]', "", s or "").strip()[:80]


def _detect_diagrams(b64_image: str, page_num: int) -> tuple:
    """
    One focused JSON-grounding call. Returns (dets, page_size) where dets
    is a list of {"box": (x,y,w,h) percent, "desc": str, "labels": str},
    [] when the call ran and found nothing, or None when both attempts
    failed. Never raises.
    """
    page_size = None
    img = None
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64_image)))
        page_size = img.size
    except Exception:
        img = None
    pw, ph = page_size if page_size else (1000, 1414)
    last_err = None
    for attempt in range(2):
        try:
            raw, _ = _ollama_chat(_DIAGRAM_SYSTEM, _DIAGRAM_USER,
                                  [b64_image], 1500, json_mode=True)
            entries = (_parse_json_loose(raw) if raw else {}).get("diagrams")
            if not isinstance(entries, list):
                entries = []
            dets = []
            for e in entries:
                if not isinstance(e, dict):
                    continue
                bb = e.get("bbox")
                if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                    continue
                try:
                    vals = [float(v) for v in bb]
                except (TypeError, ValueError):
                    continue
                box = _normalize_bbox(vals[0], vals[1], vals[2], vals[3],
                                      pw, ph)
                if box is None or box[2] * box[3] < 1.0:
                    continue
                desc = str(e.get("description") or "").strip()
                if _LOGO_ALT_RX.search(desc):
                    continue
                # Bbox area alone can't catch reference marks (a circled
                # digit + arrow + label inflates the box past the area
                # floor). Require anatomy vocabulary in the description,
                # OR a substantive (>=2-word) labels field — reference
                # marks are a single digit/letter ("1", "R"), while real
                # drawings carry a real annotation phrase ("inflamed
                # cyst", "entire breast indurated") even when the
                # description itself doesn't happen to use listed words.
                labels_word_count = len(
                    str(e.get("labels") or "").split())
                if not _ANATOMY_RX.search(desc) and labels_word_count < 2:
                    print(f"[INFO] Page {page_num}: diagram candidate "
                          f"rejected — no anatomy vocabulary and no "
                          f"substantive labels ({desc!r})")
                    continue
                # Same drawing reported twice (overlapping or nested
                # boxes): merge into one covering box instead of keeping
                # both.
                dup = next((d for d in dets
                            if _same_region(box, d["box"])), None)
                if dup is not None:
                    dup["box"] = _union_box(dup["box"], box)
                    continue
                dets.append({
                    "box": box,
                    "desc": desc,
                    "labels": str(e.get("labels") or "").strip(),
                })
                if len(dets) >= _DIAGRAM_MAX_PER_PAGE:
                    break
            # Second opinion: crop each candidate and ask what it is —
            # rejects handwriting/symbol regions the grounding call
            # mistook for drawings. Fails CLOSED: only an explicit
            # "drawing" verdict keeps a candidate (a wrong text-crop is
            # worse for the reviewer than a missed drawing).
            if img is not None and dets:
                kept = []
                for d in dets:
                    verdict = _verify_diagram_crop(img, d["box"], page_num)
                    if verdict == "drawing":
                        kept.append(d)
                    else:
                        print(f"[INFO] Page {page_num}: diagram candidate "
                              f"rejected by crop check (verdict={verdict}, "
                              f"box={d['box']})")
                dets = kept
            print(f"[INFO] Page {page_num}: diagram detector found "
                  f"{len(dets)} drawing(s)")
            return dets, page_size
        except Exception as e:
            last_err = e
            if attempt == 0:
                time.sleep(3)
    print(f"[WARNING] Page {page_num}: diagram detection failed ({last_err})")
    return None, page_size


def _merge_diagram_boxes(text: str, layout_html: str, dets,
                         page_size) -> tuple:
    """
    Reconcile the layout's img.cut tags with the detector's findings.
    Detector ran (dets is a list): its boxes are authoritative — matched
    tags get the detector's bbox, unmatched tags are deleted (logos /
    hallucinations), unmatched detections are inserted anchored beside
    their annotation text. Detector failed (dets is None): tags are kept,
    except logo-alt ones. Returns (text, layout_html); never raises.
    """
    try:
        if not layout_html:
            return text, layout_html
        pw, ph = page_size if page_size else (1000, 1414)
        tag_rx = re.compile(
            r'<img\b[^>]*\bclass\s*=\s*["\'][^"\']*\bcut\b[^"\']*["\'][^>]*>',
            re.IGNORECASE)
        bbox_rx = re.compile(r'data-bbox\s*=\s*["\']([\d.,\s]+)["\']',
                             re.IGNORECASE)
        matches = list(tag_rx.finditer(layout_html))

        if dets is None:
            for m in reversed(matches):
                if _LOGO_ALT_RX.search(_tag_alt(m.group(0))):
                    layout_html = (layout_html[:m.start()]
                                   + layout_html[m.end():])
            return text, layout_html

        consumed = set()
        for m in reversed(matches):  # right-to-left keeps spans valid
            tag = m.group(0)
            box = None
            bm = bbox_rx.search(tag)
            if bm:
                try:
                    vals = [float(v) for v in bm.group(1).split(",")]
                    if len(vals) == 4:
                        box = _normalize_bbox(vals[0], vals[1], vals[2],
                                              vals[3], pw, ph)
                except (TypeError, ValueError):
                    box = None
            best_i, best_iou = -1, 0.0
            if box is not None:
                for i, d in enumerate(dets):
                    if i in consumed:
                        continue
                    iou = _bbox_iou(box, d["box"])
                    if iou > best_iou:
                        best_i, best_iou = i, iou
            if box is not None and best_iou >= _DIAGRAM_MATCH_IOU:
                d = dets[best_i]
                consumed.add(best_i)
                x, y, w, h = d["box"]
                desc = _esc_attr(d["desc"]) or _esc_attr(_tag_alt(tag)) \
                    or "hand-drawn diagram"
                new_tag = (f'<img class="cut" data-bbox='
                           f'"{x:.1f},{y:.1f},{w:.1f},{h:.1f}" '
                           f'alt="[DIAGRAM: {desc}]">')
                layout_html = (layout_html[:m.start()] + new_tag
                               + layout_html[m.end():])
            else:
                # Unconfirmed by the detector → logo or hallucination.
                layout_html = layout_html[:m.start()] + layout_html[m.end():]

        # Missed drawings: insert new tags anchored near their labels.
        # `placed` tracks every box already present (matched tags and
        # fresh inserts) so a second detection of the same drawing can
        # never produce a duplicate crop.
        placed = [dets[i]["box"] for i in consumed]
        lines = text.splitlines() if text else []
        line_tok = [set(re.findall(r"[a-z0-9]+", ln.lower()))
                    for ln in lines]
        inserted = 0
        for i, d in enumerate(dets):
            if i in consumed:
                continue
            if any(_same_region(d["box"], p) for p in placed):
                print(f"[INFO] diagram det skipped: duplicates an "
                      f"already-placed box {d['box']}")
                continue
            x, y, w, h = d["box"]
            desc = _esc_attr(d["desc"]) or "hand-drawn diagram"
            new_tag = (f'<img class="cut" data-bbox='
                       f'"{x:.1f},{y:.1f},{w:.1f},{h:.1f}" '
                       f'alt="[DIAGRAM: {desc}]">')
            label_toks = set(re.findall(
                r"[a-z0-9]+", (d["labels"] or "").lower()))
            anchor_idx, best = -1, 0
            if label_toks:
                need = 2 if len(label_toks) >= 3 else 1
                for j, toks in enumerate(line_tok):
                    ov = len(toks & label_toks)
                    if ov > best and ov >= need:
                        best, anchor_idx = ov, j
            desc_toks = set(re.findall(r"[a-z0-9]+", desc.lower()))
            has_marker = any(
                ln.strip().upper().startswith("[DIAGRAM") and desc_toks
                and len(set(re.findall(r"[a-z0-9]+", ln.lower()))
                        & desc_toks) >= 0.7 * len(desc_toks)
                for ln in lines)
            marker = f"[DIAGRAM: {desc}]"
            if anchor_idx >= 0:
                layout_html = _insert_html_after_anchor(
                    layout_html, lines[anchor_idx], new_tag)
                if not has_marker:
                    lines.insert(anchor_idx + 1, marker)
                    line_tok.insert(anchor_idx + 1, set())
            elif y + h / 2 < 33:  # top third of the page
                layout_html = new_tag + layout_html
                if not has_marker:
                    lines.insert(0, marker)
                    line_tok.insert(0, set())
            else:
                layout_html = layout_html + new_tag
                if not has_marker:
                    lines.append(marker)
                    line_tok.append(set())
            placed.append(d["box"])
            inserted += 1
        if inserted and lines:
            text = "\n".join(lines)
        return text, layout_html
    except Exception as e:
        print(f"[WARNING] diagram merge failed ({e})")
        return text, layout_html


def _should_detect(text: str, layout_html: str) -> bool:
    """Run the diagram detector only when the page shows any signal."""
    if _page_has_handwriting(text, layout_html):
        return True
    if "[DIAGRAM" in (text or ""):
        return True
    return bool(layout_html and re.search(
        r'class\s*=\s*["\'][^"\']*\bcut\b', layout_html))


# ── Wrong-word audit pass ────────────────────────────────────────────
# Self-marked (?) uncertainty has a ceiling: a model that misreads
# confidently never marks itself. This pass shows the model the finished
# transcript AS ANOTHER TRANSCRIBER'S WORK next to the page image and
# asks it to flag words that don't match — the critic framing removes
# the self-agreement bias. Flagged words are MARKED (gain a "(?)" and
# thus the red highlight), never silently replaced: the client corrects.
_AUDIT = os.getenv("AUDIT_WORDS", "1").strip().lower() in (
    "1", "true", "yes", "on")
# A page that's overwhelmingly printed with just a trace of handwriting
# (e.g. one signed name on a lab printout) skips the full-page audit
# critique entirely — that pass was reviewing every line, including the
# printed majority, and false-flagging correct printed labels near the
# handwritten bit. Below this share of handwritten content, audit is
# skipped UNLESS the page has genuine self-marked uncertainty (?),
# which always still triggers it regardless of the fraction.
_AUDIT_MIN_HW_FRACTION = float(os.getenv("AUDIT_MIN_HW_FRACTION", "0.15"))

_AUDIT_SYSTEM = (
    "You are checking ANOTHER transcriber's work. You receive one medical "
    "document page image and their NUMBERED transcription of it. Compare "
    "every line against the image and report words that are WRONG or "
    "doubtful — misread HANDWRITING, wrong numbers or dosages, wrong "
    "clinical shorthand. Do NOT report correct words, formatting, "
    "spacing, or style choices. Do NOT report words already marked "
    "with (?). Printed or typed text — headings, titles, form labels — "
    "is clearly legible and essentially never wrong; do NOT flag it "
    "unless you can see an actual spelling error in the printed text "
    "itself. Reserve flags for content that could plausibly be "
    "misread: handwriting, numbers, dosages, and shorthand. When in "
    "doubt about a printed word, do not flag it. "
    "A printed field LABEL with no filled-in value (a blank field) is "
    "NOT inherently suspicious — do not flag a blank field's label just "
    "because the field is empty or that part of the page/photo is "
    "creased, curled, faded, or otherwise lower quality. Judge each "
    "word on its own: only flag a label if its OWN letters look "
    "different from the transcription, never because a NEARBY region "
    "of the image is harder to read. " + _NO_CROSS_PAGE_RULE + " "
    'Output ONLY JSON: {"wrong": [{"line": <line number>, '
    '"word": "<the exact word as transcribed>", '
    '"correct": "<your best reading from the image>"}]}. '
    'Maximum 25 entries. If everything matches, output {"wrong": []}.'
)

_AUDIT_USER = (
    "Their transcription of page {page} (numbered):\n\n{numbered}\n\n"
    "Compare it word by word against the page image and return the "
    "JSON now."
)


# A line that is ENTIRELY a printed field label with nothing filled in
# after it (e.g. "Pos.:", "Doctor:", "Sex:") — a blank field. There is
# nothing to verify against the image (no content was transcribed), so
# flagging its label is never useful — suppress deterministically rather
# than trust the model to keep following the equivalent prompt rule.
_BLANK_LABEL_LINE_RX = re.compile(r"^[^:]{1,40}:\s*$")


def _strip_blank_field_uncertainty(text: str, layout_html: str) -> tuple:
    """
    Remove any "(?)" marker sitting on a line that is nothing but a
    blank field's label — regardless of whether the main read
    self-marked it or the audit pass flagged it. Never raises.
    """
    if not text or "(?)" not in text:
        return text, layout_html
    try:
        changed = []
        out_lines = []
        for ln in text.splitlines():
            bare = re.sub(r"\(\?\)", "", ln).strip()
            if "(?)" in ln and _BLANK_LABEL_LINE_RX.match(bare):
                cleaned = ln.replace("(?)", "")
                changed.append((ln.strip(), cleaned.strip()))
                out_lines.append(cleaned)
            else:
                out_lines.append(ln)
        if not changed:
            return text, layout_html
        text = "\n".join(out_lines)
        if layout_html:
            for orig, cleaned in changed:
                layout_html = layout_html.replace(orig, cleaned)
        return text, layout_html
    except Exception as e:
        print(f"[WARNING] blank-field uncertainty cleanup skipped ({e})")
        return text, layout_html


def _audit_words(b64_image: str, page_num: int, text: str,
                 layout_html: str) -> tuple:
    """
    One critic call that flags misread words. Each accepted flag appends
    "(?)" to the word in text AND layout, so the existing _wrap_uncertain
    pipeline turns it into a highlighted span. Marks only — never
    replaces the transcribed word. Blank-field label lines are never
    flagged, regardless of what the model says. Never raises.
    """
    try:
        lines = text.splitlines()
        numbered = "\n".join(f"{i + 1}: {ln}" for i, ln in enumerate(lines))
        raw, _ = _ollama_chat(
            _AUDIT_SYSTEM,
            _AUDIT_USER.format(page=page_num, numbered=numbered),
            [b64_image],
            1500,
            json_mode=True,
        )
        entries = (_parse_json_loose(raw) if raw else {}).get("wrong") or []
        if not isinstance(entries, list):
            return text, layout_html

        flagged = []
        for e in entries[:25]:
            if not isinstance(e, dict):
                continue
            word = str(e.get("word") or "").strip()
            sugg = str(e.get("correct") or "").strip()
            if not word or len(word) < 2 or "(?)" in word:
                continue
            try:
                ln_no = int(e.get("line", 0)) - 1
            except (TypeError, ValueError):
                continue
            # Whole word, not already followed by a (?) marker.
            word_rx = re.compile(
                r"(?<![A-Za-z0-9])" + re.escape(word)
                + r"(?![A-Za-z0-9])(?!\(\?\))")
            hit = None
            for j in (ln_no, ln_no - 1, ln_no + 1):
                if 0 <= j < len(lines) and word_rx.search(lines[j]):
                    hit = j
                    break
            if hit is None:
                continue  # auditor hallucinated a word — reject
            if _BLANK_LABEL_LINE_RX.match(lines[hit].strip()):
                continue  # blank field label — never flagged
            lines[hit] = word_rx.sub(word + "(?)", lines[hit], count=1)
            flagged.append((word, sugg))

        if not flagged:
            print(f"[INFO] Page {page_num}: audit found no wrong words")
            return text, layout_html
        text = "\n".join(lines)

        # Layout: same word -> word(?) once, text nodes only (never
        # inside tags/attributes).
        if layout_html:
            for word, _s in flagged:
                rx = re.compile(
                    r"(?<![A-Za-z0-9])" + re.escape(word)
                    + r"(?![A-Za-z0-9])(?!\(\?\))")
                parts = re.split(r"(<[^>]*>)", layout_html)
                for idx, part in enumerate(parts):
                    if part.startswith("<"):
                        continue
                    new_part, n = rx.subn(word + "(?)", part, count=1)
                    if n:
                        parts[idx] = new_part
                        break
                layout_html = "".join(parts)

        summary = ", ".join(f"{w}→{s or '?'}" for w, s in flagged)
        print(f"[INFO] Page {page_num}: audit flagged {len(flagged)} "
              f"word(s): {summary}")
        return text, layout_html
    except Exception as e:
        print(f"[WARNING] Page {page_num}: audit pass skipped ({e})")
        return text, layout_html


# ── Sparse-page rescue ───────────────────────────────────────────────
# When a page comes back essentially empty, the most common cause is a
# rotation the orientation probe missed. Outcome-triggered: retry the
# text-only read on rotated copies and keep the first useful result.
_RESCUE_MIN_LINES = 3


def _rescue_sparse_page(b64_image: str, page_num: int):
    """
    Try 90/270/180 rotations, one text-only read each. Returns
    (text, rotated_b64) for the first rotation yielding at least
    _RESCUE_MIN_LINES non-blank lines, else None. Never raises.
    """
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64_image)))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
    except Exception as e:
        print(f"[WARNING] Page {page_num}: rescue decode failed ({e})")
        return None
    for deg in (90, 270, 180):
        try:
            rotated = img.rotate(-deg, expand=True)
            buf = io.BytesIO()
            rotated.save(buf, format="JPEG", quality=90, optimize=True)
            rb64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            raw, _ = _ollama_chat(
                _READ_SYSTEM + _LOCAL_RULES,
                _READ_USER.format(page=page_num) + _LOCAL_USER_RULES,
                [rb64],
                _VISION_MAX_TOKENS,
            )
            lines = sum(1 for ln in (raw or "").splitlines() if ln.strip())
            if lines >= _RESCUE_MIN_LINES:
                print(f"[INFO] Page {page_num}: sparse output rescued by "
                      f"{deg}° rotation ({lines} lines)")
                return raw, rb64
            print(f"[INFO] Page {page_num}: rescue rotation {deg}° gave "
                  f"only {lines} line(s)")
        except Exception as e:
            print(f"[WARNING] Page {page_num}: rescue rotation {deg}° "
                  f"failed ({e})")
    return None


# ── Deterministic checklist container ────────────────────────────────
# The bordered box around the "(Circle If Positive)" checklist must not
# depend on the model remembering the class on every run — when the
# marker is present in the text but the layout has no checklist div,
# wrap the checklist's top-level blocks in one programmatically.
_CHECKLIST_END_ANCHORS = (
    "complaints and duration", "history of present illness",
    "general examination", "clinical impression", "diagnosis",
    "investigation", "proposed")


def _split_top_level(fragment: str) -> list:
    """
    Split an HTML fragment into tag-balanced top-level chunks so ranges
    of chunks can be wrapped without breaking element nesting.
    """
    chunks = []
    depth = 0
    start = 0
    void_rx = re.compile(r"<(?:br|img|hr|input|meta|link)\b", re.IGNORECASE)
    for m in re.finditer(r"<[^>]*>", fragment):
        tag = m.group(0)
        if tag.startswith("</"):
            if depth > 0:
                depth -= 1
                if depth == 0:
                    chunks.append(fragment[start:m.end()])
                    start = m.end()
        elif tag.endswith("/>") or void_rx.match(tag) or tag.startswith("<!"):
            if depth == 0:
                chunks.append(fragment[start:m.end()])
                start = m.end()
        else:
            if depth == 0 and m.start() > start:
                chunks.append(fragment[start:m.start()])
                start = m.start()
            depth += 1
    if start < len(fragment):
        chunks.append(fragment[start:])
    return chunks


# Matches both the real form's own wording ("Circle Positive") and the
# more common generic phrasing ("Circle If Positive") — the two forms
# seen across different documents/sections, and also what a hallucinated
# echo of this project's own prompt vocabulary would produce.
_CHECKLIST_HEADING_RX = re.compile(r"circle\s*(?:if\s*)?positive", re.IGNORECASE)


def _dedupe_checklist(page_num: int, text: str, layout_html: str) -> tuple:
    """
    If the printed symptom checklist ("Circle Positive" / "Circle If
    Positive" section: GENERAL, LYMPH NODES, ... FAMILY HISTORY) appears
    more than once in the extracted text — a model/verify-pass
    duplication seen in practice — keep only the FIRST occurrence and
    remove the rest, from both text and layout. Never raises.
    """
    if not text:
        return text, layout_html
    try:
        lines = text.splitlines()
        starts = [i for i, ln in enumerate(lines)
                 if _CHECKLIST_HEADING_RX.search(ln)]
        if len(starts) < 2:
            return text, layout_html
        ranges = []
        for k, s in enumerate(starts):
            limit = starts[k + 1] if k + 1 < len(starts) else len(lines)
            end = next((i for i in range(s + 1, limit)
                       if any(a in lines[i].lower()
                              for a in _CHECKLIST_END_ANCHORS)),
                      limit)
            ranges.append((s, end))
        dupes = ranges[1:]
        removed = [lines[s:e] for s, e in dupes]
        for s, e in sorted(dupes, key=lambda r: r[0], reverse=True):
            del lines[s:e]
        text = "\n".join(lines)
        if layout_html:
            for grp in removed:
                if not grp:
                    continue
                span_start = _find_anchor_span(layout_html, grp[0])
                span_end = (_find_anchor_span(layout_html, grp[-1],
                                              start=span_start[1])
                           if span_start else None)
                if span_start and span_end and span_end[1] >= span_start[0]:
                    layout_html = (layout_html[:span_start[0]]
                                   + layout_html[span_end[1]:])
        print(f"[INFO] Page {page_num}: removed {len(dupes)} duplicate "
              f"checklist occurrence(s)")
        return text, layout_html
    except Exception as e:
        print(f"[WARNING] Page {page_num}: checklist dedupe skipped ({e})")
        return text, layout_html


def _ensure_checklist_container(text: str, layout_html: str) -> str:
    """
    Guarantee the printed checklist is wrapped in <div class="checklist">
    whenever the page has a "(Circle If Positive)" section. No-op when
    the model already emitted the container. Never raises.
    """
    if not layout_html or not text:
        return layout_html
    if not _CHECKLIST_HEADING_RX.search(text):
        return layout_html
    if re.search(r'class\s*=\s*["\'][^"\']*\bchecklist\b', layout_html):
        return layout_html
    try:
        chunks = _split_top_level(layout_html)

        def chunk_text(c):
            return re.sub(r"<[^>]*>", " ", c).lower()

        start_i = next((i for i, c in enumerate(chunks)
                        if _CHECKLIST_HEADING_RX.search(chunk_text(c))), None)
        if start_i is None:
            return layout_html
        # Both columns inside one big chunk → the model already built its
        # own structure; wrapping would swallow the handwritten column.
        if any(a in chunk_text(chunks[start_i])
               for a in _CHECKLIST_END_ANCHORS):
            return layout_html
        end_i = next((i for i in range(start_i + 1, len(chunks))
                      if any(a in chunk_text(chunks[i])
                             for a in _CHECKLIST_END_ANCHORS)),
                     len(chunks))
        # Float the checklist left so the handwritten column that follows
        # (complaints/history) naturally wraps beside it, matching the
        # source form's two-column layout. A clearing div at the very
        # end of the page bounds the float to this page only.
        wrapped = ('<div class="checklist" '
                   'style="float:left;width:46%;margin:0 3% 8px 0;'
                   'box-sizing:border-box;">'
                   + "".join(chunks[start_i:end_i]) + "</div>")
        print("[INFO] checklist container added programmatically "
              "(floated two-column)")
        return ("".join(chunks[:start_i]) + wrapped
               + "".join(chunks[end_i:]) + '<div style="clear:both"></div>')
    except Exception as e:
        print(f"[WARNING] checklist wrap skipped ({e})")
        return layout_html


# ── Deterministic shorthand repair ───────────────────────────────────
# The model keeps misreading handwritten "C/o" (Complains of) at the
# start of complaint lines as "yo"/"y/o"/"4o" despite prompt guidance.
# Line-start anchoring (with optional indentation/bullet) keeps
# "45 y/o female" (age) safe, and plain "40" is deliberately excluded
# so dosages are never touched.
_SHORTHAND_LINE_RX = re.compile(
    r"(?im)^([ \t]*(?:[-–•*]\s*)?)(?:y/o|yo|4o)\b(?=[ \t]+\S)")
# HTML variant 1: right after an hw-span opens.
_SHORTHAND_HTML_RX = re.compile(
    r'(?is)(<span[^>]*\bclass\s*=\s*"[^"]*\bhw\b[^"]*"[^>]*>\s*'
    r'(?:[-–•*]\s*)?)'
    r'(?:y/o|yo|4o)\b(?=[ \t]+\S)')
# HTML variant 2: at the start of any block element (optionally through
# inline wrapper tags and a bullet) — covers layouts that don't use the
# hw class. Never fires mid-sentence.
_SHORTHAND_HTML_BLOCK_RX = re.compile(
    r'(?is)((?:\A|<(?:div|p|li|td|th|h[1-6])[^>]*>|<br\s*/?>)\s*'
    r'(?:<[^>]+>\s*)*(?:[-–•*]\s*)?)'
    r'(?:y/o|yo|4o)\b(?=[ \t]+\S)')


def _fix_shorthand(text: str) -> str:
    return _SHORTHAND_LINE_RX.sub(r"\1C/o", text) if text else text


def _fix_shorthand_html(html: str) -> str:
    if not html:
        return html
    html = _SHORTHAND_HTML_RX.sub(r"\1C/o", html)
    return _SHORTHAND_HTML_BLOCK_RX.sub(r"\1C/o", html)


def _read_page_full(b64_image: str, page_num: int, mime: str) -> tuple:
    """
    read_page + sparse-page rescue + missed-line verification + diagram
    detection + shorthand repair + residual-script translation + unc
    wrapping.

    Returns (text, layout_html, layout_error, effective_b64) —
    effective_b64 is the page image every produced bbox refers to (it
    differs from the input only when the rescue rotated the page), and
    MUST be the image crops are cut from.
    """
    text, layout_html, layout_error, quality_flags = read_page(
        b64_image, page_num, mime)

    # Essentially-empty page: most likely a rotation the orientation
    # probe missed — retry rotated before any downstream pass.
    nonblank = sum(1 for ln in (text or "").splitlines() if ln.strip())
    if nonblank < _RESCUE_MIN_LINES:
        rescued = _rescue_sparse_page(b64_image, page_num)
        if rescued:
            text, b64_image = rescued
            text = _clean_checklist_sections(text)
            layout_html = _sanitize_html(_raw_text_to_html(text))
            layout_error = None

    if _VERIFY and text and ("cjk" in quality_flags
                             or _header_suspect(text)
                             or _page_has_handwriting(text, layout_html)):
        text, layout_html = _verify_completeness(
            b64_image, page_num, text, layout_html)
    # Top-strip check: guarantees top-of-page content (IDs, tokens,
    # titles, patient box) survives every run. "always" mode runs it on
    # every page — the merge only prepends genuinely missing lines.
    if text and (_HEADER_CHECK == "always"
                 or (_HEADER_CHECK == "suspect" and _header_suspect(text))):
        text, layout_html = _recover_header(
            b64_image, page_num, text, layout_html)
    if _DETECT_DIAGRAMS and text and _should_detect(text, layout_html):
        dets, page_size = _detect_diagrams(b64_image, page_num)
        text, layout_html = _merge_diagram_boxes(
            text, layout_html, dets, page_size)
    # Strip any hallucinated "Table 1" caption before table enforcement,
    # so it can't get swept up as part of a spliced-in table's anchor.
    text, layout_html = _strip_table_captions(text, layout_html)
    # Guaranteed table fallback: builds a real <table> in code when the
    # text is clearly tabular but the model's own layout has none (or a
    # degenerate one).
    text, layout_html = _enforce_lab_tables(page_num, text, layout_html)
    # Wrong-word audit: critic pass flags misread words with (?) so they
    # come out red for the client to correct. Runs when the page has
    # genuine self-marked uncertainty, OR when handwriting makes up a
    # meaningful share of the page (not just a trace) — a page that's
    # almost entirely printed skips the full-page critique so it can't
    # false-flag correct printed labels near a small handwritten bit.
    if _AUDIT and text and (
            "(?)" in text
            or _handwriting_fraction(text, layout_html)
            >= _AUDIT_MIN_HW_FRACTION):
        text, layout_html = _audit_words(
            b64_image, page_num, text, layout_html)
    text = _fix_shorthand(text)
    layout_html = _fix_shorthand_html(layout_html)
    # Remove any duplicate "Circle (If) Positive" checklist occurrence
    # BEFORE bordering — the border must apply to a single, deduped copy.
    text, layout_html = _dedupe_checklist(page_num, text, layout_html)
    layout_html = _ensure_checklist_container(text, layout_html)
    text, layout_html = _translate_residual_scripts(
        page_num, text, layout_html)
    # Universal backstop: a blank field's label is never highlighted,
    # whichever pass marked it — self-uncertainty during the main read,
    # or the audit critic. Runs last so it catches every source.
    text, layout_html = _strip_blank_field_uncertainty(text, layout_html)
    if layout_html:
        layout_html = _wrap_uncertain(layout_html)
    return text, layout_html, layout_error, b64_image


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
            # ExtractorError propagates. effective_b64 is the image every
            # bbox refers to — it differs from b64 only when the sparse-
            # page rescue rotated the page, and crops MUST be cut from it.
            text, layout_html, layout_error, effective_b64 = \
                page_futs[n].result()
            if layout_html:
                # Cut the real drawings out of the page image and embed
                # them where the detector/model marked data-bbox.
                layout_html = _embed_region_crops(
                    layout_html, base64.b64decode(effective_b64))
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

