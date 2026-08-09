# Medical Prescription Extraction (Offline)

Extracts text, layout, and structured data from doctor prescriptions
(image or PDF) using an **offline Ollama vision model**
(`qwen2.5vl:72b` in production on a GPU box; `qwen2.5vl:7b` /
`qwen3-vl:4b-instruct` for local dev). No cloud AI APIs — only the
optional Azure SQL feedback store leaves the machine. Runs on port
**8002**.

## Files

| File | Role |
|---|---|
| `backend.py` | FastAPI app: upload/job endpoints, correction layers, layout word-patching |
| `extractor.py` | The whole extraction pipeline: prompts + all model calls + post-passes |
| `medgemma_corrector.py` | Optional MedGemma spelling layer (disabled by default) |
| `feedback_store.py` | Azure SQL persistence: corrections, learned spellings, dictionary rules |
| `index.html` | Entire frontend (React 18 UMD + Babel + Tailwind via CDN, no build step) |
| `data/known_drugs.txt` | Drug-name whitelist protecting real drugs from "correction" |
| `.env` | All configuration (never commit; `.env.example` is the template) |

## User flow

1. **Upload** — drag & drop prescription images (JPG/PNG/WEBP) or PDFs;
   multiple files queue and extract one at a time (UI polls every 4 s).
2. **Review** — original document beside the **Reconstructed Layout**:
   - Words needing review are shown in **red** (`(?)` marker — the
     model's own doubts plus words flagged by the audit pass) with a
     count chip; everything else renders pure black regardless of pen
     ink color. The human reviews only the red words.
   - Hand-drawn clinical diagrams are cropped from the page and embedded
     at their original position (right-half drawings float right).
   - **Edit** makes the layout editable in place (Ctrl+Enter saves,
     Esc cancels). Saving diffs your edit into a text correction, stores
     it (`/api/raw-feedback`), and the system replays + learns from it.
   - Footer: page count + **PDF** and **Word** download of the
     reconstructed layout (Word = editable text, images embedded).
3. **Structured Data** — extracted fields + medicines table, CSV export.

## Extraction pipeline (per page, in order)

All in `extractor.py`; every pass degrades gracefully — a pass failure
never fails the document. Log lines identify each pass.

1. **Render** — PDF pages at 300 DPI (pypdfium2); direct images
   re-encoded. Orientation probe uprights sideways photos (180° for
   portrait, 90/270 for landscape).
2. **Combined read** (`read_page`) — ONE call returns
   `===RAW TEXT===` + `===LAYOUT HTML===` (single reading, both views
   agree). 4 attempts with recovery nudges:
   - truncation → "be more compact" retry;
   - **CJK garbage** (Chinese/Japanese chars = decode glitch) → retry
     with corrective nudge, page flagged `cjk`;
   - total failure → text-only fallback call.
3. **Sparse-page rescue** — a page with < 3 lines is re-read rotated
   90/270/180 (orientation probe missed it); the rotated image becomes
   the page's effective image so crop coordinates stay consistent.
4. **Verify pass** (`VERIFY_MISSED_LINES=1`) — pages with handwriting
   (or `cjk`/header-suspect flags) are sent again with a numbered
   transcript; completely missed lines are merged back in at the right
   position. Printed-only clean pages skip this (no extra latency).
5. **Header check** (`HEADER_CHECK=1`) — the top 30% of every page is
   transcribed separately and any missing lines (UHID/tokens/titles,
   letterhead, patient-details box) are prepended; pages the main read
   got right are untouched (dedup by token overlap).
6. **Diagram detection** (`DETECT_DIAGRAMS=1`) — a focused grounding
   call finds hand-drawn clinical drawings ONLY (anatomy sketches,
   lesion maps — never logos/emblems/barcodes/QR). Each candidate box is
   then **crop-verified** by a second small call (drawing / text /
   symbol); only explicit "drawing" survives (fail-closed — a wrong
   handwriting crop is worse than a missed diagram). Detector boxes are
   authoritative: they re-box matched layout tags, delete unconfirmed
   ones, and insert missed drawings anchored beside their annotations.
   Duplicate/nested boxes are union-merged (`_same_region`: IoU or
   intersection-over-smaller-box).
7. **Wrong-word audit** (`AUDIT_WORDS=1`) — a critic call sees the
   finished transcript as ANOTHER transcriber's work beside the page
   image and flags misread words (validated against the stated line,
   marks only — never silently replaces); flagged words gain `(?)` and
   come out red. Handwritten pages only.
8. **Deterministic repairs** (code, not model — identical every run):
   - `C/o` shorthand: line-start `yo`/`y/o`/`4o` → `C/o`;
   - checklist container: "(Circle If Positive)" section wrapped in a
     bordered `.checklist` div if the model forgot;
   - prompt-echo guard + repetition collapse.
9. **Language** — regional-script text (Kannada/Hindi/Tamil/…) is
   translated to English by prompt rules (names transliterated, never
   translated); any residual non-Latin fragments (incl. CJK noise) get
   one text-only translate/clean call.
10. **Uncertainty wrapping** — every `(?)` word is force-wrapped in
    `<span class="unc">` server-side (the red flagged-word styling never
    depends on the model remembering the class). Ink colors from the
    page are stripped — red in the output always means "verify this".
11. **Crop embedding** — `data-bbox` tags become real JPEG crops
    (data: URIs) with position-mirroring float styles; logo-alt tags are
    deleted; pixel-coordinate boxes auto-convert to percentages.

**Document-level:** one structured-fields JSON call per ≤5-page chunk
(chunks merged, fields deduped); then correction layers in `backend.py`:
MedGemma (off by default) → dictionary autocorrect → human-correction
replay (`_patch_words` anchors corrections into the layout).

## Learning loop

Human edits (Edit-in-layout → Save) are stored per document
(`raw_corrections`, keyed by file hash) and replayed on re-upload;
word-level pairs feed few-shot spelling examples into future prompts;
pairs confirmed 3+ times become deterministic dictionary rules
(Dictionary panel in the UI). Everything lives in its own SQL schema,
`PrescriptionExtractionOffline`. DB unreachable = extraction still
works, learning disabled.

## Setup

```powershell
# 1. dependencies (Python 3.10+; conda env "extract" already has them)
pip install -r requirements.txt

# 2. pull a vision model
ollama pull qwen2.5vl:72b     # production (GPU box, ~47 GB)
ollama pull qwen2.5vl:7b      # local dev (~6 GB)

# 3. copy .env.example to .env and fill in values
```

## Run

```powershell
python backend.py        # or .\run.ps1
```

Open **http://localhost:8002**. On startup the log prints
`[INFO] extractor build YYYY-MM-DD-rN` — **always check this line after
deploying** (a stale `git pull` is otherwise invisible).

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `APP_HOST` | 127.0.0.1 | Bind address (0.0.0.0 on a cloud VM) |
| `OLLAMA_HOST` | http://localhost:11434 | Ollama server (point at the GPU box if remote) |
| `OLLAMA_MODEL` | qwen2.5vl:7b | Vision model (`qwen2.5vl:72b` in production) |
| `OLLAMA_TIMEOUT` / `OLLAMA_NUM_CTX` | 900 / 32768 | Request timeout (s) / context window |
| `VERIFY_MISSED_LINES` | 1 | Second completeness pass on handwritten pages |
| `DETECT_DIAGRAMS` | 1 | Dedicated diagram grounding + crop verification |
| `HEADER_CHECK` | 1 | Top-strip re-read on every page (`suspect` = only header-less pages, `0` = off) |
| `AUDIT_WORDS` | 1 | Critic pass flags misread words in red (handwritten pages, `0` = off) |
| `CROP_MAX_EDGE` / `CROP_PAD_PCT` | 500 / 4.0 | Crop size cap (px) / padding (page %) |
| `MEDGEMMA_CORRECTION` | 0 | MedGemma spelling layer (kept off — the 72B makes it redundant) |
| `SQL_*` | — | Azure SQL feedback store (optional) |

## Production deployment (Jarvis Labs GPU box)

- **GPU:** A100-80GB fits `qwen2.5vl:72b` comfortably. Start Ollama with:
  ```bash
  export OLLAMA_NUM_PARALLEL=2      # 2 pages at once; 4 on an H200-141GB
  export OLLAMA_KV_CACHE_TYPE=q8_0  # headroom on 80 GB; omit on H200
  export OLLAMA_KEEP_ALIVE=-1       # keep the 72B loaded between documents
  nohup ollama serve > ollama.log 2>&1 &
  ```
- `cudaMalloc failed: out of memory` in the log = too many parallel
  slots for the card — drop `OLLAMA_NUM_PARALLEL` and restart Ollama.
- Run-to-run output can vary slightly at `NUM_PARALLEL>1` (GPU batching
  is not deterministic even at temperature 0). For A/B comparison runs
  use `OLLAMA_NUM_PARALLEL=1`. Structural output (checklist border,
  C/o repair, crops policy, translations) is enforced in code and does
  not vary.
- Deploy = `git pull` + restart `python backend.py` + confirm the build
  stamp in the log. `.env` is not in the repo — maintain it on the box.

## API

- `POST /api/extract` (multipart file) → `{job_id}` —
  `GET /api/jobs/{job_id}` → `{status, ...result}` when done. Result:
  `{filename, doc_hash, pages[{page,text,layout_html,layout_error}],
  raw_text, raw_text_corrected, autocorrections, medgemma_corrections,
  fields, medicines, fields_error, meta}`
- `POST /api/raw-feedback` / `DELETE /api/raw-feedback/{doc_hash}` —
  save/revert a human text correction (returns word pairs used to patch
  the layout live).
- `GET /api/dictionary`, `GET /api/health`.

## Hardware note (local dev)

On a 4 GB GPU the 7B partially offloads to CPU — expect **minutes per
page**, and the small models are a wiring test only: they under-mark
uncertainty, miss diagrams, and ignore layout rules the 72B follows.
Judge extraction quality only on the 72B.

This project is fully self-contained — everything it needs is in this
repository.
