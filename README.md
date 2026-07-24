# Offline Prescription Extractor

Extracts raw text from doctor prescriptions (image or PDF) by pushing the
image to a **local Ollama vision model (`qwen2.5vl:7b`)** — a byte-for-byte
sibling of `doctor_prescription_gpt_extractor` with only the model layer
swapped from Azure OpenAI to Ollama. Same prompts, same combined
transcript + layout-reconstruction pass, same structured-fields pass, same
correction-feedback learning and Dictionary panel.

One of three sibling demo projects — all share the same UI (only the heading
differs) and run on **different ports** so they can run at the same time:

| Project | Engine | Port |
|---|---|---|
| `doctor_prescription_gpt_extractor` | Azure OpenAI GPT vision | 8001 |
| **`doctor_prescription_offline_extractor`** | **qwen2.5vl:7b via Ollama (offline)** | **8002** |
| `doctor_prescription_azure_extractor` | Azure Document Intelligence | 8003 |

## Flow

1. **Upload** — drag & drop a prescription image (JPG/PNG/WEBP) or PDF.
2. **Review** — original document beside the Reconstructed Layout / Raw Text
   tabs; edit the raw text and **Save Correction** to teach the system.
3. **Structured Data** — extracted fields + medicines table with corrections
   and CSV export.

## Setup

```powershell
# 1. install dependencies (any Python 3.10+; conda env "extract" already has them)
pip install -r requirements.txt

# 2. pull the vision model (~6 GB)
ollama pull qwen2.5vl:7b

# 3. .env already points at the shared Azure SQL server with this project's
#    own schema (PrescriptionExtractionOffline). For a fresh setup, copy
#    .env.example to .env and fill in your values.
```

## Run

```powershell
python backend.py        # or .\run.ps1
# or
uvicorn backend:app --reload --port 8002
```

Open **http://localhost:8002**

## How it works

- Images are base64-encoded and sent to Ollama's `/api/chat`
  (`OLLAMA_MODEL`, default `qwen2.5vl:7b`; change one `.env` line to swap
  models). `temperature 0`, `num_ctx` from `OLLAMA_NUM_CTX` (default 16384).
- PDFs are rendered page-by-page (pypdfium2, 150 DPI); one combined call per
  page returns the raw transcription AND an HTML layout reconstruction; one
  extra call per document returns structured fields/medicines as JSON
  (Ollama `format: "json"`).
- Human corrections are stored in Azure SQL schema
  **`PrescriptionExtractionOffline`** (separate from the GPT sibling so each
  model learns from its own mistakes) and are injected into prompts as
  few-shot examples; corrections confirmed 3+ times become deterministic
  autocorrect rules (Dictionary panel).
- `POST /api/extract` → `{filename, doc_hash, pages, raw_text,
  raw_text_corrected, autocorrections, fields, medicines, fields_error, meta}`
- `POST /api/feedback`, `POST /api/raw-feedback`,
  `DELETE /api/raw-feedback/{doc_hash}`, `GET /api/dictionary`,
  `GET /api/health`.

## MedGemma medical-word correction

After extraction, the medically-trained **MedGemma** model
(`MEDGEMMA_MODEL`, shares the same Ollama server) reviews the extracted
text and proposes `wrong → correct` pairs for misread medical terms (drug
names, dosage units, shorthand, test names). Pairs are validated in code
(the wrong word must actually occur; no names/dates/bare numbers) and
applied as whole-word replacements — each corrected word is prefixed with
`*` in the raw text AND the layout so reviewers instantly see what the
model changed (e.g. `dallo 650` → `*Dolo 650`). The UI shows a violet
banner listing the corrections; the response carries them in
`medgemma_corrections`. Human dictionary rules run after this layer, so
human-confirmed corrections always win. Kill-switch:
`MEDGEMMA_CORRECTION=0`.

**Hardware note:** on a 4 GB GPU (Quadro T1000) the 7B model partially
offloads to CPU — expect a page to take minutes, not seconds. Point
`OLLAMA_HOST` at a bigger GPU box for faster inference.

This project is fully self-contained — no references to any other folder.
