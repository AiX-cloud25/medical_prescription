"""
backend.py — GPT Prescription Extractor
───────────────────────────────────────
FastAPI app: serves the single-page UI and one extraction endpoint.

Run (port 8002 — GPT sibling uses 8001, Azure sibling uses 8003):
    python backend.py
    # or
    uvicorn backend:app --reload --port 8002
"""

import asyncio
import hashlib
import re
import traceback
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

_DIR = Path(__file__).resolve().parent
load_dotenv(_DIR / ".env", override=True)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

import extractor
import feedback_store
import medgemma_corrector

PORT = 8002
_ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".pdf"}

app = FastAPI(title="Offline Prescription Extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ────────────────────────────────────────────────────
# Jobs are keyed by a UUID returned to the client on upload.
# Status: "queued" | "running" | "done" | "error"
_jobs: Dict[str, dict] = {}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "traceback": traceback.format_exc()},
    )


@app.on_event("startup")
def _init_feedback_db():
    try:
        feedback_store.init_db()
        print("[INFO] Feedback DB ready")
    except Exception as e:
        print(f"[WARNING] Feedback DB unavailable — corrections disabled: {e}")


@app.get("/")
def serve_ui():
    return FileResponse(_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


# Gap between anchor words in layout HTML: tags, punctuation, whitespace.
_ANCHOR_GAP = r"(?:<[^>]*>|[^A-Za-z0-9<])*"


def _apply_autocorrect(text: str, rules: list) -> tuple:
    """
    Apply graduated dictionary rules to text with whole-word matching.
    Returns (new_text, applied) where applied lists the rules that matched.
    """
    applied = []
    for pred, corr in rules:
        pred_rx = re.compile(r"(?<![A-Za-z0-9])" + re.escape(pred) + r"(?![A-Za-z0-9])")
        new_text, n = pred_rx.subn(lambda _m: corr, text)
        if n:
            text = new_text
            applied.append({"from": pred, "to": corr})
    return text, applied


def _patch_words(html: str, triples: list) -> str:
    """
    Patch corrected words into layout HTML.

    Anchor-first: each triple carries the field label preceding the change
    (e.g. 'Paranasal Sinuses :'); the first occurrence of the predicted
    value within 300 chars AFTER that label is replaced, so a '(blank)'
    corrected on one field never touches other blank fields. When the
    anchor cannot be found, falls back to a whole-word global replace —
    but only for non-placeholder values, so short tokens can't corrupt
    parts of other words and placeholders are never mass-replaced.
    """
    replaced = 0
    for pred, corr, anchor in triples:
        pred_rx = re.compile(r"(?<![A-Za-z0-9])" + re.escape(pred) + r"(?![A-Za-z0-9])")
        done = False
        words = re.findall(r"[A-Za-z0-9]+", anchor or "")
        # The anchor may start with stray words from the previous line of the
        # raw text; try progressively shorter suffixes (the tokens nearest
        # the change are the field label) until one matches the layout.
        for k in range(len(words)):
            anchor_rx = re.compile(
                _ANCHOR_GAP.join(re.escape(w) for w in words[k:]), re.IGNORECASE
            )
            m = anchor_rx.search(html)
            if not m:
                continue
            window = html[m.end(): m.end() + 300]
            # The human verified this field's value, so it wins over
            # whatever the model wrote this run. Candidates: the exact
            # predicted value, '(blank)', or a handwritten span — whichever
            # sits NEAREST to the label is this field's value.
            cands = []
            pm = pred_rx.search(window)
            if pm is not None:
                cands.append((pm.start(), pm.end()))
            if pred != "(blank)":
                bm = re.search(re.escape("(blank)"), window)
                if bm is not None:
                    cands.append((bm.start(), bm.end()))
            hw = re.search(r'<span class="hw">([^<]{1,20})</span>', window)
            if hw is not None:
                cands.append((hw.start(1), hw.end(1)))
            if cands:
                s, e = min(cands)
                html = html[:m.end() + s] + corr + html[m.end() + e:]
                done = True
            break  # anchor matched — do not retry shorter, riskier suffixes
        if not done and not any(t in pred for t in ("(blank)", "[illegible]")):
            new_html = pred_rx.sub(lambda _m: corr, html)
            done = new_html != html
            html = new_html
        if done:
            replaced += 1
    return html, replaced


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    """
    Accept ONE prescription upload and return a job_id immediately.
    The client polls GET /api/jobs/{job_id} for status and results.
    This avoids browser HTTP timeouts during long model inference.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext or 'unknown'}'. "
                   "Upload a JPG, PNG or WEBP image, or a PDF.",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "filename": file.filename}

    # Fire extraction in a background task — returns immediately to client.
    asyncio.create_task(_run_extraction(job_id, data, ext, file.filename or ""))
    return {"job_id": job_id, "status": "queued"}


async def _run_extraction(job_id: str, data: bytes, ext: str, filename: str):
    """Background task: run extraction and store result in _jobs."""
    _jobs[job_id]["status"] = "running"
    try:
        doc_hash = hashlib.sha256(data).hexdigest()
        pages, extras, meta = await asyncio.to_thread(extractor.extract, data, ext)

        # ── Persist raw model output immediately ──────────────────
        try:
            await asyncio.to_thread(
                feedback_store.save_raw_extraction, doc_hash, filename, pages
            )
            print(f"[INFO] Raw extraction saved for {filename} ({doc_hash[:12]}…)")
        except Exception as e:
            print(f"[WARNING] Could not save raw extraction (DB unavailable?): {e}")

        if len(pages) == 1:
            raw_text = pages[0]["text"]
        else:
            raw_text = "\n\n".join(f"--- Page {p['page']} ---\n{p['text']}" for p in pages)

        # ── MedGemma correction ───────────────────────────────────
        medgemma_corrections = []
        try:
            mg_pairs = await asyncio.to_thread(medgemma_corrector.get_corrections, raw_text)
            if mg_pairs:
                starred = [(w, "*" + c) for w, c in mg_pairs]
                raw_text, medgemma_corrections = _apply_autocorrect(raw_text, starred)
                for p in pages:
                    if p.get("text"):
                        p["text"], _ = _apply_autocorrect(p["text"], starred)
                    if p.get("layout_html"):
                        p["layout_html"], _ = _apply_autocorrect(p["layout_html"], starred)
                if medgemma_corrections:
                    fixes = ", ".join(f'{a["from"]} -> {a["to"]}' for a in medgemma_corrections)
                    print(f"[INFO] MedGemma corrections applied: {fixes}")
        except Exception as e:
            print(f"[WARNING] MedGemma correction skipped: {e}")

        # ── Dictionary auto-correct ───────────────────────────────
        autocorrections = []
        try:
            rules = feedback_store.get_autocorrect_rules()
            if rules:
                raw_text, autocorrections = _apply_autocorrect(raw_text, rules)
                for p in pages:
                    if p.get("text"):
                        p["text"], _ = _apply_autocorrect(p["text"], rules)
                    if p.get("layout_html"):
                        p["layout_html"], _ = _apply_autocorrect(p["layout_html"], rules)
                if autocorrections:
                    fixes = ", ".join(f'{a["from"]} -> {a["to"]}' for a in autocorrections)
                    print(f"[INFO] Dictionary auto-corrections applied: {fixes}")
        except Exception as e:
            print(f"[WARNING] Auto-correct dictionary skipped (Postgres unavailable?): {e}")

        # ── Exact-document replay ─────────────────────────────────
        raw_text_corrected = False
        try:
            stored = feedback_store.get_raw_correction(doc_hash)
            if stored is not None:
                raw_text = stored
                raw_text_corrected = True
                print(f"[INFO] Replayed stored raw-text correction for {filename} ({doc_hash[:12]}…)")
                pairs = feedback_store.get_raw_correction_pairs(doc_hash)
                if pairs:
                    total = 0
                    for p in pages:
                        if p.get("layout_html"):
                            p["layout_html"], n = _patch_words(p["layout_html"], pairs)
                            total += n
                    print(f"[INFO] Layout patch: {total}/{len(pairs)} corrected word(s) applied")
        except Exception as e:
            print(f"[WARNING] Raw-correction lookup skipped (Postgres unavailable?): {e}")

        result = {
            "filename": filename,
            "doc_hash": doc_hash,
            "pages": pages,
            "raw_text": raw_text,
            "raw_text_corrected": raw_text_corrected,
            "autocorrections": autocorrections,
            "medgemma_corrections": medgemma_corrections,
            **extras,
            "meta": meta,
        }
        _jobs[job_id] = {"status": "done", "filename": filename, "result": result}
        print(f"[INFO] Job {job_id[:8]}… done for {filename}")

    except extractor.ExtractorError as e:
        _jobs[job_id] = {"status": "error", "filename": filename, "error": str(e)}
        print(f"[ERROR] Job {job_id[:8]}… failed: {e}")
    except Exception as e:
        _jobs[job_id] = {
            "status": "error", "filename": filename,
            "error": f"Unexpected error: {e}\n{traceback.format_exc()}",
        }
        print(f"[ERROR] Job {job_id[:8]}… unexpected failure: {e}")


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    """
    Poll extraction job status.
    Returns {"status": "queued"|"running"|"done"|"error", ...result when done}
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] == "error":
        return {"status": "error", "error": job.get("error", "Unknown error")}
    if job["status"] == "done":
        return {"status": "done", **job["result"]}
    return {"status": job["status"], "filename": job.get("filename", "")}


class CorrectionItem(BaseModel):
    key: str
    predicted: str = ""
    corrected: str = ""


class FeedbackPayload(BaseModel):
    filename: Optional[str] = None
    fields: List[CorrectionItem] = []
    medicines: List[CorrectionItem] = []


@app.post("/api/feedback")
def save_feedback(payload: FeedbackPayload):
    """
    Persist human corrections (only changed values are sent by the UI).
    Future extractions retrieve these as few-shot examples for the prompt.
    """
    if not payload.fields and not payload.medicines:
        raise HTTPException(status_code=400, detail="No corrections in payload.")
    try:
        n = feedback_store.save_feedback(
            payload.filename or "",
            [f.model_dump() for f in payload.fields],
            [m.model_dump() for m in payload.medicines],
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Could not save corrections — is PostgreSQL running? ({e})",
        )
    return {"saved": n}


class RawFeedbackPayload(BaseModel):
    doc_hash: str
    filename: Optional[str] = None
    original_text: str = ""
    corrected_text: str = ""


@app.post("/api/raw-feedback")
def save_raw_feedback(payload: RawFeedbackPayload):
    """
    Persist a human-corrected raw transcription. The exact same file
    (matched by content hash) will show this text on re-upload, and word
    fixes are learned for future transcriptions of other documents.
    """
    if not payload.doc_hash:
        raise HTTPException(status_code=400, detail="doc_hash is required.")
    if not payload.corrected_text.strip():
        raise HTTPException(status_code=400, detail="Corrected text is empty.")
    if payload.corrected_text == payload.original_text:
        raise HTTPException(status_code=400, detail="Text is unchanged — nothing to save.")
    try:
        pairs, learned = feedback_store.save_raw_correction(
            payload.doc_hash,
            payload.filename or "",
            payload.original_text,
            payload.corrected_text,
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Could not save correction — is PostgreSQL running? ({e})",
        )
    return {"saved": True, "spelling_pairs": len(learned), "pairs": pairs}


@app.delete("/api/raw-feedback/{doc_hash}")
def revert_raw_feedback(doc_hash: str):
    """
    Undo a saved raw-text correction: delete the stored corrected text and
    the spelling pairs learned from it, and return the original transcription
    (None if it predates original-text recording — the UI then rebuilds it
    from the page texts).
    """
    try:
        found, original = feedback_store.delete_raw_correction(doc_hash)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Could not revert — is PostgreSQL running? ({e})",
        )
    if not found:
        raise HTTPException(status_code=404, detail="No stored correction for this document.")
    return {"reverted": True, "original_text": original}


@app.get("/api/raw-extraction/{doc_hash}")
def get_raw_extraction(doc_hash: str):
    """
    Retrieve the original unmodified model extraction for a document
    (stored automatically on every upload). Returns pages ordered by number.
    Useful for comparing the raw model output against human-corrected text.
    """
    try:
        pages = feedback_store.get_raw_extraction(doc_hash)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Raw extraction unavailable — is the database reachable? ({e})",
        )
    if not pages:
        raise HTTPException(status_code=404, detail="No stored extraction for this document.")
    return {"doc_hash": doc_hash, "pages": pages}


@app.get("/api/dictionary")
def get_dictionary():
    """
    The learned spelling dictionary, grouped by corrected word: each entry
    lists the raw misspelling variants, their confirmation counts, and
    whether the pooled votes made the correction automatic.
    """
    try:
        entries = feedback_store.get_dictionary()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Dictionary unavailable — is the database reachable? ({e})",
        )
    return {"entries": entries}


@app.get("/api/health")
def health():
    return {"status": "ok", "engine": extractor.ENGINE_NAME, "port": PORT}


if __name__ == "__main__":
    import uvicorn
    import os
    # Bind to 0.0.0.0 so Jarvis Labs proxy can reach the server.
    # Use HOST env var to override (default 0.0.0.0 for cloud, can set to
    # 127.0.0.1 for local-only).
    host = os.getenv("APP_HOST", "0.0.0.0")
    uvicorn.run(
        app,
        host=host,
        port=PORT,
        # Increase keep-alive so Jarvis proxy long-polling doesn't reset.
        timeout_keep_alive=300,
        # Allow large file uploads (PDF etc.)
        h11_max_incomplete_event_size=10 * 1024 * 1024,
    )
