"""
medgemma_corrector.py — medical-word correction layer (MedGemma via Ollama)
───────────────────────────────────────────────────────────────────────────
Post-extraction QA pass: the medically-trained MedGemma model reads the
extracted text and proposes {"wrong", "correct"} pairs for misread medical
terms (drug names, dosage units, shorthand, test names). It never rewrites
the text — pairs are validated here and applied by the backend with
whole-word replacement, prefixing each corrected word with '*' so reviewers
can see what the model changed (e.g. "dallo 650" -> "*Dolo 650").

This layer must never break extraction: any failure returns [].
"""

import json
import os
import re

import requests

_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
_MODEL = os.getenv("MEDGEMMA_MODEL", "williamljx/medgemma-4b-it-Q4_K_M-GGUF")
_ENABLED = os.getenv("MEDGEMMA_CORRECTION", "1").strip().lower() in ("1", "true", "yes", "on")
_TIMEOUT = int(os.getenv("MEDGEMMA_TIMEOUT", "300"))
_MAX_PAIRS = 15


def _load_known_drugs() -> set:
    """Whitelist of known-correct drug names — never 'corrected'."""
    path = os.path.join(os.path.dirname(__file__), "data", "known_drugs.txt")
    try:
        with open(path, encoding="utf-8") as f:
            return {
                line.strip().lower()
                for line in f
                if line.strip() and not line.startswith("#")
            }
    except OSError:
        return set()


_KNOWN_DRUGS = _load_known_drugs()

_SYSTEM = (
    "You are a medical transcription quality checker for a licensed pharmacy "
    "team. You receive OCR-extracted text of one prescription / medical "
    "document. Find ONLY misspelled or misread MEDICAL terms — drug/brand "
    "names, dosage units, medical shorthand (BD, TDS, OD, HS, p.c., a.c., "
    "R/A, c/o, O/E), lab test names, diagnoses — and return JSON:\n"
    '{"corrections": [{"wrong": "<exact text from the document>", '
    '"correct": "<the correct term>"}]}\n'
    "Rules:\n"
    "- 'wrong' must be copied EXACTLY as it appears in the text (same case, "
    "same spacing) so it can be located.\n"
    "- Only report errors you are CONFIDENT about (a real drug or term the "
    "text clearly garbles). Skip anything uncertain. Maximum "
    f"{_MAX_PAIRS} corrections.\n"
    "- Standard prescription shorthand is CORRECT as written — NEVER "
    "'correct', expand or reformat it: Tab, Cap, Syp, Inj, BD, TDS, OD, HS, "
    "SOS, PO, p.c., a.c., R/A, c/o, O/E, mg, ml, mcg, 1-0-1 and similar.\n"
    "- Only fix MISSPELLINGS (wrong letters), e.g. 'dallo'->'Dolo', "
    "'Amoxcillin'->'Amoxicillin'. Do not rephrase, expand or add words.\n"
    "- NEVER correct people's names, dates, addresses, or bare numbers.\n"
    "- Do not invent medicines that are not plausibly in the text.\n"
    '- If nothing is wrong return {"corrections": []}.\n'
    "Return ONLY the JSON object."
)


def _model_available() -> bool:
    try:
        res = requests.get(f"{_HOST}/api/tags", timeout=10)
        res.raise_for_status()
        names = [m.get("name", "") for m in res.json().get("models", [])]
        return any(n == _MODEL or n.startswith(_MODEL + ":") for n in names)
    except Exception:
        return False


def _word_rx(word: str):
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])")


def get_corrections(raw_text: str) -> list:
    """[(wrong, correct), ...] — validated pairs safe for whole-word
    replacement. Never raises; returns [] when disabled or on any failure."""
    if not _ENABLED or not (raw_text or "").strip():
        return []
    if not _model_available():
        print(f"[WARNING] MedGemma correction skipped: model '{_MODEL}' not available")
        return []
    try:
        res = requests.post(
            f"{_HOST}/api/chat",
            json={
                "model": _MODEL,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content":
                        "Check this extracted document text for misread "
                        "medical terms:\n\n" + raw_text[:6000]},
                ],
                "options": {"temperature": 0, "num_predict": 800},
            },
            timeout=_TIMEOUT,
        )
        res.raise_for_status()
        content = (res.json().get("message", {}).get("content") or "").strip()
        parsed = json.loads(content)
    except Exception as e:
        print(f"[WARNING] MedGemma correction call failed: {e}")
        return []

    pairs = []
    for item in (parsed.get("corrections") or [])[: _MAX_PAIRS * 2]:
        if not isinstance(item, dict):
            continue
        wrong = (item.get("wrong") or "").strip()
        correct = (item.get("correct") or "").strip()
        if not wrong or not correct or wrong == correct:
            continue
        if re.fullmatch(r"[\d\s.,/-]+", wrong):        # bare numbers/dates
            continue
        if len(wrong.split()) > 3:                      # phrase reformatting,
            continue                                    # not a misread word
        if wrong[0].isdigit():                          # dosage-shorthand noise
            continue
        if any(t.lower() in _KNOWN_DRUGS for t in wrong.split()):
            continue                                    # already a real drug
        if len(correct) > 3 * len(wrong) + 10:          # runaway "correction"
            continue
        if not _word_rx(wrong).search(raw_text):        # must actually occur
            continue
        pairs.append((wrong, correct))
        if len(pairs) >= _MAX_PAIRS:
            break
    return pairs
