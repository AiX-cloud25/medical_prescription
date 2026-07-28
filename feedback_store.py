"""
feedback_store.py — human-correction feedback storage (Azure SQL Database)
───────────────────────────────────────────────────────────────────────────
Persists corrections made by a human reviewer and builds the "learned
corrections" few-shot blocks that extractor.py appends to its prompts.
One shared database serves the local app AND the Azure-deployed instance.

Connection is configured via .env (same values in both environments):
    SQL_SERVER=<server>.database.windows.net
    SQL_DATABASE=<database name>
    SQL_USERNAME=<user>
    SQL_PASSWORD=<password>
    SQL_SCHEMA=PrescriptionExtraction      (tables live inside this schema)

A fresh connection is opened per operation; traffic is tiny and this avoids
stale-connection handling. login_timeout keeps an unreachable server from
stalling extraction (the free serverless tier auto-pauses — the first
connection after a long idle may be slow or fail once; callers degrade
gracefully).
"""

import difflib
import os

import pymssql

_LOGIN_TIMEOUT = 15  # seconds — fail reasonably fast, allow serverless resume


def _schema() -> str:
    return os.getenv("SQL_SCHEMA", "PrescriptionExtraction")


def _connect():
    return pymssql.connect(
        server=os.getenv("SQL_SERVER", ""),
        user=os.getenv("SQL_USERNAME", ""),
        password=os.getenv("SQL_PASSWORD", ""),
        database=os.getenv("SQL_DATABASE", ""),
        login_timeout=_LOGIN_TIMEOUT,
        timeout=30,
    )


# predicted/corrected are short word spans; bounded NVARCHAR because SQL
# Server cannot GROUP BY nvarchar(max) (the ranking queries group on them).
_VALUE_MAX_CHARS = 400


def _clip(value: str) -> str:
    return (value or "")[:_VALUE_MAX_CHARS]


def init_db():
    """Create the schema and tables if missing. Raises on failure."""
    s = _schema()
    schema_ddl = f"""
IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = '{s}')
    EXEC('CREATE SCHEMA {s}');
"""
    ddl = f"""
IF OBJECT_ID('{s}.corrections', 'U') IS NULL
CREATE TABLE {s}.corrections (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    kind            NVARCHAR(20) NOT NULL DEFAULT 'field',
    filename        NVARCHAR(400),
    doc_hash        NVARCHAR(64),
    field_key       NVARCHAR(200) NOT NULL,
    predicted_value NVARCHAR({_VALUE_MAX_CHARS}) NOT NULL,
    corrected_value NVARCHAR({_VALUE_MAX_CHARS}) NOT NULL,
    created_at      DATETIMEOFFSET NOT NULL DEFAULT SYSDATETIMEOFFSET()
);

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'idx_corrections_key'
      AND object_id = OBJECT_ID('{s}.corrections')
)
CREATE INDEX idx_corrections_key ON {s}.corrections (kind, field_key);

IF OBJECT_ID('{s}.raw_corrections', 'U') IS NULL
CREATE TABLE {s}.raw_corrections (
    id             INT IDENTITY(1,1) PRIMARY KEY,
    doc_hash       NVARCHAR(64) NOT NULL UNIQUE,
    filename       NVARCHAR(400),
    original_text  NVARCHAR(MAX),
    corrected_text NVARCHAR(MAX) NOT NULL,
    created_at     DATETIMEOFFSET NOT NULL DEFAULT SYSDATETIMEOFFSET()
);

IF OBJECT_ID('{s}.raw_extractions', 'U') IS NULL
CREATE TABLE {s}.raw_extractions (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    doc_hash        NVARCHAR(64) NOT NULL,
    filename        NVARCHAR(400),
    page_number     INT NOT NULL DEFAULT 1,
    raw_text        NVARCHAR(MAX),
    layout_html     NVARCHAR(MAX),
    layout_error    NVARCHAR(MAX),
    extracted_at    DATETIMEOFFSET NOT NULL DEFAULT SYSDATETIMEOFFSET(),
    CONSTRAINT uq_raw_extractions_hash_page UNIQUE (doc_hash, page_number)
);

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'idx_raw_extractions_hash'
      AND object_id = OBJECT_ID('{s}.raw_extractions')
)
CREATE INDEX idx_raw_extractions_hash ON {s}.raw_extractions (doc_hash);
"""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(schema_ddl)
            conn.commit()
            cur.execute(ddl)
        conn.commit()
    finally:
        conn.close()


def save_feedback(filename: str, fields: list, medicines: list) -> int:
    """
    fields / medicines: [{"key": ..., "predicted": ..., "corrected": ...}, ...]
    For medicines, "key" is the column name (medicine/dosage/frequency/
    duration/instructions). Rows where nothing changed or the corrected
    value is blank are skipped. Returns rows inserted. Raises on DB error.
    """
    rows = []
    for kind, items in (("field", fields), ("medicine", medicines)):
        for item in items or []:
            key = (item.get("key") or "").strip()
            predicted = (item.get("predicted") or "").strip()
            corrected = (item.get("corrected") or "").strip()
            if not key or not corrected or predicted == corrected:
                continue
            rows.append((kind, filename or None, key, _clip(predicted), _clip(corrected)))
    if not rows:
        return 0
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {_schema()}.corrections
                    (kind, filename, field_key, predicted_value, corrected_value)
                    VALUES (%s, %s, %s, %s, %s)""",
                rows,
            )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def _ranked_examples(kinds: list, max_examples: int) -> list:
    """
    Distinct corrections of the given kinds, ranked by frequency then
    recency; near-duplicate predicted values within the same field are
    dropped so the selection stays diverse. Raises on DB error.
    """
    placeholders = ", ".join(["%s"] * len(kinds))
    query = f"""
        SELECT TOP 40 kind, field_key, predicted_value, corrected_value,
               COUNT(*) AS times, MAX(created_at) AS last_seen
        FROM {_schema()}.corrections
        WHERE predicted_value <> corrected_value AND kind IN ({placeholders})
        GROUP BY kind, field_key, predicted_value, corrected_value
        ORDER BY times DESC, last_seen DESC
    """
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(query, tuple(kinds))
            candidates = cur.fetchall()
    finally:
        conn.close()

    kept = []
    for kind, field_key, predicted, corrected, _times, _last in candidates:
        duplicate = any(
            k_field == field_key
            and difflib.SequenceMatcher(None, predicted, k_pred).ratio() > 0.9
            for _, k_field, k_pred, _ in kept
        )
        if duplicate:
            continue
        kept.append((kind, field_key, predicted, corrected))
        if len(kept) >= max_examples:
            break
    return kept


def get_examples_for_prompt(max_examples: int = 12) -> str:
    """
    Few-shot block of field/medicine corrections for the structured-fields
    prompt, or "" when there are none. Raises on DB error (extractor wraps
    the call in try/except).
    """
    kept = _ranked_examples(["field", "medicine"], max_examples)
    if not kept:
        return ""

    lines = [
        "LEARNED CORRECTIONS (from human review of previous documents):",
        "A human reviewer corrected these past extraction mistakes. When you",
        "read a similar-looking value, prefer the corrected form if it matches",
        "what is written on the page:",
    ]
    for kind, field_key, predicted, corrected in kept:
        label = f"medicine column '{field_key}'" if kind == "medicine" else f"field '{field_key}'"
        lines.append(f'- {label}: model read "{predicted}" -> human corrected to "{corrected}"')
    return "\n".join(lines)


# A correction confirmed this many times graduates from prompt example to a
# deterministic auto-correct rule applied in code (zero prompt tokens).
_AUTOCORRECT_MIN_TIMES = 3


def _spelling_groups() -> list:
    """All distinct (predicted, corrected, times) spelling groups. Raises on DB error."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT predicted_value, corrected_value, COUNT(*) AS times
                    FROM {_schema()}.corrections
                    WHERE kind = 'spelling' AND predicted_value <> corrected_value
                    GROUP BY predicted_value, corrected_value""",
            )
            return cur.fetchall()
    finally:
        conn.close()


def get_autocorrect_rules(min_times: int = _AUTOCORRECT_MIN_TIMES) -> list:
    """
    (predicted, corrected) pairs applied deterministically to extraction
    output instead of being sent as prompt examples.

    CROSS-VARIANT GRADUATION: variants that map to the SAME corrected word
    pool their confirmations — 'Deepk' x1 + 'deepake' x2 = 3 votes for
    'Deepak', so BOTH variants graduate together once the pooled count
    reaches `min_times`. A predicted word that humans mapped to MORE THAN
    ONE distinct correction is excluded entirely (ambiguous — the model must
    judge it from context, so it stays in the prompt). Raises on DB error.
    """
    groups = _spelling_groups()

    by_pred = {}
    for pred, corr, times in groups:
        by_pred.setdefault(pred, []).append((corr, times))

    # Pool votes per corrected word, using only unambiguous variants.
    votes_by_corr = {}
    for pred, corrs in by_pred.items():
        if len(corrs) > 1:
            continue  # conflicting human corrections — not safe to automate
        corr, times = corrs[0]
        votes_by_corr.setdefault(corr, []).append((pred, times))

    rules = []
    for corr, variants in votes_by_corr.items():
        if sum(t for _p, t in variants) >= min_times:
            rules.extend((pred, corr) for pred, _t in variants)
    return rules


def get_dictionary() -> list:
    """
    Human-browsable dictionary grouped by corrected word:
        [{"corrected": "Deepak",
          "variants": [{"raw": "Deepk", "times": 1}, {"raw": "deepake", "times": 2}],
          "total": 3, "automatic": True}, ...]
    `automatic` mirrors get_autocorrect_rules (pooled votes >= threshold and
    no conflicting variants). Conflicting variants are listed with
    automatic=False so reviewers can see why. Sorted by total desc.
    Raises on DB error.
    """
    groups = _spelling_groups()

    by_pred = {}
    for pred, corr, times in groups:
        by_pred.setdefault(pred, []).append((corr, times))
    conflicted = {pred for pred, corrs in by_pred.items() if len(corrs) > 1}

    by_corr = {}
    for pred, corr, times in groups:
        by_corr.setdefault(corr, []).append((pred, times))

    entries = []
    for corr, variants in by_corr.items():
        clean = [(p, t) for p, t in variants if p not in conflicted]
        total = sum(t for _p, t in clean)
        entries.append({
            "corrected": corr,
            "variants": [
                {"raw": p, "times": t, "conflicted": p in conflicted}
                for p, t in sorted(variants, key=lambda v: -v[1])
            ],
            "total": total,
            "automatic": bool(clean) and total >= _AUTOCORRECT_MIN_TIMES,
        })
    entries.sort(key=lambda e: -e["total"])
    return entries


def get_spelling_examples_for_prompt(max_examples: int = 12) -> str:
    """
    Few-shot block for the transcription/layout prompts: word-level spelling
    corrections plus "watch out" warnings for previously misread short
    values. "" when there are none. Raises on DB error.
    """
    spelling = _ranked_examples(["spelling"], max_examples)
    ambiguous = _ranked_examples(["ambiguity"], max_examples)
    # Pairs that graduated to deterministic auto-correct rules are applied in
    # code, so they no longer need prompt tokens.
    graduated = set(get_autocorrect_rules())
    spelling = [row for row in spelling if (row[2], row[3]) not in graduated]
    if not spelling and not ambiguous:
        return ""

    lines = []
    if spelling:
        lines += [
            "LEARNED SPELLING CORRECTIONS (from human review of previous transcriptions):",
            "A human reviewer fixed these misread words in past transcriptions. When",
            "handwriting looks like the misreading, prefer the corrected form if it",
            "matches what is written on the page:",
        ]
        for _kind, _field_key, predicted, corrected in spelling:
            lines.append(f'- model read "{predicted}" -> human corrected to "{corrected}"')
    if ambiguous:
        if lines:
            lines.append("")
        lines += [
            "KNOWN MISREADINGS — EXAMINE THE STROKES EXTRA CAREFULLY:",
            "On past documents a human reviewer had to fix these short handwritten",
            "values. Do NOT blindly substitute — both readings exist on real forms —",
            "but when you see a similar value, slow down and decide letter by letter",
            "from the stroke shapes:",
        ]
        for _kind, _field_key, predicted, corrected in ambiguous:
            lines.append(f'- model wrote "{predicted}" where the reviewer verified "{corrected}"')
    return "\n".join(lines)


_MAX_DIFF_TOKENS = 4  # only small replaced spans are trustworthy word fixes

# Template placeholders — a correction touching one of these (e.g. filling a
# "(blank)" with a value) is specific to that document, not a misspelling the
# model should generalize to other documents.
_PLACEHOLDER_TOKENS = ("(blank)", "[illegible]")


# 1-2 letter tokens (NO/ND, +/-) are too ambiguous to teach the model as
# outright substitutions; they still patch this document's own layout and
# are learned as "watch out" ambiguity warnings instead.
_MIN_LEARN_CHARS = 3


def _core(token: str) -> str:
    """Token without the uncertainty marker, for length checks."""
    return token.replace("(?)", "").strip()


def _word_diffs(original: str, corrected: str) -> list:
    """
    (predicted, corrected, anchor) word-level triples via difflib over
    whitespace tokens. `anchor` is up to the last 4 unchanged tokens before
    the change (usually the field label, e.g. 'Paranasal Sinuses :') so the
    layout patch can target the right occurrence; "" when at text start.
    Placeholder pairs ARE included — needed for anchored layout patching —
    but are excluded from learning at save time.
    """
    a, b = original.split(), corrected.split()
    triples = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if op != "replace":
            continue
        if (i2 - i1) > _MAX_DIFF_TOKENS or (j2 - j1) > _MAX_DIFF_TOKENS:
            continue  # big rewrites are not spelling fixes
        pred, corr = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if not pred or not corr or pred == corr:
            continue
        anchor = " ".join(a[max(0, i1 - 4):i1])
        triples.append((pred, corr, anchor))
    return triples


def _has_placeholder(pred: str, corr: str) -> bool:
    return any(t in pred or t in corr for t in _PLACEHOLDER_TOKENS)


def _learnable(triples: list) -> list:
    """(pred, corr) pairs safe to generalize to other documents as substitutions."""
    return [
        (p, c) for p, c, _a in triples
        if not _has_placeholder(p, c)
        and len(_core(p)) >= _MIN_LEARN_CHARS and len(_core(c)) >= _MIN_LEARN_CHARS
    ]


def save_raw_extraction(doc_hash: str, filename: str, pages: list) -> None:
    """
    Persist the raw model extraction for every page of a document immediately
    after extraction, before any human correction is applied.

    Stores TWO things:
    1. raw_extractions — per-page text + layout HTML (always overwritten on re-run)
    2. raw_corrections.original_text — the full combined text, saved ONCE on first
       extraction and never overwritten (so human corrections can always be
       compared against the original model output).

    pages: [{"page": int, "text": str, "layout_html": str|None, "layout_error": str|None}]
    Never raises — failures are logged so they cannot break the extract endpoint.
    """
    if not pages:
        return

    # Build the full combined raw text (same format as raw_text in the API response)
    if len(pages) == 1:
        full_raw_text = pages[0].get("text") or ""
    else:
        full_raw_text = "\n\n".join(
            f"--- Page {p.get('page', i+1)} ---\n{p.get('text') or ''}"
            for i, p in enumerate(pages)
        )

    s = _schema()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            # ── 1. Per-page raw_extractions (always upsert) ──────────────
            for p in pages:
                page_num = p.get("page", 1)
                raw_text = p.get("text") or ""
                layout_html = p.get("layout_html") or None
                layout_error = p.get("layout_error") or None
                cur.execute(
                    f"""UPDATE {s}.raw_extractions
                        SET filename     = %s,
                            raw_text     = %s,
                            layout_html  = %s,
                            layout_error = %s,
                            extracted_at = SYSDATETIMEOFFSET()
                        WHERE doc_hash = %s AND page_number = %s""",
                    (filename or None, raw_text, layout_html, layout_error,
                     doc_hash, page_num),
                )
                if cur.rowcount == 0:
                    cur.execute(
                        f"""INSERT INTO {s}.raw_extractions
                            (doc_hash, filename, page_number, raw_text, layout_html, layout_error)
                            VALUES (%s, %s, %s, %s, %s, %s)""",
                        (doc_hash, filename or None, page_num,
                         raw_text, layout_html, layout_error),
                    )

            # ── 2. raw_corrections.original_text (only if no row yet) ────
            # We INSERT the original_text only on first extraction. If a human
            # correction row already exists, we leave it untouched — the human's
            # corrected_text is preserved and the original_text was already set.
            cur.execute(
                f"""SELECT id FROM {s}.raw_corrections WHERE doc_hash = %s""",
                (doc_hash,),
            )
            existing = cur.fetchone()
            if existing is None:
                # First time seeing this document — seed the original_text.
                # corrected_text must be NOT NULL per schema, so we use the
                # raw text as the initial corrected_text too. A human edit
                # will overwrite corrected_text later.
                cur.execute(
                    f"""INSERT INTO {s}.raw_corrections
                        (doc_hash, filename, original_text, corrected_text)
                        VALUES (%s, %s, %s, %s)""",
                    (doc_hash, filename or None, full_raw_text, full_raw_text),
                )
            else:
                # Document already has a row — update original_text only if it
                # was NULL (backfill for rows created before this logic existed).
                cur.execute(
                    f"""UPDATE {s}.raw_corrections
                        SET original_text = COALESCE(original_text, %s),
                            filename = COALESCE(filename, %s)
                        WHERE doc_hash = %s""",
                    (full_raw_text, filename or None, doc_hash),
                )

        conn.commit()
    except Exception as e:
        print(f"[WARNING] save_raw_extraction DB error: {e}")
    finally:
        conn.close()


def get_raw_extraction(doc_hash: str) -> list:
    """
    Retrieve stored raw extraction pages for a document (ordered by page).
    Returns [] if nothing stored. Raises on DB error.
    """
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT page_number, raw_text, layout_html, layout_error, extracted_at
                    FROM {_schema()}.raw_extractions
                    WHERE doc_hash = %s
                    ORDER BY page_number""",
                (doc_hash,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [
        {
            "page": row[0],
            "text": row[1],
            "layout_html": row[2],
            "layout_error": row[3],
            "extracted_at": str(row[4]) if row[4] else None,
        }
        for row in rows
    ]


def save_raw_correction(doc_hash: str, filename: str, original_text: str, corrected_text: str) -> tuple:
    """
    Upsert the corrected full raw text keyed by the document's SHA-256 hash
    (re-uploading the same file replays this text), and mine word-level
    diffs into the corrections table (kind='spelling') so the transcription
    prompt learns for other documents. Returns (patch_pairs, learned_pairs):
    patch_pairs update this document's layout, learned_pairs (long tokens
    only) teach the prompts. Raises on DB error.
    """
    pairs = _word_diffs(original_text or "", corrected_text)
    learned = _learnable(pairs)
    # Short ambiguous tokens (NO/ND) become "watch out" warnings, not
    # substitutions; placeholder fill-ins are never learned at all.
    ambiguous = [
        (p, c) for p, c, _a in pairs
        if (p, c) not in learned and not _has_placeholder(p, c)
    ]
    s = _schema()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            # Update-then-insert upsert (SQL Server has no ON CONFLICT).
            # original_text is seeded by save_raw_extraction on every run —
            # COALESCE here is just a safety net for very old rows.
            # corrected_text is always updated to the latest human edit.
            cur.execute(
                f"""UPDATE {s}.raw_corrections
                    SET filename = COALESCE(filename, %s),
                        original_text = COALESCE(original_text, %s),
                        corrected_text = %s,
                        created_at = SYSDATETIMEOFFSET()
                    WHERE doc_hash = %s""",
                (filename or None, original_text or None, corrected_text, doc_hash),
            )
            if cur.rowcount == 0:
                cur.execute(
                    f"""INSERT INTO {s}.raw_corrections
                        (doc_hash, filename, original_text, corrected_text)
                        VALUES (%s, %s, %s, %s)""",
                    (doc_hash, filename or None, original_text or None, corrected_text),
                )
            if learned:
                cur.executemany(
                    f"""INSERT INTO {s}.corrections
                        (kind, filename, doc_hash, field_key, predicted_value, corrected_value)
                        VALUES ('spelling', %s, %s, 'raw_text', %s, %s)""",
                    [(filename or None, doc_hash, _clip(p), _clip(c)) for p, c in learned],
                )
            if ambiguous:
                cur.executemany(
                    f"""INSERT INTO {s}.corrections
                        (kind, filename, doc_hash, field_key, predicted_value, corrected_value)
                        VALUES ('ambiguity', %s, %s, 'raw_text', %s, %s)""",
                    [(filename or None, doc_hash, _clip(p), _clip(c)) for p, c in ambiguous],
                )
        conn.commit()
    finally:
        conn.close()
    return pairs, learned


def get_raw_correction_pairs(doc_hash: str) -> list:
    """
    (predicted, corrected, anchor) word triples recomputed from the stored
    original and corrected text of this document — used to patch the layout
    reconstruction on replay. Empty list if nothing stored or the original
    is unknown. Raises on DB error.
    """
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT original_text, corrected_text FROM {_schema()}.raw_corrections WHERE doc_hash = %s",
                (doc_hash,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        return []
    return _word_diffs(row[0], row[1])


def get_raw_correction(doc_hash: str):
    """
    Returns the human-corrected text for this document ONLY if a human
    explicitly edited it (corrected_text differs from original_text).
    Returns None if no row exists OR if corrected_text == original_text
    (meaning it was auto-saved by the system with no human edit).
    Raises on DB error.
    """
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT corrected_text, original_text
                    FROM {_schema()}.raw_corrections
                    WHERE doc_hash = %s""",
                (doc_hash,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        return None
    corrected_text, original_text = row[0], row[1]
    # Only replay if a human actually changed the text
    if corrected_text and original_text and corrected_text.strip() == original_text.strip():
        return None  # auto-saved raw extraction — run fresh
    return corrected_text if corrected_text else None


def delete_raw_correction(doc_hash: str) -> tuple:
    """
    Remove a stored raw-text correction and the spelling pairs mined from it,
    restoring the document to un-corrected behaviour on re-upload.
    Returns (found, original_text) — original_text may be None for rows saved
    before it was recorded. Raises on DB error.
    """
    s = _schema()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT original_text FROM {s}.raw_corrections WHERE doc_hash = %s",
                (doc_hash,),
            )
            row = cur.fetchone()
            if row is None:
                return False, None
            cur.execute(
                f"DELETE FROM {s}.raw_corrections WHERE doc_hash = %s",
                (doc_hash,),
            )
            cur.execute(
                f"DELETE FROM {s}.corrections WHERE doc_hash = %s AND kind IN ('spelling', 'ambiguity')",
                (doc_hash,),
            )
        conn.commit()
    finally:
        conn.close()
    return True, row[0]
