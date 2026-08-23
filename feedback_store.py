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

IF OBJECT_ID('{s}.corrected_pages', 'U') IS NULL
CREATE TABLE {s}.corrected_pages (
    id                     INT IDENTITY(1,1) PRIMARY KEY,
    doc_hash               NVARCHAR(64) NOT NULL,
    page_number            INT NOT NULL DEFAULT 1,
    corrected_text         NVARCHAR(MAX),
    corrected_layout_html  NVARCHAR(MAX),
    created_at             DATETIMEOFFSET NOT NULL DEFAULT SYSDATETIMEOFFSET(),
    CONSTRAINT uq_corrected_pages_hash_page UNIQUE (doc_hash, page_number)
);

IF OBJECT_ID('{s}.page_images', 'U') IS NULL
CREATE TABLE {s}.page_images (
    id           INT IDENTITY(1,1) PRIMARY KEY,
    doc_hash     NVARCHAR(64) NOT NULL,
    page_number  INT NOT NULL DEFAULT 1,
    image_b64    NVARCHAR(MAX) NOT NULL,
    created_at   DATETIMEOFFSET NOT NULL DEFAULT SYSDATETIMEOFFSET(),
    CONSTRAINT uq_page_images_hash_page UNIQUE (doc_hash, page_number)
);

IF OBJECT_ID('{s}.corrected_words', 'U') IS NULL
CREATE TABLE {s}.corrected_words (
    id                INT IDENTITY(1,1) PRIMARY KEY,
    doc_hash          NVARCHAR(64) NOT NULL,
    filename          NVARCHAR(400),
    page_number       INT NOT NULL DEFAULT 1,
    predicted_word    NVARCHAR({_VALUE_MAX_CHARS}) NOT NULL,
    corrected_word    NVARCHAR({_VALUE_MAX_CHARS}) NOT NULL,
    sentence_context  NVARCHAR(1000) NOT NULL,
    crop_b64          NVARCHAR(MAX),
    crop_bbox         NVARCHAR(100),
    created_at        DATETIMEOFFSET NOT NULL DEFAULT SYSDATETIMEOFFSET()
);

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'idx_corrected_words_hash'
      AND object_id = OBJECT_ID('{s}.corrected_words')
)
CREATE INDEX idx_corrected_words_hash ON {s}.corrected_words (doc_hash);
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

    Stores ONLY in raw_extractions table (per-page text + layout HTML).
    raw_corrections is NEVER touched here — it is only written when a human
    explicitly saves a correction via the frontend /api/raw-feedback endpoint.

    pages: [{"page": int, "text": str, "layout_html": str|None, "layout_error": str|None}]
    Never raises — failures are logged so they cannot break the extract endpoint.
    """
    if not pages:
        return

    s = _schema()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            # Per-page raw_extractions — always upsert (refresh on every run)
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
    Called ONLY when a human explicitly saves a correction via the frontend.
    Stores original_text (what the model produced) and corrected_text (what
    the human fixed it to), then mines word-level diffs into the corrections
    table so the transcription prompt learns for future documents.
    Returns (patch_pairs, learned_pairs). Raises on DB error.
    """
    pairs = _word_diffs(original_text or "", corrected_text)
    learned = _learnable(pairs)
    ambiguous = [
        (p, c) for p, c, _a in pairs
        if (p, c) not in learned and not _has_placeholder(p, c)
    ]
    s = _schema()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            # Upsert: update existing row or insert new one
            cur.execute(
                f"""UPDATE {s}.raw_corrections
                    SET filename = %s,
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


def save_corrected_page(doc_hash: str, page_number: int, corrected_text: str,
                        corrected_layout_html: str) -> None:
    """
    Persist the EXACT corrected text/layout HTML for one page, as sent by
    the editor at save time — not re-derived by fuzzy patching. This is
    what makes re-upload replay exact (see get_corrected_pages) instead
    of the best-effort anchor-matched patch used elsewhere. Raises on DB
    error (caller should treat this as non-fatal to the main save).
    """
    s = _schema()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""UPDATE {s}.corrected_pages
                    SET corrected_text = %s,
                        corrected_layout_html = %s,
                        created_at = SYSDATETIMEOFFSET()
                    WHERE doc_hash = %s AND page_number = %s""",
                (corrected_text, corrected_layout_html, doc_hash, page_number),
            )
            if cur.rowcount == 0:
                cur.execute(
                    f"""INSERT INTO {s}.corrected_pages
                        (doc_hash, page_number, corrected_text, corrected_layout_html)
                        VALUES (%s, %s, %s, %s)""",
                    (doc_hash, page_number, corrected_text, corrected_layout_html),
                )
        conn.commit()
    finally:
        conn.close()


def get_corrected_pages(doc_hash: str) -> list:
    """
    [{"page": int, "text": str, "layout_html": str}, ...] for every page of
    this document that has an exact stored correction (may be a subset of
    the document's pages). [] if none. Raises on DB error.
    """
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT page_number, corrected_text, corrected_layout_html
                    FROM {_schema()}.corrected_pages
                    WHERE doc_hash = %s
                    ORDER BY page_number""",
                (doc_hash,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [{"page": r[0], "text": r[1], "layout_html": r[2]} for r in rows]


def save_page_images(doc_hash: str, filename: str, images: list) -> None:
    """
    Persist each page's already-rendered image (the same JPEG the model
    read), so a correction made later can still be cropped without
    re-uploading/re-rendering. images: [(page_number, b64_jpeg), ...].
    Never raises — failures are logged so they cannot break extraction.
    """
    if not images:
        return
    s = _schema()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            for page_number, b64 in images:
                cur.execute(
                    f"""UPDATE {s}.page_images
                        SET image_b64 = %s, created_at = SYSDATETIMEOFFSET()
                        WHERE doc_hash = %s AND page_number = %s""",
                    (b64, doc_hash, page_number),
                )
                if cur.rowcount == 0:
                    cur.execute(
                        f"""INSERT INTO {s}.page_images
                            (doc_hash, page_number, image_b64)
                            VALUES (%s, %s, %s)""",
                        (doc_hash, page_number, b64),
                    )
        conn.commit()
    except Exception as e:
        print(f"[WARNING] save_page_images DB error: {e}")
    finally:
        conn.close()


def get_page_image(doc_hash: str, page_number: int):
    """The stored base64 page image, or None if not found. Raises on DB error."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT image_b64 FROM {_schema()}.page_images
                    WHERE doc_hash = %s AND page_number = %s""",
                (doc_hash, page_number),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def _line_containing_token(text: str, token_idx: int) -> str:
    """
    The full (stripped) line of `text` containing the token_idx'th
    whitespace-split token (0-based) — used to capture a corrected
    word's surrounding sentence for training-data context.
    """
    count = 0
    for line in text.splitlines():
        n = len(line.split())
        if count <= token_idx < count + n:
            return line.strip()
        count += n
    return ""


def word_diffs_with_sentence(original: str, corrected: str) -> list:
    """
    Same word-level diff as _word_diffs, but each triple gains a 4th
    field: the full corrected-text line the change sits in. Kept
    separate from _word_diffs (rather than changing its return shape)
    since _word_diffs' 3-tuple contract is relied on elsewhere (layout
    patch replay) and training-data capture is the only caller that
    needs sentence context. Public (no leading underscore) since
    backend.py calls it directly to orchestrate word-crop capture.
    """
    a, b = original.split(), corrected.split()
    quads = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if op != "replace":
            continue
        if (i2 - i1) > _MAX_DIFF_TOKENS or (j2 - j1) > _MAX_DIFF_TOKENS:
            continue
        pred, corr = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if not pred or not corr or pred == corr:
            continue
        sentence = _line_containing_token(corrected, j1)
        quads.append((pred, corr, sentence))
    return quads


def save_corrected_word(doc_hash: str, filename: str, page_number: int,
                        predicted_word: str, corrected_word: str,
                        sentence_context: str, crop_b64, crop_bbox) -> None:
    """
    One training-data row: a corrected word, its sentence context, and
    (when the locate/crop step succeeded) an image crop of exactly where
    it sits on the source page. Never raises — failures are logged so
    they cannot break the correction-save flow that triggers this.
    """
    s = _schema()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""INSERT INTO {s}.corrected_words
                    (doc_hash, filename, page_number, predicted_word,
                     corrected_word, sentence_context, crop_b64, crop_bbox)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (doc_hash, filename or None, page_number,
                 _clip(predicted_word), _clip(corrected_word),
                 (sentence_context or "")[:1000], crop_b64, crop_bbox),
            )
        conn.commit()
    except Exception as e:
        print(f"[WARNING] save_corrected_word DB error: {e}")
    finally:
        conn.close()


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
    restoring the document to un-corrected behaviour on re-upload. Also
    clears corrected_pages — otherwise the re-upload short-circuit in
    backend.py would keep serving the now-reverted correction from cache.
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
            cur.execute(
                f"DELETE FROM {s}.corrected_pages WHERE doc_hash = %s",
                (doc_hash,),
            )
        conn.commit()
    finally:
        conn.close()
    return True, row[0]
