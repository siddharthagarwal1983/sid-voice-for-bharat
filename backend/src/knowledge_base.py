"""Lightweight RAG (retrieval-augmented generation) over a local knowledge
base, backed by SQLite FTS5 full-text search.

Ships with a handful of SAMPLE reference documents standing in for real
scheme PDFs / crop advisories / syllabus material — this is placeholder
content to make the retrieval path concrete and testable, not an official
source. Swap it out (or add to it) with real extracted document text via
add_document() for production use.

Retrieval is keyword-based full-text search (BM25 ranking) rather than
embeddings — no extra ML dependency, consistent with the "SQLite is simple
and good enough" approach used elsewhere in this backend, and works well
for terminology-heavy reference material (scheme names, PHC, ORS, etc.).
"""

import re
import sqlite3

import db

_TABLE = "kb_chunks"

_SAMPLE_DOCUMENTS = [
    {
        "source": "PM-JAY Scheme Guide",
        "title": "Eligibility",
        "text": (
            "आयुष्मान भारत PM-JAY योजना उन परिवारों के लिए है जो सामाजिक-आर्थिक जनगणना (SECC) 2011 की "
            "वंचन श्रेणियों में आते हैं — जैसे भूमिहीन खेत मजदूर, कच्चे मकान में रहने वाले परिवार, और "
            "अनुसूचित जाति/जनजाति के परिवार। पात्रता की पुष्टि नजदीकी आयुष्मान भारत सेवा केंद्र या "
            "pmjay.gov.in पर आधार कार्ड से की जा सकती है।"
        ),
    },
    {
        "source": "PM-JAY Scheme Guide",
        "title": "Coverage & documents",
        "text": (
            "PM-JAY के अंतर्गत पात्र परिवार को प्रति वर्ष 5 लाख रुपये तक का कैशलेस अस्पताल इलाज मिलता है, "
            "जो सूचीबद्ध सरकारी और निजी अस्पतालों में मान्य है। आवेदन के लिए आधार कार्ड, राशन कार्ड, और "
            "परिवार की पहचान का प्रमाण साथ ले जाना आवश्यक है।"
        ),
    },
    {
        "source": "Home Care Advisory",
        "title": "ORS aur dehydration",
        "text": (
            "हल्के दस्त या उल्टी में डिहाइड्रेशन से बचने के लिए ORS (ओरल रीहाइड्रेशन सॉल्यूशन) का घोल "
            "थोड़ी-थोड़ी मात्रा में बार-बार दें। अगर 24 घंटे में सुधार न हो, पेशाब कम आए, या बहुत कमजोरी "
            "महसूस हो, तो तुरंत नजदीकी पीएचसी में दिखाएं।"
        ),
    },
    {
        "source": "PHC Referral Guideline",
        "title": "Kab PHC jaayein",
        "text": (
            "साधारण बुखार, हल्की खांसी-जुकाम, या मामूली चोट का इलाज घर पर या आंगनवाड़ी सहायता से किया जा "
            "सकता है। लगातार तेज बुखार, सांस लेने में तकलीफ, गंभीर दर्द, या घाव से खून बहना — इनमें से किसी "
            "भी लक्षण पर नजदीकी पीएचसी (PHC) में जाने की सलाह दी जाती है।"
        ),
    },
]


def init_kb() -> None:
    conn = db.get_connection()
    try:
        conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {_TABLE} USING fts5(source, title, text)")
        count = conn.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()[0]
        if count == 0:
            for doc in _SAMPLE_DOCUMENTS:
                conn.execute(
                    f"INSERT INTO {_TABLE} (source, title, text) VALUES (?, ?, ?)",
                    (doc["source"], doc["title"], doc["text"]),
                )
            conn.commit()
    finally:
        conn.close()


def add_document(source: str, title: str, text: str) -> None:
    """Add a document (or chunk) to the knowledge base.

    Call this yourself to ingest real reference material — extract text
    from a scheme PDF, crop advisory, or syllabus (e.g. with a PDF-parsing
    library), split it into reasonably-sized chunks, and call this once per
    chunk. `source` should identify where it came from (e.g. the document
    name), since search() surfaces it alongside the matched text.
    """
    conn = db.get_connection()
    try:
        conn.execute(
            f"INSERT INTO {_TABLE} (source, title, text) VALUES (?, ?, ?)",
            (source, title, text),
        )
        conn.commit()
    finally:
        conn.close()


def _fts_query(query: str) -> str:
    """Build a safe FTS5 MATCH expression from free-form text.

    Quoting each token as a literal string (rather than passing the raw
    query straight through) avoids FTS5 query-syntax errors on punctuation
    like hyphens (e.g. "PM-JAY" would otherwise be parsed as a column
    filter) and lets any one matching word surface a result.
    """
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    if not tokens:
        return '""'
    return " OR ".join(f'"{token}"' for token in tokens)


def search(query: str, top_k: int = 3) -> list[dict]:
    conn = db.get_connection()
    try:
        rows = conn.execute(
            f"""
            SELECT source, title, text, bm25({_TABLE}) AS score
            FROM {_TABLE}
            WHERE {_TABLE} MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (_fts_query(query), top_k),
        ).fetchall()
        return [{"source": r["source"], "title": r["title"], "text": r["text"]} for r in rows]
    except sqlite3.OperationalError:
        # A pathological query (e.g. only stopword-like tokens) can still
        # trip FTS5 syntax in edge cases — fail closed to "no results"
        # rather than raising into the conversation.
        return []
    finally:
        conn.close()
