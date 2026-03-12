import os, shutil, json, csv, mimetypes, time, asyncio, requests, hashlib, datetime, random
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from docx import Document
from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI, OpenAI
from agents import (
    Agent, Runner, function_tool,
    set_default_openai_client, set_default_openai_api,
    set_tracing_disabled, OpenAIChatCompletionsModel,
)

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DOCS_DIR    = "my_docs"
MODEL_NAME  = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM   = 3072
INDEX_NAME  = "local-docs-index"

CHAT_NAMESPACE    = "chat_history"
DOCS_NAMESPACE    = "documents"
CHAT_HISTORY_DAYS = 30
CHAT_DUMMY_VECTOR = [0.001] * EMBED_DIM  # dummy vector for chat saves — no Gemini call

SEED_TOPICS = [
    "what happen in cancer",
    "cancer treatment",
    "what happen in diabetes",
    "diabetes treatment",
]

ABSTRACTS_PER_TOPIC  = 10
RATE_LIMIT_DELAY     = 4        # base seconds for Gemini 429 backoff
MAX_RETRIES          = 4
SIMILARITY_THRESHOLD = 0.50
MAX_ABSTRACT_CHARS   = 2500     # ~1 full PubMed abstract
MAX_NCBI_CHARS       = ABSTRACTS_PER_TOPIC * MAX_ABSTRACT_CHARS

NCBI_TOOL    = os.getenv("NCBI_TOOL",  "ncbi_medical_chatbot")
NCBI_EMAIL   = os.getenv("NCBI_EMAIL", "your_email@example.com")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
NCBI_DELAY   = 0.15 if NCBI_API_KEY else 0.5   # 10 req/s with key, 3 req/s without

os.makedirs(DOCS_DIR, exist_ok=True)

# ── Clients ───────────────────────────────────────────────────────────────────

gemini_async_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
gemini_sync_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

set_default_openai_client(gemini_async_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

gemini_model = OpenAIChatCompletionsModel(
    model=MODEL_NAME,
    openai_client=gemini_async_client,
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
vector_index = pc.Index(INDEX_NAME)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""],
)

# ── In-memory caches ──────────────────────────────────────────────────────────

_embedding_cache: dict[str, list[float]] = {}
_answer_cache:    dict[str, str]         = {}
# Filenames indexed this session. Prevents re-embedding on repeated fallback queries.
# Resets on restart, which is correct since upload_* vectors are also cleared on restart.
_indexed_files:   set[str]               = set()

# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed_sync(text: str) -> list[float]:
    """Embed text synchronously with exponential backoff. Used at index time."""
    for attempt in range(MAX_RETRIES):
        try:
            return gemini_sync_client.embeddings.create(
                input=text, model=EMBED_MODEL
            ).data[0].embedding
        except Exception as e:
            if any(x in str(e).lower() for x in ("429", "quota", "rate")):
                wait = RATE_LIMIT_DELAY * (2 ** attempt)
                print(f"⚠️  Gemini rate limited — retrying in {wait}s ({attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Embedding failed after max retries.")


async def _embed_async(text: str) -> list[float]:
    """Embed text asynchronously with cache + exponential backoff. Used at query time."""
    key = hashlib.md5(text.encode()).hexdigest()
    if key in _embedding_cache:
        return _embedding_cache[key]

    for attempt in range(MAX_RETRIES):
        try:
            resp = await gemini_async_client.embeddings.create(input=text, model=EMBED_MODEL)
            emb = resp.data[0].embedding
            _embedding_cache[key] = emb
            return emb
        except Exception as e:
            if any(x in str(e).lower() for x in ("429", "quota", "rate")):
                wait = RATE_LIMIT_DELAY * (2 ** attempt)
                print(f"⚠️  Gemini rate limited — retrying in {wait}s ({attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError("Async embedding failed after max retries.")

# ── NCBI helpers ──────────────────────────────────────────────────────────────

def _ncbi_params() -> dict:
    params = {"tool": NCBI_TOOL, "email": NCBI_EMAIL}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params


def _ncbi_delay() -> None:
    time.sleep(NCBI_DELAY + random.uniform(0, 0.3))


def _ncbi_get(url: str, params: dict, timeout: int = 15) -> Optional[requests.Response]:
    """GET request to NCBI with retries on 429, 503, timeout, and connection errors."""
    headers = {"User-Agent": f"{NCBI_TOOL}/1.0 ({NCBI_EMAIL})"}
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code in (429, 503):
                wait = 2 ** (attempt + 1)
                print(f"⚠️  NCBI {resp.status_code} — backing off {wait}s ({attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            return resp
        except requests.exceptions.Timeout:
            wait = 2 ** (attempt + 1)
            print(f"⚠️  NCBI timeout — retrying in {wait}s ({attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)
        except requests.exceptions.ConnectionError as e:
            wait = 2 ** (attempt + 1)
            print(f"⚠️  NCBI connection error — retrying in {wait}s ({attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(wait)
    print(f"❌  NCBI request failed after {MAX_RETRIES} attempts.")
    return None

# ── File helpers ──────────────────────────────────────────────────────────────

def _topic_filename(topic: str) -> str:
    """Convert a topic string to its .txt filename on disk."""
    return topic.replace(" ", "_").replace("/", "_") + ".txt"


def _active_seed_filenames() -> set[str]:
    return {_topic_filename(t) for t in SEED_TOPICS}


def _read_file_chunks(path: str) -> list[str]:
    """Read any supported file and return text chunks. Returns [] on failure."""
    mime_type, _ = mimetypes.guess_type(path)
    text = ""
    try:
        if mime_type == "application/pdf":
            text = "".join(p.extract_text() or "" for p in PdfReader(path).pages)
        elif mime_type in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ):
            text = "\n".join(p.text for p in Document(path).paragraphs)
        elif mime_type == "text/csv":
            with open(path, encoding="utf-8") as f:
                text = "\n".join(",".join(row) for row in csv.reader(f))
        elif mime_type == "application/json":
            with open(path, encoding="utf-8") as f:
                text = json.dumps(json.load(f))
        else:
            with open(path, encoding="utf-8") as f:
                text = f.read()
    except Exception as e:
        print(f"⚠️  Could not read {os.path.basename(path)}: {e}")
        return []
    return text_splitter.split_text(text) if text.strip() else []

# ── NCBI fetching ─────────────────────────────────────────────────────────────

def fetch_and_save_ncbi(query: str, max_results: int = ABSTRACTS_PER_TOPIC) -> Optional[str]:
    """
    Fetch PubMed abstracts for a query and save to disk.
    Returns the local file path, or None on failure.
    Skips the NCBI call entirely if the file already exists on disk.
    """
    path = os.path.join(DOCS_DIR, _topic_filename(query))
    if os.path.exists(path):
        print(f"   📁  Cached on disk — skipping NCBI: {query[:60]}")
        return path

    base   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    params = _ncbi_params()

    try:
        search = _ncbi_get(
            base + "esearch.fcgi",
            {**params, "db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"},
            timeout=10,
        )
        if not search:
            return None

        ids = search.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return None

        _ncbi_delay()

        fetch = _ncbi_get(
            base + "efetch.fcgi",
            {**params, "db": "pubmed", "id": ",".join(ids), "retmode": "text", "rettype": "abstract"},
            timeout=15,
        )
        if not fetch:
            return None

        # Truncate at sentence boundary so we never cut mid-abstract
        text = fetch.text
        if len(text) > MAX_NCBI_CHARS:
            text = text[:MAX_NCBI_CHARS]
            last_period = text.rfind(".")
            if last_period > MAX_NCBI_CHARS * 0.8:
                text = text[:last_period + 1]
            print(f"   ✂️  Truncated to {len(text):,} chars ({max_results} × {MAX_ABSTRACT_CHARS} max)")

        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    except Exception as e:
        print(f"❌  NCBI fetch failed for '{query}': {e}")
        return None

# ── Indexing ──────────────────────────────────────────────────────────────────
#
# Vector ID scheme:
#   seed_<md5>   — active SEED_TOPICS (stable across restarts, never auto-deleted)
#   upload_<md5> — user uploads + fallback NCBI fetches (cleared on every restart)
#
# Content-hash IDs are deterministic: same chunk → same ID forever.
# On restart, we fetch existing IDs from Pinecone and skip any chunk already there,
# so seed embeddings are preserved indefinitely with zero Gemini calls on re-runs.

def _fetch_existing_ids(id_prefix: str) -> set[str]:
    """Return all vector IDs in DOCS_NAMESPACE that start with the given prefix."""
    existing = set()
    try:
        for batch in vector_index.list(prefix=id_prefix, namespace=DOCS_NAMESPACE):
            existing.update(batch if isinstance(batch, list) else [batch])
    except Exception as e:
        print(f"⚠️  Could not fetch existing IDs for prefix '{id_prefix}': {e}")
    return existing


def _index_files(filenames: set[str], id_prefix: str) -> tuple[int, int]:
    """
    Embed and upsert chunks for the given filenames.
    Chunks whose vector ID already exists in Pinecone are skipped entirely —
    no Gemini embedding call is made for them.
    Registers processed filenames in _indexed_files.
    Returns (indexed_count, skipped_count).
    """
    existing_ids = _fetch_existing_ids(id_prefix)

    vectors = []
    indexed = 0
    skipped = 0

    for filename in filenames:
        path = os.path.join(DOCS_DIR, filename)
        if not os.path.isfile(path):
            continue
        for chunk in _read_file_chunks(path):
            chunk_id = f"{id_prefix}_{hashlib.md5(chunk.encode()).hexdigest()}"
            if chunk_id in existing_ids:
                skipped += 1
                continue
            vectors.append({
                "id":       chunk_id,
                "values":   _embed_sync(chunk),
                "metadata": {"text": chunk, "source": filename},
            })
            indexed += 1
            if indexed % 10 == 0:
                print(f"   Embedded {indexed} new chunks...")
                time.sleep(RATE_LIMIT_DELAY)
        _indexed_files.add(filename)

    for i in range(0, len(vectors), 100):
        vector_index.upsert(vectors=vectors[i:i + 100], namespace=DOCS_NAMESPACE)

    return indexed, skipped


def index_seed_documents() -> tuple[int, int]:
    return _index_files(_active_seed_filenames(), id_prefix="seed")


def index_uploaded_documents() -> tuple[int, int]:
    all_files    = {f for f in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, f))}
    upload_files = all_files - _active_seed_filenames()
    return _index_files(upload_files, id_prefix="upload")


def index_all_documents() -> tuple[int, int]:
    si, ss = index_seed_documents()
    ui, us = index_uploaded_documents()
    return si + ui, ss + us

# ── Vector cleanup ────────────────────────────────────────────────────────────

def _list_doc_ids() -> list[str]:
    all_ids = []
    for batch in vector_index.list(namespace=DOCS_NAMESPACE):
        all_ids.extend(batch if isinstance(batch, list) else [batch])
    return all_ids


def _delete_doc_ids(ids: list[str]) -> None:
    for i in range(0, len(ids), 100):
        vector_index.delete(ids=ids[i:i + 100], namespace=DOCS_NAMESPACE)


def delete_uploaded_vectors() -> int:
    """Delete all upload_* vectors. Seed vectors are never touched here."""
    try:
        ids = [v for v in _list_doc_ids() if v.startswith("upload_")]
        if not ids:
            print("    ✅ No upload vectors to remove.")
            return 0
        _delete_doc_ids(ids)
        print(f"    🗑️  Deleted {len(ids)} upload vectors.")
        return len(ids)
    except Exception as e:
        print(f"    ⚠️  Could not delete upload vectors: {e}")
        return 0


def delete_stale_seed_vectors() -> int:
    """Delete seed_* vectors for topics removed from SEED_TOPICS."""
    try:
        existing = {v for v in _list_doc_ids() if v.startswith("seed_")}
        if not existing:
            print("    ✅ No seed vectors to check.")
            return 0

        expected      = set()
        files_on_disk = 0
        for filename in _active_seed_filenames():
            path = os.path.join(DOCS_DIR, filename)
            if os.path.isfile(path):
                files_on_disk += 1
                for chunk in _read_file_chunks(path):
                    expected.add(f"seed_{hashlib.md5(chunk.encode()).hexdigest()}")

        # Skip if no seed files are on disk yet — avoids wiping valid vectors
        # on first run before NCBI files have been fetched.
        if files_on_disk == 0:
            print("    ✅ No seed files on disk yet — skipping stale vector check.")
            return 0

        stale = list(existing - expected)
        if not stale:
            print("    ✅ No stale seed vectors.")
            return 0

        _delete_doc_ids(stale)
        print(f"    🗑️  Deleted {len(stale)} stale seed vectors.")
        return len(stale)
    except Exception as e:
        print(f"    ⚠️  Could not delete stale seed vectors: {e}")
        return 0

# ── Disk cleanup ──────────────────────────────────────────────────────────────

def cleanup_docs_folder() -> tuple[int, int]:
    """
    Keep ONLY active seed .txt files in DOCS_DIR — delete everything else.
    This includes stale seed files, user uploads (.pdf, .docx, .csv, etc.),
    and any fallback NCBI files fetched at query time.
    Returns (kept, deleted).
    """
    seed_files    = _active_seed_filenames()
    kept = deleted = 0
    for filename in os.listdir(DOCS_DIR):
        filepath = os.path.join(DOCS_DIR, filename)
        if not os.path.isfile(filepath):
            continue
        if filename in seed_files:
            kept += 1
        else:
            os.remove(filepath)
            deleted += 1
            print(f"   🗑️  Removed: {filename}")
    return kept, deleted

# ── Chat history ──────────────────────────────────────────────────────────────

async def save_chat_message(role: str, content: str, timestamp: str) -> None:
    """Save a chat message to Pinecone using a dummy vector (zero Gemini calls)."""
    try:
        msg_id = hashlib.md5(f"{timestamp}_{role}_{content[:80]}".encode()).hexdigest()
        vector_index.upsert(
            vectors=[{
                "id":       msg_id,
                "values":   CHAT_DUMMY_VECTOR,
                "metadata": {"role": role, "content": content, "timestamp": timestamp},
            }],
            namespace=CHAT_NAMESPACE,
        )
    except Exception as e:
        print(f"⚠️  Could not save chat message: {e}")


def fetch_chat_history() -> list[dict]:
    """Return all chat messages from the last 30 days, sorted oldest-first."""
    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=CHAT_HISTORY_DAYS)).isoformat()
    try:
        all_ids = []
        for batch in vector_index.list(namespace=CHAT_NAMESPACE):
            all_ids.extend(batch if isinstance(batch, list) else [batch])
        if not all_ids:
            return []

        messages = []
        for i in range(0, len(all_ids), 100):
            resp = vector_index.fetch(ids=all_ids[i:i + 100], namespace=CHAT_NAMESPACE)
            for vid, vec in resp.vectors.items():
                meta = vec.metadata or {}
                ts   = meta.get("timestamp", "")
                if ts >= cutoff:
                    messages.append({
                        "id":        vid,
                        "role":      meta.get("role", "bot"),
                        "content":   meta.get("content", ""),
                        "timestamp": ts,
                    })

        messages.sort(key=lambda m: m["timestamp"])
        return messages
    except Exception as e:
        print(f"❌  Could not fetch chat history: {e}")
        return []


def cleanup_old_chat_history() -> int:
    """Delete chat messages older than CHAT_HISTORY_DAYS. Returns count deleted."""
    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=CHAT_HISTORY_DAYS)).isoformat()
    try:
        all_ids = []
        for batch in vector_index.list(namespace=CHAT_NAMESPACE):
            all_ids.extend(batch if isinstance(batch, list) else [batch])
        if not all_ids:
            print("💬  Chat history empty — nothing to expire.")
            return 0

        expired = []
        for i in range(0, len(all_ids), 100):
            resp = vector_index.fetch(ids=all_ids[i:i + 100], namespace=CHAT_NAMESPACE)
            for vid, vec in resp.vectors.items():
                ts = (vec.metadata or {}).get("timestamp", "")
                if ts and ts < cutoff:
                    expired.append(vid)

        if not expired:
            print(f"💬  No messages older than {CHAT_HISTORY_DAYS} days.")
            return 0

        for i in range(0, len(expired), 100):
            vector_index.delete(ids=expired[i:i + 100], namespace=CHAT_NAMESPACE)
        print(f"💬  Expired {len(expired)} old chat messages.")
        return len(expired)
    except Exception as e:
        print(f"⚠️  Could not clean chat history: {e}")
        return 0

# ── Agents ────────────────────────────────────────────────────────────────────

@function_tool
async def retrieve_context(query: str) -> str:
    """Search Pinecone for document chunks relevant to the query."""
    emb     = await _embed_async(query)
    results = vector_index.query(
        vector=emb, top_k=5, include_metadata=True, namespace=DOCS_NAMESPACE
    )
    docs = [
        m["metadata"]["text"]
        for m in results.get("matches", [])
        if m["score"] > SIMILARITY_THRESHOLD
    ]
    return "\n\n---\n\n".join(docs) if docs else ""


@function_tool
async def fetch_pubmed_and_index(query: str) -> str:
    """
    Fallback tool: fetch PubMed abstracts for an unknown query and index them.
    Guards against redundant work:
      1. Already indexed this session → return instantly
      2. File already on disk → skip NCBI network call
      3. Already in Pinecone → skip Gemini embedding calls
      4. Index ONLY the new file, not the entire corpus
    """
    filename = _topic_filename(query)

    if filename in _indexed_files:
        return f"Already indexed this session: {filename}"

    # Both operations are blocking — run off the async event loop
    loop = asyncio.get_running_loop()
    path = await loop.run_in_executor(None, lambda: fetch_and_save_ncbi(query))
    if not path:
        return "No PubMed articles found for this query."

    indexed, skipped = await loop.run_in_executor(
        None, lambda: _index_files({filename}, id_prefix="upload")
    )
    return f"Indexed {indexed} new chunks for: {query}" + (f" ({skipped} already existed)" if skipped else "")


retriever_agent = Agent(
    name="RetrieverAgent",
    instructions=(
        "You are a medical knowledge retrieval specialist. When given a question:\n"
        "1. Call retrieve_context with the question.\n"
        "2. If retrieve_context returns empty, call fetch_pubmed_and_index with the "
        "   key medical term, then call retrieve_context again.\n"
        "3. Return ONLY the raw retrieved context — no commentary."
    ),
    tools=[retrieve_context, fetch_pubmed_and_index],
    model=gemini_model,
)

answer_agent = Agent(
    name="AnswerAgent",
    instructions=(
        "You are a helpful medical assistant. "
        "You will receive document chunks and a question. "
        "Answer clearly and concisely based only on the provided chunks. "
        "If the context lacks enough information, say so honestly. "
        "Output only the final answer."
    ),
    model=gemini_model,
)

# ── Startup ───────────────────────────────────────────────────────────────────

async def _startup():
    print(f"🚀  NCBI Medical Chatbot v13")
    print(f"    NCBI: tool='{NCBI_TOOL}' | email='{NCBI_EMAIL}' | key={'set' if NCBI_API_KEY else 'not set'}")
    print(f"    Char limit: {MAX_NCBI_CHARS:,} ({ABSTRACTS_PER_TOPIC} abstracts × {MAX_ABSTRACT_CHARS})\n")

    # Step 1: Remove .txt files for topics no longer in SEED_TOPICS
    print("🧹  Step 1/4 — Disk cleanup...")
    kept, deleted = cleanup_docs_folder()
    print(f"    Kept {kept} | Deleted {deleted} stale files\n")

    # Step 2: Expire chat messages older than CHAT_HISTORY_DAYS
    print("💬  Step 2/4 — Chat history cleanup...")
    expired = cleanup_old_chat_history()
    print(f"    {expired} old messages removed\n")

    # Step 3: Remove upload_* vectors (always) + seed_* vectors for removed topics
    print("🗑️   Step 3/4 — Vector cleanup...")
    del_uploads = delete_uploaded_vectors()
    del_stale   = delete_stale_seed_vectors()
    print(f"    {del_uploads} upload vectors + {del_stale} stale seed vectors removed\n")

    # Fetch any seed topic files not yet on disk
    missing = [t for t in SEED_TOPICS if not os.path.exists(os.path.join(DOCS_DIR, _topic_filename(t)))]
    on_disk = len(SEED_TOPICS) - len(missing)
    print(f"📁  {on_disk}/{len(SEED_TOPICS)} topics already on disk.")
    if missing:
        print(f"📥  Fetching {len(missing)} missing topic(s) from NCBI...\n")

    fetched = 0
    for i, topic in enumerate(SEED_TOPICS):
        is_new = topic in missing
        path   = fetch_and_save_ncbi(topic)
        fetched += bool(path)
        status  = "✅" if path else "⚠️ "
        print(f"   [{i+1}/{len(SEED_TOPICS)}] {status} {topic[:65]}")
        if is_new:
            _ncbi_delay()

    # Step 4: Index seed topics — skips chunks already in Pinecone (zero Gemini calls)
    print(f"\n📦  Step 4/4 — Indexing seed topics (skipping already-embedded chunks)...")
    indexed, skipped = index_seed_documents()

    if skipped > 0 and indexed == 0:
        print(f"    ⚡ All {skipped} seed chunks already in Pinecone — zero Gemini calls needed.")
    elif skipped > 0:
        print(f"    ⚡ {indexed} new chunks embedded | {skipped} already in Pinecone (skipped).")
    else:
        print(f"    📄 {indexed} chunks embedded and indexed.")

    print(f"\n✅  Startup complete — {fetched} seed topics ready\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _startup()
    yield


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="NCBI Medical Chatbot",
    description="RAG-powered medical chatbot with PubMed knowledge",
    version="13.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/health")
def health_check():
    return {
        "status":            "ok",
        "topics_seeded":     len(SEED_TOPICS),
        "indexed_files":     len(_indexed_files),
        "cached_answers":    len(_answer_cache),
        "cached_embeddings": len(_embedding_cache),
        "ncbi_api_key_set":  bool(NCBI_API_KEY),
        "ncbi_email":        NCBI_EMAIL,
    }


@app.get("/chat-history")
def get_chat_history():
    return {"history": fetch_chat_history()}


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    saved = []
    for file in files:
        if not file.filename.lower().endswith((".pdf", ".txt", ".docx", ".csv", ".json", ".md", ".html")):
            continue
        dest = os.path.join(DOCS_DIR, file.filename)
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(file.filename)
    if not saved:
        raise HTTPException(status_code=400, detail="No valid files uploaded")
    return {"message": "Files uploaded successfully", "files": saved}


@app.post("/index")
def index_documents():
    indexed, skipped = index_all_documents()
    if indexed == 0 and skipped == 0:
        raise HTTPException(status_code=400, detail="No documents found to index")
    return {"message": "Indexed successfully", "chunks_indexed": indexed, "chunks_skipped": skipped}


@app.post("/scrape-ncbi")
def scrape_ncbi(payload: QuestionRequest):
    query = payload.question.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    path = fetch_and_save_ncbi(query)
    if not path:
        return {"message": "No articles found"}
    return {"message": f"Scraped PubMed for '{query}'", "file": path}


@app.post("/ask")
async def ask(payload: QuestionRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    now = datetime.datetime.utcnow().isoformat()
    await save_chat_message("user", question, now)

    cache_key = question.lower()
    if cache_key in _answer_cache:
        cached = _answer_cache[cache_key]
        print(f"✅  Cache hit: '{question}'")
        await save_chat_message("bot", cached, datetime.datetime.utcnow().isoformat())
        return {"answer": cached, "source": "cache"}

    retrieval = await Runner.run(retriever_agent, question)
    context   = retrieval.final_output.strip()

    if not context:
        answer = "I could not find relevant medical information for your question."
        await save_chat_message("bot", answer, datetime.datetime.utcnow().isoformat())
        return {"answer": answer}

    result = await Runner.run(answer_agent, f"Document chunks:\n{context}\n\nQuestion: {question}")
    answer = result.final_output.strip()

    _answer_cache[cache_key] = answer
    await save_chat_message("bot", answer, datetime.datetime.utcnow().isoformat())
    return {"answer": answer, "source": "agents"}