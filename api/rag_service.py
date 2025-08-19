from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import logging

# reuse your pipeline (uses Factory under the hood)
from rag.RagPipeline import RagPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("rag_service")
ROOT = Path(__file__).resolve().parent.parent  # project root
PUBLIC_DIR = ROOT / "server/public"
INDEX_FILE = PUBLIC_DIR / "index.html"

app = FastAPI(title="RAG Service")

# initialize once (loads embedder/LLM/Qdrant from rag/config.yaml / ENV)
pipe = RagPipeline()

# ---------- API models ----------
class QARequest(BaseModel):
    query: str
    top_k: int = 5

class Hit(BaseModel):
    score: float
    doc_id: Optional[str] = None
    title: Optional[str] = None
    text: Optional[str] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    source_url: Optional[str] = None
    tags: Optional[List[str]] = None
    importance: Optional[float] = None
    metadata: Optional[dict] = None

class QAResponse(BaseModel):
    answer: str
    hits: List[Hit]

# ---------- API endpoint ----------
@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest):
    logger.info(f"Incoming query: '{req.query}' (top_k={req.top_k})")
    
    hits = pipe.retrieve(req.query, top_k=req.top_k)
    if not hits:
        return QAResponse(answer="No results in index.", hits=[])
    contexts = [h.payload.get("text", "") for h in hits]
    answer = pipe.answer(req.query, contexts)
    norm = [
        Hit(
            score=float(h.score),
            doc_id=p.get("doc_id"),
            title=p.get("title"),
            text=p.get("text"),
            chunk_index=p.get("chunk_index"),
            total_chunks=p.get("total_chunks"),
            source_url=p.get("source_url"),
            tags=p.get("tags"),
            importance=p.get("importance"),
            metadata=p.get("metadata"),
        ) for h in hits for p in [h.payload]
    ]
    return QAResponse(answer=answer, hits=norm)

# ---------- static site ----------
app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")

@app.get("/")
def root():
    # serve your UI
    return FileResponse(str(INDEX_FILE))
