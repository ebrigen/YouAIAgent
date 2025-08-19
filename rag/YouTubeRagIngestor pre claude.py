from __future__ import annotations
import os
import uuid
from typing import Dict, List, Tuple, Optional, Callable, Iterable
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

from rag.config import Factory  # <- YAML/ENV driven factory


@dataclass
class IngestStats:
    files: int
    chunks: int
    upserted: int


class YouTubeRagIngestor:
    """
    Ingest transcripts into a Qdrant collection using config-driven Factory.
    - Embeddings: SentenceTransformers (from config.embedding)
    - Qdrant: host/port/collection, auto-create with proper dim (from config.qdrant)
    - Chunking: size/overlap + importance keywords (from config.ingestion)
    - Payload: generic, RAG-oriented (doc_id, source, text, chunk_index, total_chunks, tags, importance, metadata)
    """

    def __init__(self, factory: Factory | None = None):
        self.factory = factory or Factory()
        self.cfg = self.factory.cfg

        # models/clients
        self.embedder: SentenceTransformer = self.factory.embedder()
        self.qdrant: QdrantClient = self.factory.qdrant(
            dim=self.embedder.get_sentence_embedding_dimension()
        )

        # collection + knobs
        self.collection = self.cfg.qdrant.collection
        self.chunk_size = self.cfg.ingestion.chunk_size
        self.chunk_overlap = self.cfg.ingestion.chunk_overlap
        self.importance_keywords = list(self.cfg.ingestion.importance_keywords or [])

    # ---------- helpers ----------
    def _chunk_text(self, text: str) -> List[Tuple[int, int, str]]:
        """Return list of (start_char, end_char, chunk_text)."""
        chunks: List[Tuple[int, int, str]] = []
        size, overlap = self.chunk_size, self.chunk_overlap
        step = max(1, size - overlap)
        i = 0
        L = len(text)
        while i < L:
            start = i
            end = min(i + size, L)
            chunk = text[start:end]
            # try not to cut mid-sentence
            period = chunk.rfind(".")
            if period != -1 and period > int(size * 0.5):
                end = min(start + period + 1, L)
                chunk = text[start:end]
            chunk = chunk.strip()
            if chunk:
                chunks.append((start, end, chunk))
            i = end if overlap == 0 else max(end - overlap, end) if end == L else end - overlap
        return chunks

    def _importance(self, text: str) -> float:
        base = 1.0
        if not self.importance_keywords:
            return base
        bonus = sum(0.5 for kw in self.importance_keywords if kw.lower() in text.lower())
        return base + bonus

    @staticmethod
    def _stable_uuid(*parts: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, "::".join(parts)))

    @staticmethod
    def _default_meta(doc_id: str, title: str) -> Dict:
        return {
            "channel_title": "UnknownChannel",
            "published_at": None,
            "view_count": None,
            "like_count": None,
            "duration": None,
            "description": f"Transcript for {doc_id} ({title})",
        }

    def _build_payload(
        self,
        *,
        doc_id: str,
        source_url: str,
        title: str,
        chunk_index: int,
        total_chunks: int,
        start_char: int,
        end_char: int,
        text: str,
        tags: List[str],
        importance: float,
        metadata: Dict,
    ) -> Dict:
        return {
            "doc_id": doc_id,
            "source": "youtube",
            "source_url": source_url,
            "title": title,
            "text": text,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "start_char": start_char,
            "end_char": end_char,
            "timestamp_sec": metadata.get("timestamp_sec"),  # optional if you have it
            "tags": tags,
            "importance": importance,
            "metadata": {
                "channel_title": metadata.get("channel_title"),
                "published_at": metadata.get("published_at"),
                "view_count": metadata.get("view_count"),
                "like_count": metadata.get("like_count"),
                "duration": metadata.get("duration"),
                "description": metadata.get("description"),
            },
            # keep a human-readable external id (useful for debugging)
            "external_id": f"{doc_id}_{chunk_index}",
        }

    # ---------- public API ----------
    def ingest_transcripts_folder(
        self,
        transcripts_dir: str = "transcripts",
        *,
        default_tags: Optional[List[str]] = None,
        video_meta_provider: Optional[Callable[[str], Dict]] = None,
        title_provider: Optional[Callable[[str], str]] = None,
        source_url_provider: Optional[Callable[[str], str]] = None,
        dry_run: bool = False,
        batch_size: int = 512,
    ) -> IngestStats:
        """
        Ingest all *.txt files from a folder.
        - default_tags: will be added to each chunk's payload
        - video_meta_provider(video_id) -> Dict: supply real YouTube stats if you have them
        - title_provider(video_id) -> str
        - source_url_provider(video_id) -> str (e.g., https://youtu.be/<id>)
        - dry_run: don't upsert, just return stats
        - batch_size: upsert in batches to avoid huge requests
        """
        default_tags = list(default_tags or ["youtube", "transcript"])
        files = [f for f in os.listdir(transcripts_dir) if f.endswith(".txt")]

        total_files = 0
        total_chunks = 0
        upserted = 0
        buffer: List[PointStruct] = []

        if not files:
            print(f"⚠️ No .txt files in '{transcripts_dir}'.")
            return IngestStats(files=0, chunks=0, upserted=0)

        for fname in files:
            total_files += 1
            video_id = os.path.splitext(fname)[0]
            title = title_provider(video_id) if title_provider else f"Video {video_id}"
            source_url = source_url_provider(video_id) if source_url_provider else f"https://youtu.be/{video_id}"
            meta = video_meta_provider(video_id) if video_meta_provider else self._default_meta(video_id, title)

            # read transcript
            with open(os.path.join(transcripts_dir, fname), "r", encoding="utf-8") as f:
                full_text = f.read().replace("\n", " ").strip()
            if not full_text:
                continue

            # chunk & embed
            chunks = self._chunk_text(full_text)
            total = len(chunks)
            if total == 0:
                continue

            texts = [c[2] for c in chunks]
            vectors = self.embedder.encode(texts, show_progress_bar=True)

            for idx, ((start, end, txt), vec) in enumerate(zip(chunks, vectors)):
                importance = self._importance(txt)
                doc_id = f"yt_{video_id}"
                payload = self._build_payload(
                    doc_id=doc_id,
                    source_url=source_url,
                    title=title,
                    chunk_index=idx,
                    total_chunks=total,
                    start_char=start,
                    end_char=end,
                    text=txt,
                    tags=default_tags,
                    importance=importance,
                    metadata=meta,
                )
                point_id = self._stable_uuid(doc_id, str(idx))  # ✅ valid UUID id
                buffer.append(PointStruct(id=point_id, vector=list(map(float, vec)), payload=payload))
                total_chunks += 1

                # batch upsert
                if not dry_run and len(buffer) >= batch_size:
                    self.qdrant.upsert(collection_name=self.collection, points=buffer)
                    upserted += len(buffer)
                    buffer.clear()

        # flush remainder
        if not dry_run and buffer:
            self.qdrant.upsert(collection_name=self.collection, points=buffer)
            upserted += len(buffer)
            buffer.clear()

        print(f"✅ Ingested files={total_files}, chunks={total_chunks}, upserted={upserted} into '{self.collection}'")
        return IngestStats(files=total_files, chunks=total_chunks, upserted=upserted)


# ---------------- quick test ----------------
if __name__ == "__main__":
    """
    Smoke test:
    - Reads config from rag/config.yaml (and ENV)
    - Ensures collection exists with correct dim
    - Ingests transcripts/ folder
    """
    ing = YouTubeRagIngestor()  # uses Factory under the hood
    stats = ing.ingest_transcripts_folder("transcripts")
    print(stats)
