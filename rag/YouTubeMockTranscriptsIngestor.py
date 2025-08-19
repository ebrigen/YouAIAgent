from __future__ import annotations
import uuid
import random
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

from config import Factory


@dataclass
class YTMockStats:
    videos: int
    chunks: int
    upserted: int
    sec: float


class YouTubeMockTranscriptsIngestor:
    """
    Genera transcript mock per video YouTube (>=10 chunk per video) e li indicizza in Qdrant.
    - Usa Factory per caricare embedder e Qdrant client/collection.
    - point_id = UUID valido (stabile su doc_id+chunk_index).
    - payload 'generico' orientato RAG, con source='youtube' e meta (view_count, like_count, ...).
    """

    def __init__(self, factory: Factory | None = None):
        self.factory = factory or Factory()
        self.cfg = self.factory.cfg

        self.embedder: SentenceTransformer = self.factory.embedder()
        self.qdrant: QdrantClient = self.factory.qdrant(
            dim=self.embedder.get_sentence_embedding_dimension()
        )
        self.collection = self.cfg.qdrant.collection

        # keywords che aumentano l'importance (per reranking)
        self.keywords = self.cfg.ingestion.importance_keywords or [
            "ingredienti", "ricetta", "passaggi", "tutorial", "riassunto", "conclusioni"
        ]

    # ---------- generazione testo stile "video YouTube" ----------
    @staticmethod
    def _sent_bank(topic: str) -> List[str]:
        return [
            f"Benvenuti! Oggi parliamo di {topic} con una guida passo-passo.",
            f"Prima cosa: setup iniziale per {topic}, così eviti errori comuni.",
            f"Vediamo i passaggi pratici per {topic} con esempi semplici.",
            f"Tips & tricks su {topic} per velocizzare il flusso di lavoro.",
            f"Attenzione: gli errori più frequenti su {topic} e come prevenirli.",
            f"FAQ rapide su {topic} con le domande che fate più spesso.",
            f"Mini riepilogo dei concetti chiave su {topic}.",
            f"Best practice consolidate per {topic} secondo l'esperienza sul campo.",
            f"Dimostrazione pratica: applichiamo {topic} in un caso reale.",
            f"Conclusioni e prossimi step per approfondire {topic}.",
        ]

    def _make_video(self, topic: str, chunks_per_video: int) -> Tuple[str, str, List[str], Dict]:
        """
        Crea un 'video' con N chunk (>=10) generando frasi varie sul topic.
        """
        n = max(10, chunks_per_video)  # ✅ minimo 10 chunk
        base = self._sent_bank(topic)
        paragraphs: List[str] = []
        # per dare varietà, mischia e compone frasi finché superi ~450-600 caratteri/pezzo
        for _ in range(n):
            random.shuffle(base)
            acc, out = 0, []
            for s in base:
                spice = random.choice(["ingredienti", "passaggi", "tutorial", "riassunto", "conclusioni"])
                sent = f"{s} Nota importante: {spice}."
                out.append(sent)
                acc += len(sent) + 1
                if acc >= random.randint(450, 600):
                    break
            paragraphs.append(" ".join(out))

        video_id = f"demo_{uuid.uuid4().hex[:12]}"
        title = f"{topic.title()} — Tutorial rapido"
        meta = {
            "channel_title": random.choice(["ProTips Italia", "Dev&Food", "StudioLab", "Sprint Tutorial"]),
            "published_at": "2025-01-01T12:00:00Z",
            "view_count": random.randint(2_000, 1_200_000),
            "like_count": random.randint(80, 50_000),
            "duration": random.randint(180, 2400),
            "description": f"Transcript mock sul tema '{topic}'.",
        }
        return video_id, title, paragraphs, meta

    def _importance(self, text: str) -> float:
        base = 1.0
        return base + sum(0.5 for k in self.keywords if k in text.lower())

    @staticmethod
    def _uuid_from(*parts: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, "::".join(parts)))

    # ---------- ingest pubblico ----------
    def ingest(
        self,
        n_videos: int = 20,
        chunks_per_video: int = 12,
        *,
        topic_pool: List[str] | None = None,
        tags: List[str] | None = None,
        batch_size_embed: int = 256,
        batch_size_upsert: int = 512,
        show_progress: bool = True,
    ) -> YTMockStats:
        """
        Genera n_videos transcript mock (>=10 chunk ciascuno) e li upserta in Qdrant.
        """
        topic_pool = topic_pool or [
            "pasta fatta in casa", "python list comprehension", "allenamento HIIT",
            "finanza personale base", "pattern MVC", "frontend con React",
            "SQL join e indicizzazione", "fotografia notturna", "montaggio video", "grafica con Figma"
        ]
        tags = tags or ["youtube", "transcript", "mock"]

        t0 = time.perf_counter()
        buffer: List[PointStruct] = []
        total_chunks = 0
        upserted = 0

        batch_payloads: List[Tuple[str, str, str, int, int, Dict]] = []  # (vid, title, text, idx, total, meta)

        def _flush_embed_upsert():
            nonlocal total_chunks, upserted, batch_payloads, buffer
            if not batch_payloads:
                return
            texts = [b[2] for b in batch_payloads]
            vectors = self.embedder.encode(texts, show_progress_bar=show_progress)
            for (vid, title, text, idx, total, meta), vec in zip(batch_payloads, vectors):
                doc_id = f"yt_{vid}"
                payload = {
                    "doc_id": doc_id,
                    "source": "youtube",
                    "source_url": f"https://youtu.be/{vid}",
                    "title": title,
                    "text": text,
                    "chunk_index": idx,
                    "total_chunks": total,
                    "start_char": 0,
                    "end_char": len(text),
                    "timestamp_sec": None,
                    "tags": tags,
                    "importance": self._importance(text),
                    "metadata": meta,
                    "external_id": f"{doc_id}_{idx}",
                }
                pid = self._uuid_from(doc_id, str(idx))  # ✅ UUID valido per Qdrant
                buffer.append(PointStruct(id=pid, vector=list(map(float, vec)), payload=payload))
            total_chunks += len(batch_payloads)
            batch_payloads = []

            if len(buffer) >= batch_size_upsert:
                self.qdrant.upsert(collection_name=self.collection, points=buffer)
                upserted += len(buffer)
                buffer = []

        # genera dataset mock
        for _ in range(n_videos):
            topic = random.choice(topic_pool)
            vid, title, chunks, meta = self._make_video(topic, chunks_per_video)
            total = len(chunks)  # >= 10
            for idx, text in enumerate(chunks):
                batch_payloads.append((vid, title, text, idx, total, meta))
                if len(batch_payloads) >= batch_size_embed:
                    _flush_embed_upsert()

        # flush finali
        _flush_embed_upsert()
        if buffer:
            self.qdrant.upsert(collection_name=self.collection, points=buffer)
            upserted += len(buffer)
            buffer = []

        sec = time.perf_counter() - t0
        print(f"✅ YouTube mock ingestion: videos={n_videos}, chunks={total_chunks}, upserted={upserted}, time={sec:.2f}s")
        return YTMockStats(videos=n_videos, chunks=total_chunks, upserted=upserted, sec=sec)


if __name__ == "__main__":
    """
    Esempio:
      docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
      # Configura rag/config.yaml (collection, modelli, ecc.)
      python rag/youtube_mock_ingestor.py
    """
    ing = YouTubeMockTranscriptsIngestor()
    # per la tua richiesta: 20 video, >=10 chunk ciascuno
    ing.ingest(n_videos=20, chunks_per_video=12, batch_size_embed=256, batch_size_upsert=512)
