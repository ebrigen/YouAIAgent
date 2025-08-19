import uuid
import random
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct

class MockRagIngestor:
    """
    Generate synthetic 'YouTube-like' chunks with realistic payload and push them to Qdrant.
    Useful for demos/tests when you don't have transcripts yet.
    """
    def __init__(
        self,
        collection: str = "youtube_rag",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        host: str = "localhost",
        port: int = 6333,
        force_recreate: bool = False,
    ):
        self.collection = collection
        self.embedder = SentenceTransformer(embed_model)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.client = QdrantClient(host, port=port)
        self._ensure_collection(force_recreate)

    def _ensure_collection(self, force: bool):
        from qdrant_client.models import VectorParams
        names = [c.name for c in self.client.get_collections().collections]
        if force and self.collection in names:
            self.client.delete_collection(self.collection)
            names.remove(self.collection)
        if self.collection not in names:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance="Cosine"),
            )
            print(f"✅ Created collection '{self.collection}'")
        else:
            print(f"ℹ️ Using existing collection '{self.collection}'")

    @staticmethod
    def _mk_video(topic: str, n_chunks: int) -> List[str]:
        """Build a synthetic transcript split in n_chunks paragraphs on a topic."""
        templates = [
            f"Introduzione al tema: {topic}. In questa parte diamo il contesto e gli obiettivi.",
            f"Punti chiave su {topic}: elenco di concetti fondamentali e definizioni pratiche.",
            f"Esempio pratico legato a {topic}, con passaggi operativi e accorgimenti.",
            f"Errori comuni su {topic} e come evitarli in situazioni reali.",
            f"Riepilogo e conclusioni su {topic}, con best practice e prossimi passi."
        ]
        # ensure n_chunks paragraphs by repeating/expanding
        base = []
        while len(base) < n_chunks:
            for t in templates:
                if len(base) >= n_chunks:
                    break
                # shuffle in a keyword to make importance kick in
                spice = random.choice(["ingredienti", "passaggi", "tutorial", "riassunto", "conclusioni"])
                base.append(f"{t} Nota importante: {spice}.")
        return base[:n_chunks]

    @staticmethod
    def _importance(text: str) -> float:
        kw = ["ingredienti", "ricetta", "passaggi", "tutorial", "riassunto", "conclusioni"]
        score = 1.0 + sum(0.5 for k in kw if k in text.lower())
        return score

    def generate_and_upsert(
        self,
        n_videos: int = 2,
        chunks_per_video: int = 5,
        topic_pool: List[str] = None
    ) -> int:
        topic_pool = topic_pool or ["carbonara", "python list comprehensions", "allenamento HIIT"]
        all_points: List[PointStruct] = []
        total = 0

        for v in range(n_videos):
            video_id = f"demo_{uuid.uuid4().hex[:8]}"
            topic = random.choice(topic_pool)
            title = f"{topic.title()} — Guida pratica"
            source_url = f"https://youtu.be/{video_id}"
            tags = ["mock", "youtube", topic.split()[0].lower()]

            # fake top-level metadata (per-video)
            meta: Dict = {
                "channel_title": "DemoChannel",
                "published_at": "2025-01-01T00:00:00Z",
                "view_count": random.randint(1_000, 500_000),
                "like_count": random.randint(50, 25_000),
                "duration": random.randint(180, 1800),
                "description": f"Demo transcript for topic '{topic}'."
            }

            chunks = self._mk_video(topic, chunks_per_video)
            vectors = self.embedder.encode(chunks, show_progress_bar=False)

            for idx, (txt, vec) in enumerate(zip(chunks, vectors)):
                payload = {
                    "doc_id": f"yt_{video_id}",
                    "source": "youtube",
                    "source_url": source_url,
                    "title": title,
                    "text": txt,
                    "chunk_index": idx,
                    "total_chunks": chunks_per_video,
                    "start_char": 0,   # mock
                    "end_char": len(txt),
                    "timestamp_sec": None,  # could be filled if you map time
                    "tags": tags,
                    "importance": self._importance(txt),
                    "metadata": meta,
                }
                
                all_points.append(
                    PointStruct(
                       # id=f"{video_id}_{idx}",
                        id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}_{idx}")),
                        vector=vec.tolist(),
                        payload=payload
                    )
                )
                total += 1

        if all_points:
            self.client.upsert(collection_name=self.collection, points=all_points)
            print(f"✅ Upserted {total} mock chunks into '{self.collection}'")
        return total

if __name__ == "__main__":
    # quick test
    ing = MockRagIngestor(collection="youtube_rag", force_recreate=False)
    ing.generate_and_upsert(n_videos=3, chunks_per_video=4)
