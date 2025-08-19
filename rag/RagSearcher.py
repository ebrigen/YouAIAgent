from __future__ import annotations
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from rag.config import Factory  # central loader/factories
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

@dataclass
class SearchHit:
    id: str
    score: float
    payload: Dict[str, Any]

class RagSearcher:
    """
    Semantic search against a Qdrant collection using config-driven Factory.
    - Loads Qdrant host/port/collection and embedding model from config.yaml / ENV.
    - Ensures collection exists (dimension taken from embedder).
    """
    def __init__(self, factory: Factory | None = None):
        self.factory = factory or Factory()
        self.cfg = self.factory.cfg

        # Models/clients from Factory
        self.embedder = self.factory.embedder()
        self.qdrant: QdrantClient = self.factory.qdrant(
            dim=self.embedder.get_sentence_embedding_dimension()
        )
        self.collection = self.cfg.qdrant.collection

    @staticmethod
    def _build_filter(
        tag_filter: Optional[str],
        min_importance: Optional[float],
        doc_id: Optional[str],
    ) -> Optional[Filter]:
        must = []
        if tag_filter:
            must.append(FieldCondition(key="tags", match=MatchValue(value=tag_filter)))
        if min_importance is not None:
            must.append(FieldCondition(key="importance", range=Range(gte=float(min_importance))))
        if doc_id:
            must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
        return Filter(must=must) if must else None

    def search(
        self,
        question: str,
        top_k: int = 5,
        tag_filter: Optional[str] = None,
        min_importance: Optional[float] = None,
        doc_id: Optional[str] = None,
        *,
        # optional per-call overrides (don’t touch global config file)
        qdrant_override: Optional[dict] = None,
        embed_override: Optional[dict] = None,
    ) -> List[SearchHit]:
        # Optional per-call overrides
        qdrant = self.qdrant
        embedder = self.embedder
        collection = self.collection

        if embed_override:
            embedder = self.factory.embedder(override=embed_override)
        if qdrant_override:
            # if you override collection or host/port, re-create client on the fly
            collection = qdrant_override.get("collection", collection)
            qdrant = self.factory.qdrant(
                dim=embedder.get_sentence_embedding_dimension(),
                override=qdrant_override
            )

        q_vec = embedder.encode(question).tolist()
        flt = self._build_filter(tag_filter, min_importance, doc_id)

        raw_hits = qdrant.search(
            collection_name=collection,
            query_vector=q_vec,
            limit=top_k,
            filter=flt,
        )

        return [
            SearchHit(id=str(h.id), score=float(h.score), payload=h.payload)
            for h in raw_hits
        ]

if __name__ == "__main__":
    # quick interactive smoke test
    s = RagSearcher()  # loads config via Factory
    print(f"🔎 Using collection: {s.collection}")

    try:
        while True:
            q = input("\n❓ Query (empty to exit): ").strip()
            if not q:
                break
            tag = input("🔖 tag filter (enter to skip): ").strip() or None
            imp = input("⭐ min importance (e.g., 1.5, enter to skip): ").strip()
            imp = float(imp) if imp else None
            doc = input("📄 doc_id (enter to skip): ").strip() or None

            hits = s.search(q, top_k=5, tag_filter=tag, min_importance=imp, doc_id=doc)
            if not hits:
                print("— no results —")
                continue
            for i, h in enumerate(hits, 1):
                p = h.payload
                title = p.get("title", "")
                snippet = (p.get("text") or "")[:200].replace("\n", " ")
                idx = p.get("chunk_index")
                tot = p.get("total_chunks")
                print(f"[{i}] score={h.score:.3f}  {title}  ({idx+1}/{tot})")
                print("    ", snippet, "…")
    except KeyboardInterrupt:
        print("\n👋 bye")
