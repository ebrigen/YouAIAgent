import logging
from typing import List
from dataclasses import dataclass
from rag.config import Factory
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# --- Logging setup ---
logger = logging.getLogger("rag_pipeline")

@dataclass
class RetrievedHit:
    score: float
    text: str
    payload: dict

class RagPipeline:
    def __init__(self, factory: Factory | None = None):
        self.factory = factory or Factory()
        self.cfg = self.factory.cfg

        # Embeddings + Qdrant
        self.embedder: SentenceTransformer = self.factory.embedder()
        self.qdrant: QdrantClient = self.factory.qdrant(
            dim=self.embedder.get_sentence_embedding_dimension()
        )
        self.collection = self.cfg.qdrant.collection

        # LLM
        self.tokenizer: AutoTokenizer
        self.llm: AutoModelForSeq2SeqLM
        self.tokenizer, self.llm = self.factory.llm()

        # Device
        want = self.cfg.llm.device
        self.device = torch.device(want if (want == "cpu" or torch.cuda.is_available()) else "cpu")
        self.llm.to(self.device)

        self.max_in = int(self.cfg.llm.max_input_tokens)
        self.max_out = int(self.cfg.llm.max_new_tokens)

        logger.info(f"Initialized RagPipeline with collection={self.collection}, "
                    f"embedder={self.cfg.embedding.model}, llm={self.cfg.llm.name}, device={self.device}")

    # --- Retrieve ---
    def retrieve(self, question: str, top_k: int = 3) -> List[RetrievedHit]:
        logger.info(f"Retrieving top {top_k} docs for query: {question[:80]}...")
        q_vec = self.embedder.encode(question).tolist()

        hits = self.qdrant.search(collection_name=self.collection, query_vector=q_vec, limit=top_k)
        logger.info(f"Retrieved {len(hits)} results from Qdrant")

        return [RetrievedHit(score=float(h.score), text=h.payload.get("text", ""), payload=h.payload) for h in hits]

    # --- Answer ---
    def answer(self, question: str, contexts: List[str]) -> str:
        logger.info(f"Generating answer for query: {question[:80]} with {len(contexts)} contexts")

        context = "\n\n".join([c for c in contexts if c])
        prompt = f"""Usa SOLO le informazioni seguenti per rispondere in modo chiaro e conciso.
Se l'informazione non è presente, dì che non è nel contesto.

Contesto:
\"\"\"
{context}
\"\"\"

Domanda: {question}
Risposta:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_in,
        ).to(self.device)

        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=self.max_out,
            do_sample=False
        )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("Answer generated successfully")
        return answer

    def qa(self, question: str, top_k: int = 3) -> tuple[str, List[RetrievedHit]]:
        logger.info(f"QA pipeline started for query: {question}")
        hits = self.retrieve(question, top_k=top_k)
        if not hits:
            logger.warning("No results found in Qdrant")
            return "Nessun risultato in indice.", []
        answer = self.answer(question, [h.text for h in hits])
        logger.info("QA pipeline completed")
        return answer, hits

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    pipe = RagPipeline()
    q = "how to make a perfect picture?"
    ans, hits = pipe.qa(q, top_k=3)
    print("\n— Answer —\n", ans)
