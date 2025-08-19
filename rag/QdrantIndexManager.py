from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams
from rag.config import Factory
    
class QdrantIndexManager:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host, port=port)

    def create_if_missing(self, collection: str, dim: int, distance: str = "Cosine") -> None:
        names = [c.name for c in self.client.get_collections().collections]
        if collection in names:
            print(f"ℹ️ Collection '{collection}' already exists.")
            return
        self.client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=distance),
        )
        print(f"✅ Created collection '{collection}' (dim={dim}, distance={distance}).")

    def recreate(self, collection: str, dim: int, distance: str = "Cosine") -> None:
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=distance),
        )
        print(f"✅ Recreated collection '{collection}'.")

    def drop(self, collection: str) -> None:
        self.client.delete_collection(collection)
        print(f"🗑️ Dropped collection '{collection}'.")

if __name__ == "__main__":
    # quick test
    mgr = QdrantIndexManager()
    mgr.create_if_missing("youtube_rag", 384, "Cosine")
