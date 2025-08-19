import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct

# === CONFIG ===
COLLECTION_NAME = "youtube_rag_mock"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS = [
    "In questa lezione spieghiamo come preparare la carbonara.",
    "È importante mescolare le uova fuori dal fuoco.",
    "Aggiungere il pecorino gradualmente mentre si manteca.",
    "Non si usa la panna nella carbonara originale."
]
VIDEO_METADATA = {
    "video_id": "demo123",
    "title": "Come cucinare la carbonara perfetta",
    "description": "Un tutorial semplice e veloce per cucinare la carbonara.",
    "channel_title": "CucinaFacile",
    "published_at": "2024-10-01T12:00:00Z",
    "duration": 480,
    "view_count": 123456,
    "like_count": 7890,
    "tags": ["ricetta", "carbonara", "tutorial"]
}

# === STEP 1: Connessione Qdrant
    qdrant = QdrantClient("localhost", port=6333)
if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    embed_dim = SentenceTransformer(EMBED_MODEL).get_sentence_embedding_dimension()
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=embed_dim, distance="Cosine")
    )
    print(f"✅ Collezione '{COLLECTION_NAME}' creata.")
else:
    print(f"ℹ️ Collezione '{COLLECTION_NAME}' già esistente.")

# === STEP 2: Calcola gli embedding
model = SentenceTransformer(EMBED_MODEL)
points = []
for idx, chunk in enumerate(CHUNKS):
    vector = model.encode(chunk).tolist()
    payload = {
        "video_id": VIDEO_METADATA["video_id"],
        "title": VIDEO_METADATA["title"],
        "description": VIDEO_METADATA["description"],
        "channel_title": VIDEO_METADATA["channel_title"],
        "published_at": VIDEO_METADATA["published_at"],
        "duration": VIDEO_METADATA["duration"],
        "view_count": VIDEO_METADATA["view_count"],
        "like_count": VIDEO_METADATA["like_count"],
        "tags": VIDEO_METADATA["tags"],
        "chunk_index": idx,
        "total_chunks": len(CHUNKS),
        "text": chunk,
        "importance": 1.5 if "pecorino" in chunk.lower() or "uova" in chunk.lower() else 1.0
    }
    points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

# === STEP 3: Carica in Qdrant
qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"✅ Inseriti {len(points)} chunk mock nella collezione '{COLLECTION_NAME}'")
