import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { QdrantClient } from '@qdrant/js-client-rest';
import { pipeline } from '@xenova/transformers';

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3001;
const QDRANT_URL = process.env.QDRANT_URL || 'http://localhost:6333';
const QDRANT_API_KEY = process.env.QDRANT_API_KEY || undefined;
const COLLECTION = process.env.QDRANT_COLLECTION || 'youtube_rag';

// --- Init Qdrant client ---
const qdrant = new QdrantClient({
  url: QDRANT_URL,
  apiKey: QDRANT_API_KEY,
});

// --- Load local embedding model (384-dim) ---
// First call will download the model weights to a local cache (~100MB)
console.log('Loading embedding model (Xenova/all-MiniLM-L6-v2)...');
const embedder = await pipeline(
  'feature-extraction',
  'Xenova/all-MiniLM-L6-v2'
);
console.log('Model ready.');

// in server.js
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
app.use(express.static(path.join(__dirname, './public')));

// Utility: compute normalized mean-pooled embedding
async function embed(text) {
  const out = await embedder(text, { pooling: 'mean', normalize: true });
  // out is a Tensor-like object with .data Float32Array
  return Array.from(out.data); // 384 floats
}

// Build Qdrant filter from query params
function buildFilter({ tag, minImportance, docId, source }) {
  const must = [];
  if (tag) {
    must.push({ key: 'tags', match: { value: tag } });
  }
  if (minImportance) {
    const num = Number(minImportance);
    if (!Number.isNaN(num)) {
      must.push({ key: 'importance', range: { gte: num } });
    }
  }
  if (docId) {
    must.push({ key: 'doc_id', match: { value: docId } });
  }
  if (source) {
    must.push({ key: 'source', match: { value: source } }); // e.g. 'youtube'
  }
  return must.length ? { must } : undefined;
}

// Health
app.get('/api/health', async (_req, res) => {
  try {
    const info = await qdrant.getCollections();
    res.json({ ok: true, collections: info.collections?.map(c => c.name) });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e) });
  }
});

// Main search endpoint
app.post('/api/search', async (req, res) => {
  try {
    const { query, topK = 5, tag, minImportance, docId, source } = req.body || {};
    if (!query || typeof query !== 'string') {
      return res.status(400).json({ error: 'query (string) is required' });
    }

    // 1) embed query
    const vector = await embed(query);

    // 2) build filter
    const filter = buildFilter({ tag, minImportance, docId, source });

    // 3) Qdrant vector search
    const result = await qdrant.search(COLLECTION, {
      vector,
      limit: Number(topK) || 5,
      filter,
      with_payload: true,
      with_vector: false,
    });

    res.json({
      count: result.length,
      hits: result.map(h => ({
        id: String(h.id),
        score: h.score,
        payload: h.payload,
      })),
    });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e) });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 API listening on http://localhost:${PORT}`);
  console.log(`   Qdrant: ${QDRANT_URL}  Collection: ${COLLECTION}`);
});
