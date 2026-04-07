# Quick Reference - RAG Implementation

## 🚀 Getting Started (5 minutes)

### Step 1: Install
```bash
pip install sentence-transformers torch faiss-cpu
```

### Step 2: Run Setup
```bash
python setup_rag.py
```

### Step 3: Test
```bash
python examples/rag_integration_example.py
```

---

## 📖 Key Concepts (Read First)

**What is RAG?**
- Converts text →  numerical vectors (embeddings)
- Finds similar vectors = finds semantically similar content
- Enables: "phone" matches "cell phone" ✓

**Why do you need it?**
- Current: Only exact keyword matches
- RAG: Understands meaning, handles synonyms

---

## 🔧 Integration Points

### 1. Generate & Store Embeddings
```python
from memory.embeddings import create_embedder

embedder = create_embedder()

# Generate embedding for scene description
embedding = embedder.embed_text("Phone on wooden nightstand")

# Store it
memory.store_embedding(frame_id, embedding)
```

### 2. Search Semantically
```python
from query.engine import create_query_engine

query_engine = create_query_engine(memory, use_semantic=True)

# Semantic search
results = query_engine.semantic_search("phone on table", k=5)

# Smart search (keyword + semantic hybrid)
response = query_engine.smart_search("Where is my phone?")
```

### 3. Use in Streamlit
```python
# In app_streamlit.py
query_engine = create_query_engine(memory, use_semantic=True)

if user_query:
    response = query_engine.smart_search(user_query)
    st.write(response)
```

---

## 📋 Configuration

Add to `config.py`:

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"        # Fast (22MB)
EMBEDDING_DIM = 384
SEMANTIC_SIMILARITY_THRESHOLD = 0.65        # 0-1, higher=stricter
SEMANTIC_SEARCH_TOP_K = 5                   # Return top 5
ENABLE_FAISS = True                         # Speed up search
```

---

## 🎯 What Works Now

| Scenario | Works? |
|----------|--------|
| "Where is my phone?" | ✅ Yes |
| "Have you seen my cell phone?" (NEW!) | ✅ Yes |
| "Find my mobile" (NEW!) | ✅ Yes |
| "Objects on the nightstand" (NEW!) | ✅ Yes |

---

## ⚠️ Important Notes

1. **Backward Compatibility**: Existing code still works (RAG is opt-in)
2. **Database**: No schema changes (uses existing `embedding_vector` column)
3. **Performance**: Semantic search ~100-200ms (slower than keywords, but smart)
4. **Raspberry Pi**: Use `all-MiniLM-L6-v2` model (lightweight)

---

## 🐛 Troubleshooting

**Error: "sentence-transformers not installed"**
```bash
pip install sentence-transformers
```

**Slow on Raspberry Pi?**
- Use FAISS: Set `ENABLE_FAISS = True`
- Use smaller model: `all-MiniLM-L6-v2`
- Reduce top-k: `SEMANTIC_SEARCH_TOP_K = 2`

**Embeddings not working?**
- Check: `embedder.model` is not None
- Check: Scene description is being generated
- Check: `use_semantic=True` in query engine

---

## 📚 Full Documentation

- **[RAG_IMPLEMENTATION_GUIDE.md](RAG_IMPLEMENTATION_GUIDE.md)** - Complete guide
- **[RAG_STATUS.md](RAG_STATUS.md)** - Before/after comparison
- **[examples/rag_integration_example.py](examples/rag_integration_example.py)** - Code examples

---

## ✅ Verification

After setup, you should see:
```
✓ Embedder initialized with dimension 384
✓ Vector store ready with FAISS index
✓ Semantic search returning results
✓ Smart search combining keyword + semantic
```

Run `python setup_rag.py` to verify all components.

---

**Ready?** Start with: `python setup_rag.py`
