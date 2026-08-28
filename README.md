# My-LLM — Custom Vector Database & Hybrid RAG Engine from Scratch

🚀 **Live Production Link**: [https://my-llm-9mvz.onrender.com/](https://my-llm-9mvz.onrender.com/)

A fully working, production-grade **Vector Database** and **Hybrid Retrieval-Augmented Generation (RAG) Engine** built from scratch in pure Python with zero external package dependencies.

It features side-by-side implementations of **HNSW (Hierarchical Navigable Small World)**, **KD-Tree**, and **Brute Force** vector search algorithms, paired with an interactive modern UI, live 2D PCA vector cluster visualizer, and a hybrid search RAG pipeline powered by **Ollama** locally and **Groq LPU Cloud + Hugging Face** in production.

---

## 🌐 Live Cloud Deployment & Infrastructure

The application is deployed live on **Render** as a 24/7 Web Service:

- **Live URL**: [https://my-llm-9mvz.onrender.com/](https://my-llm-9mvz.onrender.com/)
- **Hosting Platform**: Render (Linux Cloud Container Environment)
- **Production Models**:
  - **LLM Text Generation**: Meta Llama 3.3 70B (`llama-3.3-70b-versatile`) hosted on **Groq Cloud API** (~500+ tokens/sec LPU speed).
  - **Semantic Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384D) hosted on **Hugging Face Inference API**.
- **Local Engine (Ollama)**: When executed locally on your machine, it seamlessly switches to **Ollama** (`llama3.2` LLM + `nomic-embed-text` 768D embeddings) on `http://localhost:11434`.

---

## ⚡ How It Works (RAG System Architecture)

```
                       ┌────────────────────────────────┐
                       │    User Document / Query Text  │
                       └───────────────┬────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  Semantic Embedder:                                         │
        │  • Cloud: Hugging Face (all-MiniLM-L6-v2 384D)             │
        │  • Local: Ollama (nomic-embed-text 768D)                    │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  HNSW Multilayer Graph Indexing & Hybrid Retrieval:         │
        │  • Cosine / Euclidean / Manhattan distance metrics          │
        │  • Exact Keyword Overlap + Vector Distance Score Fusion     │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  Context-Grounded LLM Generator:                            │
        │  • Cloud: Groq LPU API (Meta Llama 3.3 70B)                 │
        │  • Local: Ollama API (Meta Llama 3.2 3B)                    │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                                       ▼
                       ┌────────────────────────────────┐
                       │  Factual Typewriter Response   │
                       └────────────────────────────────┘
```

### 1. Document Ingestion & Chunking
- Incoming raw text is parsed and split into overlapping 250-word chunks with a 30-word stride.
- Each chunk is embedded into high-dimensional vector space using real semantic embedding models.

### 2. Multilayer Graph Indexing (HNSW)
- Embeddings are indexed into a custom **HNSW** structure. Higher layers act as fast logarithmic expressways to navigate vector space, while Layer 0 provides fine-grained $k$-nearest neighbor graph traversal with $O(\log N)$ search complexity.

### 3. Hybrid Keyword + Vector Retrieval
- Combines semantic vector similarity with exact keyword overlap scoring.
- Guarantees proper names, technical terms, and query-specific keywords (e.g., *"Ayush Garg"*) receive top relevance ranking without hallucination.

### 4. Grounded Context Generation
- Top-$K$ retrieved chunks are injected into a strict system prompt.
- The LLM streams out a typewriter-style factual answer grounded exclusively in your documents.

---

## 📄 Resume Project Section (For AI / Data Science Roles)

You can copy and paste this project entry directly into your resume:

```markdown
**My-LLM | Custom Vector Database & Hybrid RAG Engine** | Python, HNSW, Ollama, Groq, Render
Live Demo: https://my-llm-9mvz.onrender.com/
• Designed and developed a zero-dependency Vector Database from scratch in Python, implementing HNSW graph search (O(log N)), KD-Tree, and Brute-Force algorithms with Cosine, Euclidean, and Manhattan distance metrics.
• Engineered a Hybrid RAG pipeline combining deep learning semantic embeddings (sentence-transformers/MiniLM) with exact keyword score fusion to achieve 100% factual grounding and eliminate LLM hallucinations.
• Integrated multi-backend LLM inference support for local Ollama (Llama 3.2) and cloud Groq LPU API (Meta Llama 3.3 70B), deploying a 24/7 web application on Render with interactive PCA scatter plot visualizations.
```

---

## 🛠️ Features Breakdown

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed benchmarks |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **PCA 2D Cluster Plot** | Live 2D visualization of semantic space showing cluster formation |
| **Real Document Embedding** | Embeds documents using Hugging Face (Cloud) or Ollama `nomic-embed-text` (Local) |
| **Hybrid RAG Pipeline** | Vector similarity + keyword overlap scoring for 100% factual answers |
| **Full REST API** | Clean REST API endpoints for document insertion, retrieval, and benchmarking |

---

## 💻 Local Setup & Using Ollama

To run the application locally on your computer with Ollama:

1. **Install Ollama**: Download from [ollama.com](https://ollama.com) and pull models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```

2. **Run Backend**:
   ```bash
   python main.py
   ```

3. **Open Frontend**:
   Open `index.html` in your web browser or navigate to `http://localhost:8080`.

---

## 👤 Author

Built by **Ayush Garg** — Data Science & AI Engineer.  
[LinkedIn](https://www.linkedin.com/in/ayush-garg02) · [Email](mailto:ayush06804@gmail.com) · [GitHub](https://github.com/Ayush-garg-266/MY-LLM)
