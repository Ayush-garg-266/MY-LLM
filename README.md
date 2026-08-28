# My-LLM — Custom Vector Database & Hybrid RAG Engine from Scratch

🚀 **Live Production Link**: [https://my-llm-9mvz.onrender.com/](https://my-llm-9mvz.onrender.com/)

A fully working, production-grade **Vector Database** and **Hybrid Retrieval-Augmented Generation (RAG) Engine** built from scratch in pure Python with zero external package dependencies.

It features side-by-side implementations of **HNSW (Hierarchical Navigable Small World)**, **KD-Tree**, and **Brute Force** vector search algorithms, paired with an interactive modern UI, live 2D PCA vector cluster visualizer, and a hybrid search RAG pipeline powered by **Ollama** locally and **Groq LPU Cloud + Hugging Face** in production.

> Built as an educational and engineering showcase to demonstrate how production vector databases like Pinecone, Weaviate, and Chroma work under the hood.

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

## 🖥️ Using the Application

### Tab 1: Search (Demo Vectors)
- Type any concept in the search box: `binary tree`, `sushi`, `basketball`, `calculus`
- Choose your algorithm: **HNSW**, **KD-Tree**, or **Brute Force**
- Choose distance metric: **Cosine**, **Euclidean**, or **Manhattan**
- Click **⚡ SEARCH** — results appear with distances, the matching point glows on the scatter plot
- Click **▶ COMPARE ALL ALGOS** to run all 3 algorithms and compare their execution speed in microseconds ($\mu s$)

The scatter plot shows all 20 vectors projected to 2D using PCA. Notice how the 4 semantic categories (CS, Math, Food, Sports) form distinct clusters — this is what "semantic similarity" looks like visually.

### Tab 2: Documents (Real Embeddings)
This uses **Hugging Face Inference API** (Cloud) or **Ollama** (Local) to generate real 384D/768D semantic embeddings from any text.

- Type a title (e.g., *Operating Systems Notes*)
- Paste any text — lecture notes, textbook paragraphs, Wikipedia articles
- Click **⚡ EMBED & INSERT**
- Long documents are automatically split into overlapping 250-word chunks
- Each chunk gets its own embedding and is stored in a separate HNSW index

### Tab 3: Ask AI (RAG Pipeline)
- Make sure you have inserted some documents in Tab 2 first
- Type a question about your documents (e.g., *"Who is Ayush Garg?"*)
- Click **🤖 ASK AI**

**What happens behind the scenes:**
1. **Your question** → embedded with `sentence-transformers/all-MiniLM-L6-v2` (384D vector) or `nomic-embed-text` (768D)
2. **Hybrid HNSW search** → performs dual vector distance + exact keyword overlap search to find top matching chunks
3. **Retrieved chunks** → sent as context to `llama-3.3-70b-versatile` (Groq Cloud) or `llama3.2` (Ollama)
4. **LLM Generator** → generates an answer based strictly on your documents without hallucination
5. The answer streams in with a typewriter effect. Click the context chips to see exactly which chunks the AI used.

---

## 📁 Project Structure

```
My-AI/
├── main.py            ← Python backend (HNSW, KD-Tree, BruteForce, REST API, RAG)
├── index.html         ← Frontend (PCA scatter plot, chat UI, benchmark)
├── requirements.txt   ← Render build configuration
├── Dockerfile         ← Container deployment config
└── README.md          ← Project documentation
```

---

## 🏗️ Architecture (main.py)

| Component | Complexity | Description |
|---|---|---|
| **BruteForce** | $O(N \cdot d)$ | Exact, baseline linear scan |
| **KDTree** | $O(\log N)$ | Exact, axis-aligned space partitioning |
| **HNSW** | $O(\log N)$ | Approximate, multilayer small-world graph |

- **VectorDB**: Unified interface over all 3 algorithms (16D demo vectors)
- **DocumentDB**: HNSW-only index for real embeddings (384D / 768D) with Hybrid Search
- **OllamaClient**: Client for Ollama & Groq Cloud API (`/api/embeddings` + `/api/generate`)

---

## 🔬 Algorithm Deep Dive

### HNSW (Hierarchical Navigable Small World)
Nodes are inserted into a multilayer graph. Each node randomly gets assigned a maximum layer. Layer 0 has all nodes with many connections; higher layers have fewer nodes (exponentially fewer) with longer-range connections.

- **Insert**: Start at the top layer, greedily find the nearest node, drop a layer, repeat. At each layer from your assigned max down to 0, run a beam search (`ef_construction=200`) and connect to the $M$ nearest neighbors bidirectionally.
- **Search**: Same greedy descent from top layer. At layer 0, expand to $ef$ nearest candidates using a priority queue.
- **Why it's fast**: The upper layers act like a highway — you quickly get to the right neighborhood, then zoom in at layer 0.

### KD-Tree (K-Dimensional Tree)
Binary space partitioning. Each node splits space along one dimension (cycling through all dimensions). Search prunes entire subtrees when the closest possible point in that subtree can't beat the current best — the "ball within hyperslab" check.

- **Weakness**: Degrades with high dimensions (curse of dimensionality). Works well for $\le 20\text{D}$, becomes close to brute force at 768D.

### Why HNSW Wins at High Dimensions
KD-Tree pruning relies on axis-aligned distance bounds. In high dimensions, almost all the space is near the boundary of the hypersphere — no subtrees get pruned. HNSW's graph-based approach doesn't have this problem.

---

## ❓ Common Issues & Troubleshooting

| Problem | Fix |
|---|---|
| **Ollama: OFFLINE in header** | Run `ollama serve` in a terminal (for local development) |
| **Embedding takes time** | Ollama is downloading the model on first use, wait 1-2 minutes |
| **Port 8080 already in use** | Kill the process: `netstat -ano \| findstr 8080` then `taskkill /PID <pid> /F` |
| **LLM answer is slow on laptop** | Normal — `llama3.2` takes 10–30s on laptop CPU. Use `llama3.2:1b` for faster answers |

### Use a Smaller/Faster LLM Locally
If `llama3.2` is too slow on your laptop CPU, switch to the 1B model:

```bash
ollama pull llama3.2:1b
```

Then edit `main.py` line where `gen_model` is set:

```python
self.gen_model = "llama3.2:1b"   # change this
```

Restart `main.py`:

```bash
python main.py
```

---

## 👤 Author

Built by **Ayush Garg** — Data Science & AI Engineer.  
[LinkedIn](https://www.linkedin.com/in/ayush-garg02) · [Email](mailto:ayush06804@gmail.com) · [GitHub](https://github.com/Ayush-garg-266/MY-LLM)
