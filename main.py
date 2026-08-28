import http.server
import json
import math
import random
import re
import time
import urllib.parse
import urllib.request
from threading import Lock
import os

DIMS = 16  # Demo vectors dimension

# =====================================================================
#  DATA TYPES
# =====================================================================

class VectorItem:
    def __init__(self, item_id, metadata, category, emb):
        self.id = item_id
        self.metadata = metadata
        self.category = category
        self.emb = emb

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a)
    nb = sum(y * y for y in b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))

def manhattan(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))

def get_dist_fn(metric):
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean

# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items = []

    def insert(self, item):
        self.items.append(item)

    def knn(self, q, k, dist_fn):
        res = [(dist_fn(q, v.emb), v.id) for v in self.items]
        res.sort(key=lambda x: x[0])
        return res[:k]

    def remove(self, item_id):
        self.items = [v for v in self.items if v.id != item_id]

# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    def __init__(self, item):
        self.item = item
        self.left = None
        self.right = None

class KDTree:
    def __init__(self, dims):
        self.dims = dims
        self.root = None

    def _ins(self, node, item, depth):
        if not node:
            return KDNode(item)
        axis = depth % self.dims
        if item.emb[axis] < node.item.emb[axis]:
            node.left = self._ins(node.left, item, depth + 1)
        else:
            node.right = self._ins(node.right, item, depth + 1)
        return node

    def insert(self, item):
        self.root = self._ins(self.root, item, 0)

    def _knn(self, node, q, k, depth, dist_fn, heap):
        if not node:
            return
        dn = dist_fn(q, node.item.emb)
        # Store as (distance, id) in max-heap simulation
        heap.append((dn, node.item.id))
        heap.sort(key=lambda x: x[0])
        if len(heap) > k:
            heap.pop()

        axis = depth % self.dims
        diff = q[axis] - node.item.emb[axis]
        closer = node.left if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left

        self._knn(closer, q, k, depth + 1, dist_fn, heap)
        if len(heap) < k or abs(diff) < heap[-1][0]:
            self._knn(farther, q, k, depth + 1, dist_fn, heap)

    def knn(self, q, k, dist_fn):
        heap = []
        self._knn(self.root, q, k, 0, dist_fn, heap)
        heap.sort(key=lambda x: x[0])
        return heap

    def rebuild(self, items):
        self.root = None
        for v in items:
            self.insert(v)

# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class HNSWNode:
    def __init__(self, item, max_lyr):
        self.item = item
        self.max_lyr = max_lyr
        self.nbrs = [[] for _ in range(max_lyr + 1)]

class HNSW:
    def __init__(self, m=16, ef_build=200):
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(float(m))
        self.top_layer = -1
        self.entry_pt = -1
        self.G = {}  # id -> HNSWNode
        self.rng = random.Random(42)

    def rand_level(self):
        u = self.rng.random()
        while u == 0.0:
            u = self.rng.random()
        return int(math.floor(-math.log(u) * self.mL))

    def search_layer(self, q, ep, ef, lyr, dist_fn):
        vis = {ep: True}
        cands = [(dist_fn(q, self.G[ep].item.emb), ep)]  # min list sorting for candidates
        found = [(cands[0][0], ep)]  # sorted by dist ascending

        while cands:
            cands.sort(key=lambda x: x[0])
            cd, cid = cands.pop(0)

            found.sort(key=lambda x: x[0])
            if len(found) >= ef and cd > found[-1][0]:
                break

            node = self.G[cid]
            if lyr >= len(node.nbrs):
                continue

            for nid in node.nbrs[lyr]:
                if nid in vis or nid not in self.G:
                    continue
                vis[nid] = True
                nd = dist_fn(q, self.G[nid].item.emb)
                found.sort(key=lambda x: x[0])
                if len(found) < ef or nd < found[-1][0]:
                    cands.append((nd, nid))
                    found.append((nd, nid))
                    found.sort(key=lambda x: x[0])
                    if len(found) > ef:
                        found.pop()

        found.sort(key=lambda x: x[0])
        return found

    def select_nbrs(self, cands, max_m):
        return [cid for _, cid in cands[:max_m]]

    def insert(self, item, dist_fn):
        item_id = item.id
        lvl = self.rand_level()
        self.G[item_id] = HNSWNode(item, lvl)

        if self.entry_pt == -1:
            self.entry_pt = item_id
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(self.G[ep].nbrs):
                W = self.search_layer(item.emb, ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W = self.search_layer(item.emb, ep, self.ef_build, lc, dist_fn)
            max_m = self.M0 if lc == 0 else self.M
            sel = self.select_nbrs(W, max_m)
            self.G[item_id].nbrs[lc] = list(sel)

            for nid in sel:
                if nid not in self.G:
                    continue
                target_node = self.G[nid]
                if len(target_node.nbrs) <= lc:
                    target_node.nbrs.extend([[] for _ in range(lc + 1 - len(target_node.nbrs))])
                conn = target_node.nbrs[lc]
                conn.append(item_id)
                if len(conn) > max_m:
                    ds = [(dist_fn(target_node.item.emb, self.G[c].item.emb), c) for c in conn if c in self.G]
                    ds.sort(key=lambda x: x[0])
                    target_node.nbrs[lc] = [c for _, c in ds[:max_m]]
            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = item_id

    def knn(self, q, k, ef, dist_fn):
        if self.entry_pt == -1 or not self.G:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                W = self.search_layer(q, ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]
        W = self.search_layer(q, ep, max(ef, k), 0, dist_fn)
        return W[:k]

    def remove(self, item_id):
        if item_id not in self.G:
            return
        for nid, node in self.G.items():
            for layer in node.nbrs:
                if item_id in layer:
                    layer.remove(item_id)
        if self.entry_pt == item_id:
            self.entry_pt = -1
            for nid in self.G:
                if nid != item_id:
                    self.entry_pt = nid
                    break
        del self.G[item_id]

    def get_info(self):
        top_layer = self.top_layer
        node_count = len(self.G)
        max_l = max(top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l

        nodes = []
        edges = []

        for item_id, node in self.G.items():
            nodes.append({
                "id": item_id,
                "metadata": node.item.metadata,
                "category": node.item.category,
                "maxLyr": node.max_lyr
            })
            for lc in range(min(node.max_lyr + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(node.nbrs):
                    for nid in node.nbrs[lc]:
                        if item_id < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": item_id, "dst": nid, "lyr": lc})

        return {
            "topLayer": top_layer,
            "nodeCount": node_count,
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges
        }

    def size(self):
        return len(self.G)

# =====================================================================
#  VECTOR DATABASE (demo 16D index)
# =====================================================================

class VectorDB:
    def __init__(self, dims):
        self.dims = dims
        self.store = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.lock = Lock()
        self.next_id = 1

    def insert(self, meta, cat, emb, dist_fn=None):
        if dist_fn is None:
            dist_fn = get_dist_fn("cosine")
        with self.lock:
            item = VectorItem(self.next_id, meta, cat, emb)
            self.next_id += 1
            self.store[item.id] = item
            self.bf.insert(item)
            self.kdt.insert(item)
            self.hnsw.insert(item, dist_fn)
            return item.id

    def remove(self, item_id):
        with self.lock:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.bf.remove(item_id)
            self.hnsw.remove(item_id)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q, k, metric="cosine", algo="hnsw"):
        with self.lock:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter()

            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)

            latency_us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for dist_val, item_id in raw:
                if item_id in self.store:
                    item = self.store[item_id]
                    hits.append({
                        "id": item.id,
                        "metadata": item.metadata,
                        "category": item.category,
                        "embedding": item.emb,
                        "distance": dist_val
                    })
            return {
                "results": hits,
                "latencyUs": latency_us,
                "algo": algo,
                "metric": metric
            }

    def benchmark(self, q, k, metric="cosine"):
        with self.lock:
            dfn = get_dist_fn(metric)

            def measure(fn):
                t0 = time.perf_counter()
                fn()
                return int((time.perf_counter() - t0) * 1_000_000)

            bf_us = measure(lambda: self.bf.knn(q, k, dfn))
            kd_us = measure(lambda: self.kdt.knn(q, k, dfn))
            hnsw_us = measure(lambda: self.hnsw.knn(q, k, 50, dfn))

            return {
                "bruteforceUs": bf_us,
                "kdtreeUs": kd_us,
                "hnswUs": hnsw_us,
                "itemCount": len(self.store)
            }

    def all(self):
        with self.lock:
            return list(self.store.values())

    def hnsw_info(self):
        with self.lock:
            return self.hnsw.get_info()

    def size(self):
        with self.lock:
            return len(self.store)

# =====================================================================
#  DOCUMENT DATABASE — HNSW over real Ollama embeddings
# =====================================================================

class DocItem:
    def __init__(self, doc_id, title, text, emb):
        self.id = doc_id
        self.title = title
        self.text = text
        self.emb = emb

class DocumentDB:
    def __init__(self):
        self.store = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()
        self.lock = Lock()
        self.next_id = 1
        self.dims = 0

    def insert(self, title, text, emb):
        with self.lock:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(self.next_id, title, text, emb)
            self.next_id += 1
            self.store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q, k, max_dist=0.7):
        with self.lock:
            if not self.store:
                return []
            raw = self.bf.knn(q, k, cosine) if len(self.store) < 10 else self.hnsw.knn(q, k, 50, cosine)
            out = []
            for d_val, item_id in raw:
                if item_id in self.store and d_val <= max_dist:
                    out.append((d_val, self.store[item_id]))
            return out

    def remove(self, item_id):
        with self.lock:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.hnsw.remove(item_id)
            self.bf.remove(item_id)
            return True

    def all(self):
        with self.lock:
            return list(self.store.values())

    def size(self):
        with self.lock:
            return len(self.store)

    def get_dims(self):
        return self.dims

# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text, chunk_words=250, overlap_words=30):
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step = chunk_words - overlap_words
    for i in range(0, len(words), step):
        end = min(i + chunk_words, len(words))
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        if end == len(words):
            break
    return chunks

# =====================================================================
#  OLLAMA CLIENT — wraps local Ollama REST API
# =====================================================================

class OllamaClient:
    def __init__(self, host="127.0.0.1", port=11434):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "").strip().strip('"').strip("'")
        if self.groq_api_key:
            self.gen_model = "Llama 3 (Groq Cloud)"

    def is_available(self):
        if self.groq_api_key:
            return True
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _embed_local_hash(self, text, dim=768):
        words = text.lower().split()
        vec = [0.0] * dim
        for w in words:
            h = hash(w)
            for i in range(dim):
                vec[i] += math.sin((h + i * 17) % 10000)
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            vec = [x / norm for x in vec]
        return vec

    def embed(self, text):
        if not self.groq_api_key:
            url = f"{self.base_url}/api/embeddings"
            data = json.dumps({"model": self.embed_model, "prompt": text}).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    if resp.status == 200:
                        body = json.loads(resp.read().decode('utf-8'))
                        emb = body.get("embedding", [])
                        if emb:
                            return emb
            except Exception:
                pass
        return self._embed_local_hash(text, 768)

    def generate(self, prompt):
        if self.groq_api_key:
            url = "https://api.groq.com/openai/v1/chat/completions"
            models = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "llama-3.2-3b-preview", "mixtral-8x7b-32768", "gemma2-9b-it", "llama3-8b-8192"]
            last_err = "ERROR Groq API: Unable to reach models"
            for m in models:
                data = json.dumps({
                    "model": m,
                    "messages": [{"role": "user", "content": prompt}]
                }).encode('utf-8')
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.groq_api_key}",
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    }
                )
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        if resp.status == 200:
                            body = json.loads(resp.read().decode('utf-8'))
                            choices = body.get("choices", [])
                            if choices:
                                return choices[0]["message"]["content"]
                except urllib.error.HTTPError as e:
                    err_body = e.read().decode('utf-8') if e.fp else ""
                    try:
                        err_json = json.loads(err_body)
                        msg = err_json.get("error", {}).get("message", err_body)
                        last_err = f"ERROR Groq API ({e.code}): {msg}"
                    except Exception:
                        last_err = f"ERROR Groq API ({e.code}): {e.reason}"
                    if e.code in (400, 404):
                        continue
                    return last_err
                except Exception as e:
                    last_err = f"ERROR Groq API: {str(e)}"
                    return last_err
            return last_err

        url = f"{self.base_url}/api/generate"
        data = json.dumps({"model": self.gen_model, "prompt": prompt, "stream": False}).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                if resp.status != 200:
                    return "ERROR: Ollama unavailable. Run: ollama serve"
                body = json.loads(resp.read().decode('utf-8'))
                return body.get("response", "")
        except Exception:
            return "ERROR: Ollama unavailable. Run: ollama serve"

# =====================================================================
#  DEMO DATA (16D categorical vectors)
# =====================================================================

def load_demo(db):
    dist_fn = get_dist_fn("cosine")
    demo_data = [
        ("Linked List: nodes connected by pointers", "cs",
         [0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs",
         [0.88, 0.82, 0.78, 0.74, 0.15, 0.10, 0.08, 0.12, 0.06, 0.07, 0.08, 0.05, 0.09, 0.06, 0.07, 0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs",
         [0.82, 0.76, 0.88, 0.80, 0.20, 0.18, 0.12, 0.09, 0.07, 0.06, 0.08, 0.07, 0.08, 0.09, 0.06, 0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs",
         [0.85, 0.80, 0.75, 0.82, 0.18, 0.14, 0.10, 0.08, 0.06, 0.09, 0.07, 0.06, 0.10, 0.08, 0.09, 0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs",
         [0.87, 0.78, 0.70, 0.76, 0.13, 0.11, 0.09, 0.14, 0.08, 0.07, 0.06, 0.08, 0.07, 0.10, 0.08, 0.09]),
        ("Calculus: derivatives integrals and limits", "math",
         [0.12, 0.15, 0.18, 0.10, 0.91, 0.86, 0.78, 0.72, 0.08, 0.06, 0.07, 0.09, 0.07, 0.08, 0.06, 0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
         [0.20, 0.18, 0.15, 0.12, 0.88, 0.90, 0.82, 0.76, 0.09, 0.07, 0.08, 0.06, 0.10, 0.07, 0.08, 0.09]),
        ("Probability: distributions random variables Bayes theorem", "math",
         [0.15, 0.12, 0.20, 0.18, 0.84, 0.80, 0.88, 0.82, 0.07, 0.08, 0.06, 0.10, 0.09, 0.06, 0.09, 0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math",
         [0.22, 0.16, 0.14, 0.20, 0.80, 0.85, 0.76, 0.90, 0.08, 0.09, 0.07, 0.06, 0.08, 0.10, 0.07, 0.06]),
        ("Combinatorics: permutations combinations generating functions", "math",
         [0.18, 0.20, 0.16, 0.14, 0.86, 0.78, 0.84, 0.80, 0.06, 0.07, 0.09, 0.08, 0.06, 0.09, 0.10, 0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
         [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.90, 0.86, 0.78, 0.72, 0.08, 0.06, 0.09, 0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food",
         [0.06, 0.08, 0.07, 0.09, 0.09, 0.06, 0.08, 0.07, 0.86, 0.90, 0.82, 0.76, 0.07, 0.09, 0.06, 0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
         [0.09, 0.07, 0.06, 0.08, 0.08, 0.09, 0.07, 0.06, 0.82, 0.78, 0.90, 0.84, 0.09, 0.07, 0.08, 0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
         [0.07, 0.09, 0.08, 0.06, 0.06, 0.07, 0.09, 0.08, 0.78, 0.82, 0.86, 0.90, 0.06, 0.08, 0.07, 0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food",
         [0.06, 0.07, 0.10, 0.09, 0.10, 0.06, 0.07, 0.10, 0.85, 0.80, 0.76, 0.82, 0.09, 0.07, 0.10, 0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
         [0.09, 0.07, 0.08, 0.10, 0.08, 0.09, 0.07, 0.06, 0.08, 0.07, 0.09, 0.06, 0.91, 0.85, 0.78, 0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports",
         [0.07, 0.09, 0.06, 0.08, 0.09, 0.07, 0.10, 0.08, 0.07, 0.09, 0.08, 0.07, 0.87, 0.89, 0.82, 0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
         [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.09, 0.06, 0.07, 0.08, 0.83, 0.80, 0.88, 0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports",
         [0.25, 0.20, 0.22, 0.18, 0.22, 0.18, 0.20, 0.15, 0.06, 0.08, 0.07, 0.09, 0.80, 0.84, 0.78, 0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
         [0.06, 0.08, 0.07, 0.09, 0.08, 0.06, 0.09, 0.07, 0.10, 0.08, 0.06, 0.07, 0.85, 0.82, 0.86, 0.80])
    ]
    for meta, cat, emb in demo_data:
        db.insert(meta, cat, emb, dist_fn)

# =====================================================================
#  GLOBAL SERVER INSTANCES
# =====================================================================

db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()
load_demo(db)

# =====================================================================
#  HTTP SERVER HANDLER
# =====================================================================

class VectorDBRequestHandler(http.server.BaseHTTPRequestHandler):
    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def parse_vec(self, s):
        if not s:
            return []
        try:
            return [float(x) for x in s.split(',') if x.strip()]
        except Exception:
            return []

    def get_body_json(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length).decode('utf-8')
        try:
            return json.loads(body)
        except Exception:
            return {}

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        path = parsed.path

        if path == "/":
            try:
                with open("index.html", "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(content)
            except Exception:
                self.send_response(404)
                self.end_headers()
            return

        if path == "/search":
            v_param = params.get("v", [""])[0]
            q = self.parse_vec(v_param)
            if len(q) != DIMS:
                self.send_json({"error": f"need {DIMS}D vector"}, status=400)
                return
            k = int(params.get("k", [5])[0])
            metric = params.get("metric", ["cosine"])[0]
            algo = params.get("algo", ["hnsw"])[0]

            out = db.search(q, k, metric, algo)
            self.send_json(out)
            return

        if path == "/items":
            items = db.all()
            out = [{
                "id": v.id,
                "metadata": v.metadata,
                "category": v.category,
                "embedding": v.emb
            } for v in items]
            self.send_json(out)
            return

        if path == "/benchmark":
            v_param = params.get("v", [""])[0]
            q = self.parse_vec(v_param)
            if len(q) != DIMS:
                self.send_json({"error": f"need {DIMS}D vector"}, status=400)
                return
            k = int(params.get("k", [5])[0])
            metric = params.get("metric", ["cosine"])[0]
            out = db.benchmark(q, k, metric)
            self.send_json(out)
            return

        if path == "/hnsw-info":
            gi = db.hnsw_info()
            self.send_json(gi)
            return

        if path == "/doc/list":
            docs = doc_db.all()
            out = []
            for d in docs:
                preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
                words = len(d.text.split())
                out.append({
                    "id": d.id,
                    "title": d.title,
                    "preview": preview,
                    "words": words
                })
            self.send_json(out)
            return

        if path == "/status":
            up = ollama.is_available()
            self.send_json({
                "ollamaAvailable": up,
                "embedModel": ollama.embed_model,
                "genModel": ollama.gen_model,
                "docCount": doc_db.size(),
                "docDims": doc_db.get_dims(),
                "demoDims": DIMS,
                "demoCount": db.size()
            })
            return

        if path == "/stats":
            self.send_json({
                "count": db.size(),
                "dims": DIMS,
                "algorithms": ["bruteforce", "kdtree", "hnsw"],
                "metrics": ["euclidean", "cosine", "manhattan"]
            })
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        body = self.get_body_json()

        if path == "/insert":
            meta = body.get("metadata", "")
            cat = body.get("category", "")
            emb = body.get("embedding", [])
            if not meta or not emb or len(emb) != DIMS:
                self.send_json({"error": "invalid body"}, status=400)
                return
            item_id = db.insert(meta, cat, emb, get_dist_fn("cosine"))
            self.send_json({"id": item_id})
            return

        if path == "/doc/insert":
            title = body.get("title", "")
            text = body.get("text", "")
            if not title or not text:
                self.send_json({"error": "need title and text"}, status=400)
                return

            chunks = chunk_text(text, 250, 30)
            ids = []
            for i, chunk in enumerate(chunks):
                emb = ollama.embed(chunk)
                if not emb:
                    self.send_json({
                        "error": "Ollama unavailable. Install from https://ollama.com then run: ollama pull nomic-embed-text && ollama pull llama3.2"
                    }, status=400)
                    return
                chunk_title = f"{title} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else title
                ids.append(doc_db.insert(chunk_title, chunk, emb))

            self.send_json({
                "ids": ids,
                "chunks": len(chunks),
                "dims": doc_db.get_dims()
            })
            return

        if path == "/doc/search":
            question = body.get("question", "")
            k = int(body.get("k", 3))
            if not question:
                self.send_json({"error": "need question"}, status=400)
                return

            q_emb = ollama.embed(question)
            if not q_emb:
                self.send_json({"error": "Ollama unavailable"}, status=400)
                return

            hits = doc_db.search(q_emb, k)
            contexts = [{
                "id": doc.id,
                "title": doc.title,
                "distance": float(f"{d:.4f}")
            } for d, doc in hits]
            self.send_json({"contexts": contexts})
            return

        if path == "/doc/ask":
            question = body.get("question", "")
            k = int(body.get("k", 3))
            if not question:
                self.send_json({"error": "need question"}, status=400)
                return

            q_emb = ollama.embed(question)
            if not q_emb:
                self.send_json({"error": "Ollama unavailable"}, status=400)
                return

            hits = doc_db.search(q_emb, k)
            ctx_str = ""
            for i, (d, doc) in enumerate(hits):
                ctx_str += f"[{i+1}] {doc.title}:\n{doc.text}\n\n"

            prompt = (
                "You are a helpful assistant. Answer the user's question directly. "
                "Use the provided context if it contains relevant information. "
                "If it doesn't, just use your own general knowledge. "
                "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
                "Just answer the question naturally.\n\n"
                f"Context:\n{ctx_str}"
                f"Question: {question}\n\n"
                "Answer:"
            )

            answer = ollama.generate(prompt)

            contexts = [{
                "id": doc.id,
                "title": doc.title,
                "text": doc.text,
                "distance": float(f"{d:.4f}")
            } for d, doc in hits]

            self.send_json({
                "answer": answer,
                "model": ollama.gen_model,
                "contexts": contexts,
                "docCount": doc_db.size()
            })
            return

        self.send_response(404)
        self.end_headers()

    def do_DELETE(self):
        path = urllib.parse.urlparse(self.path).path
        m_del = re.match(r"^/delete/(\d+)$", path)
        if m_del:
            item_id = int(m_del.group(1))
            ok = db.remove(item_id)
            self.send_json({"ok": ok})
            return

        m_doc_del = re.match(r"^/doc/delete/(\d+)$", path)
        if m_doc_del:
            item_id = int(m_doc_del.group(1))
            ok = doc_db.remove(item_id)
            self.send_json({"ok": ok})
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        # Silence standard HTTP access logging to keep terminal output clean
        pass

# =====================================================================
#  MAIN ENTRYPOINT
# =====================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    ollama_up = ollama.is_available()
    print("=== My-LLM Engine (Python) ===")
    print(f"http://localhost:{port}")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), VectorDBRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down My-LLM server.")
