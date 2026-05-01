"""
plato_mythos.py — Pure Python tile embedding model (no torch).

Models the plato-mythos architecture concepts:
  - rooms_as_experts: Tag-based routing → domain expert groups
  - tiles_as_kv:     Latent compression via random projection (256-dim)
  - deadband_act:    Priority-aware halting (P0=critical, P1=standard, P2=low)
  - curriculum_loop: Multi-step iterative refinement

Pure Python + numpy. No torch, no external ML deps.
The tile store is at ~/jetsonclaw1-vessel/memory/tiles/ (10 .md files).
"""

import math
import os
import re
import glob
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

LATENT_DIM = 256          # Compressed latent vector dimension
VOCAB_SIZE = 4096         # Max vocab for TF-IDF
DFLT_K = 5                # Default top-K results
DFLT_LOOPS = 3            # Default curriculum loop iterations

# Priority tiers (DeadbandACT)
PRIORITY_TIERS = {
    "P0": 0.99,  # Critical — high relevance threshold
    "P1": 0.80,  # Standard — normal relevance
    "P2": 0.50,  # Low     — wide net
}

# Domain → Room mapping (expert groups)
# Matches the 14 room types from the MUD.
# Each domain routes to one or more expert rooms.
DOMAIN_ROOMS = {
    "research":     ["science lab", "library"],
    "engineering":  ["engine room", "workshop"],
    "fleet":        ["harbor", "tactical"],
    "health":       ["sickbay"],
    "knowledge":    ["library"],
    "training":     ["dojo", "workshop"],
    "status":       ["bridge", "quarterdeck", "tactical"],
    "sandbox":      ["holodeck"],
    "storage":      ["cargo bay"],
    "external":     ["airlock"],
    "ai":           ["science lab", "holodeck"],
    "hardware":     ["engine room"],
    "cuda":         ["engine room"],
    "gpu":          ["engine room"],
    "jetson":       ["engine room", "workshop"],
    "product":      ["quarterdeck", "bridge"],
    "architecture": ["bridge", "library"],
    "mud":          ["holodeck", "library"],
    "network":      ["harbor", "tactical"],
    "cpp":          ["workshop", "engine room"],
    "format":       ["library"],
    "gguf":         ["workshop"],
    "inference":    ["science lab", "engine room"],
    "machine_learning": ["science lab"],
    "git":          ["tactical", "workshop"],
    "agent":        ["bridge", "tactical"],
    "bottle":       ["harbor"],
    "oracle1":      ["harbor", "bridge"],
    "plato":        ["bridge", "library"],
    "cocapn":       ["bridge", "tactical"],
    "edge":         ["bridge", "holodeck"],
    "cloudflare":   ["airlock", "bridge"],
    "api":          ["airlock", "tactical"],
    "sdk":          ["workshop"],
    "product":      ["quarterdeck"],
}

# Reverse: all unique domain names
ALL_DOMAINS = sorted(set(DOMAIN_ROOMS.keys()))


# ═══════════════════════════════════════════════════════════════
# Tile Record
# ═══════════════════════════════════════════════════════════════

class TileRecord:
    """A parsed knowledge tile with structured metadata."""

    __slots__ = ("id", "path", "title", "tags", "created", "updated",
                 "body", "body_text", "domain", "domains")

    def __init__(self, filepath: str):
        self.path = filepath
        self.id = ""
        self.title = ""
        self.tags = []
        self.created = ""
        self.updated = ""
        self.body = ""
        self.body_text = ""   # Lowered, stripped for embedding
        self.domain = "general"   # Primary domain
        self.domains = set()      # All matched domains

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        self._parse_frontmatter(raw)
        self._infer_domains()

    def _parse_frontmatter(self, raw: str):
        """Extract YAML frontmatter fields and body."""
        lines = raw.split("\n")
        in_fm = False
        fm_lines = []
        body_lines = []
        found_opener = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not found_opener and stripped == "---":
                found_opener = True
                in_fm = True
                continue
            if in_fm:
                if stripped == "---":
                    in_fm = False
                    continue
                fm_lines.append(line)
            else:
                body_lines.append(line)

        # Parse frontmatter
        for line in fm_lines:
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip().lower()
                val = val.strip()

                if key == "id":
                    self.id = val
                elif key == "tags":
                    # Handle both [tag1, tag2] and ['tag1', 'tag2'] formats
                    val = val.strip("[]").strip("'\" ")
                    self.tags = [t.strip().strip("'\"").strip()
                                for t in val.split(",") if t.strip()]
                elif key == "created":
                    self.created = val
                elif key == "updated":
                    self.updated = val
                elif key == "title":
                    self.title = val

        # Fallback title from first heading
        if not self.title:
            for line in body_lines:
                if line.startswith("# "):
                    self.title = line.strip("# ").strip()
                    break

        # Fallback ID from filename
        if not self.id:
            self.id = os.path.splitext(os.path.basename(self.path))[0]

        # Body text — remove headings, strip whitespace
        text_lines = []
        for line in body_lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue   # skip headings
            if stripped:
                text_lines.append(stripped)
        self.body = "\n".join(body_lines).strip()
        self.body_text = " ".join(text_lines).lower()

    def _infer_domains(self):
        """Map tags to expert domains via DOMAIN_ROOMS."""
        matched = set()
        for tag in self.tags:
            tag_key = tag.lower().replace(" ", "_")
            if tag_key in DOMAIN_ROOMS:
                matched.add(tag_key)
            # Also try the raw tag
            if tag.lower() in ALL_DOMAINS:
                matched.add(tag.lower())

        if not matched:
            # Content-based fallback: scan body for domain keywords
            for domain, rooms in DOMAIN_ROOMS.items():
                if domain in self.body_text:
                    matched.add(domain)

        self.domains = matched
        if matched:
            self.domain = sorted(matched)[0]
        else:
            self.domain = "general"

    def __repr__(self):
        return f"<TileRecord '{self.id}' [{self.domain}] {len(self.tags)} tags>"


# ═══════════════════════════════════════════════════════════════
# TF-IDF Embedding
# ═══════════════════════════════════════════════════════════════

class TfidfEmbedder:
    """
    Simple TF-IDF vectorizer.
    No external deps — pure Python + numpy.
    """

    def __init__(self, max_features: int = VOCAB_SIZE):
        self.max_features = max_features
        self.vocab = {}
        self.idf = np.array([], dtype=np.float32)
        self._fitted = False
        self._stopwords = {
            "the", "a", "an", "is", "it", "in", "to", "for", "of", "and",
            "on", "at", "by", "with", "from", "as", "be", "was", "are",
            "were", "been", "being", "has", "have", "had", "do", "does",
            "did", "but", "or", "if", "so", "not", "no", "all", "each",
            "can", "will", "this", "that", "these", "those", "its", "i",
            "my", "we", "our", "you", "your", "he", "she", "they", "them",
        }

    def _tokenize(self, text: str) -> list:
        """Tokenize and filter stopwords."""
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        return [t for t in tokens if len(t) > 2 and t not in self._stopwords]

    def fit(self, documents: list):
        """Build vocab and IDF from a list of text documents."""
        # Count document frequency
        df = Counter()
        all_tokens = []
        for doc in documents:
            tokens = self._tokenize(doc)
            for t in set(tokens):  # per-document presence
                df[t] += 1
            all_tokens.append(tokens)

        # Take most frequent max_features
        vocab = {t for t, _ in df.most_common(self.max_features)}
        self.vocab = {t: i for i, t in enumerate(sorted(vocab))}
        self.vocab_size = len(self.vocab)

        n_docs = len(documents)
        self.idf = np.zeros(self.vocab_size, dtype=np.float32)
        for term, idx in self.vocab.items():
            doc_freq = df.get(term, 1)
            self.idf[idx] = math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

        self._fitted = True
        return self

    def transform(self, text: str) -> np.ndarray:
        """Transform text to sparse-ish TF-IDF vector."""
        if not self._fitted:
            return np.zeros(LATENT_DIM, dtype=np.float32)

        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.vocab_size, dtype=np.float32)

        # Term frequency in this document
        tf = Counter(tokens)
        max_tf = max(tf.values())

        vec = np.zeros(self.vocab_size, dtype=np.float32)
        for term, count in tf.items():
            if term in self.vocab:
                idx = self.vocab[term]
                vec[idx] = (0.5 + 0.5 * count / max_tf) * self.idf[idx]

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec


# ═══════════════════════════════════════════════════════════════
# Random Projection (Johnson-Lindenstrauss)
# ═══════════════════════════════════════════════════════════════

class RandomProjection:
    """
    Johnson-Lindenstrauss random projection.
    Compresses TF-IDF vectors into latent_dim space.
    """

    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM, seed: int = 42):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        rng = np.random.RandomState(seed)

        # Achlioptas: entries in {-sqrt(3), 0, +sqrt(3)} with prob {1/6, 2/3, 1/6}
        self.matrix = rng.choice(
            [-math.sqrt(3), 0, math.sqrt(3)],
            size=(input_dim, latent_dim),
            p=[1.0/6, 2.0/3, 1.0/6]
        ).astype(np.float32)

    def project(self, vec: np.ndarray) -> np.ndarray:
        """Project high-dim vector into latent space. L2-normalized output."""
        latent = vec @ self.matrix  # (latent_dim,)
        norm = np.linalg.norm(latent)
        if norm > 0:
            latent /= norm
        return latent.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# PlatoTileModel — Main Embedding Index
# ═══════════════════════════════════════════════════════════════

class PlatoTileModel:
    """
    Pure-Python tile embedding model implementing plato-mythos concepts.

    Architecture (mythos forward pass):
        embed → domain router → iterative search loop → deadband halt → results

    Features:
        - rooms_as_experts: Domain routing via tag → room mappings
        - tiles_as_kv:      Latent compression (TF-IDF → random projection)
        - deadband_act:     Priority tiers (P0=0.99, P1=0.80, P2=0.50)
        - curriculum_loop:  Multi-step search refinement
    """

    def __init__(self, tiles_dir: str = None):
        if tiles_dir is None:
            # Check openclaw workspace first (full 10-tile set), then legacy path
            ocw = os.path.expanduser("~/.openclaw/workspace/memory/tiles")
            legacy = os.path.expanduser("~/jetsonclaw1-vessel/memory/tiles")
            if os.path.isdir(ocw):
                tiles_dir = ocw
            else:
                tiles_dir = legacy

        self.tiles_dir = os.path.expanduser(tiles_dir)
        self.tiles = []               # List[TileRecord]
        self.index = {}               # tile_id -> int index
        self.latent_vectors = None    # np.ndarray (n_tiles, LATENT_DIM)
        self.domain_vectors = {}      # domain_name -> list of latent vectors
        self.tfidf = TfidfEmbedder()
        self.projection = None
        self._fitted = False
        self._graph = None            # Cached from _graph.json if present
        self.stats = {
            "total_tiles": 0,
            "total_domains": 0,
            "domains": defaultdict(int),
            "last_rebuild": None,
        }

    # ── Public API ──

    def load_and_build(self, rebuild_vocab: bool = True):
        """
        Load all tiles from disk and build the embedding index.
        Call this once at startup or on @mythos rebuild.
        """
        tile_paths = sorted(glob.glob(os.path.join(self.tiles_dir, "*.md")))

        # Skip _graph.json and other non-tile files
        tile_paths = [p for p in tile_paths
                     if not os.path.basename(p).startswith("_")]

        self.tiles = []
        for path in tile_paths:
            try:
                tile = TileRecord(path)
                self.tiles.append(tile)
            except Exception as e:
                import traceback
                traceback.print_exc()

        if not self.tiles:
            self._fitted = False
            self.latent_vectors = np.zeros((0, LATENT_DIM), dtype=np.float32)
            self.stats["total_tiles"] = 0
            return

        # Build TF-IDF vocab from all tile bodies + tag text
        if rebuild_vocab:
            documents = []
            for t in self.tiles:
                enriched = t.body_text + " " + " ".join(t.tags) + " " + t.title
                documents.append(enriched)
            self.tfidf.fit(documents)

        # Encode all tiles
        vectors = []
        for t in self.tiles:
            enriched = t.body_text + " " + " ".join(t.tags)
            vec = self.encode_tile(enriched, t.tags, t.domain)
            vectors.append(vec)

        self.latent_vectors = np.array(vectors, dtype=np.float32)

        # Build per-domain indices
        if self.tfidf._fitted and self.tfidf.vocab_size > 0:
            if self.projection is None:
                self.projection = RandomProjection(self.tfidf.vocab_size)

        # Build domain lookup
        self.domain_vectors = defaultdict(list)
        self.domain_indices = defaultdict(list)
        for i, tile in enumerate(self.tiles):
            for d in tile.domains:
                self.domain_vectors[d].append(self.latent_vectors[i])
                self.domain_indices[d].append(i)
            # Also add to "general" domain
            self.domain_vectors["general"].append(self.latent_vectors[i])
            self.domain_indices["general"].append(i)

        # Build id → index mapping
        self.index = {t.id: i for i, t in enumerate(self.tiles)}

        # Load graph metadata if available
        graph_path = os.path.join(self.tiles_dir, "_graph.json")
        if os.path.exists(graph_path):
            try:
                import json
                with open(graph_path) as f:
                    self._graph = json.load(f)
            except:
                self._graph = None

        self._fitted = True
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.stats["total_tiles"] = len(self.tiles)
        self.stats["last_rebuild"] = now

        domain_counts = Counter()
        for t in self.tiles:
            for d in t.domains:
                domain_counts[d] += 1
        self.stats["domains"] = dict(domain_counts)
        self.stats["total_domains"] = len(domain_counts)

    def encode_tile(self, text: str, tags: list = None,
                    domain: str = None) -> np.ndarray:
        """
        Encode text into a latent vector (256-dim).
        Steps: tokenize → TF-IDF → random projection → L2 normalize.
        """
        if tags:
            text = text + " " + " ".join(tags)

        tfidf_vec = self.tfidf.transform(text)

        if self.projection is None:
            # Fallback: return TF-IDF padded/truncated
            if len(tfidf_vec) >= LATENT_DIM:
                return tfidf_vec[:LATENT_DIM].astype(np.float32)
            pad = np.zeros(LATENT_DIM, dtype=np.float32)
            pad[:len(tfidf_vec)] = tfidf_vec
            return pad

        latent = self.projection.project(tfidf_vec)
        return latent

    def search(self, query: str, k: int = DFLT_K,
               domain: str = None) -> list:
        """
        Search tiles by query with semantic embedding scores.

        Supports rooms_as_experts via domain filtering.
        Returns list of dicts with: tile, score, domain, priority.
        """
        if not self._fitted or len(self.tiles) == 0:
            return []

        query_vec = self.encode_tile(query, tags=[], domain=domain)

        # Domain routing (rooms_as_experts)
        if domain and domain in self.domain_indices:
            candidates = self.domain_indices[domain]
            if len(candidates) == 0:
                return []
            vecs = self.latent_vectors[candidates]
        else:
            candidates = list(range(len(self.tiles)))
            vecs = self.latent_vectors

        if len(candidates) == 0:
            return []

        # Cosine similarity
        scores = vecs @ query_vec  # dot product (L2 normalized = cosine)

        # Top-K indices
        top_k = min(k, len(scores))
        if top_k == 0:
            return []

        top_idx = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_idx:
            candidate_idx = candidates[idx]
            tile = self.tiles[candidate_idx]
            score = float(scores[idx])

            # Deadband priority assignment
            if score >= PRIORITY_TIERS["P0"]:
                priority = "P0"
            elif score >= PRIORITY_TIERS["P1"]:
                priority = "P1"
            else:
                priority = "P2"

            # Rooms routing
            rooms = set()
            for d in tile.domains:
                if d in DOMAIN_ROOMS:
                    for room in DOMAIN_ROOMS[d]:
                        rooms.add(room)
            if not rooms:
                rooms.add("library")  # default fallback

            results.append({
                "tile": tile,
                "score": score,
                "domain": tile.domain,
                "domains": list(tile.domains),
                "rooms": list(rooms),
                "priority": priority,
                "rank": len(results) + 1,
            })

        return results

    def get_neighbors(self, tile_id: str, k: int = 3,
                      domain: str = None) -> list:
        """Find nearest-neighbor tiles by latent vector similarity."""
        if tile_id not in self.index:
            return []

        idx = self.index[tile_id]
        query_vec = self.latent_vectors[idx]

        # All other tiles
        candidates = list(range(len(self.tiles)))
        candidates.remove(idx)

        if domain and domain in self.domain_indices:
            candidates = [c for c in candidates
                         if domain in self.tiles[c].domains]
            if not candidates:
                return []

        vecs = self.latent_vectors[candidates]
        scores = vecs @ query_vec

        top_k = min(k, len(scores))
        if top_k == 0:
            return []

        top_idx = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_idx:
            tile = self.tiles[candidates[idx]]
            score = float(scores[idx])
            results.append({
                "tile": tile,
                "score": score,
                "domain": tile.domain,
                "priority": "P0" if score >= 0.99 else "P1" if score >= 0.80 else "P2",
            })

        return results

    def curriculum_loop(self, query: str, loops: int = DFLT_LOOPS,
                        domain: str = None, k: int = DFLT_K) -> dict:
        """
        Multi-step iterative search refinement.

        Each loop:
        1. Search tiles with current query
        2. Extract top-domain signal from results
        3. Refine query by appending domain context
        4. Re-search with refined query

        Returns:
            {
                "final_results": [...],
                "loop_results": [[...], [...], ...],
                "converged": bool,
                "loops_used": int,
                "domain_path": [str, ...],
            }
        """
        loop_results = []
        current_query = query
        domain_path = [domain] if domain else ["general"]
        prev_top_id = None
        converged = False

        for loop_num in range(loops):
            results = self.search(
                current_query, k=k, domain=domain_path[-1]
            )
            loop_results.append(results)

            if not results:
                break

            # Extract top domain signal
            top_domains = Counter()
            for r in results:
                for d in r.get("domains", []):
                    top_domains[d] += 1

            if top_domains:
                best_domain = top_domains.most_common(1)[0][0]
                if best_domain != domain_path[-1]:
                    domain_path.append(best_domain)

            # Refine query: append top result's tile text
            top_result = results[0]
            top_tile = top_result["tile"]

            # Check convergence: same top tile as previous loop
            if prev_top_id is not None and top_tile.id == prev_top_id:
                converged = True
                break
            prev_top_id = top_tile.id

            # Refine: append tile title and tags for next loop
            refinement = f" {top_tile.title} {' '.join(top_tile.tags)}"
            current_query = query + refinement

        return {
            "final_results": loop_results[-1] if loop_results else [],
            "all_loops": loop_results,
            "converged": converged,
            "loops_used": loop_num + 1,
            "domain_path": domain_path,
            "query": query,
        }

    def trace(self, tile_id: str) -> dict:
        """
        Show which rooms/experts were activated for a tile.

        Returns:
            {
                "tile": TileRecord,
                "domains": [str, ...],
                "rooms": [str, ...],
                "neighbors": [...],
                "tags": [str, ...],
            }
        """
        if tile_id not in self.index:
            return {"error": f"Tile '{tile_id}' not found"}

        tile = self.tiles[self.index[tile_id]]

        rooms = set()
        for d in tile.domains:
            if d in DOMAIN_ROOMS:
                for room in DOMAIN_ROOMS[d]:
                    rooms.add(room)

        neighbors = self.get_neighbors(tile_id, k=3)

        return {
            "tile": tile,
            "id": tile.id,
            "title": tile.title,
            "tags": tile.tags,
            "domains": list(tile.domains),
            "primary_domain": tile.domain,
            "rooms": list(rooms),
            "neighbors": neighbors,
        }

    def get_stats(self) -> dict:
        """Get index statistics."""
        if not self._fitted:
            return {"status": "not loaded", "total_tiles": 0}

        total = len(self.tiles)
        if total == 0:
            return self.stats

        avg_confidence = 0.0
        # Rough estimate: average of cosine similarities between
        # each tile and its nearest neighbor
        if total > 1:
            sims = []
            for i in range(min(total, 50)):  # sample for speed
                vec = self.latent_vectors[i]
                others = np.delete(self.latent_vectors, i, axis=0)
                scores = others @ vec
                sims.append(float(np.max(scores)))
            avg_confidence = sum(sims) / len(sims) if sims else 0.0

        return {
            "status": "loaded",
            "total_tiles": total,
            "latent_dim": LATENT_DIM,
            "vocab_size": self.tfidf.vocab_size if self.tfidf._fitted else 0,
            "domains": self.stats.get("domains", {}),
            "total_domains": len(self.stats.get("domains", {})),
            "last_rebuild": self.stats.get("last_rebuild", "never"),
            "avg_confidence": round(avg_confidence, 4),
            "total_rooms": 14,
        }

    def get_room_for_domain(self, domain: str) -> list:
        """Resolve a domain to its expert rooms."""
        return DOMAIN_ROOMS.get(domain, ["library"])


# ═══════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════

plato_model = PlatoTileModel()
