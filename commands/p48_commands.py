"""
p48_commands.py — P48 Vector Search for Evennia

Commands:
  @p48-search <text>  — Search tiles by P48 exact distance
  @p48-status          — P48 index status

Depends on p48-index-server.py running (port 8846) or reading /dev/shm/p48-index/.
"""

import json, struct, os, math, re, urllib.request, urllib.parse
from evennia import Command, default_cmds

# 90 keywords matching warp-room.c
EDGE_KW = ["jetson","cpu","gpu","memory","temperature","load","uptime",
           "disk","thermal","fan","power","nvidia","cuda","nvcc",
           "arm64","aarch64","swap","network","interface","sensor",
           "telemetry","hardware","clock","throttle","edge","device"]
RESEARCH_KW = ["research","paper","study","findings","analysis","experiment",
               "benchmark","performance","test","comparison","evaluation",
               "learn","training","dataset","model","inference","llm",
               "neural","embedding","vector","similarity","tile",
               "investigation","methodology","result","conclusion","algorithm"]
FLEET_KW = ["fleet","agent","oracle","forge","vessel","bottle","matrix",
            "heartbeat","sync","mesh","iron","coordination","bridge",
            "pki","cert","trust","deadman","migration","protocol",
            "lighthouse","beacon","dm","conduit","message"]
JC1_KW = ["jc1","jetsonclaw","plato","evennia","flato","mythos",
          "cocapn","libllama","gguf","sovereign","infer","think","vessel"]

ALL_KW = EDGE_KW + RESEARCH_KW + FLEET_KW + JC1_KW
KW_SET = set(ALL_KW)
N_DIMS = len(ALL_KW)  # 90

P48_SERVER = "http://localhost:8846"


def _query_p48_server(query, top_k=10):
    """Query P48 index server via HTTP."""
    url = f"{P48_SERVER}/search?q={urllib.parse.quote(query)}&top_k={top_k}"
    try:
        resp = urllib.request.urlopen(url, timeout=3)
        return json.loads(resp.read())
    except Exception:
        return None


def _status_from_server():
    """Get status from P48 index server."""
    try:
        resp = urllib.request.urlopen(f"{P48_SERVER}/status", timeout=2)
        return json.loads(resp.read())
    except Exception:
        return None


class CmdP48Search(Command):
    """
    @p48-search <text> — Search knowledge tiles by P48 exact distance.
    
    Uses Pythagorean48 6-bit vector encoding for exact integer 
    nearest-neighbor search. Results are ordered by distance (lower = better).
    Compatible with warp-room and the Fleet Math ecosystem.
    
    Usage:
      @p48-search <query text>
    
    Example:
      @p48-search fleet deadman protocol
      @p48-search jetson gpu temperature
    """
    
    key = "@p48-search"
    aliases = ["@p48s", "@vector-search"]
    help_category = "Knowledge"
    locks = "cmd:all()"
    arg_regex = r"\s.+"
    
    def func(self):
        query = self.args.strip()
        if not query:
            self.msg("Usage: @p48-search <query text>")
            self.msg("Example: @p48-search fleet deadman protocol")
            return
        
        # Query P48 index server
        data = _query_p48_server(query, top_k=8)
        if data is None:
            self.msg("|rP48 index server not available.|n Start: p48-index-server.py --daemon")
            return
        
        results = data.get("results", [])
        if not results:
            self.msg("No matching tiles found.")
            return
        
        self.msg(f"|wP48 Exact Search|n: |c{query[:60]}|n")
        self.msg("-" * 70)
        for r in results:
            tile = r.get("tile", {})
            q = tile.get("question", "?")[:45]
            room = tile.get("room", "?")
            dist = r.get("distance", -1)
            self.msg(f"  |d#{dist:>5}|n |c{room:>10}|n  |w{q}|n")


class CmdP48Status(Command):
    """
    @p48-status — Show P48 vector index status.
    
    Displays number of indexed vectors, keyword dimensions, room breakdown,
    and shared memory path. Indicates warp-room compatibility.
    """
    
    key = "@p48-status"
    aliases = ["@p48st"]
    help_category = "Knowledge"
    locks = "cmd:all()"
    
    def func(self):
        s = _status_from_server()
        if not s:
            # Fall back to checking /dev/shm directly
            shm_path = "/dev/shm/p48-index"
            meta_file = f"{shm_path}/index.json"
            if os.path.exists(meta_file):
                with open(meta_file) as f:
                    s = json.load(f)
                rooms = s.get("room_counts", {})
                rooms_str = ", ".join(f"{k}: {v}" for k, v in rooms.items())
                self.msg(
                    f"|wP48 Vector Index|n\n"
                    f"  Vectors:   |c{s.get('n_vectors', 0)}|n\n"
                    f"  Keywords:  |c{s.get('n_keywords', 90)}|n\n"
                    f"  Dims:      |c{s.get('n_p48_dims', 12)}|n packed\n"
                    f"  Rooms:     {rooms_str}\n"
                    f"  SHM:       {shm_path}\n"
                    f"  Server:    |roffline|n"
                )
            else:
                self.msg("|rP48 index not found.|n Build: p48-index-server.py --index")
            return
        
        rooms = s.get("rooms", {})
        rooms_str = ", ".join(f"{k}: {v}" for k, v in rooms.items())
        self.msg(
            f"|wP48 Vector Index|n\n"
            f"  Vectors:   |c{s.get('n_vectors', 0)}|n\n"
            f"  Keywords:  |c{s.get('n_keywords', 90)}|n\n"
            f"  Dims:      |c{s.get('n_p48_dims', 12)}|n packed\n"
            f"  Rooms:     {rooms_str}\n"
            f"  SHM:       {s.get('shm_path', '/dev/shm/p48-index')}\n"
            f"  Server:    |g@localhost:8846|n"
        )
