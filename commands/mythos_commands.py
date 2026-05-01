"""
mythos_commands.py — @mythos commands for the Plato MUD.

Implements the plato-mythos architecture in the Evennia MUD:
  - @mythos query    — Semantic tile search with domain routing
  - @mythos ask      — Iterative curriculum-loop reasoning
  - @mythos trace    — Show expert room activation for a tile
  - @mythos stats    — Index statistics
  - @mythos rebuild  — Rebuild embedding index from disk

Each result includes domain routing path, confidence score, and
deadband priority (P0=critical, P1=standard, P2=low).
"""

import os
import threading
from collections import Counter

from evennia import Command, default_cmds
from evennia.utils import logger

from commands.plato_mythos import plato_model, PRIORITY_TIERS


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

PRIORITY_ICONS = {
    "P0": "🔥",  # Critical
    "P1": "⭐",  # Standard
    "P2": "💠",  # Low
}

def _format_result(r: dict, show_body: bool = False) -> str:
    """Format a single search result for MUD display."""
    tile = r["tile"]
    priority_icon = PRIORITY_ICONS.get(r["priority"], "💠")
    score_pct = f"{r['score'] * 100:.1f}%"
    rooms_str = ", ".join(r.get("rooms", ["library"]))
    domains_str = ", ".join(r.get("domains", [r["domain"]]))
    title = tile.title or tile.id

    lines = [
        f"  {priority_icon} #{r['rank']:2d} | {title[:50]:50s} | {score_pct:>6s} | {r['priority']}",
        f"       ├─ Domain: {domains_str}",
        f"       └─ Rooms:  {rooms_str}",
    ]
    if show_body and tile.body_text:
        snippet = tile.body_text[:200].replace("\n", " ")
        if len(snippet) >= 200:
            snippet += "..."
        lines.append(f"       ┌─ Preview: {snippet}")

    return "\n".join(lines)


def _build_result_from_hit(hit: dict, rank: int) -> dict:
    """Build a result dict from a raw hit."""
    tile = hit["tile"]
    priority = hit.get("priority", "P1")

    # Resolve rooms from domains
    from commands.plato_mythos import DOMAIN_ROOMS
    rooms = set()
    for d in hit.get("domains", [tile.domain]):
        if d in DOMAIN_ROOMS:
            for room in DOMAIN_ROOMS[d]:
                rooms.add(room)
    if not rooms:
        rooms.add("library")

    return {
        "tile": tile,
        "score": hit["score"],
        "domain": hit.get("domain", tile.domain),
        "domains": hit.get("domains", []),
        "rooms": list(rooms),
        "priority": priority,
        "rank": rank,
    }


# ═══════════════════════════════════════════════════════════════
# @mythos command — dispatch to subcommands
# ═══════════════════════════════════════════════════════════════

class CmdMythos(default_cmds.MuxCommand):
    """
    Knowledge retrieval using the plato-mythos semantic index.

    Usage:
      @mythos query [--k 5] [--domain research] <question>
      @mythos ask [--loops 3] [--domain engineering] <question>
      @mythos trace <tile-id>
      @mythos stats
      @mythos rebuild

    Subcommands:

      query    — Search tiles using semantic embedding + domain routing.
                 Returns results with confidence scores and deadband priority.

      ask      — Iterative curriculum-loop reasoning.
                 The model "thinks" by doing multiple passes through the index.
                 Each loop refines the query using top results.

      trace    — Show which rooms/experts were activated for a tile.
                 Displays domains, expert rooms, and nearest neighbors.

      stats    — Show index statistics: total tiles, per-domain counts,
                 average confidence, latent dimension.

      rebuild  — Rebuild the embedding index from all tile files on disk.

    Flags:
      --k N        Number of results (default: 5)
      --domain D   Filter by domain/room (research, engineering, fleet, etc.)
      --loops N    Curriculum loop depth (default: 3)
      --body       Show body preview in results

    Examples:
      @mythos query --k 3 --domain research jetpack attention
      @mythos ask --loops 5 --domain engineering why is the CUDA OOM
      @mythos trace cocapn-architecture
      @mythos stats
    """
    key = "@mythos"
    locks = "cmd:all()"
    help_category = "Plato"

    def func(self):
        caller = self.caller
        args = self.args.strip()

        if not args:
            caller.msg(self.__doc__)
            return

        # Extract subcommand
        parts = args.split()
        subcmd = parts[0].lower()

        if subcmd == "query":
            self._cmd_query(caller, parts[1:])
        elif subcmd == "ask":
            self._cmd_ask(caller, parts[1:])
        elif subcmd == "trace":
            self._cmd_trace(caller, parts[1:])
        elif subcmd == "stats":
            self._cmd_stats(caller)
        elif subcmd == "rebuild":
            self._cmd_rebuild(caller)
        else:
            caller.msg(f"Unknown subcommand: '{subcmd}'. Try @mythos query, ask, trace, stats, or rebuild.")

    def _parse_flags(self, args: list) -> dict:
        """Parse --flag style arguments from arg list."""
        opts = {
            "k": 5,
            "domain": None,
            "loops": 3,
            "body": False,
            "query_parts": [],
        }
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--k" and i + 1 < len(args):
                try:
                    opts["k"] = max(1, int(args[i + 1]))
                except ValueError:
                    pass
                i += 2
            elif arg == "--domain" and i + 1 < len(args):
                opts["domain"] = args[i + 1]
                i += 2
            elif arg == "--loops" and i + 1 < len(args):
                try:
                    opts["loops"] = max(1, int(args[i + 1]))
                except ValueError:
                    pass
                i += 2
            elif arg == "--body":
                opts["body"] = True
                i += 1
            else:
                opts["query_parts"].append(arg)
                i += 1
        return opts

    def _ensure_loaded(self, caller) -> bool:
        """Ensure the model index is loaded. Return False if not."""
        if not plato_model._fitted or len(plato_model.tiles) == 0:
            caller.msg(
                "🧠 plato-mythos index not loaded.\n"
                "Type @mythos rebuild to build the embedding index from tiles."
            )
            return False
        return True

    def _cmd_query(self, caller, args: list):
        """@mythos query — semantic tile search."""
        opts = self._parse_flags(args)
        query = " ".join(opts["query_parts"]).strip()

        if not query:
            caller.msg("Usage: @mythos query [--k N] [--domain D] <question>")
            return
        if not self._ensure_loaded(caller):
            return

        k = opts["k"]
        domain = opts["domain"]

        caller.msg(
            f"🔍 Mythos Query\n"
            f"{'─'*60}\n"
            f"  Query:   {query[:80]}"
        )
        if domain:
            rooms = plato_model.get_room_for_domain(domain)
            caller.msg(f"  Domain:  {domain} → {', '.join(rooms)}")
        caller.msg(f"  Top-K:   {k}\n")

        # Run search in a background thread for responsiveness
        def _search():
            try:
                results = plato_model.search(query, k=k, domain=domain)

                if not results:
                    caller.msg("📭 No matching tiles found. Try a broader search or @mythos rebuild.")
                    return

                for r in results:
                    r["rank"] = results.index(r) + 1
                    caller.msg(_format_result(r, show_body=opts["body"]))

                # Summary
                priorities = Counter(r["priority"] for r in results)
                summary_parts = []
                for p in ["P0", "P1", "P2"]:
                    if priorities[p]:
                        summary_parts.append(f"{PRIORITY_ICONS[p]} {priorities[p]} {p}")
                caller.msg(
                    f"\n{'─'*60}"
                    f"\n  Results: {len(results)} | "
                    f"{' | '.join(summary_parts)}"
                    f"\n  💡 Use @mythos trace <tile-id> for expert activation details"
                )

            except Exception as e:
                caller.msg(f"❌ Search error: {e}")

        t = threading.Thread(target=_search, daemon=True)
        t.start()

    def _cmd_ask(self, caller, args: list):
        """@mythos ask — iterative curriculum-loop reasoning."""
        opts = self._parse_flags(args)
        query = " ".join(opts["query_parts"]).strip()

        if not query:
            caller.msg("Usage: @mythos ask [--loops N] [--domain D] <question>")
            return
        if not self._ensure_loaded(caller):
            return

        loops = opts["loops"]
        domain = opts["domain"]
        k = opts["k"]

        caller.msg(
            f"🧠 Mythos Curriculum Loop\n"
            f"{'─'*60}\n"
            f"  Query:  {query[:80]}"
        )
        if domain:
            caller.msg(f"  Domain: {domain}")
        caller.msg(f"  Loops:  {loops}\n")

        def _think():
            try:
                result = plato_model.curriculum_loop(
                    query, loops=loops, domain=domain, k=k
                )

                # Show loop evolution
                for loop_num, loop_results in enumerate(result["all_loops"]):
                    caller.msg(
                        f"\n{'─'*60}\n"
                        f"  📍 Loop {loop_num + 1}/{result['loops_used']}"
                    )
                    if result["domain_path"] and loop_num < len(result["domain_path"]):
                        d = result["domain_path"][loop_num]
                        rooms = plato_model.get_room_for_domain(d)
                        caller.msg(f"     Domain path: {d} → {', '.join(rooms)}")

                    if not loop_results:
                        caller.msg("     (no results)")
                        continue

                    for r in loop_results[:2]:  # Show top 2 per loop
                        r["rank"] = loop_results.index(r) + 1
                        r_rooms = set()
                        from commands.plato_mythos import DOMAIN_ROOMS
                        for d in r.get("domains", [r["domain"]]):
                            if d in DOMAIN_ROOMS:
                                r_rooms.update(DOMAIN_ROOMS[d])
                        r["rooms"] = list(r_rooms) if r_rooms else ["library"]
                        caller.msg(_format_result(r))

                    if len(loop_results) > 2:
                        caller.msg(f"     ... and {len(loop_results) - 2} more results")

                # Final results
                final = result["final_results"]
                caller.msg(
                    f"\n{'═'*60}\n"
                    f"  ✅ Curriculum Loop Complete\n"
                    f"     {'Converged' if result['converged'] else 'Max loops reached'}"
                    f" after {result['loops_used']} loops"
                )
                if result["domain_path"]:
                    caller.msg(f"     Domain path: {' → '.join(result['domain_path'])}")

                if final:
                    caller.msg(f"\n  Top {len(final)} final results:")
                    for r in final:
                        r["rank"] = final.index(r) + 1
                        r_rooms = set()
                        from commands.plato_mythos import DOMAIN_ROOMS
                        for d in r.get("domains", [r["domain"]]):
                            if d in DOMAIN_ROOMS:
                                r_rooms.update(DOMAIN_ROOMS[d])
                        r["rooms"] = list(r_rooms) if r_rooms else ["library"]
                        caller.msg(_format_result(r))

            except Exception as e:
                caller.msg(f"❌ Curriculum loop error: {e}")

        t = threading.Thread(target=_think, daemon=True)
        t.start()

    def _cmd_trace(self, caller, args: list):
        """@mythos trace — show expert room activation for a tile."""
        if not args:
            caller.msg("Usage: @mythos trace <tile-id>")
            return
        if not self._ensure_loaded(caller):
            return

        tile_id = args[0].strip().lower()
        # Check if it's a filename
        if tile_id.endswith(".md"):
            tile_id = tile_id[:-3]

        trace = plato_model.trace(tile_id)
        if "error" in trace:
            caller.msg(f"❌ {trace['error']}")
            # Suggest similar tiles
            matches = [t for t in plato_model.tiles if tile_id in t.id]
            if matches:
                caller.msg(
                    "Did you mean one of these?\n  "
                    + "\n  ".join(f"• {m.id}" for m in matches[:5])
                )
            return

        tile = trace["tile"]
        priority = "P0"  # Trace is always critical
        priority_icon = PRIORITY_ICONS["P0"]

        rooms_str = ", ".join(trace.get("rooms", ["library"]))
        domains_str = ", ".join(trace.get("domains", ["general"]))

        caller.msg(
            f"{priority_icon} Expert Activation Trace\n"
            f"{'─'*60}\n"
            f"  Tile:     {tile.id}"
        )
        if tile.title:
            caller.msg(f"  Title:    {tile.title}")
        caller.msg(
            f"  Tags:     [{', '.join(trace['tags'])}]\n"
            f"  Domains:  {domains_str}\n"
            f"  Rooms:    {rooms_str}\n"
            f"  Created:  {tile.created}"
        )

        # Show nearest neighbors
        neighbors = trace.get("neighbors", [])
        if neighbors:
            caller.msg(
                f"\n  🧩 Nearest Neighbors (expert co-activation):"
            )
            for n in neighbors:
                nt = n["tile"]
                pct = f"{n['score'] * 100:.1f}%"
                caller.msg(
                    f"     #{neighbors.index(n) + 1} | {nt.id[:40]:40s} "
                    f"| {pct:>6s} | {n['priority']}"
                )
                # Show which rooms this neighbor activates
                n_rooms = set()
                from commands.plato_mythos import DOMAIN_ROOMS
                for d in nt.domains:
                    if d in DOMAIN_ROOMS:
                        n_rooms.update(DOMAIN_ROOMS[d])
                if n_rooms:
                    caller.msg(f"         Rooms: {', '.join(sorted(n_rooms))}")

        # Body preview
        if tile.body_text:
            snippet = tile.body_text[:300].replace("\n", " ")
            if len(snippet) >= 300:
                snippet += "..."
            caller.msg(
                f"\n  📄 Content Preview:\n"
                f"     {snippet}"
            )

        caller.msg(f"\n{'─'*60}")
        deadband_str = (
            "Priority tiers: P0≥0.99 critical, P1≥0.80 standard, P2≥0.50 low"
        )
        caller.msg(f"  {deadband_str}")

    def _cmd_stats(self, caller):
        """@mythos stats — show index statistics."""
        if not self._ensure_loaded(caller):
            return

        stats = plato_model.get_stats()

        caller.msg(
            f"📊 PlatoMythos Index Stats\n"
            f"{'─'*60}\n"
            f"  Total tiles:      {stats['total_tiles']}\n"
            f"  Latent dimension: {stats['latent_dim']} (random projection)\n"
            f"  Vocab size:       {stats['vocab_size']}\n"
            f"  Total domains:    {stats['total_domains']}\n"
            f"  Total rooms:      {stats['total_rooms']} (expert groups)\n"
            f"  Avg confidence:   {stats['avg_confidence']}\n"
            f"  Last rebuild:     {stats['last_rebuild']}\n"
        )

        # Per-domain stats
        domains = stats.get("domains", {})
        if domains:
            caller.msg(f"  ── Domain Distribution ──")
            sorted_domains = sorted(domains.items(), key=lambda x: -x[1])
            for domain, count in sorted_domains:
                rooms = ", ".join(plato_model.get_room_for_domain(domain))
                bar = "█" * count
                caller.msg(f"     {domain:20s} | {count:3d} tiles | {bar}")

        caller.msg(
            f"\n  ── Architecture ──\n"
            f"  • TilesAsKV:      TF-IDF → Random Projection (256-dim)\n"
            f"  • RoomsAsExperts: Tag-based domain routing to 14 expert groups\n"
            f"  • DeadbandACT:    P0≥0.99 🔥 / P1≥0.80 ⭐ / P2≥0.50 💠\n"
            f"  • CurriculumLoop: Iterative multi-step search refinement\n"
        )

    def _cmd_rebuild(self, caller):
        """@mythos rebuild — rebuild the embedding index from disk."""
        caller.msg(
            f"🔨 Rebuilding plato-mythos index from {plato_model.tiles_dir}..."
        )

        def _rebuild():
            try:
                plato_model.load_and_build(rebuild_vocab=True)
                stats = plato_model.get_stats()
                caller.msg(
                    f"✅ Index rebuilt!\n"
                    f"   {stats['total_tiles']} tiles loaded\n"
                    f"   {stats['total_domains']} domain groups\n"
                    f"   {stats['vocab_size']} vocab features\n"
                    f"   Latent dim: {stats['latent_dim']}"
                )
            except Exception as e:
                caller.msg(f"❌ Rebuild error: {e}")

        t = threading.Thread(target=_rebuild, daemon=True)
        t.start()
