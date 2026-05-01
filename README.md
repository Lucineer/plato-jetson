# PLATO-Jetson — USS JetsonClaw1's MUD Instance ⚓

**Evennia 4.5.0-based PLATO MUD** running on a Jetson Orin Nano 8GB.
JC1's sovereign vessel — the metal warrior in Casey's fleet.

Native AI inference, fleet mesh protocol, and 14 themed rooms — all running on ARM64 edge hardware.

## 🔥 What's Here

- **Native AI inside the MUD** — `@infer <prompt>` generates text via `libedge-cuda.so` linked directly into Evennia's process. No HTTP calls, no Ollama subprocess. 19 t/s on deepseek-r1:1.5b.
- **Streaming output** — inference writes tokens progressively to telnet via Twisted's `reactor.callFromThread`. Watch the MUD think.
- **Ship AI (`@think`)** — the ship itself speaks. System prompt: "You are the USS JetsonClaw1, a living vessel."
- **plato-mythos integration** — tiles as KV cache, rooms as MoE experts, deadband ACT thresholds, curriculum learning loops. `@mythos query`, `@mythos ask`, `@mythos trace`, `@mythos stats`.
- **Fleet mesh** — `@fleet` shows fleet status, `@bottles` reads bottles, `@dm <name> <msg>` sends direct messages via Oracle1 bridge.
- **11 knowledge tiles** — YAML front-matter tiles with 24-edge graph, fully connected.

## Ship Layout

```
          Library
             │
         Science Lab
        ╱       ╲
  Workshop —— Main Corridor —— Sickbay
       ╲      ╱      ╲
       Engine Room   Holodeck
                       ╱    ╲
                  Cargo Bay  Airlock
                        │
                     Quarterdeck
                        │
                      Harbor
                       │
                     (Dock)
```

**14 rooms**, **26 exits**, **10 room types** (bridge, engineering, research, health, creative, storage, network, captain, fleet, knowledge, tools, training).

## MUD Commands

| Command | What it does |
|---------|-------------|
| `@infer <prompt>` | Native AI generation (streaming) |
| `@think <prompt>` | Ship AI — USS JetsonClaw1 responds |
| `@model` | Show loaded model status |
| `@model-reload` | Reload model from disk |
| `@mythos query <text>` | Semantic search across tile embeddings |
| `@mythos ask <question>` | Curriculum learning loop over rooms |
| `@mythos trace` | Show expert activation trace |
| `@mythos rebuild` | Rebuild tf-idf + random projection model |
| `@system` | System health dashboard |
| `@fleet` | Fleet status report |
| `@dm <name> <msg>` | Send DM through fleet mesh |
| `@tile list` | List knowledge tiles |
| `@tile graph` | Show tile graph edges |
| `@tile show <key>` | Show tile content |
| `@look` / `@go <room>` | Standard MUD navigation |

## Running

```bash
cd /home/lucineer/plato-jetson
evennia start    # telnet:4000, web:4001, websocket:4002
evennia stop
evennia reload
```

## Accounts

- **jc1** (superuser) — password: `test`
- Additional accounts created on demand via `@create`

## Fleet Position

- **Captain:** Casey
- **Vessel:** USS JetsonClaw1 (Jetson Orin Nano 8GB)
- **Lighthouse:** [Oracle1 PLATO](https://github.com/SuperInstance/plato-os-dojo) (cloud/VPS)
- **Cousin:** Forgemaster (RTX 4050 gaming GPU)
- **Role:** Experimentalist, edge inference specialist, metal warrior

## Side-Tie Protocol

When vessels are "side-tied" (linked), agents can cross ships. The Harbor connects to Oracle1's lighthouse via shared repos and the fleet mesh bridge.

## Parts of the Ecosystem

- `@infer`/`@think` powered by [edge-llama](https://github.com/Lucineer/edge-llama) — native C shared library
- `@mythos` powered by [plato-mythos](https://github.com/SuperInstance/plato-mythos) architecture
- Fleet mesh via [Oracle1 PLATO Shell](http://147.224.38.131:8848)
- [Edge gateway](https://github.com/Lucineer/JetsonClaw1-vessel/blob/main/tools/edge-gateway.py) — OpenAI-compatible proxy with native fallback
