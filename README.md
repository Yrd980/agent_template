AgentX — Extensible Multi‑Provider Agent (TUI)

Overview
- Python backend with a minimal TUI REPL frontend.
- Pluggable providers behind a clean Strategy interface.
- Config via .env + JSON; logging with redaction and noise filters.
- Session management with autosave; live reload and runtime toggles.

Status
- Implemented providers: openai, deepseek (OpenAI‑compatible chat/completions).
- TUI REPL with streaming and cancellation.
- Error handling: retries, backoff, timeouts (per‑provider overrides).

Install
- Python 3.10+ required.
- From source:
  - Optional venv: `python -m venv .venv && . .venv/bin/activate`
  - Install: `pip install -e .`
  - Run: `agentx` or `python -m agentx.cli`

Quickstart
1) Configure credentials
   - Copy `.env.example` → `.env` and set keys (e.g., `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`).
2) Configure runtime (optional)
   - Copy `config.json.example` → `config.json` and adjust:
     - `provider`, `model`, `endpoints`, `streaming`, `timeouts`, `retries`, `session.autosave`, `tui.history_limit`.
3) Start REPL: `agentx`

Core Concepts
- Providers (Strategy):
  - `openai` and `deepseek` implemented; more can be added via `agentx.providers.<name>`.
  - Each implements `complete()` and `stream()` on unified `ChatRequest`/`ChatResponse` types.
- Factory + Registry:
  - `ProviderFactory.create(name, ...)` instantiates a provider (auto‑imports `agentx.providers.<name>` when missing).
- Events (Pub/Sub):
  - Config reload/update, provider/model set, history clear, token stream, sessions.
- Agent facade:
  - Maintains history, coordinates streaming tokens and cancellation, applies system prompt first.

Config
- Load order: `.env` → `config.json` (project) → `~/.agent/config.json` (fallback) → CLI flags.
- JSON keys (high‑level):
  - `provider`, `model`, `streaming`
  - `endpoints`: `{ "openai": "https://api.openai.com/v1", "deepseek": "https://api.deepseek.com/v1" }`
  - `timeouts`: `{ "connect_ms": 10000, "read_ms": 60000 }`
  - `retries`: `{ "enabled": true, "max_attempts": 3, "status_codes": [408,429], "include_5xx": true, "backoff": { "base_ms": 200, "factor": 2.0, "jitter": true }, "providers": { "deepseek": { "status_codes": [429] }}}`
  - `session`: `{ "autosave": false }`
  - `tui`: `{ "history_limit": 200, "show_timestamps": false }`
  - `logging`: `{ "level": "INFO", "filter": null, "file": null }`
- API keys via env: `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, etc.

CLI
- `agentx [--provider NAME] [--model NAME] [--log LEVEL] [--log-file PATH] [--config PATH]`

REPL Commands
- Help: `/help [cmd]`, `/status`, `/config`
- Provider/Model: `/provider [name]`, `/model [name]`
- Streaming: `/stream [on|off]`, cancel with `/cancel` or Ctrl+C
- System prompt: `/system [text|clear]` (stored at the start of history)
- Config persist: `/set key=value [key=value ...]` (writes to JSON, supports dotted keys)
- Reload config: `/reload`
- History/session: `/history`, `/clear`, `/sessions`, `/session [new|save|load|rename]`
- Simple history files: `/save [path]`, `/load [path]`
- Retries: `/retry [on|off]`, Autosave: `/autosave [on|off]`

Examples
- Switch to DeepSeek and stream:
  - `/provider deepseek`
  - `/model deepseek-chat`
  - `Explain bubble sort simply.`
- Turn off streaming for full responses:
  - `/stream off` → ask a prompt → prints a single full reply
- Persist settings with `/set`:
  - `/set retries.max_attempts=5 retries.backoff.base_ms=300`
  - `/reload` (also auto‑triggered by /set)

Sessions
- Stored at `~/.agent/sessions/` as JSON: `{ id, name, created_at, updated_at, messages: [...] }`
- Autosave (`session.autosave: true`) saves after each assistant reply.

Logging
- Redacts API keys and auth headers; filters out keepalive noise.
- Enable debug: run with `--log DEBUG` or set in `config.json`.

Development
- Run tests: `pytest -q`
- Add a provider: create `src/agentx/providers/<name>.py`, implement `Provider`, register via `register_provider("<name>", Class)`, and it will auto‑load.

Troubleshooting
- 404 on DeepSeek: ensure `endpoints.deepseek` includes `/v1`.
- No streaming: check `/stream on` and that your model supports streaming.
- System prompt ignored: `/system` moves it to the start of history; clear and reset if needed.

