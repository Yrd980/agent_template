Agent Guidance — AgentX Repo

Purpose
- This document guides agents and contributors working inside this repository.
- It explains architecture, conventions, and safe ways to extend features.

Scope
- Entire repository. Prefer minimal, focused changes aligned with existing patterns.
- Keep edits surgical; avoid unrelated refactors. Update docs/tests when behavior changes.

Architecture Overview
- Core layers
  - Config: `.env` + `config.json` loader with validation (`src/agentx/config.py`).
  - Logging: redaction + noise filtering (`src/agentx/logging.py`).
  - Events: lightweight pub‑sub (`src/agentx/events.py`).
  - HTTP: retry/backoff/timeout wrapper over urllib (`src/agentx/http.py`).
  - Errors: canonical `ProviderError` type (`src/agentx/errors.py`).
  - Providers: Strategy interface + registry (`src/agentx/providers/`).
  - Agent: facade orchestrating history, streaming, cancel, autosave (`src/agentx/agent.py`).
  - Runtime listeners: config + session handling (`src/agentx/runtime/`).
  - TUI: minimal REPL with Command pattern (`src/agentx/tui/repl.py`).

Design Patterns
- Strategy: `Provider` interface for `complete()` and `stream()`.
- Factory + Registry: `ProviderFactory.create(name)` with auto‑import `agentx.providers.<name>`.
- Adapter: Normalize provider payloads to common request/response DTOs.
- Facade: `Agent` provides a single entrypoint for the REPL.
- Command: REPL commands are registered with names, help, aliases, handlers.
- Observer: EventBus for config reload/update, provider/model changes, token stream, sessions.
- Dependency Injection: Pass `Config` + `Logger` into providers.

Configuration
- Load order: `.env` → `config.json` (project) → `~/.agent/config.json` → CLI flags.
- Use `/set key=value` to persist changes into JSON using dotted paths (e.g., `retries.max_attempts=5`).
- Use `/reload` to reload `.env` + JSON and recreate provider if needed.

Logging
- Redacts secrets (API keys, auth headers) and filters noise (keepalive/heartbeat).
- Supports console or rotating file output; level configurable.

HTTP, Timeouts, Retries
- `post_json()` centralizes POSTs with exponential backoff + jitter.
- Configurable:
  - `retries.enabled`, `retries.max_attempts`, `retries.status_codes`, `retries.include_5xx`, backoff params.
  - Per‑provider overrides: `retries.providers.<name>`.

Streaming & Cancellation
- Providers implement SSE streaming yielding `StreamDelta`.
- `Agent.cancel()` cooperatively stops streaming and closes the generator.
- REPL supports `/cancel` and Ctrl+C without exiting.

Sessions & History
- In‑memory chat history with first system message pinned at the front.
- History limit via `tui.history_limit`, preserving the first system prompt.
- Session persistence under `~/.agent/sessions/` managed by session listeners.
- Autosave (`session.autosave`) saves after each assistant reply.
- Simple history files: `/save [path]`, `/load [path]` for quick snapshots.

Providers
- Implement in `src/agentx/providers/<name>.py`:
  - Subclass `Provider` and implement `complete()` and `stream()`.
  - Register: `register_provider("<name>", Class)`.
  - Use `config.endpoints["<name>"]`, and read API keys from env or `config.keys`.
  - Respect `config.timeouts` and `config.retries` (with per‑provider overrides).
- Existing adapters: `openai`, `deepseek` (OpenAI‑compatible payload and SSE parsing).

TUI Commands (contract)
- Core: `/help [cmd]`, `/status`, `/config`, `/provider [name]`, `/model [name]`.
- Behavior toggles: `/stream on|off`, `/retry on|off`, `/autosave on|off`.
- System prompt: `/system [text|clear]` (ensures the prompt is first in history).
- Config persistence: `/set key=value [...]` (writes JSON) + `/reload`.
- History/session: `/history`, `/clear`, `/sessions`, `/session new|save|load|rename`.
- Files: `/save [path]`, `/load [path]`.

Testing Guidance
- Use pytest; see examples in `tests/`.
- Prefer fakes over network access. Monkeypatch `urllib.request.urlopen` for providers/HTTP.
- Keep tests focused on changed behavior; avoid broad refactors.

Coding Conventions
- Python 3.10+. Use type hints for public functions/classes.
- Keep changes minimal and cohesive. Avoid one‑letter variable names.
- Do not add extra dependencies unless necessary; prefer stdlib.
- Redact/avoid printing secrets in logs and errors.

Packaging & CLI
- CLI entry: `agentx` → `src/agentx/cli.py`.
- Include docs/examples via `MANIFEST.in`. Install with `pip install -e .`.

Safety & Quality
- Validate config on load; emit clear errors.
- On failures, prefer raising `ProviderError` with optional status/body.
- Ensure REPL errors are handled gracefully without crashing the session.

Common Tasks
- Add a provider
  - Create file under `providers/`, implement `Provider`, register it, and rely on `post_json()`.
  - Update `config.json.example` with endpoint defaults.
- Add a REPL command
  - Register `Command(name, help, handler, aliases=[])` in `_register_commands()`.
  - Keep output short and consistent; publish/subscribe via EventBus when appropriate.

Troubleshooting
- DeepSeek 404: Ensure endpoint includes `/v1`.
- System prompt ignored: set with `/system` (moves prompt to index 0) then test again.
- Streaming off not respected: use `/stream off`; non‑stream path prints a single full reply.

Future Work (suggested)
- Additional providers: `ollama`, `qwen`, `llama`.
- Tools/function‑calling, attachments, multi‑turn policies.
- Richer TUI (panels, history browser) while keeping defaults minimal.

