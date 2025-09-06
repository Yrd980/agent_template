prompt

@CLAUDE.md @./scripts/dev.sh

./scripts/dev.sh server

my system is arch linux
i have config deepseek api in .env and config proxy in my computer

NO_PROXY=127.0.0.1,localhost curl -X POST http://127.0.0.1:8000/api/v1/messages -H "Content-Type: application/json" -d '{"content": "Hello, this is a test message to verify DeepSeek connection"}'

use vscode debug

.vscode/launch.json

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Attach to agent-server (uv run + debugpy)",
      "type": "debugpy",
      "request": "attach",
      "connect": { "host": "127.0.0.1", "port": 5678 },
      "justMyCode": true
    }
  ]
}
```

```sh
uv run python -m debugpy --listen 5678 --wait-for-client -m agent_template.cli --debug server

```
