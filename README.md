# Serena CLI - Go Edition

A lean coding assistant powered by OpenAI compatible inference endpoints LLMs and Serena MCP.

## Features

- 🚀 Powered by any OpenAI compatible endpoint
- 🛠️ Full Serena MCP tool integration
- ⚡ Fast compiled Go binary
- 🔧 Configuration management

## Installation

```bash
go install github.com/unixsysdev/serena-cli-go/cmd/serena@latest
```

## Configuration

Create a `serena-cli.yaml` file (copy `serena-cli.yaml.example` from this repo). Config is loaded from:

- `./serena-cli.yaml`
- `~/.serena-cli/serena-cli.yaml`
- `~/.config/serena-cli/serena-cli.yaml`

Example minimal config:

```yaml
llm:
  api_key: "your-api-key"
  base_url: "https://llm.chutes.ai/v1"
  model: "zai-org/GLM-4.7-TEE"
  compaction_model: "Qwen/Qwen3-VL-235B-A22B-Instruct"
  timeout_seconds: 300

serena:
  command: "uvx"
  tool_timeout_seconds: 300
  enable_web_dashboard: false
  enable_gui_log_window: false
  max_tool_answer_chars: 20000
  args:
    - "--from"
    - "git+https://github.com/oraios/serena"
    - "serena"
    - "start-mcp-server"
```

Optional: set `serena.context` or `serena.project_path` if you want to force them;
leaving them empty lets Serena manage context and project activation.

Or set environment variables:

```bash
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://llm.chutes.ai/v1"
export LLM_MODEL="zai-org/GLM-4.7-TEE"
export LLM_COMPACTION_MODEL="Qwen/Qwen3-VL-235B-A22B-Instruct"
export LLM_TIMEOUT_SECONDS="300"
export SERENA_TOOL_TIMEOUT_SECONDS="300"
export SERENA_ENABLE_WEB_DASHBOARD="false"
export SERENA_ENABLE_GUI_LOG_WINDOW="false"
export SERENA_MAX_TOOL_ANSWER_CHARS="20000"
```

## Usage

```bash
# Run in current directory
serena

# Run in specific project directory
cd /path/to/project
serena

# Show configuration (API key is masked)
serena --config

# Show version
serena --version

# One-shot prompt
serena "summarize the repository"
```

REPL commands:

```
/model
/model 3
/model "moonshotai/Kimi-K2-Instruct-0905"
/tools
/status
/context
/trace 5
/summary
/session list
/session new experiment
/session switch experiment
/compact
@context ./README.md
```

Tip: press `Ctrl+C` while a tool or model request is running to cancel it.

Sessions are stored under `~/.serena-cli/sessions/<project-name>` so you can switch contexts.
On startup, the CLI prints a short summary of the session history (generated with the compaction model).

### Built-in models

- deepseek-ai/DeepSeek-V3.2-Speciale-TEE
- MiniMaxAI/MiniMax-M2.1-TEE
- Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8-TEE
- moonshotai/Kimi-K2-Thinking-TEE
- moonshotai/Kimi-K2-Instruct-0905
- deepseek-ai/DeepSeek-V3.2-TEE
- zai-org/GLM-4.7-TEE

## Development

```bash
# Install dependencies
go mod download

# Run tests
go test ./...

# Build
go build -o bin/serena ./cmd/serena

# Run locally
go run ./cmd/serena
```

## License

MIT
