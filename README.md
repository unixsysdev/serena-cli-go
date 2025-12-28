# Serena CLI - Go Edition

A lean coding assistant powered by GLM 4.7 and Serena MCP.

## Features

- 🚀 Powered by GLM 4.7 for superior coding capabilities
- 🛠️ Full Serena MCP tool integration
- ⚡ Fast compiled Go binary
- 🔧 Professional configuration management

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
glm:
  api_key: "your-api-key"
  base_url: "https://llm.chutes.ai/v1"
  model: "zai-org/GLM-4.7-TEE"

serena:
  context: "claude-code"
  project_path: "."
  command: "uvx"
  args:
    - "--from"
    - "git+https://github.com/oraios/serena"
    - "serena"
    - "start-mcp-server"
```

Or set environment variables:

```bash
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://llm.chutes.ai/v1"
export LLM_MODEL="zai-org/GLM-4.7-TEE"
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
```

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
