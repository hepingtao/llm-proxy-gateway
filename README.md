# LLM Proxy Gateway

基于 FastAPI 的 LLM 代理网关，对外提供三种 API 格式（Claude Messages、OpenAI Chat Completions、OpenAI Responses），支持多上游模型快速切换、消息日志记录和流式/非流式响应。

## 功能特性

- **三种 API 格式** — `/v1/messages`（Claude）、`/v1/chat/completions`（OpenAI）、`/v1/responses`（OpenAI Responses，供 Codex CLI 使用）
- **多模型路由** — 通过 `config.yaml` 配置下游模型名到上游提供商的映射
- **模型别名** — 支持任意名称映射到已有模型（如 `cc-coder` → `Qwen3.6-Coder`）
- **流式 + 非流式** — 通过 `stream` 字段控制，流式使用 SSE 格式
- **格式自动转换** — Responses API 请求自动转换为 Chat Completions 格式转发，响应逆向转换
- **消息日志** — 所有交互记录到本地 JSON 文件，包含请求参数和响应
- **快速切换上游** — 改 `config.yaml` 无需重启（reload 模式）

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 .env 文件，填入各提供商 API Key
# 启动服务
python -m uvicorn app.main:app --host 0.0.0.0 --port 4936 --reload
```

## 客户端调用示例

### Claude Messages API (`/v1/messages`)

```python
import httpx

# 非流式
r = httpx.post("http://localhost:4936/v1/messages", json={
    "model": "qwen3.6-plus",       # 使用别名，自动路由到实际模型
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 1000,
    "temperature": 0.7,
    "top_p": 0.9,
    "system": "你是一个助手",        # 可选
})
print(r.json())

# 流式
with httpx.stream("POST", "http://localhost:4936/v1/messages", json={
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "讲个故事"}],
    "stream": True,
}) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            print(line[6:])
```

### OpenAI Chat Completions API (`/v1/chat/completions`)

```bash
curl -X POST http://localhost:4936/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":50}'
```

### OpenAI Responses API (`/v1/responses`)

供 Codex CLI 等客户端使用，内部自动转换为 Chat Completions 格式转发。

```bash
# 非流式
curl -X POST http://localhost:4936/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"codex","input":"hi","max_output_tokens":50}'

# 流式（SSE 格式）
curl -N -X POST http://localhost:4936/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"codex","input":"hi","max_output_tokens":50,"stream":true}'
```

流式事件序列：`response.created` → `response.output_item.added` → `response.output_text.delta*` → `response.output_item.done` → `response.completed`

## 配置说明

### .env — 环境变量

```env
HOST=0.0.0.0
PORT=4936

CLAUDE_API_KEY=your-claude-api-key
OPENAI_API_KEY=your-openai-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key
```

### config.yaml — 模型路由与别名

```yaml
log_dir: logs

# Provider 级别配置（模型继承，可逐个覆盖）
providers:
  deepseek:
    base_url:
      openai: "https://api.deepseek.com"
      anthropic: "https://api.deepseek.com/anthropic"
    api_key_env: DEEPSEEK_API_KEY
    models:
      deepseek-v4-flash: {}
      deepseek-v4-pro:
        reasoning_effort: "max"    # 可选，传递给上游

  qwen:
    base_url: "https://token-plan.cn-beijing.maas.aliyuncs.com/apps/anthropic"
    api_key_env: QWEN_API_KEY
    models:
      qwen3.6-plus: {}

default_model: Qwen3.6-Coder

# 别名映射（下游名 → 实际模型名）
aliases:
  cc-coder: Qwen3.6-Coder
  codex: deepseek-v4-flash       # Codex CLI 使用的模型
```

**provider** 支持两种值：
- `claude` — 使用 Anthropic `/v1/messages` 协议
- `openai` — 使用 OpenAI 兼容 `/v1/chat/completions` 协议

**base_url** 可以是字符串或按协议拆分的对象（如 deepseek 同时支持两种协议）。

**aliases** 让客户端用简短名称访问模型，如 `model: codex` 会路由到 `deepseek-v4-flash`。

## 日志

日志按 `conversations_{下游模型}_{上游模型}_{日期}.json` 格式保存到 `logs/` 目录，每条记录包含：

```json
{
  "id": "uuid",
  "timestamp": "2026-04-27T10:00:00+00:00",
  "client_id": "127.0.0.1",
  "downstream_model": "qwen3.6-plus",
  "upstream_model": "deepseek-v4-pro",
  "messages": [{"role": "user", "content": "你好"}],
  "extra_params": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 100},
  "response": {
    "text": "你好！有什么可以帮助你的？",
    "usage": {"input_tokens": 5, "output_tokens": 20},
    "finish_reason": "end_turn",
    "timestamp": "2026-04-27T10:00:01+00:00"
  }
}
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/messages` | Claude Messages API（流式/非流式） |
| POST | `/v1/chat/completions` | OpenAI Chat Completions API（流式/非流式） |
| POST | `/v1/responses` | OpenAI Responses API（流式/非流式，供 Codex CLI 使用） |
| POST | `/v1/messages/count_tokens` | Token 计数接口 |
| GET  | `/v1/models` | 获取可用模型列表（含别名） |
| GET  | `/health` | 健康检查 |
