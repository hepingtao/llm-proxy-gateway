# LLM Proxy Gateway

基于 FastAPI 的 LLM 代理网关，对外提供 Claude 格式 API，支持多上游模型快速切换、消息日志记录和流式/非流式响应。

## 功能特性

- **多模型路由** — 通过 `config.yaml` 配置下游模型名到上游提供商的映射
- **模型别名** — 支持任意名称映射到已有模型（如 `qwen3.6-plus` → `deepseek-v4-pro-anthropic`）
- **流式 + 非流式** — 统一 `/v1/messages` 接口，通过 `stream` 字段控制
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

```python
import httpx

# 非流式
r = httpx.post("http://localhost:4936/v1/messages", json={
    "model": "qwen3.6-plus",       # 使用别名，自动路由到 deepseek-v4-pro-anthropic
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
models:
  deepseek-v4-pro:
    provider: openai             # 或 claude
    upstream_model: deepseek-v4-pro
    api_key_env: DEEPSEEK_API_KEY
    base_url: "https://api.deepseek.com"

aliases:
  qwen3.6-plus: deepseek-v4-pro-anthropic   # 别名 → 实际模型名

default_model: deepseek-v4-pro-anthropic
log_dir: logs
```

**provider** 支持两种值：
- `claude` — 使用 Anthropic `/v1/messages` 协议
- `openai` — 使用 OpenAI 兼容 `/v1/chat/completions` 协议

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
| POST | `/v1/messages` | 对话接口（流式/非流式） |
| GET  | `/v1/models` | 获取可用模型列表（含别名） |
| GET  | `/health` | 健康检查 |
