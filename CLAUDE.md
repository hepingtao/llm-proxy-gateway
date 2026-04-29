# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## 项目概述

基于 FastAPI 的 LLM 代理网关，对外提供 Claude 格式 API，将请求转发到上游大模型（Anthropic Claude 或 OpenAI 兼容协议），所有交互记录到本地 JSON 日志文件。

## 常用命令

```bash
# 启动开发服务（端口 4936，自动重载）
python -m uvicorn app.main:app --host 0.0.0.0 --port 4936 --reload

# 查找并杀掉占用 4936 端口的进程
netstat -ano | grep :4936
taskkill //PID <pid> //F

# 快速测试
python -c "import httpx; r=httpx.post('http://localhost:4936/v1/messages', json={'model':'deepseek-v4-pro-anthropic','messages':[{'role':'user','content':'hi'}],'max_tokens':50}); print(r.json())"
```

## 架构

### 请求流程

1. 客户端 POST `/v1/messages`，携带 `model` 字段（可以是别名）
2. `main.py` 通过 `config.py` 解析模型名：别名 → 实际模型 → 上游信息
3. `proxy.py` 根据 `provider` 字段路由到对应提供商（`call_claude` 或 `call_openai`）
4. 响应统一转换为 Claude 格式返回，`logger.py` 记录完整对话

### 文件职责

| 文件 | 职责 |
|------|------|
| `app/main.py` | FastAPI 路由、请求校验、流式响应编排 |
| `app/config.py` | 加载 `.env`（Settings）和 `config.yaml`（ModelConfig），处理别名解析 |
| `app/proxy.py` | HTTP 转发到上游提供商，Claude 与 OpenAI 格式互转 |
| `app/logger.py` | JSON 文件日志，文件名格式 `conversations_{下游}_{上游}_{日期}.json` |
| `app/models.py` | Pydantic 请求/响应模型 |
| `config.yaml` | 模型定义（provider、upstream_model、base_url、api_key_env, reasoning_effort）和别名映射 |
| `.env` | API Key 和服务配置 |

### 关键设计

- **别名解析** — `ModelConfig._resolve()` 在查找模型前先将别名映射到实际模型 key（如 `cc-coder` → `deepseek-v4-pro-anthropic`）
- **两种提供商** — `claude` 使用 Anthropic `/v1/messages` 协议；`openai` 使用 `/v1/chat/completions` 并自动转换为 Claude 响应格式
- **流式日志** — 在 SSE 流式传输过程中收集文本块，最终写入完整响应到日志
- **日志目录** — `logs/`，按下游模型名、上游模型名和日期分文件
- **参数转发** — `top_p`、`top_k`、`stop_sequences`、`tools`、`tool_choice` 等参数会透传到上游请求
- **配置热更新** — `--reload` 模式下修改 `config.yaml` 后服务自动重启加载
