# LLM Proxy Gateway — QWEN.md

## 项目概述

基于 FastAPI 的 LLM 代理网关，对外提供 Claude 格式 API，支持多上游模型快速切换、消息日志记录和流式/非流式响应。

### 核心功能
- **多模型路由** — 通过 `config.yaml` 配置下游模型名到上游提供商的映射
- **模型别名** — 支持任意名称映射到已有模型
- **双协议支持** — Claude (`/v1/messages`) 和 OpenAI (`/v1/chat/completions`) 协议
- **消息日志** — 所有交互记录到本地 JSON 文件
- **流式响应** — SSE 格式流式输出

### 技术栈
- Python + FastAPI
- Pydantic (数据验证)
- httpx (异步 HTTP 客户端)
- YAML (配置管理)

## 项目结构

```
D:\ProjectsMy\X\
├── app/
│   ├── main.py      # FastAPI 应用入口，API 端点定义
│   ├── config.py    # 配置加载，Settings + ModelConfig
│   ├── models.py    # Pydantic 数据模型 (请求/响应)
│   ├── proxy.py     # 上游模型调用逻辑 (Claude/OpenAI)
│   └── logger.py    # 对话日志记录
├── config.yaml      # 模型路由与别名配置
├── .env             # 环境变量 (API Keys)
└── logs/            # 日志输出目录
```

## 构建与运行

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务 (开发模式)
python -m uvicorn app.main:app --host 0.0.0.0 --port 4936 --reload

# 或直接运行
python app/main.py
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/messages` | 对话接口 (支持 `stream: true` 流式) |
| GET | `/v1/models` | 获取可用模型列表 |
| GET | `/health` | 健康检查 |

## 配置说明

### config.yaml
- `models` — 定义上游模型连接信息 (provider/base_url/api_key_env)
- `aliases` — 下游模型名到实际模型的映射
- `default_model` — 默认使用的模型

### .env
定义各提供商 API Key 环境变量：
- `CLAUDE_API_KEY`
- `OPENAI_API_KEY`
- `DEEPSEEK_API_KEY`
- `QWEN_API_KEY`

## 高级参数

### reasoning_effort (思考强度)
支持值：`low` | `medium` | `high` | `xhigh` | `max`

**配置方式（优先级从高到低）：**
1. **客户端请求参数** - 请求中传入的 `reasoning_effort` 或 `output_config`
2. **config.yaml 配置** - 模型配置中的 `reasoning_effort` 字段（客户端不传时使用）

支持两种传递格式（自动转换）：
- **简化格式**: `{"reasoning_effort": "high"}`
- **Claude 原生格式**: `{"output_config": {"effort": "high"}}`

代理会根据上游协议自动转换：
- **Claude 协议**: 统一转为 `{"output_config": {"effort": "high"}}`
- **OpenAI 协议**: 统一转为 `{"reasoning_effort": "high"}`

### config.yaml 配置示例
```yaml
models:
  deepseek-v4-pro-anthropic:
    provider: claude
    upstream_model: deepseek-v4-pro
    api_key_env: DEEPSEEK_API_KEY
    base_url: "https://api.deepseek.com/anthropic"
    reasoning_effort: "high"  # 默认思考强度
```

### 客户端调用示例
```python
# 方式 1：简化格式
r = httpx.post("http://localhost:4936/v1/messages", json={
    "model": "qwen3.6-plus",
    "messages": [{"role": "user", "content": "你好"}],
    "reasoning_effort": "high",
})

# 方式 2：Claude 原生格式
r = httpx.post("http://localhost:4936/v1/messages", json={
    "model": "qwen3.6-plus",
    "messages": [{"role": "user", "content": "你好"}],
    "output_config": {"effort": "high"},
})

# 方式 3：使用 config.yaml 配置的默认值（客户端不传 reasoning_effort）
r = httpx.post("http://localhost:4936/v1/messages", json={
    "model": "deepseek-v4-pro-anthropic",  # 此模型配置了 reasoning_effort: "high"
    "messages": [{"role": "user", "content": "你好"}],
    # 不传 reasoning_effort，自动使用配置的 "high"
})
```

## 开发规范

### 代码风格
- 使用 Python 类型注解
- 遵循 PEP 8 命名规范
- 异步函数使用 `async/await`

### 测试实践
- 当前无自动化测试
- 手动测试通过客户端调用验证

### 日志格式
日志按 `conversations_{downstream}_{upstream}_{date}.json` 保存，每条记录包含：
- 请求 ID、时间戳、客户端信息
- 下游/上游模型名
- 消息内容与额外参数
- 响应文本、token 使用、结束原因

## 注意事项

- 修改 `config.yaml` 后需重启服务或使用 `--reload` 模式
- API Key 变更需更新 `.env` 并重启
- 日志文件按日期分割，避免单文件过大
