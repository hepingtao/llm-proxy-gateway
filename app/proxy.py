from typing import AsyncGenerator

import httpx

from .models import ClaudeRequest


# ── Helper functions ────────────────────────────────────────────

def _convert_to_dict(obj):
    """Convert Pydantic model or list of models to dict/list."""
    if obj is None:
        return None
    if isinstance(obj, list):
        return [item.model_dump() if hasattr(item, "model_dump") else item for item in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


# ── Anthropic provider ────────────────────────────────────────────

async def call_claude(req: ClaudeRequest, upstream: dict) -> dict:
    url = f"{upstream['base_url']}/v1/messages"
    headers = {
        "x-api-key": upstream["api_key"],
        "anthropic-version": upstream["api_version"],
        "content-type": "application/json",
    }
    body = {
        "model": upstream["upstream_model"],
        "messages": _to_claude_messages(req),
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }
    # Extract system text if present (handle both string and array format)
    system_text = _extract_system(req.system)
    if system_text:
        body["system"] = system_text
    if req.top_p != 1.0:
        body["top_p"] = req.top_p
    if req.top_k:
        body["top_k"] = req.top_k
    if req.stop_sequences:
        body["stop_sequences"] = req.stop_sequences
    if req.tools:
        body["tools"] = _convert_to_dict(req.tools)
    if req.tool_choice:
        body["tool_choice"] = _convert_to_dict(req.tool_choice)
    # Handle reasoning_effort: support both {"reasoning_effort": "high"} and {"output_config": {"effort": "high"}}
    if req.output_config:
        body["output_config"] = req.output_config
    elif req.reasoning_effort:
        body["output_config"] = {"effort": req.reasoning_effort}

    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()


async def stream_claude(req: ClaudeRequest, upstream: dict) -> AsyncGenerator[str, None]:
    url = f"{upstream['base_url']}/v1/messages"
    headers = {
        "x-api-key": upstream["api_key"],
        "anthropic-version": upstream["api_version"],
        "content-type": "application/json",
    }
    body = {
        "model": upstream["upstream_model"],
        "messages": _to_claude_messages(req),
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": True,
    }
    # Extract system text if present (handle both string and array format)
    system_text = _extract_system(req.system)
    if system_text:
        body["system"] = system_text
    if req.top_p != 1.0:
        body["top_p"] = req.top_p
    if req.top_k:
        body["top_k"] = req.top_k
    if req.stop_sequences:
        body["stop_sequences"] = req.stop_sequences
    if req.tools:
        body["tools"] = _convert_to_dict(req.tools)
    if req.tool_choice:
        body["tool_choice"] = _convert_to_dict(req.tool_choice)
    # Handle reasoning_effort: support both {"reasoning_effort": "high"} and {"output_config": {"effort": "high"}}
    if req.output_config:
        body["output_config"] = req.output_config
    elif req.reasoning_effort:
        body["output_config"] = {"effort": req.reasoning_effort}

    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        async with client.stream("POST", url, headers=headers, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                yield f"data: {data}\n\n"


# ── OpenAI provider ──────────────────────────────────────────────

async def call_openai(req: ClaudeRequest, upstream: dict) -> dict:
    url = f"{upstream['base_url']}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {upstream['api_key']}",
        "content-type": "application/json",
    }
    body = {
        "model": upstream["upstream_model"],
        "messages": _to_openai_messages(req),
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }
    if req.top_p != 1.0:
        body["top_p"] = req.top_p
    if req.stop_sequences:
        body["stop"] = req.stop_sequences
    if req.tools:
        body["tools"] = _convert_to_dict(req.tools)
    if req.tool_choice:
        body["tool_choice"] = _convert_to_dict(req.tool_choice)
    # Handle reasoning_effort: support both {"reasoning_effort": "high"} and {"output_config": {"effort": "high"}}
    # For OpenAI, convert output_config.effort to reasoning_effort
    if req.output_config and "effort" in req.output_config:
        body["reasoning_effort"] = req.output_config["effort"]
    elif req.reasoning_effort:
        body["reasoning_effort"] = req.reasoning_effort

    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()


async def stream_openai(req: ClaudeRequest, upstream: dict) -> AsyncGenerator[str, None]:
    url = f"{upstream['base_url']}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {upstream['api_key']}",
        "content-type": "application/json",
    }
    body = {
        "model": upstream["upstream_model"],
        "messages": _to_openai_messages(req),
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": True,
    }
    if req.top_p != 1.0:
        body["top_p"] = req.top_p
    if req.stop_sequences:
        body["stop"] = req.stop_sequences
    if req.tools:
        body["tools"] = _convert_to_dict(req.tools)
    if req.tool_choice:
        body["tool_choice"] = _convert_to_dict(req.tool_choice)
    # Handle reasoning_effort: support both {"reasoning_effort": "high"} and {"output_config": {"effort": "high"}}
    # For OpenAI, convert output_config.effort to reasoning_effort
    if req.output_config and "effort" in req.output_config:
        body["reasoning_effort"] = req.output_config["effort"]
    elif req.reasoning_effort:
        body["reasoning_effort"] = req.reasoning_effort

    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        async with client.stream("POST", url, headers=headers, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                yield line + "\n\n"


# ── Format converters ────────────────────────────────────────────

def _extract_content(content) -> str:
    """Extract text from content which may be string or [{type, text/thinking}] array."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "text":
                    texts.append(item.get("text", ""))
                elif item_type == "thinking":
                    texts.append(item.get("thinking", ""))
                elif item_type == "tool_result" and isinstance(item.get("content"), str):
                    texts.append(item.get("content", ""))
        return "".join(texts)
    return str(content)


def _extract_system(system) -> str | None:
    """Extract system text from system which may be string or [{type, text, cache_control}] array."""
    if system is None:
        return None
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        texts = []
        for item in system:
            if isinstance(item, dict):
                texts.append(item.get("text", ""))
        return "".join(texts)
    return str(system)


def _to_claude_messages(req: ClaudeRequest) -> list[dict]:
    messages = []
    for msg in req.messages:
        if msg.role == "system":
            continue  # system handled separately
        # Convert content to dict format for JSON serialization
        if isinstance(msg.content, list):
            content = [_convert_to_dict(c) for c in msg.content]
        else:
            content = msg.content
        messages.append({"role": msg.role, "content": content})
    return messages


def _to_openai_messages(req: ClaudeRequest) -> list[dict]:
    messages = []
    for msg in req.messages:
        # OpenAI expects string content, extract from Claude's array format if needed
        content = _extract_content(msg.content)
        messages.append({"role": msg.role, "content": content})
    
    # Add system message if present
    if req.system:
        system_text = _extract_system(req.system)
        if system_text:
            messages.insert(0, {"role": "system", "content": system_text})
    
    return messages


