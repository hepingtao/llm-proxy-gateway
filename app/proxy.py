import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from .models import ClaudeRequest, OpenAIRequest, RequestFormat, ResponsesRequest


def _dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    if isinstance(obj, list):
        return [_dump(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _dump(value) for key, value in obj.items() if value is not None}
    return obj


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _now() -> int:
    return int(time.time())


def _response_id(prefix: str = "resp") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            block = _dump(block)
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in ("text", "input_text", "output_text"):
                parts.append(block.get("text") or "")
            elif block_type == "thinking":
                parts.append(block.get("thinking") or "")
            elif block_type == "tool_result":
                parts.append(_extract_text(block.get("content")))
        return "".join(parts)
    return str(content)


def _extract_system(system: Any) -> str | None:
    text = _extract_text(system)
    return text or None


def _normalize_stop(stop: str | list[str] | None) -> list[str] | None:
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return stop


def _finish_to_stop_reason(finish_reason: str | None) -> str:
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "stop_sequence",
    }.get(finish_reason or "stop", finish_reason or "end_turn")


def _stop_reason_to_finish(stop_reason: str | None) -> str:
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }.get(stop_reason or "end_turn", stop_reason or "stop")


def _openai_tool_to_anthropic(tool: dict) -> dict:
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        function = tool["function"]
        return {
            "name": function.get("name", ""),
            "description": function.get("description", ""),
            "input_schema": function.get("parameters", {"type": "object", "properties": {}}),
        }
    return tool


def _anthropic_tool_to_openai(tool: dict) -> dict:
    if "function" in tool:
        return tool
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


def _responses_tool_to_chat(tool: dict) -> dict:
    if tool.get("type") == "function" and "function" not in tool:
        return {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        }
    return tool


def _chat_tool_to_responses(tool: dict) -> dict:
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        function = tool["function"]
        return {
            "type": "function",
            "name": function.get("name", ""),
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {"type": "object", "properties": {}}),
        }
    return tool


def _openai_tool_choice_to_anthropic(tool_choice: str | dict | None) -> dict | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "none":
            return {"type": "none"}
        if tool_choice == "required":
            return {"type": "any"}
        return {"type": "tool", "name": tool_choice}
    if tool_choice.get("type") == "function":
        function = tool_choice.get("function") or {}
        return {"type": "tool", "name": function.get("name", "")}
    return tool_choice


def _anthropic_tool_choice_to_openai(tool_choice: dict | None) -> str | dict | None:
    if tool_choice is None:
        return None
    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "none":
        return "none"
    if choice_type == "any":
        return "required"
    if choice_type == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    return tool_choice


def _anthropic_content_to_openai(content: Any) -> tuple[Any, list[dict] | None]:
    if isinstance(content, str):
        return content, None

    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in _dump(content) or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in ("text", "thinking"):
            text_parts.append(block.get("text") or block.get("thinking") or "")
        elif block_type == "tool_use":
            tool_calls.append({
                "id": block.get("id") or _response_id("call"),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": _json_dumps(block.get("input") or {}),
                },
            })

    return "".join(text_parts), (tool_calls or None)


def _openai_message_to_anthropic(message: dict) -> list[dict] | str:
    content = message.get("content")
    blocks: list[dict] = []
    if isinstance(content, list):
        for block in content:
            block_type = block.get("type") if isinstance(block, dict) else None
            if block_type in ("text", "input_text", "output_text"):
                blocks.append({"type": "text", "text": block.get("text", "")})
            elif isinstance(block, dict):
                blocks.append(block)
    elif content:
        blocks.append({"type": "text", "text": str(content)})

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        arguments = function.get("arguments") or "{}"
        try:
            tool_input = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            tool_input = {"arguments": arguments}
        blocks.append({
            "type": "tool_use",
            "id": tool_call.get("id") or _response_id("call"),
            "name": function.get("name", ""),
            "input": tool_input or {},
        })

    if not blocks:
        return ""
    return blocks if any(block.get("type") != "text" for block in blocks) else blocks[0]["text"]


def _chat_messages_to_anthropic(messages: list[Any]) -> tuple[str | None, list[dict]]:
    system_parts: list[str] = []
    anthropic_messages: list[dict] = []
    for raw_message in messages:
        message = _dump(raw_message)
        role = message.get("role", "user")
        if role == "developer":
            role = "system"
        if role == "system":
            system_parts.append(_extract_text(message.get("content")))
            continue
        if role == "tool":
            anthropic_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id", ""),
                    "content": message.get("content") or "",
                }],
            })
            continue
        if role not in ("user", "assistant"):
            role = "user"
        anthropic_messages.append({"role": role, "content": _openai_message_to_anthropic(message)})
    return ("\n\n".join(part for part in system_parts if part) or None), anthropic_messages


def _anthropic_messages_to_chat(messages: list[Any], system: Any = None) -> list[dict]:
    chat_messages: list[dict] = []
    system_text = _extract_system(system)
    if system_text:
        chat_messages.append({"role": "system", "content": system_text})

    for raw_message in messages:
        message = _dump(raw_message)
        role = message.get("role", "user")
        content, tool_calls = _anthropic_content_to_openai(message.get("content"))
        if role == "assistant" and tool_calls:
            chat_message = {"role": "assistant", "content": content or None, "tool_calls": tool_calls}
        else:
            chat_message = {"role": role, "content": content}
        chat_messages.append(chat_message)
    return chat_messages


def _responses_input_to_chat_messages(req: ResponsesRequest) -> list[dict]:
    messages: list[dict] = []
    if req.instructions:
        messages.append({"role": "system", "content": req.instructions})

    if isinstance(req.input, str):
        messages.append({"role": "user", "content": req.input})
        return messages

    for item in req.input or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            role = item.get("role", "user")
            content = item.get("content", "")
            messages.append({"role": role, "content": _extract_text(content)})
        elif item_type == "function_call_output":
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id") or item.get("id", ""),
                "content": item.get("output", ""),
            })
        elif item_type == "function_call":
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item.get("call_id") or item.get("id") or _response_id("call"),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                }],
            })
        else:
            messages.append({"role": item.get("role", "user"), "content": _extract_text(item.get("content", item))})
    return messages


def _base_chat_body(req: OpenAIRequest, upstream: dict) -> dict:
    body = req.model_dump(exclude_none=True, exclude={"model"})
    body["model"] = upstream["upstream_model"]
    if upstream.get("reasoning_effort") and "reasoning_effort" not in body:
        body["reasoning_effort"] = upstream["reasoning_effort"]
    return body


def _base_anthropic_body(req: ClaudeRequest, upstream: dict) -> dict:
    body = req.model_dump(exclude_none=True, exclude={"model"})
    body["model"] = upstream["upstream_model"]
    return body


def _anthropic_to_chat_body(req: ClaudeRequest, upstream: dict) -> dict:
    body: dict[str, Any] = {
        "model": upstream["upstream_model"],
        "messages": _anthropic_messages_to_chat(req.messages, req.system),
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }
    if req.top_p != 1.0:
        body["top_p"] = req.top_p
    if req.stop_sequences:
        body["stop"] = req.stop_sequences
    if req.tools:
        body["tools"] = [_anthropic_tool_to_openai(tool) for tool in _dump(req.tools)]
    if req.tool_choice:
        body["tool_choice"] = _anthropic_tool_choice_to_openai(_dump(req.tool_choice))
    if req.output_config and "effort" in req.output_config:
        body["reasoning_effort"] = req.output_config["effort"]
    elif req.reasoning_effort:
        body["reasoning_effort"] = req.reasoning_effort
    elif upstream.get("reasoning_effort"):
        body["reasoning_effort"] = upstream["reasoning_effort"]
    return body


def _chat_to_anthropic_body(req: OpenAIRequest, upstream: dict) -> dict:
    system, messages = _chat_messages_to_anthropic(req.messages)
    body: dict[str, Any] = {
        "model": upstream["upstream_model"],
        "messages": messages,
        "max_tokens": req.max_tokens or 1024,
        "temperature": req.temperature,
    }
    if system:
        body["system"] = system
    if req.top_p != 1.0:
        body["top_p"] = req.top_p
    if req.stop:
        body["stop_sequences"] = _normalize_stop(req.stop)
    if req.tools:
        body["tools"] = [_openai_tool_to_anthropic(tool) for tool in req.tools]
    if req.tool_choice:
        body["tool_choice"] = _openai_tool_choice_to_anthropic(req.tool_choice)
    if req.reasoning_effort:
        body["output_config"] = {"effort": req.reasoning_effort}
    elif upstream.get("reasoning_effort"):
        body["output_config"] = {"effort": upstream["reasoning_effort"]}
    return body


def _responses_to_chat_body(req: ResponsesRequest, upstream: dict) -> dict:
    body: dict[str, Any] = {
        "model": upstream["upstream_model"],
        "messages": _responses_input_to_chat_messages(req),
    }
    if req.max_output_tokens is not None:
        body["max_tokens"] = req.max_output_tokens
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.stream:
        body["stream"] = True
    if req.reasoning_effort:
        body["reasoning_effort"] = req.reasoning_effort
    elif upstream.get("reasoning_effort"):
        body["reasoning_effort"] = upstream["reasoning_effort"]
    if req.tools:
        body["tools"] = [_responses_tool_to_chat(tool) for tool in req.tools]
    if req.tool_choice:
        body["tool_choice"] = req.tool_choice
    return body


def _responses_to_anthropic_body(req: ResponsesRequest, upstream: dict) -> dict:
    chat_req = OpenAIRequest(**_responses_to_chat_body(req, {"upstream_model": req.model or upstream["upstream_model"]}))
    return _chat_to_anthropic_body(chat_req, upstream)


def _request_to_body(req: ClaudeRequest | OpenAIRequest | ResponsesRequest, source_format: str, target_format: str, upstream: dict) -> dict:
    if source_format == target_format == RequestFormat.ANTHROPIC_MESSAGES.value:
        return _base_anthropic_body(req, upstream)  # type: ignore[arg-type]
    if source_format == target_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value:
        return _base_chat_body(req, upstream)  # type: ignore[arg-type]
    if source_format == target_format == RequestFormat.OPENAI_RESPONSES.value:
        body = req.model_dump(exclude_none=True, exclude={"model"})  # type: ignore[union-attr]
        body["model"] = upstream["upstream_model"]
        return body

    if source_format == RequestFormat.ANTHROPIC_MESSAGES.value and target_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value:
        return _anthropic_to_chat_body(req, upstream)  # type: ignore[arg-type]
    if source_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value and target_format == RequestFormat.ANTHROPIC_MESSAGES.value:
        return _chat_to_anthropic_body(req, upstream)  # type: ignore[arg-type]
    if source_format == RequestFormat.OPENAI_RESPONSES.value and target_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value:
        return _responses_to_chat_body(req, upstream)  # type: ignore[arg-type]
    if source_format == RequestFormat.OPENAI_RESPONSES.value and target_format == RequestFormat.ANTHROPIC_MESSAGES.value:
        return _responses_to_anthropic_body(req, upstream)  # type: ignore[arg-type]

    if target_format == RequestFormat.OPENAI_RESPONSES.value:
        body = _request_to_body(req, source_format, RequestFormat.OPENAI_CHAT_COMPLETIONS.value, upstream)
        response_body = _chat_body_to_responses_body(body)
        response_body["model"] = upstream["upstream_model"]
        return response_body

    raise ValueError(f"Unsupported conversion: {source_format} -> {target_format}")


def _chat_body_to_responses_body(body: dict) -> dict:
    input_items: list[dict] = []
    instructions: str | None = None
    for message in body.get("messages") or []:
        role = message.get("role")
        if role == "system":
            instructions = (instructions + "\n\n" if instructions else "") + _extract_text(message.get("content"))
            continue
        input_items.append({
            "type": "message",
            "role": role,
            "content": [{"type": "input_text", "text": _extract_text(message.get("content"))}],
        })

    response_body: dict[str, Any] = {"input": input_items}
    if instructions:
        response_body["instructions"] = instructions
    if body.get("max_tokens") is not None:
        response_body["max_output_tokens"] = body["max_tokens"]
    for key in ("temperature", "top_p", "stream", "reasoning_effort", "tools", "tool_choice"):
        if body.get(key) is not None:
            response_body[key] = [_chat_tool_to_responses(tool) for tool in body[key]] if key == "tools" else body[key]
    return response_body


async def _post_json(url: str, headers: dict, body: dict) -> dict:
    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()


def _anthropic_headers(upstream: dict) -> dict:
    return {
        "x-api-key": upstream["api_key"],
        "anthropic-version": upstream.get("api_version") or "2023-06-01",
        "content-type": "application/json",
    }


def _openai_headers(upstream: dict) -> dict:
    return {
        "Authorization": f"Bearer {upstream['api_key']}",
        "content-type": "application/json",
    }


async def call_anthropic_passthrough(req: ClaudeRequest, upstream: dict) -> dict:
    body = _base_anthropic_body(req, upstream)
    return await _post_json(f"{upstream['base_url']}/v1/messages", _anthropic_headers(upstream), body)


async def call_openai_chat_passthrough(req: OpenAIRequest, upstream: dict) -> dict:
    body = _base_chat_body(req, upstream)
    return await _post_json(f"{upstream['base_url']}/v1/chat/completions", _openai_headers(upstream), body)


async def call_openai_responses_passthrough(req: ResponsesRequest, upstream: dict) -> dict:
    body = req.model_dump(exclude_none=True, exclude={"model"})
    body["model"] = upstream["upstream_model"]
    return await _post_json(f"{upstream['base_url']}/v1/responses", _openai_headers(upstream), body)


async def call_with_conversion(req: ClaudeRequest | OpenAIRequest | ResponsesRequest, source_format: str, actual_format: str, upstream: dict) -> dict:
    body = _request_to_body(req, source_format, actual_format, upstream)
    if actual_format == RequestFormat.ANTHROPIC_MESSAGES.value:
        raw = await _post_json(f"{upstream['base_url']}/v1/messages", _anthropic_headers(upstream), body)
    elif actual_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value:
        raw = await _post_json(f"{upstream['base_url']}/v1/chat/completions", _openai_headers(upstream), body)
    elif actual_format == RequestFormat.OPENAI_RESPONSES.value:
        raw = await _post_json(f"{upstream['base_url']}/v1/responses", _openai_headers(upstream), body)
    else:
        raise ValueError(f"Unsupported upstream format: {actual_format}")
    return _response_to_format(raw, actual_format, source_format, getattr(req, "model", "") or upstream["upstream_model"])


def _chat_to_anthropic_response(resp: dict, model: str) -> dict:
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = _openai_message_to_anthropic(message)
    if isinstance(content, str):
        content = [{"type": "text", "text": content}] if content else []
    usage = resp.get("usage") or {}
    return {
        "id": resp.get("id") or _response_id("msg"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _finish_to_stop_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
        },
    }


def _anthropic_to_chat_response(resp: dict, model: str) -> dict:
    content, tool_calls = _anthropic_content_to_openai(resp.get("content") or [])
    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = resp.get("usage") or {}
    return {
        "id": resp.get("id") or _response_id("chatcmpl"),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": _stop_reason_to_finish(resp.get("stop_reason")),
        }],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
            "total_tokens": usage.get("total_tokens", usage.get("input_tokens", 0) + usage.get("output_tokens", 0)),
        },
    }


def _chat_to_responses_response(resp: dict, model: str) -> dict:
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output: list[dict] = []
    content = message.get("content")
    if content:
        output.append({
            "id": _response_id("msg"),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": _extract_text(content), "annotations": []}],
        })
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        output.append({
            "id": tool_call.get("id") or _response_id("fc"),
            "type": "function_call",
            "status": "completed",
            "call_id": tool_call.get("id") or _response_id("call"),
            "name": function.get("name", ""),
            "arguments": function.get("arguments", "{}"),
        })
    usage = resp.get("usage") or {}
    return {
        "id": resp.get("id") or _response_id("resp"),
        "object": "response",
        "created_at": resp.get("created", _now()),
        "status": "completed",
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


def _anthropic_to_responses_response(resp: dict, model: str) -> dict:
    return _chat_to_responses_response(_anthropic_to_chat_response(resp, model), model)


def _responses_to_chat_response(resp: dict, model: str) -> dict:
    content_parts: list[str] = []
    tool_calls: list[dict] = []
    for item in resp.get("output") or []:
        if item.get("type") == "message":
            content_parts.append(_extract_text(item.get("content")))
        elif item.get("type") == "function_call":
            tool_calls.append({
                "id": item.get("call_id") or item.get("id") or _response_id("call"),
                "type": "function",
                "function": {"name": item.get("name", ""), "arguments": item.get("arguments", "{}")},
            })
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = resp.get("usage") or {}
    return {
        "id": resp.get("id") or _response_id("chatcmpl"),
        "object": "chat.completion",
        "created": int(resp.get("created_at") or _now()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


def _responses_to_anthropic_response(resp: dict, model: str) -> dict:
    return _chat_to_anthropic_response(_responses_to_chat_response(resp, model), model)


def _response_to_format(resp: dict, actual_format: str, target_format: str, model: str) -> dict:
    if actual_format == target_format:
        return resp
    if actual_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value and target_format == RequestFormat.ANTHROPIC_MESSAGES.value:
        return _chat_to_anthropic_response(resp, model)
    if actual_format == RequestFormat.ANTHROPIC_MESSAGES.value and target_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value:
        return _anthropic_to_chat_response(resp, model)
    if actual_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value and target_format == RequestFormat.OPENAI_RESPONSES.value:
        return _chat_to_responses_response(resp, model)
    if actual_format == RequestFormat.ANTHROPIC_MESSAGES.value and target_format == RequestFormat.OPENAI_RESPONSES.value:
        return _anthropic_to_responses_response(resp, model)
    if actual_format == RequestFormat.OPENAI_RESPONSES.value and target_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value:
        return _responses_to_chat_response(resp, model)
    if actual_format == RequestFormat.OPENAI_RESPONSES.value and target_format == RequestFormat.ANTHROPIC_MESSAGES.value:
        return _responses_to_anthropic_response(resp, model)
    raise ValueError(f"Unsupported response conversion: {actual_format} -> {target_format}")


async def _stream_lines(url: str, headers: dict, body: dict) -> AsyncGenerator[str, None]:
    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        async with client.stream("POST", url, headers=headers, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    yield line


async def stream_anthropic_passthrough(req: ClaudeRequest, upstream: dict) -> AsyncGenerator[str, None]:
    body = _base_anthropic_body(req, upstream)
    body["stream"] = True
    async for line in _stream_lines(f"{upstream['base_url']}/v1/messages", _anthropic_headers(upstream), body):
        yield line + "\n"


async def stream_openai_chat_passthrough(req: OpenAIRequest, upstream: dict) -> AsyncGenerator[str, None]:
    body = _base_chat_body(req, upstream)
    body["stream"] = True
    async for line in _stream_lines(f"{upstream['base_url']}/v1/chat/completions", _openai_headers(upstream), body):
        yield line + "\n"


async def stream_with_conversion(req: ClaudeRequest | OpenAIRequest | ResponsesRequest, source_format: str, actual_format: str, upstream: dict) -> AsyncGenerator[str, None]:
    body = _request_to_body(req, source_format, actual_format, upstream)
    body["stream"] = True
    if actual_format == RequestFormat.ANTHROPIC_MESSAGES.value:
        url = f"{upstream['base_url']}/v1/messages"
        headers = _anthropic_headers(upstream)
    elif actual_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value:
        url = f"{upstream['base_url']}/v1/chat/completions"
        headers = _openai_headers(upstream)
    elif actual_format == RequestFormat.OPENAI_RESPONSES.value:
        url = f"{upstream['base_url']}/v1/responses"
        headers = _openai_headers(upstream)
    else:
        raise ValueError(f"Unsupported upstream format: {actual_format}")

    if actual_format == source_format:
        async for line in _stream_lines(url, headers, body):
            yield line + "\n"
        return

    async for chunk in _convert_stream(_stream_lines(url, headers, body), actual_format, source_format, getattr(req, "model", "") or upstream["upstream_model"]):
        yield chunk


async def _convert_stream(lines: AsyncGenerator[str, None], actual_format: str, target_format: str, model: str) -> AsyncGenerator[str, None]:
    if target_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value:
        async for chunk in _stream_to_chat(lines, actual_format, model):
            yield chunk
    elif target_format == RequestFormat.ANTHROPIC_MESSAGES.value:
        async for chunk in _stream_to_anthropic(lines, actual_format, model):
            yield chunk
    elif target_format == RequestFormat.OPENAI_RESPONSES.value:
        async for chunk in _stream_to_responses(lines, actual_format, model):
            yield chunk
    else:
        raise ValueError(f"Unsupported stream target: {target_format}")


def _parse_sse_data(line: str) -> dict | None:
    if not line.startswith("data: "):
        return None
    data = line[6:]
    if data == "[DONE]":
        return {"__done__": True}
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def _chat_delta_text(event: dict) -> str:
    return ((event.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""


def _anthropic_delta_text(event: dict) -> str:
    if event.get("type") == "content_block_delta":
        delta = event.get("delta") or {}
        if delta.get("type") == "text_delta":
            return delta.get("text") or ""
    return ""


def _responses_delta_text(event: dict) -> str:
    if event.get("type") == "response.output_text.delta":
        return event.get("delta") or ""
    return ""


def _extract_stream_text(event: dict, actual_format: str) -> str:
    if actual_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value:
        return _chat_delta_text(event)
    if actual_format == RequestFormat.ANTHROPIC_MESSAGES.value:
        return _anthropic_delta_text(event)
    if actual_format == RequestFormat.OPENAI_RESPONSES.value:
        return _responses_delta_text(event)
    return ""


async def _stream_to_chat(lines: AsyncGenerator[str, None], actual_format: str, model: str) -> AsyncGenerator[str, None]:
    chunk_id = _response_id("chatcmpl")
    async for line in lines:
        event = _parse_sse_data(line)
        if not event:
            continue
        if event.get("__done__"):
            break
        text = _extract_stream_text(event, actual_format)
        if text:
            payload = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": _now(),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            yield f"data: {_json_dumps(payload)}\n\n"
    done = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {_json_dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_to_anthropic(lines: AsyncGenerator[str, None], actual_format: str, model: str) -> AsyncGenerator[str, None]:
    msg_id = _response_id("msg")
    yield f"event: message_start\ndata: {_json_dumps({'type':'message_start','message':{'id':msg_id,'type':'message','role':'assistant','model':model,'content':[],'stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':0,'output_tokens':0}}})}\n\n"
    yield f"event: content_block_start\ndata: {_json_dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"
    output_tokens = 0
    async for line in lines:
        event = _parse_sse_data(line)
        if not event:
            continue
        if event.get("__done__"):
            break
        text = _extract_stream_text(event, actual_format)
        if text:
            output_tokens += 1
            payload = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
            yield f"event: content_block_delta\ndata: {_json_dumps(payload)}\n\n"
    yield f"event: content_block_stop\ndata: {_json_dumps({'type':'content_block_stop','index':0})}\n\n"
    yield f"event: message_delta\ndata: {_json_dumps({'type':'message_delta','delta':{'stop_reason':'end_turn','stop_sequence':None},'usage':{'output_tokens':output_tokens}})}\n\n"
    yield f"event: message_stop\ndata: {_json_dumps({'type':'message_stop'})}\n\n"


async def _stream_to_responses(lines: AsyncGenerator[str, None], actual_format: str, model: str) -> AsyncGenerator[str, None]:
    resp_id = _response_id("resp")
    item_id = _response_id("msg")
    created = {
        "type": "response.created",
        "response": {"id": resp_id, "object": "response", "created_at": _now(), "status": "in_progress", "model": model, "output": []},
    }
    yield f"event: response.created\ndata: {_json_dumps(created)}\n\n"
    added = {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {"id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []},
    }
    yield f"event: response.output_item.added\ndata: {_json_dumps(added)}\n\n"
    async for line in lines:
        event = _parse_sse_data(line)
        if not event:
            continue
        if event.get("__done__"):
            break
        text = _extract_stream_text(event, actual_format)
        if text:
            payload = {"type": "response.output_text.delta", "item_id": item_id, "output_index": 0, "content_index": 0, "delta": text}
            yield f"event: response.output_text.delta\ndata: {_json_dumps(payload)}\n\n"
    done = {"type": "response.output_item.done", "output_index": 0, "item": {"id": item_id, "type": "message", "status": "completed", "role": "assistant"}}
    yield f"event: response.output_item.done\ndata: {_json_dumps(done)}\n\n"
    completed = {"type": "response.completed", "response": {"id": resp_id, "object": "response", "status": "completed", "model": model}}
    yield f"event: response.completed\ndata: {_json_dumps(completed)}\n\n"
