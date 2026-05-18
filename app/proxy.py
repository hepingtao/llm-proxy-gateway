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


def _extract_reasoning_content(item: dict) -> str:
    reasoning = item.get("reasoning_content") or item.get("reasoning") or ""
    if reasoning:
        return str(reasoning)

    parts: list[str] = []
    for summary in item.get("summary") or []:
        if isinstance(summary, dict):
            parts.append(summary.get("text") or "")
    return "".join(parts)


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
            "input_schema": function.get(
                "parameters", {"type": "object", "properties": {}}
            ),
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
            "parameters": tool.get(
                "input_schema", {"type": "object", "properties": {}}
            ),
        },
    }


def _responses_tool_to_chat(tool: dict) -> dict | None:
    """Convert a single Responses-API tool to Chat-Completions format.

    Returns None for tool types that OpenAI-compatible upstreams cannot handle
    (e.g. ``web_search``, ``image_generation``, ``custom``).  ``namespace``
    tools are NOT handled here — use ``_responses_tools_to_chat`` which
    recursively expands their sub-tools.
    """
    tool_type = tool.get("type")
    if tool_type == "function":
        if "function" not in tool:
            # Flat Responses-API format → wrap into Chat-Completions format
            return {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                },
            }
        # Already in Chat-Completions nested format
        return tool
    # All other types (custom, web_search, image_generation, namespace, …)
    # are Codex-specific and not supported by standard OpenAI-compatible APIs.
    return None


def _responses_tools_to_chat(tools: list[dict]) -> list[dict]:
    """Convert a list of Responses-API tools to Chat-Completions format.

    * ``function`` tools are wrapped into ``{type, function: {…}}`` form.
    * ``namespace`` tools are expanded: their nested sub-tools are processed
      recursively so callers in MCP namespaces remain available.
    * All other types (``web_search``, ``image_generation``, ``custom``, …)
      are silently dropped because standard OpenAI-compatible upstreams do
      not understand them.
    """
    result: list[dict] = []
    for tool in tools:
        if tool.get("type") == "namespace":
            # Expand sub-tools from the namespace group
            sub_tools = tool.get("tools") or []
            result.extend(_responses_tools_to_chat(sub_tools))
        else:
            converted = _responses_tool_to_chat(tool)
            if converted is not None:
                result.append(converted)
    return result


def _chat_tool_to_responses(tool: dict) -> dict:
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        function = tool["function"]
        return {
            "type": "function",
            "name": function.get("name", ""),
            "description": function.get("description", ""),
            "parameters": function.get(
                "parameters", {"type": "object", "properties": {}}
            ),
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
            tool_calls.append(
                {
                    "id": block.get("id") or _response_id("call"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": _json_dumps(block.get("input") or {}),
                    },
                }
            )

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
            tool_input = (
                json.loads(arguments) if isinstance(arguments, str) else arguments
            )
        except json.JSONDecodeError:
            tool_input = {"arguments": arguments}
        blocks.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id") or _response_id("call"),
                "name": function.get("name", ""),
                "input": tool_input or {},
            }
        )

    if not blocks:
        return ""
    return (
        blocks
        if any(block.get("type") != "text" for block in blocks)
        else blocks[0]["text"]
    )


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
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id", ""),
                            "content": message.get("content") or "",
                        }
                    ],
                }
            )
            continue
        if role not in ("user", "assistant"):
            role = "user"
        anthropic_messages.append(
            {"role": role, "content": _openai_message_to_anthropic(message)}
        )
    return (
        "\n\n".join(part for part in system_parts if part) or None
    ), anthropic_messages


def _anthropic_messages_to_chat(messages: list[Any], system: Any = None) -> list[dict]:
    chat_messages: list[dict] = []
    system_text = _extract_system(system)
    if system_text:
        chat_messages.append({"role": "system", "content": system_text})

    for raw_message in messages:
        message = _dump(raw_message)
        role = message.get("role", "user")
        content_blocks = message.get("content")
        if role == "user" and isinstance(content_blocks, list):
            text_parts: list[str] = []
            tool_messages: list[dict] = []
            for block in content_blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": _extract_text(block.get("content")),
                        }
                    )
                elif block.get("type") in ("text", "input_text", "output_text"):
                    text_parts.append(block.get("text") or "")
            if tool_messages:
                text = "".join(text_parts)
                if text:
                    chat_messages.append({"role": role, "content": text})
                chat_messages.extend(tool_messages)
                continue

        content, tool_calls = _anthropic_content_to_openai(message.get("content"))
        if role == "assistant" and tool_calls:
            chat_message = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": tool_calls,
            }
        else:
            chat_message = {"role": role, "content": content}
        chat_messages.append(chat_message)
    return chat_messages


def _responses_content_to_chat(content: Any) -> str | list[dict]:
    """Convert Responses API message content to Chat Completions content.

    - input_text → text
    - input_image → image_url
    - other text-like blocks → text
    If only text blocks are present, returns a plain string.
    If image blocks are present, returns a content list.
    When images are present, always includes at least one ``text`` block —
    some providers (e.g. mimo) reject image-only content lists.
    """
    if not isinstance(content, list):
        return _extract_text(content)

    has_image = False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "input_image":
            has_image = True
            break

    if not has_image:
        return _extract_text(content)

    # Build a content list with both text and image_url parts
    parts: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in ("text", "input_text", "output_text"):
            text = block.get("text") or ""
            parts.append({"type": "text", "text": text})
        elif block_type == "input_image":
            # Responses API: {"type": "input_image", "image_url": {"url": "..."} or "data:..."}
            # Chat Completions:  {"type": "image_url", "image_url": {"url": "..."}}
            img_url = block.get("image_url")
            if img_url:
                # image_url may be a string (data URL) or a dict {"url": "..."}
                if isinstance(img_url, str):
                    img_url = {"url": img_url}
                parts.append({"type": "image_url", "image_url": img_url})

    # Ensure at least one text block exists alongside image_url blocks —
    # providers like mimo reject content lists that have only image_url entries.
    has_text = any(p.get("type") == "text" for p in parts if isinstance(p, dict))
    if not has_text and parts:
        parts.insert(0, {"type": "text", "text": ""})

    return parts or ""


def _responses_input_to_chat_messages(req: ResponsesRequest) -> list[dict]:
    messages: list[dict] = []
    system_parts: list[str] = []
    if req.instructions:
        system_parts.append(req.instructions)

    if isinstance(req.input, str):
        messages.append({"role": "user", "content": req.input})
        return messages

    pending_reasoning = ""
    for item in req.input or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "reasoning":
            pending_reasoning += _extract_reasoning_content(item)
            continue
        if item_type == "message":
            role = item.get("role", "user")
            if role == "developer":
                system_parts.append(_extract_text(item.get("content", "")))
                continue
            if role not in ("user", "assistant", "system", "tool"):
                role = "user"
            content = item.get("content", "")
            chat_content = _responses_content_to_chat(content)
            # For assistant messages with no actual text, use None instead of empty string
            if role == "assistant" and chat_content == "":
                chat_content = None
            message = {"role": role, "content": chat_content}
            if role == "assistant" and pending_reasoning:
                message["reasoning_content"] = pending_reasoning
                pending_reasoning = ""
            messages.append(message)
        elif item_type == "function_call_output":
            output = item.get("output", "")
            # output may contain input_image blocks (e.g. from screenshot tools)
            if isinstance(output, list):
                output = _responses_content_to_chat(output)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or item.get("id", ""),
                    "content": output,
                }
            )
        elif item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    **(
                        {"reasoning_content": pending_reasoning}
                        if pending_reasoning
                        else {}
                    ),
                    "tool_calls": [
                        {
                            "id": item.get("call_id")
                            or item.get("id")
                            or _response_id("call"),
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "{}"),
                            },
                        }
                    ],
                }
            )
            pending_reasoning = ""
        elif item_type == "input_image":
            # Top-level input_image item → convert to user message with image_url
            img_url = item.get("image_url")
            if img_url:
                messages.append({
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": img_url}],
                })
        else:
            extracted = _extract_text(item.get("content", item))
            messages.append(
                {
                    "role": item.get("role", "user"),
                    "content": extracted or None,
                }
            )

    # Prepend accumulated system content (from instructions + developer messages)
    # at the beginning, since Chat Completions requires system messages first.
    if system_parts:
        system_text = "\n\n".join(part for part in system_parts if part)
        messages.insert(0, {"role": "system", "content": system_text})

    return messages


def _upstream_requires_reasoning_content(upstream: dict) -> bool:
    provider = upstream.get("provider_name") or upstream.get("provider") or ""
    return str(provider).lower() in {"mimo"}


def _add_missing_reasoning_content(messages: list[dict], upstream: dict) -> None:
    if not _upstream_requires_reasoning_content(upstream):
        return
    for message in messages:
        if message.get("role") == "assistant" and "reasoning_content" not in message:
            message["reasoning_content"] = ""


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
    messages = _responses_input_to_chat_messages(req)
    _add_missing_reasoning_content(messages, upstream)
    body: dict[str, Any] = {
        "model": upstream["upstream_model"],
        "messages": messages,
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
        converted_tools = _responses_tools_to_chat(req.tools)
        if converted_tools:
            body["tools"] = converted_tools
    if req.tool_choice:
        body["tool_choice"] = req.tool_choice
    return body


def _responses_to_anthropic_body(req: ResponsesRequest, upstream: dict) -> dict:
    chat_req = OpenAIRequest(
        **_responses_to_chat_body(
            req, {"upstream_model": req.model or upstream["upstream_model"]}
        )
    )
    return _chat_to_anthropic_body(chat_req, upstream)


def _request_has_images(req: ClaudeRequest | OpenAIRequest | ResponsesRequest, source_format: str) -> bool:
    """Check whether a request contains image content.

    Detects ``image_url`` (OpenAI), ``input_image`` (Responses), and ``image``
    (Anthropic) blocks inside message content.
    """
    # Collect all content payloads to scan
    contents: list[Any] = []

    if isinstance(req, ResponsesRequest):
        inp = req.input
        if isinstance(inp, str):
            return False
        for item in inp or []:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "input_image":
                return True
            if item_type == "message":
                contents.append(item.get("content"))
            elif item_type == "function_call_output":
                output = item.get("output")
                if isinstance(output, list):
                    for block in output:
                        if isinstance(block, dict) and block.get("type") == "input_image":
                            return True
    elif isinstance(req, OpenAIRequest):
        for msg in req.messages or []:
            contents.append(msg.content if hasattr(msg, "content") else msg.get("content"))
    elif isinstance(req, ClaudeRequest):
        for msg in req.messages or []:
            contents.append(msg.content if hasattr(msg, "content") else msg.get("content"))
        if req.system:
            contents.append(req.system)

    for content in contents:
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in ("image_url", "input_image", "image"):
                return True

    return False


def _normalize_content_for_upstream(body: dict, supports_vision: bool) -> dict:
    """Normalize message content for upstream compatibility.

    * When ``supports_vision`` is False: strips ``image_url`` blocks (OpenAI format)
      and ``image`` blocks (Anthropic format) from message content.  If only text
      blocks remain, collapses content to a plain string.

    * When ``supports_vision`` is True: ensures every message that contains
      ``image_url`` / ``image`` blocks also has at least one ``text`` block —
      some providers (e.g. mimo) reject image-only content lists.
    """
    messages = body.get("messages") or []

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        has_image_url = any(
            isinstance(b, dict) and b.get("type") == "image_url"
            for b in content
        )
        has_image_anthropic = any(
            isinstance(b, dict) and b.get("type") == "image"
            for b in content
        )

        if not has_image_url and not has_image_anthropic:
            continue  # No image blocks — nothing to do

        if not supports_vision:
            # ---- Strip image blocks ----
            text_blocks = [
                b for b in content
                if isinstance(b, dict) and b.get("type") not in ("image_url", "image")
            ]
            img_count = len(content) - len(text_blocks)
            if img_count > 0:
                print(
                    f"[proxy] STRIP {img_count} image block(s) from message "
                    f"(upstream does not support vision)",
                    flush=True,
                )
            if not text_blocks:
                msg["content"] = ""
            elif all(b.get("type") == "text" for b in text_blocks if isinstance(b, dict)):
                msg["content"] = "\n".join(
                    b.get("text", "") for b in text_blocks if isinstance(b, dict)
                )
            else:
                msg["content"] = text_blocks
        else:
            # ---- Vision-capable: ensure text block exists ----
            has_text = any(
                isinstance(b, dict) and b.get("type") == "text"
                for b in content
            )
            if not has_text:
                # Insert an empty text block so the provider doesn't reject
                content.insert(0, {"type": "text", "text": ""})

    return body


def _request_to_body(
    req: ClaudeRequest | OpenAIRequest | ResponsesRequest,
    source_format: str,
    target_format: str,
    upstream: dict,
) -> dict:
    if source_format == target_format == RequestFormat.ANTHROPIC_MESSAGES.value:
        body = _base_anthropic_body(req, upstream)  # type: ignore[arg-type]
    elif source_format == target_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value:
        body = _base_chat_body(req, upstream)  # type: ignore[arg-type]
    elif source_format == target_format == RequestFormat.OPENAI_RESPONSES.value:
        body = req.model_dump(exclude_none=True, exclude={"model"})  # type: ignore[union-attr]
        body["model"] = upstream["upstream_model"]

    elif (
        source_format == RequestFormat.ANTHROPIC_MESSAGES.value
        and target_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value
    ):
        body = _anthropic_to_chat_body(req, upstream)  # type: ignore[arg-type]
    elif (
        source_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value
        and target_format == RequestFormat.ANTHROPIC_MESSAGES.value
    ):
        body = _chat_to_anthropic_body(req, upstream)  # type: ignore[arg-type]
    elif (
        source_format == RequestFormat.OPENAI_RESPONSES.value
        and target_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value
    ):
        body = _responses_to_chat_body(req, upstream)  # type: ignore[arg-type]
    elif (
        source_format == RequestFormat.OPENAI_RESPONSES.value
        and target_format == RequestFormat.ANTHROPIC_MESSAGES.value
    ):
        body = _responses_to_anthropic_body(req, upstream)  # type: ignore[arg-type]

    elif target_format == RequestFormat.OPENAI_RESPONSES.value:
        body = _request_to_body(
            req, source_format, RequestFormat.OPENAI_CHAT_COMPLETIONS.value, upstream
        )
        response_body = _chat_body_to_responses_body(body)
        response_body["model"] = upstream["upstream_model"]
        body = response_body

    else:
        raise ValueError(f"Unsupported conversion: {source_format} -> {target_format}")

    # Normalize image content for upstream compatibility
    body = _normalize_content_for_upstream(body, upstream.get("supports_vision", False))

    return body


def _chat_body_to_responses_body(body: dict) -> dict:
    input_items: list[dict] = []
    instructions: str | None = None
    for message in body.get("messages") or []:
        role = message.get("role")
        if role in ("system", "developer"):
            instructions = (
                instructions + "\n\n" if instructions else ""
            ) + _extract_text(message.get("content"))
            continue
        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id", ""),
                    "output": _extract_text(message.get("content")),
                }
            )
            continue

        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if role == "assistant" and reasoning:
            input_items.append(
                {
                    "id": _response_id("rs"),
                    "type": "reasoning",
                    "summary": [],
                    "reasoning_content": str(reasoning),
                }
            )

        text = _extract_text(message.get("content"))
        if text or not message.get("tool_calls"):
            input_items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": [{"type": "input_text", "text": text}],
                }
            )

        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            input_items.append(
                {
                    "type": "function_call",
                    "call_id": tool_call.get("id") or _response_id("call"),
                    "name": function.get("name", ""),
                    "arguments": function.get("arguments", "{}"),
                }
            )

    response_body: dict[str, Any] = {"input": input_items}
    if instructions:
        response_body["instructions"] = instructions
    if body.get("max_tokens") is not None:
        response_body["max_output_tokens"] = body["max_tokens"]
    for key in (
        "temperature",
        "top_p",
        "stream",
        "reasoning_effort",
        "tools",
        "tool_choice",
    ):
        if body.get(key) is not None:
            response_body[key] = (
                [_chat_tool_to_responses(tool) for tool in body[key]]
                if key == "tools"
                else body[key]
            )
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
    return await _post_json(
        f"{upstream['base_url']}/v1/messages", _anthropic_headers(upstream), body
    )


async def call_openai_chat_passthrough(req: OpenAIRequest, upstream: dict) -> dict:
    body = _base_chat_body(req, upstream)
    return await _post_json(
        f"{upstream['base_url']}/v1/chat/completions", _openai_headers(upstream), body
    )


async def call_openai_responses_passthrough(
    req: ResponsesRequest, upstream: dict
) -> dict:
    body = req.model_dump(exclude_none=True, exclude={"model"})
    body["model"] = upstream["upstream_model"]
    return await _post_json(
        f"{upstream['base_url']}/v1/responses", _openai_headers(upstream), body
    )


async def call_with_conversion(
    req: ClaudeRequest | OpenAIRequest | ResponsesRequest,
    source_format: str,
    actual_format: str,
    upstream: dict,
) -> dict:
    body = _request_to_body(req, source_format, actual_format, upstream)
    if actual_format == RequestFormat.ANTHROPIC_MESSAGES.value:
        raw = await _post_json(
            f"{upstream['base_url']}/v1/messages", _anthropic_headers(upstream), body
        )
    elif actual_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value:
        raw = await _post_json(
            f"{upstream['base_url']}/v1/chat/completions",
            _openai_headers(upstream),
            body,
        )
    elif actual_format == RequestFormat.OPENAI_RESPONSES.value:
        raw = await _post_json(
            f"{upstream['base_url']}/v1/responses", _openai_headers(upstream), body
        )
    else:
        raise ValueError(f"Unsupported upstream format: {actual_format}")
    return _response_to_format(
        raw,
        actual_format,
        source_format,
        getattr(req, "model", "") or upstream["upstream_model"],
    )


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
            "output_tokens": usage.get(
                "output_tokens", usage.get("completion_tokens", 0)
            ),
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
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _stop_reason_to_finish(resp.get("stop_reason")),
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            "completion_tokens": usage.get(
                "completion_tokens", usage.get("output_tokens", 0)
            ),
            "total_tokens": usage.get(
                "total_tokens",
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            ),
        },
    }


def _chat_to_responses_response(resp: dict, model: str) -> dict:
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output: list[dict] = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        output.append(
            {
                "id": _response_id("rs"),
                "type": "reasoning",
                "status": "completed",
                "summary": [],
                "reasoning_content": str(reasoning),
            }
        )
    content = message.get("content")
    if content:
        output.append(
            {
                "id": _response_id("msg"),
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": _extract_text(content),
                        "annotations": [],
                    }
                ],
            }
        )
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        call_id = tool_call.get("id") or _response_id("call")
        output.append(
            {
                "id": call_id,
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": function.get("name", ""),
                "arguments": function.get("arguments", "{}"),
            }
        )
    usage = resp.get("usage") or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    return {
        "id": resp.get("id") or _response_id("resp"),
        "object": "response",
        "created_at": resp.get("created", _now()),
        "status": "completed",
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
        },
    }


def _anthropic_to_responses_response(resp: dict, model: str) -> dict:
    return _chat_to_responses_response(_anthropic_to_chat_response(resp, model), model)


def _responses_to_chat_response(resp: dict, model: str) -> dict:
    content_parts: list[str] = []
    tool_calls: list[dict] = []
    reasoning_content = ""
    for item in resp.get("output") or []:
        if item.get("type") == "reasoning":
            reasoning_content += _extract_reasoning_content(item)
        elif item.get("type") == "message":
            content_parts.append(_extract_text(item.get("content")))
        elif item.get("type") == "function_call":
            tool_calls.append(
                {
                    "id": item.get("call_id") or item.get("id") or _response_id("call"),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                }
            )
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts) or None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    usage = resp.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion_tokens = usage.get(
        "completion_tokens", usage.get("output_tokens", 0)
    )
    return {
        "id": resp.get("id") or _response_id("chatcmpl"),
        "object": "chat.completion",
        "created": int(resp.get("created_at") or _now()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get(
                "total_tokens", prompt_tokens + completion_tokens
            ),
        },
    }


def _responses_to_anthropic_response(resp: dict, model: str) -> dict:
    return _chat_to_anthropic_response(_responses_to_chat_response(resp, model), model)


def _response_to_format(
    resp: dict, actual_format: str, target_format: str, model: str
) -> dict:
    if actual_format == target_format:
        return resp
    if (
        actual_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value
        and target_format == RequestFormat.ANTHROPIC_MESSAGES.value
    ):
        return _chat_to_anthropic_response(resp, model)
    if (
        actual_format == RequestFormat.ANTHROPIC_MESSAGES.value
        and target_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value
    ):
        return _anthropic_to_chat_response(resp, model)
    if (
        actual_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value
        and target_format == RequestFormat.OPENAI_RESPONSES.value
    ):
        return _chat_to_responses_response(resp, model)
    if (
        actual_format == RequestFormat.ANTHROPIC_MESSAGES.value
        and target_format == RequestFormat.OPENAI_RESPONSES.value
    ):
        return _anthropic_to_responses_response(resp, model)
    if (
        actual_format == RequestFormat.OPENAI_RESPONSES.value
        and target_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value
    ):
        return _responses_to_chat_response(resp, model)
    if (
        actual_format == RequestFormat.OPENAI_RESPONSES.value
        and target_format == RequestFormat.ANTHROPIC_MESSAGES.value
    ):
        return _responses_to_anthropic_response(resp, model)
    raise ValueError(
        f"Unsupported response conversion: {actual_format} -> {target_format}"
    )


async def _stream_lines(
    url: str, headers: dict, body: dict
) -> AsyncGenerator[str, None]:
    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        async with client.stream("POST", url, headers=headers, json=body) as resp:
            if resp.status_code >= 400:
                # Read the full error body before raising so that
                # exc.response.text is available in _format_upstream_error.
                # Without this, httpx raises ResponseNotRead and the
                # upstream's error details are silently lost.
                await resp.aread()
                resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    yield line


async def stream_anthropic_passthrough(
    req: ClaudeRequest, upstream: dict
) -> AsyncGenerator[str, None]:
    body = _base_anthropic_body(req, upstream)
    body["stream"] = True
    async for line in _stream_lines(
        f"{upstream['base_url']}/v1/messages", _anthropic_headers(upstream), body
    ):
        yield line + "\n"


async def stream_openai_chat_passthrough(
    req: OpenAIRequest, upstream: dict
) -> AsyncGenerator[str, None]:
    body = _base_chat_body(req, upstream)
    body["stream"] = True
    async for line in _stream_lines(
        f"{upstream['base_url']}/v1/chat/completions", _openai_headers(upstream), body
    ):
        yield line + "\n"


async def stream_with_conversion(
    req: ClaudeRequest | OpenAIRequest | ResponsesRequest,
    source_format: str,
    actual_format: str,
    upstream: dict,
) -> AsyncGenerator[str, None]:
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

    async for chunk in _convert_stream(
        _stream_lines(url, headers, body),
        actual_format,
        source_format,
        getattr(req, "model", "") or upstream["upstream_model"],
    ):
        yield chunk


async def _convert_stream(
    lines: AsyncGenerator[str, None], actual_format: str, target_format: str, model: str
) -> AsyncGenerator[str, None]:
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


def _stream_key(index: Any = None, item_id: Any = None) -> str:
    if item_id is not None:
        return str(item_id)
    if index is not None:
        return str(index)
    return _response_id("stream")


def _stream_finish_to_anthropic(finish_reason: str | None, saw_tool: bool) -> str:
    if saw_tool:
        return "tool_use"
    return _finish_to_stop_reason(finish_reason)


async def _iter_stream_events(
    lines: AsyncGenerator[str, None], actual_format: str
) -> AsyncGenerator[dict, None]:
    if actual_format == RequestFormat.OPENAI_CHAT_COMPLETIONS.value:
        async for event in _chat_stream_events(lines):
            yield event
    elif actual_format == RequestFormat.ANTHROPIC_MESSAGES.value:
        async for event in _anthropic_stream_events(lines):
            yield event
    elif actual_format == RequestFormat.OPENAI_RESPONSES.value:
        async for event in _responses_stream_events(lines):
            yield event


async def _chat_stream_events(
    lines: AsyncGenerator[str, None],
) -> AsyncGenerator[dict, None]:
    tools: dict[str, dict[str, Any]] = {}
    finish_reason: str | None = None
    async for line in lines:
        event = _parse_sse_data(line)
        if not event:
            continue
        if event.get("__done__"):
            break

        choice = (event.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        finish_reason = choice.get("finish_reason") or finish_reason
        content = delta.get("content") or ""
        if content:
            yield {"type": "text_delta", "delta": content}

        for tool_call in delta.get("tool_calls") or []:
            index = tool_call.get("index", 0)
            key = _stream_key(index)
            state = tools.setdefault(
                key,
                {
                    "index": index,
                    "id": tool_call.get("id") or _response_id("call"),
                    "name": "",
                    "arguments": "",
                    "started": False,
                },
            )
            if tool_call.get("id"):
                state["id"] = tool_call["id"]
            function = tool_call.get("function") or {}
            if function.get("name"):
                state["name"] = function["name"]

            if not state["started"]:
                state["started"] = True
                yield {
                    "type": "tool_start",
                    "key": key,
                    "id": state["id"],
                    "name": state["name"],
                }

            arguments_delta = function.get("arguments") or ""
            if arguments_delta:
                state["arguments"] += arguments_delta
                yield {
                    "type": "tool_arguments_delta",
                    "key": key,
                    "id": state["id"],
                    "name": state["name"],
                    "delta": arguments_delta,
                }

    for key, state in tools.items():
        if state.get("started"):
            yield {
                "type": "tool_done",
                "key": key,
                "id": state["id"],
                "name": state["name"],
                "arguments": state["arguments"],
            }
    yield {"type": "message_done", "finish_reason": finish_reason}


async def _anthropic_stream_events(
    lines: AsyncGenerator[str, None],
) -> AsyncGenerator[dict, None]:
    blocks: dict[int, dict[str, Any]] = {}
    stop_reason: str | None = None
    async for line in lines:
        event = _parse_sse_data(line)
        if not event:
            continue
        event_type = event.get("type")
        index = event.get("index", 0)

        if event_type == "content_block_start":
            block = event.get("content_block") or {}
            block_type = block.get("type")
            if block_type == "tool_use":
                key = _stream_key(index)
                blocks[index] = {
                    "type": "tool_use",
                    "key": key,
                    "id": block.get("id") or _response_id("call"),
                    "name": block.get("name", ""),
                    "arguments": "",
                }
                yield {
                    "type": "tool_start",
                    "key": key,
                    "id": blocks[index]["id"],
                    "name": blocks[index]["name"],
                }
            else:
                blocks[index] = {"type": block_type or "text"}
            continue

        if event_type == "content_block_delta":
            delta = event.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text = delta.get("text") or ""
                if text:
                    yield {"type": "text_delta", "delta": text}
            elif delta_type == "input_json_delta":
                block = blocks.setdefault(
                    index,
                    {
                        "type": "tool_use",
                        "key": _stream_key(index),
                        "id": _response_id("call"),
                        "name": "",
                        "arguments": "",
                    },
                )
                partial_json = delta.get("partial_json") or ""
                block["arguments"] += partial_json
                if partial_json:
                    yield {
                        "type": "tool_arguments_delta",
                        "key": block["key"],
                        "id": block["id"],
                        "name": block["name"],
                        "delta": partial_json,
                    }
            continue

        if event_type == "content_block_stop":
            block = blocks.get(index)
            if block and block.get("type") == "tool_use":
                yield {
                    "type": "tool_done",
                    "key": block["key"],
                    "id": block["id"],
                    "name": block["name"],
                    "arguments": block["arguments"],
                }
            continue

        if event_type == "message_delta":
            delta = event.get("delta") or {}
            stop_reason = delta.get("stop_reason") or stop_reason
            continue

        if event_type == "message_stop":
            break

    yield {"type": "message_done", "finish_reason": _stop_reason_to_finish(stop_reason)}


async def _responses_stream_events(
    lines: AsyncGenerator[str, None],
) -> AsyncGenerator[dict, None]:
    tools: dict[str, dict[str, Any]] = {}
    async for line in lines:
        event = _parse_sse_data(line)
        if not event:
            continue
        event_type = event.get("type")

        if event_type == "response.output_text.delta":
            text = event.get("delta") or ""
            if text:
                yield {"type": "text_delta", "delta": text}
            continue

        if event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                key = _stream_key(event.get("output_index"), item.get("id"))
                tools[key] = {
                    "id": item.get("call_id") or item.get("id") or _response_id("call"),
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments") or "",
                }
                yield {
                    "type": "tool_start",
                    "key": key,
                    "id": tools[key]["id"],
                    "name": tools[key]["name"],
                }
            continue

        if event_type == "response.function_call_arguments.delta":
            key = _stream_key(event.get("output_index"), event.get("item_id"))
            if key not in tools:
                tools[key] = {
                    "id": event.get("item_id") or _response_id("call"),
                    "name": "",
                    "arguments": "",
                    "started": True,
                    "done": False,
                }
                yield {
                    "type": "tool_start",
                    "key": key,
                    "id": tools[key]["id"],
                    "name": "",
                }
            state = tools[key]
            delta = event.get("delta") or ""
            state["arguments"] += delta
            if delta:
                yield {
                    "type": "tool_arguments_delta",
                    "key": key,
                    "id": state["id"],
                    "name": state["name"],
                    "delta": delta,
                }
            continue

        if event_type == "response.function_call_arguments.done":
            key = _stream_key(event.get("output_index"), event.get("item_id"))
            if key not in tools:
                tools[key] = {
                    "id": event.get("item_id") or _response_id("call"),
                    "name": "",
                    "arguments": "",
                    "started": True,
                    "done": False,
                }
                yield {
                    "type": "tool_start",
                    "key": key,
                    "id": tools[key]["id"],
                    "name": "",
                }
            state = tools[key]
            state["name"] = event.get("name") or state["name"]
            state["arguments"] = event.get("arguments") or state["arguments"]
            state["done"] = True
            yield {
                "type": "tool_done",
                "key": key,
                "id": state["id"],
                "name": state["name"],
                "arguments": state["arguments"],
            }
            continue

        if event_type == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                key = _stream_key(event.get("output_index"), item.get("id"))
                if key not in tools:
                    tools[key] = {
                        "id": item.get("call_id") or item.get("id") or _response_id("call"),
                        "name": item.get("name", ""),
                        "arguments": "",
                        "started": True,
                        "done": False,
                    }
                    yield {
                        "type": "tool_start",
                        "key": key,
                        "id": tools[key]["id"],
                        "name": tools[key]["name"],
                    }
                state = tools[key]
                state["name"] = item.get("name") or state["name"]
                full_arguments = item.get("arguments") or state["arguments"]
                if not state.get("done"):
                    current_arguments = state.get("arguments") or ""
                    if full_arguments != current_arguments:
                        delta = (
                            full_arguments[len(current_arguments):]
                            if full_arguments.startswith(current_arguments)
                            else full_arguments
                        )
                        if delta:
                            yield {
                                "type": "tool_arguments_delta",
                                "key": key,
                                "id": state["id"],
                                "name": state["name"],
                                "delta": delta,
                            }
                    state["arguments"] = full_arguments
                    state["done"] = True
                    yield {
                        "type": "tool_done",
                        "key": key,
                        "id": state["id"],
                        "name": state["name"],
                        "arguments": state["arguments"],
                    }
            continue

        if event_type in ("response.completed", "response.failed", "error"):
            break

    yield {"type": "message_done", "finish_reason": None}


async def _stream_to_chat(
    lines: AsyncGenerator[str, None], actual_format: str, model: str
) -> AsyncGenerator[str, None]:
    chunk_id = _response_id("chatcmpl")
    tool_indexes: dict[str, int] = {}
    saw_tool = False
    finish_reason: str | None = None
    async for event in _iter_stream_events(lines, actual_format):
        event_type = event.get("type")
        if event_type == "message_done":
            finish_reason = event.get("finish_reason") or finish_reason
            continue
        if event_type == "text_delta":
            payload = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": _now(),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": event.get("delta") or ""},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {_json_dumps(payload)}\n\n"
            continue

        if event_type == "tool_start":
            saw_tool = True
            key = event.get("key") or event.get("id") or str(len(tool_indexes))
            tool_index = tool_indexes.setdefault(key, len(tool_indexes))
            payload = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": _now(),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": tool_index,
                                    "id": event.get("id") or _response_id("call"),
                                    "type": "function",
                                    "function": {
                                        "name": event.get("name") or "",
                                        "arguments": "",
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {_json_dumps(payload)}\n\n"
            continue

        if event_type == "tool_arguments_delta":
            saw_tool = True
            key = event.get("key") or event.get("id") or str(len(tool_indexes))
            tool_index = tool_indexes.setdefault(key, len(tool_indexes))
            payload = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": _now(),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": tool_index,
                                    "function": {"arguments": event.get("delta") or ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {_json_dumps(payload)}\n\n"

    final_finish = finish_reason or ("tool_calls" if saw_tool else "stop")
    done = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": final_finish}],
    }
    yield f"data: {_json_dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_to_anthropic(
    lines: AsyncGenerator[str, None], actual_format: str, model: str
) -> AsyncGenerator[str, None]:
    msg_id = _response_id("msg")
    yield f"event: message_start\ndata: {_json_dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
    current_block: str | None = None
    current_index: int | None = None
    next_index = 0
    output_tokens = 0
    saw_tool = False
    finish_reason: str | None = None

    async def close_current() -> AsyncGenerator[str, None]:
        nonlocal current_block, current_index
        if current_block is not None and current_index is not None:
            yield f"event: content_block_stop\ndata: {_json_dumps({'type': 'content_block_stop', 'index': current_index})}\n\n"
        current_block = None
        current_index = None

    async for event in _iter_stream_events(lines, actual_format):
        event_type = event.get("type")
        if event_type == "message_done":
            finish_reason = event.get("finish_reason") or finish_reason
            continue
        if event_type == "text_delta":
            if current_block != "text":
                async for chunk in close_current():
                    yield chunk
                current_index = next_index
                next_index += 1
                current_block = "text"
                payload = {
                    "type": "content_block_start",
                    "index": current_index,
                    "content_block": {"type": "text", "text": ""},
                }
                yield f"event: content_block_start\ndata: {_json_dumps(payload)}\n\n"
            output_tokens += 1
            payload = {
                "type": "content_block_delta",
                "index": current_index,
                "delta": {"type": "text_delta", "text": event.get("delta") or ""},
            }
            yield f"event: content_block_delta\ndata: {_json_dumps(payload)}\n\n"
            continue

        if event_type == "tool_start":
            saw_tool = True
            async for chunk in close_current():
                yield chunk
            current_index = next_index
            next_index += 1
            current_block = "tool_use"
            payload = {
                "type": "content_block_start",
                "index": current_index,
                "content_block": {
                    "type": "tool_use",
                    "id": event.get("id") or _response_id("call"),
                    "name": event.get("name") or "",
                    "input": {},
                },
            }
            yield f"event: content_block_start\ndata: {_json_dumps(payload)}\n\n"
            continue

        if event_type == "tool_arguments_delta":
            saw_tool = True
            if current_block != "tool_use":
                async for chunk in close_current():
                    yield chunk
                current_index = next_index
                next_index += 1
                current_block = "tool_use"
                payload = {
                    "type": "content_block_start",
                    "index": current_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": event.get("id") or _response_id("call"),
                        "name": event.get("name") or "",
                        "input": {},
                    },
                }
                yield f"event: content_block_start\ndata: {_json_dumps(payload)}\n\n"
            payload = {
                "type": "content_block_delta",
                "index": current_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": event.get("delta") or "",
                },
            }
            yield f"event: content_block_delta\ndata: {_json_dumps(payload)}\n\n"
            continue

        if event_type == "tool_done" and current_block == "tool_use":
            async for chunk in close_current():
                yield chunk

    async for chunk in close_current():
        yield chunk
    stop_reason = _stream_finish_to_anthropic(finish_reason, saw_tool)
    yield f"event: message_delta\ndata: {_json_dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
    yield f"event: message_stop\ndata: {_json_dumps({'type': 'message_stop'})}\n\n"


async def _stream_to_responses(
    lines: AsyncGenerator[str, None], actual_format: str, model: str
) -> AsyncGenerator[str, None]:
    resp_id = _response_id("resp")
    text_item_id = _response_id("msg")
    text_output_index: int | None = None
    text_parts: list[str] = []
    output_items: list[dict] = []
    tool_items: dict[str, dict[str, Any]] = {}
    next_output_index = 0
    created = {
        "type": "response.created",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": _now(),
            "status": "in_progress",
            "model": model,
            "output": [],
        },
    }
    yield f"event: response.created\ndata: {_json_dumps(created)}\n\n"

    async def ensure_text_item() -> AsyncGenerator[str, None]:
        nonlocal text_output_index, next_output_index
        if text_output_index is None:
            text_output_index = next_output_index
            next_output_index += 1
            payload = {
                "type": "response.output_item.added",
                "output_index": text_output_index,
                "item": {
                    "id": text_item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            }
            yield f"event: response.output_item.added\ndata: {_json_dumps(payload)}\n\n"

    async for event in _iter_stream_events(lines, actual_format):
        event_type = event.get("type")
        if event_type == "message_done":
            continue
        if event_type == "text_delta":
            async for chunk in ensure_text_item():
                yield chunk
            text = event.get("delta") or ""
            text_parts.append(text)
            payload = {
                "type": "response.output_text.delta",
                "item_id": text_item_id,
                "output_index": text_output_index,
                "content_index": 0,
                "delta": text,
            }
            yield f"event: response.output_text.delta\ndata: {_json_dumps(payload)}\n\n"
            continue

        if event_type == "tool_start":
            key = event.get("key") or event.get("id") or _response_id("tool")
            if key not in tool_items:
                output_index = next_output_index
                next_output_index += 1
                item_id = event.get("id") or _response_id("fc")
                tool_items[key] = {
                    "id": item_id,
                    "call_id": event.get("id") or item_id,
                    "name": event.get("name") or "",
                    "arguments": "",
                    "output_index": output_index,
                    "done": False,
                }
                payload = {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "id": item_id,
                        "type": "function_call",
                        "status": "in_progress",
                        "call_id": tool_items[key]["call_id"],
                        "name": tool_items[key]["name"],
                        "arguments": "",
                    },
                }
                yield f"event: response.output_item.added\ndata: {_json_dumps(payload)}\n\n"
            continue

        if event_type == "tool_arguments_delta":
            key = event.get("key") or event.get("id") or _response_id("tool")
            if key not in tool_items:
                output_index = next_output_index
                next_output_index += 1
                item_id = event.get("id") or _response_id("fc")
                tool_items[key] = {
                    "id": item_id,
                    "call_id": event.get("id") or item_id,
                    "name": event.get("name") or "",
                    "arguments": "",
                    "output_index": output_index,
                    "done": False,
                }
                payload = {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "id": item_id,
                        "type": "function_call",
                        "status": "in_progress",
                        "call_id": tool_items[key]["call_id"],
                        "name": tool_items[key]["name"],
                        "arguments": "",
                    },
                }
                yield f"event: response.output_item.added\ndata: {_json_dumps(payload)}\n\n"
            state = tool_items[key]
            delta = event.get("delta") or ""
            state["arguments"] += delta
            payload = {
                "type": "response.function_call_arguments.delta",
                "item_id": state["id"],
                "output_index": state["output_index"],
                "delta": delta,
            }
            yield f"event: response.function_call_arguments.delta\ndata: {_json_dumps(payload)}\n\n"
            continue

        if event_type == "tool_done":
            key = event.get("key") or event.get("id") or _response_id("tool")
            if key not in tool_items:
                output_index = next_output_index
                next_output_index += 1
                item_id = event.get("id") or _response_id("fc")
                tool_items[key] = {
                    "id": item_id,
                    "call_id": event.get("id") or item_id,
                    "name": event.get("name") or "",
                    "arguments": "",
                    "output_index": output_index,
                    "done": False,
                }
            state = tool_items[key]
            state["name"] = event.get("name") or state["name"]
            state["arguments"] = event.get("arguments") or state["arguments"]
            done_args = {
                "type": "response.function_call_arguments.done",
                "item_id": state["id"],
                "output_index": state["output_index"],
                "name": state["name"],
                "arguments": state["arguments"],
            }
            yield f"event: response.function_call_arguments.done\ndata: {_json_dumps(done_args)}\n\n"
            item = {
                "id": state["id"],
                "type": "function_call",
                "status": "completed",
                "call_id": state["call_id"],
                "name": state["name"],
                "arguments": state["arguments"],
            }
            output_items.append(item)
            payload = {
                "type": "response.output_item.done",
                "output_index": state["output_index"],
                "item": item,
            }
            yield f"event: response.output_item.done\ndata: {_json_dumps(payload)}\n\n"
            state["done"] = True

    if text_output_index is not None:
        item = {
            "id": text_item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "".join(text_parts),
                    "annotations": [],
                }
            ],
        }
        output_items.insert(0, item)
        done = {
            "type": "response.output_item.done",
            "output_index": text_output_index,
            "item": item,
        }
        yield f"event: response.output_item.done\ndata: {_json_dumps(done)}\n\n"

    for state in tool_items.values():
        if state.get("done"):
            continue
        done_args = {
            "type": "response.function_call_arguments.done",
            "item_id": state["id"],
            "output_index": state["output_index"],
            "name": state["name"],
            "arguments": state["arguments"],
        }
        yield f"event: response.function_call_arguments.done\ndata: {_json_dumps(done_args)}\n\n"
        item = {
            "id": state["id"],
            "type": "function_call",
            "status": "completed",
            "call_id": state["call_id"],
            "name": state["name"],
            "arguments": state["arguments"],
        }
        output_items.append(item)
        done = {
            "type": "response.output_item.done",
            "output_index": state["output_index"],
            "item": item,
        }
        yield f"event: response.output_item.done\ndata: {_json_dumps(done)}\n\n"

    completed = {
        "type": "response.completed",
        "response": {
            "id": resp_id,
            "object": "response",
            "status": "completed",
            "model": model,
            "output": output_items,
        },
    }
    yield f"event: response.completed\ndata: {_json_dumps(completed)}\n\n"
