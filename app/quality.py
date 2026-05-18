"""
Response quality checking for Auto mode cascade routing.

Evaluates upstream responses against configurable criteria to decide
whether to accept the result or fall back to the next model in the cascade.
"""

from .models import RequestFormat


def check_response_quality(
    result: dict,
    request_format: RequestFormat,
    quality_config: dict,
) -> tuple[bool, str]:
    """
    Check if an upstream response meets quality criteria.
    Returns (True, "ok") if passed, (False, reason) if failed.

    Checks are evaluated in fail-fast order:
    1. Error presence
    2. Empty response
    3. Minimum response length
    4. Truncation (stop_reason == max_tokens)
    """
    # 1. Error check
    if quality_config.get("check_error", True):
        err = result.get("error")
        if err:
            err_msg = err.get("message", "") if isinstance(err, dict) else str(err)
            return False, f"upstream_error: {err_msg[:200]}"

    # Extract text and stop_reason based on format
    text, stop_reason, has_tool_use, has_thinking = _extract(result, request_format)

    # 2. Empty check (skip if tool_use or thinking is present)
    if quality_config.get("check_empty", True) and not has_tool_use and not has_thinking:
        if not text or not text.strip():
            return False, "empty_response"

    # 3. Min length check (skip if tool_use or thinking is present)
    min_length = quality_config.get("min_response_length", 10)
    if min_length > 0 and not has_tool_use and not has_thinking:
        if len(text.strip()) < min_length:
            return False, f"response_too_short: {len(text.strip())} < {min_length}"

    # 4. Truncation check (skip if thinking is present — thinking consumes token budget)
    if quality_config.get("check_truncation", True) and not has_thinking:
        if stop_reason == "max_tokens":
            return False, "truncated_max_tokens"

    return True, "ok"


def _extract(result: dict, request_format: RequestFormat) -> tuple[str, str, bool, bool]:
    """Extract (text, stop_reason, has_tool_use, has_thinking) from a response dict."""
    if request_format == RequestFormat.ANTHROPIC_MESSAGES:
        return _extract_claude(result)
    else:
        return _extract_openai(result)


def _extract_claude(result: dict) -> tuple[str, str, bool, bool]:
    """Extract from Claude Messages API response."""
    content = result.get("content", [])
    stop_reason = result.get("stop_reason", "")

    text_parts = []
    has_tool_use = False
    has_thinking = False
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")
        if block_type == "text":
            text_parts.append(block.get("text", "") or "")
        elif block_type == "tool_use":
            has_tool_use = True
        elif block_type == "thinking":
            has_thinking = True

    text = "".join(text_parts)
    # If there's thinking but no text, treat thinking as valid output
    if not text and has_thinking:
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                text_parts.append(block.get("thinking", "") or "")
        text = "".join(text_parts)

    return text, stop_reason, has_tool_use, has_thinking


def _extract_openai(result: dict) -> tuple[str, str, bool, bool]:
    """Extract from OpenAI Chat Completions or Responses API response."""
    # Check if this is a Responses API response (has "output" list)
    output = result.get("output")
    if isinstance(output, list):
        return _extract_responses_api(result)

    # Standard OpenAI Chat Completions
    choices = result.get("choices", [])
    if not choices:
        return "", "", False, False

    choice = choices[0]
    message = choice.get("message", {})
    text = message.get("content", "") or ""
    stop_reason = choice.get("finish_reason", "")

    # Check for tool calls
    tool_calls = message.get("tool_calls")
    has_tool_use = bool(tool_calls)

    # Check for reasoning_content (thinking) in OpenAI-compatible responses
    has_thinking = bool(message.get("reasoning_content"))

    return text, stop_reason, has_tool_use, has_thinking


def _extract_responses_api(result: dict) -> tuple[str, str, bool, bool]:
    """Extract from OpenAI Responses API response format."""
    output = result.get("output", [])
    text_parts = []
    has_tool_use = False

    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "")
        if item_type == "message":
            for content_block in item.get("content", []):
                if isinstance(content_block, dict) and content_block.get("type") == "output_text":
                    text_parts.append(content_block.get("text", "") or "")
        elif item_type == "function_call":
            has_tool_use = True

    # Responses API doesn't have a direct stop_reason field at the response level
    # If it completed, we treat it as non-truncated
    stop_reason = result.get("stop_reason", "stop")

    return "".join(text_parts), stop_reason, has_tool_use, False
