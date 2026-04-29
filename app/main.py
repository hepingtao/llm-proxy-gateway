import json
import time
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import get_model_config, get_settings
from .logger import logger
from .models import ClaudeRequest, OpenAIRequest, RequestFormat
from .proxy import call_claude, call_openai, stream_claude, stream_openai


def _convert_to_dict(obj):
    """Convert Pydantic model or list of models to dict/list."""
    if obj is None:
        return None
    if isinstance(obj, list):
        return [item.model_dump() if hasattr(item, "model_dump") else item for item in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


settings = get_settings()
model_config = get_model_config()

app = FastAPI(title="LLM Proxy Gateway", version="1.0.0")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)

    # Access log line (replaces uvicorn's default access log)
    client = request.client
    client_addr = f"{client.host}:{client.port}" if client else "unknown"
    print(f"INFO:     {client_addr} - \"{request.method} {request.url.path} HTTP/1.1\" {response.status_code}")

    # Proxy-specific REQ/RES lines
    info = getattr(request.state, "_proxy_info", None)
    if info:
        start_ts = info["start"].strftime("%Y-%m-%d %H:%M:%S")
        end_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        usage = info["usage"] or {}
        in_t = usage.get("input_tokens", "?")
        out_t = usage.get("output_tokens", "?")
        stop = info["stop"] or "unknown"
        print(f"[{start_ts}] REQ → {info['base_url']}{info['path']} [{info['upstream_model']}]")
        print(f"[{end_ts}] RES ← {in_t}/{out_t} tokens, {stop}")
    return response


def _collect_extra_params(req: ClaudeRequest) -> dict:
    params = {}
    for key in ("temperature", "top_p", "max_tokens", "stop_sequences", "top_k", "metadata", "reasoning_effort", "output_config"):
        val = getattr(req, key, None)
        if val is not None:
            params[key] = val
    
    # Handle system field which may be string or ContentBlock array
    if req.system is not None:
        if isinstance(req.system, list):
            params["system"] = [_convert_to_dict(s) for s in req.system]
        else:
            params["system"] = req.system
    
    # Handle tools and tool_choice which may be Pydantic models
    if req.tools:
        params["tools"] = _convert_to_dict(req.tools)
    if req.tool_choice:
        params["tool_choice"] = _convert_to_dict(req.tool_choice)
    
    return params


def _format_upstream_error(exc: Exception, model_name: str, upstream: dict, request_url: str | None = None, request_format: RequestFormat = RequestFormat.ANTHROPIC) -> tuple[int, dict]:
    upstream_model = upstream.get("upstream_model", "unknown")
    provider = upstream.get("provider", "unknown")

    if isinstance(exc, httpx.HTTPStatusError):
        upstream_status = exc.response.status_code
        try:
            response_text = (exc.response.text or "").strip()
        except httpx.ResponseNotRead:
            response_text = ""
        if len(response_text) > 1000:
            response_text = response_text[:1000] + "...<truncated>"
        return 502, {
            "error": {
                "type": "upstream_http_error",
                "message": f"Upstream provider returned HTTP {upstream_status}",
                "provider": provider,
                "request_format": request_format,
                "model": model_name,
                "upstream_model": upstream_model,
                "upstream_status": upstream_status,
                "upstream_body": response_text or None,
                "request_url": request_url,
            }
        }

    if isinstance(exc, httpx.TimeoutException):
        return 504, {
            "error": {
                "type": "upstream_timeout",
                "message": "Upstream provider request timed out",
                "provider": provider,
                "request_format": request_format,
                "model": model_name,
                "upstream_model": upstream_model,
                "request_url": request_url,
            }
        }

    if isinstance(exc, httpx.RequestError):
        return 502, {
            "error": {
                "type": "upstream_request_error",
                "message": str(exc),
                "provider": provider,
                "request_format": request_format,
                "model": model_name,
                "upstream_model": upstream_model,
                "request_url": request_url,
            }
        }

    return 500, {
        "error": {
            "type": "internal_proxy_error",
            "message": str(exc),
            "provider": provider,
            "model": model_name,
            "upstream_model": upstream_model,
            "request_url": request_url,
        }
    }


@app.post("/v1/messages/count_tokens")
async def count_tokens(req: Request):
    body = await req.json()
    model_name = body.get("model", "") or model_config._default
    upstream = model_config.get_upstream_info(model_name, request_format=RequestFormat.ANTHROPIC)

    url = f"{upstream['base_url']}/v1/messages/count_tokens"
    headers = {
        "x-api-key": upstream["api_key"],
        "anthropic-version": upstream.get("api_version", "2023-06-01"),
        "content-type": "application/json",
    }

    # Try upstream first, fall back to local estimation
    try:
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        pass

    # Local fallback: rough token estimation (~4 chars per token)
    total_chars = 0
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total_chars += len(block.get("text", "") or block.get("thinking", "") or "")
        total_chars += len(msg.get("role", ""))
    if body.get("system"):
        sys_text = body["system"]
        if isinstance(sys_text, str):
            total_chars += len(sys_text)
        elif isinstance(sys_text, list):
            for block in sys_text:
                if isinstance(block, dict):
                    total_chars += len(block.get("text", ""))
    return {"input_tokens": max(1, total_chars // 4)}


@app.post("/v1/messages")
async def create_message(req: ClaudeRequest, request: Request):
    model_name = req.model or model_config._default
    upstream = model_config.get_upstream_info(model_name, request_format=RequestFormat.ANTHROPIC)
    client_id = request.client.host if request.client else "unknown"

    # Apply default reasoning_effort from config if client doesn't provide one
    if not req.reasoning_effort and not req.output_config:
        config_reasoning_effort = upstream.get("reasoning_effort")
        if config_reasoning_effort:
            req.reasoning_effort = config_reasoning_effort

    # Convert messages to dict, handling both string and ContentBlock array content
    messages = []
    for m in req.messages:
        msg_dict = {"role": m.role}
        if isinstance(m.content, list):
            msg_dict["content"] = [_convert_to_dict(c) for c in m.content]
        else:
            msg_dict["content"] = m.content
        messages.append(msg_dict)

    extra = _collect_extra_params(req)
    conv_id = logger.log_request(model_name, upstream["upstream_model"], messages, client_id, extra, upstream["base_url"])

    # Store request info for logging after response
    request.state._proxy_info = {
        "start": datetime.now(timezone.utc),
        "model": model_name,
        "upstream_model": upstream["upstream_model"],
        "base_url": upstream["base_url"],
        "path": request.url.path,
        "conv_id": conv_id,
        "usage": None,
        "stop": None,
    }

    if req.stream:
        async def event_stream():
            text_parts = []
            finish_reason = "end_turn"
            has_error = False
            try:
                async for chunk in stream_claude(req, upstream):
                    yield chunk
                    try:
                        import json as _json
                        data = chunk[6:].strip()
                        if data and data != "[DONE]":
                            evt = _json.loads(data)
                            if evt.get("type") == "content_block_delta":
                                delta = evt.get("delta", {})
                                text_parts.append(delta.get("text", "") or delta.get("thinking", ""))
                    except Exception:
                        pass
            except Exception as e:
                has_error = True
                status_code, error_payload = _format_upstream_error(e, model_name, upstream, str(request.url), RequestFormat.ANTHROPIC)
                logger.log_response(
                    model_name,
                    upstream["upstream_model"],
                    conv_id,
                    "",
                    {},
                    f"error_status_{status_code}: {error_payload['error']['type']}",
                )
                yield f"data: {json.dumps({'type': 'error', 'error': error_payload['error']})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if not has_error:
                    full_text = "".join(text_parts)
                    logger.log_response(model_name, upstream["upstream_model"], conv_id, full_text, {"input_tokens": 0, "output_tokens": 0}, finish_reason)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming
    try:
        result = await call_claude(req, upstream)
    except Exception as e:
        status_code, error_payload = _format_upstream_error(e, model_name, upstream, str(request.url), RequestFormat.ANTHROPIC)
        logger.log_response(
            model_name,
            upstream["upstream_model"],
            conv_id,
            "",
            {},
            f"error_status_{status_code}: {error_payload['error']['type']}",
        )
        request.state._proxy_info["stop"] = f"error_{status_code}"
        return JSONResponse(status_code=status_code, content=error_payload)

    text = ""
    for block in result.get("content", []):
        text += block.get("text") or ""

    usage = result.get("usage", {})
    logger.log_response(model_name, upstream["upstream_model"], conv_id, text, usage, result.get("stop_reason", "end_turn"))
    request.state._proxy_info["usage"] = usage
    request.state._proxy_info["stop"] = result.get("stop_reason", "end_turn")
    return result


@app.post("/v1/chat/completions")
async def create_chat_completion(req: OpenAIRequest, request: Request):
    model_name = req.model or model_config._default
    upstream = model_config.get_upstream_info(model_name, request_format=RequestFormat.OPENAI)
    client_id = request.client.host if request.client else "unknown"

    # Convert OpenAI request to ClaudeRequest for internal handling
    from .models import ClaudeRequest, Message

    messages = [Message(role=m.role, content=m.content) for m in req.messages]
    claude_req = ClaudeRequest(
        model=req.model,
        messages=messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stream=req.stream,
        top_p=req.top_p,
        stop_sequences=req.stop,
        reasoning_effort=req.reasoning_effort,
    )

    extra = {"temperature": req.temperature, "top_p": req.top_p, "max_tokens": req.max_tokens}
    if req.stop:
        extra["stop"] = req.stop
    if req.reasoning_effort:
        extra["reasoning_effort"] = req.reasoning_effort

    conv_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    conv_id = logger.log_request(model_name, upstream["upstream_model"], conv_messages, client_id, extra, upstream["base_url"])

    request.state._proxy_info = {
        "start": datetime.now(timezone.utc),
        "model": model_name,
        "upstream_model": upstream["upstream_model"],
        "base_url": upstream["base_url"],
        "path": request.url.path,
        "conv_id": conv_id,
        "usage": None,
        "stop": None,
    }

    if req.stream:
        async def event_stream():
            text_parts = []
            finish_reason = "stop"
            has_error = False
            try:
                async for chunk in stream_openai(claude_req, upstream):
                    yield chunk
                    try:
                        import json as _json
                        data = chunk[6:].strip()
                        if data and data != "[DONE]":
                            evt = _json.loads(data)
                            delta = evt.get("choices", [{}])[0].get("delta", {})
                            text_parts.append(delta.get("content", "") or "")
                    except Exception:
                        pass
            except Exception as e:
                has_error = True
                status_code, error_payload = _format_upstream_error(e, model_name, upstream, str(request.url), RequestFormat.OPENAI)
                logger.log_response(
                    model_name, upstream["upstream_model"], conv_id, "", {},
                    f"error_status_{status_code}: {error_payload['error']['type']}",
                )
                yield f"data: {json.dumps({'error': error_payload['error']})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if not has_error:
                    full_text = "".join(text_parts)
                    logger.log_response(model_name, upstream["upstream_model"], conv_id, full_text, {"input_tokens": 0, "output_tokens": 0}, finish_reason)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming
    try:
        result = await call_openai(claude_req, upstream)
    except Exception as e:
        status_code, error_payload = _format_upstream_error(e, model_name, upstream, str(request.url), RequestFormat.OPENAI)
        logger.log_response(
            model_name, upstream["upstream_model"], conv_id, "", {},
            f"error_status_{status_code}: {error_payload['error']['type']}",
        )
        request.state._proxy_info["stop"] = f"error_{status_code}"
        return JSONResponse(status_code=status_code, content=error_payload)

    text = result.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    usage = result.get("usage", {})
    finish_reason = result["choices"][0].get("finish_reason", "stop") if result.get("choices") else "stop"

    logger.log_response(model_name, upstream["upstream_model"], conv_id, text, usage, finish_reason)
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] {model_name} ← {usage.get('prompt_tokens',0)}/{usage.get('completion_tokens',0)} tokens, stop={finish_reason}")
    return result


@app.get("/v1/models")
async def list_models():
    models = []
    for name in model_config.list_models():
        models.append({"id": name, "object": "model", "owned_by": "proxy"})
    return {"data": models}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True, access_log=False)
