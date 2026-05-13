import asyncio
import json
import time
from datetime import datetime, timezone, timedelta

_CST = timezone(timedelta(hours=8))

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import get_model_config, get_settings
from .logger import logger
from .models import ClaudeRequest, OpenAIRequest, RequestFormat, ResponsesRequest
from .proxy import (
    call_anthropic_passthrough,
    call_openai_chat_passthrough,
    call_with_conversion,
    stream_anthropic_passthrough,
    stream_openai_chat_passthrough,
    stream_with_conversion,
)
from .quality import check_response_quality


settings = get_settings()


class _LazyModelConfig:
    """Delegates all attribute access to a hot-reloaded ModelConfig instance."""
    def __getattr__(self, name):
        return getattr(get_model_config(), name)


model_config = _LazyModelConfig()

app = FastAPI(title="LLM Proxy Gateway", version="1.0.0")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)

    client = request.client
    client_addr = f"{client.host}:{client.port}" if client else "unknown"
    print(f"INFO:     {client_addr} - \"{request.method} {request.url.path} HTTP/1.1\" {response.status_code}")

    info = getattr(request.state, "_proxy_info", None)
    if info:
        start_ts = info["start"].strftime("%Y-%m-%d %H:%M:%S")
        end_ts = datetime.now(_CST).strftime("%Y-%m-%d %H:%M:%S")
        usage = info["usage"] or {}
        in_t = usage.get("input_tokens", usage.get("prompt_tokens", "?"))
        out_t = usage.get("output_tokens", usage.get("completion_tokens", "?"))
        stop = info["stop"] or "unknown"
        print(f"[{start_ts}] REQ -> {info['base_url']}{info['path']} [{info['upstream_model']}]")
        print(f"[{end_ts}] RES <- {in_t}/{out_t} tokens, {stop}")
    return response


def _format_config_error(exc: Exception, model_name: str, request_format: RequestFormat) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "type": "unsupported_model_or_format",
                "message": str(exc),
                "model": model_name,
                "request_format": request_format.value,
            }
        },
    )


def _format_upstream_error(
    exc: Exception,
    model_name: str,
    upstream: dict,
    request_url: str | None = None,
    request_format: RequestFormat = RequestFormat.ANTHROPIC_MESSAGES,
) -> tuple[int, dict]:
    upstream_model = upstream.get("upstream_model", "unknown")
    provider = upstream.get("provider_name", upstream.get("provider", "unknown"))
    actual_format = upstream.get("actual_format", "unknown")

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
                "request_format": request_format.value,
                "actual_format": actual_format,
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
                "request_format": request_format.value,
                "actual_format": actual_format,
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
                "request_format": request_format.value,
                "actual_format": actual_format,
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
            "request_format": request_format.value,
            "actual_format": actual_format,
            "model": model_name,
            "upstream_model": upstream_model,
            "request_url": request_url,
        }
    }


def _response_usage(result: dict) -> dict:
    return result.get("usage") or {}


def _response_stop(result: dict, request_format: RequestFormat) -> str:
    """Extract stop/finish reason from a response in the given request_format."""
    if request_format in (RequestFormat.OPENAI_CHAT_COMPLETIONS, RequestFormat.OPENAI_RESPONSES):
        choices = result.get("choices") or []
        if choices:
            return choices[0].get("finish_reason") or "stop"
        return "stop"
    # anthropic-messages
    return result.get("stop_reason") or "end_turn"


def _set_proxy_info(request: Request, model_name: str, upstream: dict, upstream_path: str = ""):
    request.state._proxy_info = {
        "start": datetime.now(_CST),
        "model": model_name,
        "upstream_model": upstream["upstream_model"],
        "base_url": upstream["base_url"],
        "path": upstream_path or request.url.path,
        "usage": None,
        "stop": None,
    }


def _path_for_format(request_format: str) -> str:
    return {
        RequestFormat.ANTHROPIC_MESSAGES.value: "/v1/messages",
        RequestFormat.OPENAI_CHAT_COMPLETIONS.value: "/v1/chat/completions",
        RequestFormat.OPENAI_RESPONSES.value: "/v1/responses",
    }.get(request_format, "")


def _cascade_alias_for(model_name: str) -> str | None:
    """Return the cascade alias name if model_name should use cascade routing, else None."""
    alias = model_config.get_cascade_alias_name(model_name)
    if alias:
        return alias
    if model_config.should_use_cascade(model_name):
        return "auto"
    return None


async def _cascade_non_stream(
    req,
    request_format: RequestFormat,
    client_id: str,
    request_data: dict,
    alias_name: str = "auto",
) -> tuple[dict | JSONResponse, dict | None]:
    """
    Cascade through models for non-streaming requests.
    Returns (result_dict, upstream_info) on success,
    or (JSONResponse_error, None) on failure.
    """
    alias_cfg = model_config.get_cascade_config(alias_name)
    quality_cfg = alias_cfg.get("quality_checks", {})
    max_total_timeout = alias_cfg.get("max_total_timeout", 180.0)

    try:
        cascade_list = model_config.get_cascade_upstream_list(request_format, alias_name)
    except ValueError as e:
        return _format_config_error(e, alias_name, request_format), None

    start_time = time.monotonic()
    last_error = None
    attempted_models = []

    for idx, upstream in enumerate(cascade_list):
        elapsed = time.monotonic() - start_time
        if elapsed >= max_total_timeout:
            return JSONResponse(status_code=504, content={"error": {
                "type": "cascade_timeout",
                "message": f"Cascade timeout after {elapsed:.1f}s (budget: {max_total_timeout}s)",
                "attempted_models": attempted_models,
            }}), None

        cascade_model = upstream["_cascade_model_name"]
        actual_format = upstream["actual_format"]
        attempted_models.append(f"{cascade_model} ({actual_format})")
        remaining_timeout = max_total_timeout - elapsed

        attempt_conv_id = logger.log_request(
            f"{alias_name}/{cascade_model}", upstream["upstream_model"],
            request_data, client_id, upstream["base_url"],
        )

        try:
            result = await asyncio.wait_for(
                call_with_conversion(req, request_format.value, actual_format, upstream),
                timeout=remaining_timeout,
            )
        except asyncio.TimeoutError:
            last_error = f"timeout for {cascade_model}"
            logger.log_response(f"{alias_name}/{cascade_model}", upstream["upstream_model"],
                              attempt_conv_id, {"error": {"type": "cascade_timeout", "model": cascade_model}})
            continue
        except Exception as e:
            last_error = f"error for {cascade_model}: {e}"
            _, error_payload = _format_upstream_error(e, cascade_model, upstream, None, request_format)
            logger.log_response(f"{alias_name}/{cascade_model}", upstream["upstream_model"],
                              attempt_conv_id, {"error": error_payload["error"]})
            continue

        # Quality check — response is in request_format
        passed, reason = check_response_quality(result, request_format, quality_cfg)
        if passed:
            logger.log_response(f"{alias_name}/{cascade_model}", upstream["upstream_model"],
                              attempt_conv_id, result)
            return result, upstream
        else:
            last_error = f"quality fail for {cascade_model}: {reason}"
            logger.log_response(f"{alias_name}/{cascade_model}", upstream["upstream_model"],
                              attempt_conv_id, {"cascade_quality_fail": reason, "result_preview": str(result)[:500]})
            continue

    return JSONResponse(status_code=502, content={"error": {
        "type": "cascade_exhausted",
        "message": f"All cascade models failed. Last error: {last_error}",
        "attempted_models": attempted_models,
    }}), None


@app.post("/v1/messages/count_tokens")
async def count_tokens(req: Request):
    body = await req.json()
    model_name = body.get("model", "") or model_config._default
    try:
        upstream = model_config.get_upstream_info(model_name, request_format=RequestFormat.ANTHROPIC_MESSAGES)
    except ValueError as e:
        return _format_config_error(e, model_name, RequestFormat.ANTHROPIC_MESSAGES)

    url = f"{upstream['base_url']}/v1/messages/count_tokens"
    headers = {
        "x-api-key": upstream["api_key"],
        "anthropic-version": upstream.get("api_version") or "2023-06-01",
        "content-type": "application/json",
    }
    upstream_body = dict(body)
    upstream_body["model"] = upstream["upstream_model"]

    try:
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            resp = await client.post(url, headers=headers, json=upstream_body)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        pass

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
    request_format = RequestFormat.ANTHROPIC_MESSAGES

    # ===== CASCADE ROUTING =====
    cascade_alias = _cascade_alias_for(model_name)
    if cascade_alias:
        client_id = request.client.host if request.client else "unknown"
        request_data = req.model_dump(exclude_none=True)

        if req.stream:
            try:
                cascade_list = model_config.get_cascade_upstream_list(request_format, cascade_alias)
                upstream = cascade_list[0]
            except ValueError as e:
                return _format_config_error(e, cascade_alias, request_format)

            cascade_model = upstream["_cascade_model_name"]
            actual_format = upstream["actual_format"]
            conv_id = logger.log_request(f"{cascade_alias}/{cascade_model}", upstream["upstream_model"],
                                        request_data, client_id, upstream["base_url"])
            _set_proxy_info(request, model_name, upstream)

            async def event_stream():
                chunks = []
                has_error = False
                try:
                    async for chunk in stream_with_conversion(req, request_format.value, actual_format, upstream):
                        chunks.append(chunk)
                        yield chunk
                except Exception as e:
                    has_error = True
                    status_code, error_payload = _format_upstream_error(
                        e, model_name, upstream, str(request.url), request_format)
                    logger.log_response(f"{cascade_alias}/{cascade_model}", upstream["upstream_model"],
                                      conv_id, {"error": error_payload["error"]})
                    yield f"data: {json.dumps({'type': 'error', 'error': error_payload['error']})}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    if not has_error:
                        logger.log_response(f"{cascade_alias}/{cascade_model}", upstream["upstream_model"],
                                          conv_id, {"stream_chunks": chunks})

            resp = StreamingResponse(event_stream(), media_type="text/event-stream")
            resp.headers["x-proxy-selected-model"] = cascade_model
            return resp

        # Non-streaming cascade
        result, upstream = await _cascade_non_stream(
            req, request_format, client_id, request_data, cascade_alias,
        )
        if upstream is None:
            return result

        _set_proxy_info(request, model_name, upstream)
        request.state._proxy_info["usage"] = _response_usage(result)
        request.state._proxy_info["stop"] = _response_stop(result, request_format)
        resp = JSONResponse(content=result)
        resp.headers["x-proxy-selected-model"] = upstream["_cascade_model_name"]
        return resp

    # ===== DIRECT ROUTING =====
    try:
        upstream = model_config.get_upstream_info(model_name, request_format=request_format)
    except ValueError as e:
        return _format_config_error(e, model_name, request_format)

    actual_format = upstream["actual_format"]
    client_id = request.client.host if request.client else "unknown"
    request_data = req.model_dump(exclude_none=True)
    conv_id = logger.log_request(model_name, upstream["upstream_model"], request_data, client_id, upstream["base_url"])
    _set_proxy_info(request, model_name, upstream)

    if req.stream:
        async def event_stream():
            chunks = []
            has_error = False
            try:
                async for chunk in stream_with_conversion(req, request_format.value, actual_format, upstream):
                    chunks.append(chunk)
                    yield chunk
            except Exception as e:
                has_error = True
                status_code, error_payload = _format_upstream_error(e, model_name, upstream, str(request.url), request_format)
                logger.log_response(model_name, upstream["upstream_model"], conv_id, {"error": error_payload["error"]})
                yield f"data: {json.dumps({'type': 'error', 'error': error_payload['error']})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if not has_error:
                    logger.log_response(model_name, upstream["upstream_model"], conv_id, {"stream_chunks": chunks})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        result = await call_with_conversion(req, request_format.value, actual_format, upstream)
    except Exception as e:
        status_code, error_payload = _format_upstream_error(e, model_name, upstream, str(request.url), request_format)
        logger.log_response(model_name, upstream["upstream_model"], conv_id, {"error": error_payload["error"]})
        request.state._proxy_info["stop"] = f"error_{status_code}"
        return JSONResponse(status_code=status_code, content=error_payload)

    logger.log_response(model_name, upstream["upstream_model"], conv_id, result)
    request.state._proxy_info["usage"] = _response_usage(result)
    request.state._proxy_info["stop"] = _response_stop(result, request_format)
    return result


@app.post("/v1/chat/completions")
async def create_chat_completion(req: OpenAIRequest, request: Request):
    model_name = req.model or model_config._default
    request_format = RequestFormat.OPENAI_CHAT_COMPLETIONS

    # ===== CASCADE ROUTING =====
    cascade_alias = _cascade_alias_for(model_name)
    if cascade_alias:
        client_id = request.client.host if request.client else "unknown"
        request_data = req.model_dump(exclude_none=True)

        if req.stream:
            try:
                cascade_list = model_config.get_cascade_upstream_list(request_format, cascade_alias)
                upstream = cascade_list[0]
            except ValueError as e:
                return _format_config_error(e, cascade_alias, request_format)

            cascade_model = upstream["_cascade_model_name"]
            actual_format = upstream["actual_format"]
            conv_id = logger.log_request(f"{cascade_alias}/{cascade_model}", upstream["upstream_model"],
                                        request_data, client_id, upstream["base_url"])
            _set_proxy_info(request, model_name, upstream)

            async def event_stream():
                chunks = []
                has_error = False
                try:
                    async for chunk in stream_with_conversion(req, request_format.value, actual_format, upstream):
                        chunks.append(chunk)
                        yield chunk
                except Exception as e:
                    has_error = True
                    status_code, error_payload = _format_upstream_error(
                        e, model_name, upstream, str(request.url), request_format)
                    logger.log_response(f"{cascade_alias}/{cascade_model}", upstream["upstream_model"],
                                      conv_id, {"error": error_payload["error"]})
                    yield f"data: {json.dumps({'error': error_payload['error']})}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    if not has_error:
                        logger.log_response(f"{cascade_alias}/{cascade_model}", upstream["upstream_model"],
                                          conv_id, {"stream_chunks": chunks})

            resp = StreamingResponse(event_stream(), media_type="text/event-stream")
            resp.headers["x-proxy-selected-model"] = cascade_model
            return resp

        # Non-streaming cascade
        result, upstream = await _cascade_non_stream(
            req, request_format, client_id, request_data, cascade_alias,
        )
        if upstream is None:
            return result

        _set_proxy_info(request, model_name, upstream)
        request.state._proxy_info["usage"] = _response_usage(result)
        request.state._proxy_info["stop"] = _response_stop(result, request_format)
        resp = JSONResponse(content=result)
        resp.headers["x-proxy-selected-model"] = upstream["_cascade_model_name"]
        return resp

    # ===== DIRECT ROUTING =====
    try:
        upstream = model_config.get_upstream_info(model_name, request_format=request_format)
    except ValueError as e:
        return _format_config_error(e, model_name, request_format)

    actual_format = upstream["actual_format"]
    client_id = request.client.host if request.client else "unknown"
    request_data = req.model_dump(exclude_none=True)
    conv_id = logger.log_request(model_name, upstream["upstream_model"], request_data, client_id, upstream["base_url"])
    _set_proxy_info(request, model_name, upstream)

    if req.stream:
        async def event_stream():
            chunks = []
            has_error = False
            try:
                async for chunk in stream_with_conversion(req, request_format.value, actual_format, upstream):
                    chunks.append(chunk)
                    yield chunk
            except Exception as e:
                has_error = True
                status_code, error_payload = _format_upstream_error(e, model_name, upstream, str(request.url), request_format)
                logger.log_response(model_name, upstream["upstream_model"], conv_id, {"error": error_payload["error"]})
                yield f"data: {json.dumps({'error': error_payload['error']})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if not has_error:
                    logger.log_response(model_name, upstream["upstream_model"], conv_id, {"stream_chunks": chunks})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        result = await call_with_conversion(req, request_format.value, actual_format, upstream)
    except Exception as e:
        status_code, error_payload = _format_upstream_error(e, model_name, upstream, str(request.url), request_format)
        logger.log_response(model_name, upstream["upstream_model"], conv_id, {"error": error_payload["error"]})
        request.state._proxy_info["stop"] = f"error_{status_code}"
        return JSONResponse(status_code=status_code, content=error_payload)

    logger.log_response(model_name, upstream["upstream_model"], conv_id, result)
    request.state._proxy_info["usage"] = _response_usage(result)
    request.state._proxy_info["stop"] = _response_stop(result, request_format)
    return result


@app.post("/v1/responses")
async def create_response(req: ResponsesRequest, request: Request):
    model_name = req.model or model_config._default
    request_format = RequestFormat.OPENAI_RESPONSES

    # ===== CASCADE ROUTING =====
    cascade_alias = _cascade_alias_for(model_name)
    if cascade_alias:
        client_id = request.client.host if request.client else "unknown"
        request_data = req.model_dump(exclude_none=True)

        if req.stream:
            try:
                cascade_list = model_config.get_cascade_upstream_list(request_format, cascade_alias)
                upstream = cascade_list[0]
            except ValueError as e:
                return _format_config_error(e, cascade_alias, request_format)

            cascade_model = upstream["_cascade_model_name"]
            actual_format = upstream["actual_format"]
            conv_id = logger.log_request(f"{cascade_alias}/{cascade_model}", upstream["upstream_model"],
                                        request_data, client_id, upstream["base_url"])
            _set_proxy_info(
                request, model_name, upstream, _path_for_format(actual_format)
            )

            async def event_stream():
                has_error = False
                try:
                    async for chunk in stream_with_conversion(req, request_format.value, actual_format, upstream):
                        yield chunk
                except Exception as e:
                    has_error = True
                    status_code, error_payload = _format_upstream_error(
                        e, model_name, upstream, str(request.url), request_format)
                    logger.log_response(f"{cascade_alias}/{cascade_model}", upstream["upstream_model"],
                                      conv_id, {"error": error_payload["error"]})
                    yield f"event: response.failed\ndata: {json.dumps({'type': 'response.failed', 'response': {'status': 'failed', 'error': error_payload['error']}})}\n\n"
                finally:
                    if not has_error:
                        logger.log_response(f"{cascade_alias}/{cascade_model}", upstream["upstream_model"],
                                          conv_id, {"format": "responses_stream"})

            resp = StreamingResponse(event_stream(), media_type="text/event-stream")
            resp.headers["x-proxy-selected-model"] = cascade_model
            return resp

        # Non-streaming cascade
        result, upstream = await _cascade_non_stream(
            req, request_format, client_id, request_data, cascade_alias,
        )
        if upstream is None:
            return result

        _set_proxy_info(
            request,
            model_name,
            upstream,
            _path_for_format(upstream["actual_format"]),
        )
        request.state._proxy_info["usage"] = _response_usage(result)
        request.state._proxy_info["stop"] = "stop"
        resp = JSONResponse(content=result)
        resp.headers["x-proxy-selected-model"] = upstream["_cascade_model_name"]
        return resp

    # ===== DIRECT ROUTING =====
    try:
        upstream = model_config.get_upstream_info(model_name, request_format=request_format)
    except ValueError as e:
        return _format_config_error(e, model_name, request_format)

    actual_format = upstream["actual_format"]
    client_id = request.client.host if request.client else "unknown"
    request_data = req.model_dump(exclude_none=True)
    conv_id = logger.log_request(model_name, upstream["upstream_model"], request_data, client_id, upstream["base_url"])
    _set_proxy_info(request, model_name, upstream, _path_for_format(actual_format))

    if req.stream:
        async def event_stream():
            has_error = False
            try:
                async for chunk in stream_with_conversion(req, request_format.value, actual_format, upstream):
                    yield chunk
            except Exception as e:
                has_error = True
                status_code, error_payload = _format_upstream_error(e, model_name, upstream, str(request.url), request_format)
                logger.log_response(model_name, upstream["upstream_model"], conv_id, {"error": error_payload["error"]})
                yield f"event: response.failed\ndata: {json.dumps({'type': 'response.failed', 'response': {'status': 'failed', 'error': error_payload['error']}})}\n\n"
            finally:
                if not has_error:
                    logger.log_response(model_name, upstream["upstream_model"], conv_id, {"format": "responses_stream"})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        result = await call_with_conversion(req, request_format.value, actual_format, upstream)
    except Exception as e:
        status_code, error_payload = _format_upstream_error(e, model_name, upstream, str(request.url), request_format)
        logger.log_response(model_name, upstream["upstream_model"], conv_id, {"error": error_payload["error"]})
        request.state._proxy_info["stop"] = f"error_{status_code}"
        return JSONResponse(status_code=status_code, content=error_payload)

    logger.log_response(model_name, upstream["upstream_model"], conv_id, result)
    request.state._proxy_info["usage"] = _response_usage(result)
    request.state._proxy_info["stop"] = "stop"
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


@app.get("/")
async def root_get():
    return {"service": app.title, "version": app.version}


@app.api_route("/", methods=["HEAD"])
async def root_head():
    return JSONResponse(content=None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True, access_log=False)
