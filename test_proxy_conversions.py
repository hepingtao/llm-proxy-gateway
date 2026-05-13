import asyncio
import json
import unittest
from unittest.mock import patch

from app.models import ClaudeRequest, OpenAIRequest, RequestFormat, ResponsesRequest
from app.proxy import (
    _chat_to_responses_response,
    _convert_stream,
    _responses_to_chat_response,
    call_with_conversion,
)


UPSTREAM = {
    "actual_format": RequestFormat.OPENAI_CHAT_COMPLETIONS.value,
    "upstream_model": "upstream-model",
    "api_key": "test-key",
    "base_url": "http://upstream.test",
}


async def _line_source(lines):
    for line in lines:
        yield line


async def _collect_stream(lines, source_format, target_format):
    return [
        chunk
        async for chunk in _convert_stream(
            _line_source(lines), source_format, target_format, "downstream-model"
        )
    ]


def _data_payloads(chunks):
    payloads = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                continue
            payloads.append(json.loads(data))
    return payloads


class ProxyConversionTests(unittest.TestCase):
    def test_claude_request_to_openai_chat_and_back(self):
        req = ClaudeRequest(
            model="downstream-model",
            system="You are concise.",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            tools=[{
                "name": "get_weather",
                "description": "weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }],
        )
        chat_response = {
            "id": "chatcmpl_1",
            "created": 1,
            "model": "upstream-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

        async def run():
            with patch("app.proxy._post_json", return_value=chat_response) as post_json:
                result = await call_with_conversion(
                    req,
                    RequestFormat.ANTHROPIC_MESSAGES.value,
                    RequestFormat.OPENAI_CHAT_COMPLETIONS.value,
                    UPSTREAM,
                )
                body = post_json.call_args.args[2]
                self.assertEqual(body["messages"][0], {"role": "system", "content": "You are concise."})
                self.assertEqual(body["messages"][1], {"role": "user", "content": "hi"})
                self.assertEqual(body["tools"][0]["type"], "function")

            self.assertEqual(result["type"], "message")
            self.assertEqual(result["content"][0]["text"], "hello")
            self.assertEqual(result["usage"], {"input_tokens": 2, "output_tokens": 3})

        asyncio.run(run())

    def test_openai_chat_request_to_claude_and_back_with_tool_call(self):
        req = OpenAIRequest(
            model="downstream-model",
            messages=[
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "weather?"},
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }],
        )
        claude_response = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "upstream-model",
            "content": [{
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "Beijing"},
            }],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }

        async def run():
            upstream = {**UPSTREAM, "actual_format": RequestFormat.ANTHROPIC_MESSAGES.value}
            with patch("app.proxy._post_json", return_value=claude_response) as post_json:
                result = await call_with_conversion(
                    req,
                    RequestFormat.OPENAI_CHAT_COMPLETIONS.value,
                    RequestFormat.ANTHROPIC_MESSAGES.value,
                    upstream,
                )
                body = post_json.call_args.args[2]
                self.assertEqual(body["system"], "You are concise.")
                self.assertEqual(body["messages"][0]["content"], "weather?")
                self.assertEqual(body["tools"][0]["name"], "get_weather")

            message = result["choices"][0]["message"]
            self.assertEqual(result["choices"][0]["finish_reason"], "tool_calls")
            self.assertEqual(message["tool_calls"][0]["function"]["name"], "get_weather")

        asyncio.run(run())

    def test_responses_request_to_chat_and_back_with_function_call(self):
        req = ResponsesRequest(
            model="downstream-model",
            instructions="You are concise.",
            input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "weather?"}]}],
            max_output_tokens=64,
            tools=[{
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }],
        )
        chat_response = {
            "id": "chatcmpl_1",
            "created": 1,
            "model": "upstream-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Beijing\"}"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

        async def run():
            with patch("app.proxy._post_json", return_value=chat_response) as post_json:
                result = await call_with_conversion(
                    req,
                    RequestFormat.OPENAI_RESPONSES.value,
                    RequestFormat.OPENAI_CHAT_COMPLETIONS.value,
                    UPSTREAM,
                )
                body = post_json.call_args.args[2]
                self.assertEqual(body["messages"][0], {"role": "system", "content": "You are concise."})
                self.assertEqual(body["messages"][1], {"role": "user", "content": "weather?"})
                self.assertEqual(body["max_tokens"], 64)
                self.assertEqual(body["tools"][0]["function"]["name"], "get_weather")

            self.assertEqual(result["object"], "response")
            self.assertEqual(result["output"][0]["type"], "function_call")
            self.assertEqual(result["output"][0]["name"], "get_weather")

        asyncio.run(run())

    def test_claude_request_to_responses_upstream_and_back(self):
        req = ClaudeRequest(model="downstream-model", messages=[{"role": "user", "content": "hi"}])
        responses_response = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "upstream-model",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            }],
            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        }

        async def run():
            upstream = {**UPSTREAM, "actual_format": RequestFormat.OPENAI_RESPONSES.value}
            with patch("app.proxy._post_json", return_value=responses_response) as post_json:
                result = await call_with_conversion(
                    req,
                    RequestFormat.ANTHROPIC_MESSAGES.value,
                    RequestFormat.OPENAI_RESPONSES.value,
                    upstream,
                )
                body = post_json.call_args.args[2]
                self.assertEqual(body["input"][0]["content"][0]["text"], "hi")

            self.assertEqual(result["type"], "message")
            self.assertEqual(result["content"][0]["text"], "hello")

        asyncio.run(run())

    def test_claude_request_to_responses_preserves_tool_result_history(self):
        req = ClaudeRequest(
            model="downstream-model",
            messages=[
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Beijing"},
                    }],
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "sunny",
                    }],
                },
            ],
        )
        responses_response = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "upstream-model",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "It is sunny."}],
            }],
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }

        async def run():
            upstream = {**UPSTREAM, "actual_format": RequestFormat.OPENAI_RESPONSES.value}
            with patch("app.proxy._post_json", return_value=responses_response) as post_json:
                result = await call_with_conversion(
                    req,
                    RequestFormat.ANTHROPIC_MESSAGES.value,
                    RequestFormat.OPENAI_RESPONSES.value,
                    upstream,
                )
                body = post_json.call_args.args[2]

            self.assertEqual(body["input"][0]["type"], "message")
            self.assertEqual(body["input"][1]["type"], "function_call")
            self.assertEqual(body["input"][1]["call_id"], "toolu_1")
            self.assertEqual(body["input"][2]["type"], "function_call_output")
            self.assertEqual(body["input"][2]["call_id"], "toolu_1")
            self.assertEqual(result["usage"], {"input_tokens": 2, "output_tokens": 3})

        asyncio.run(run())

    def test_chat_request_to_responses_upstream_and_back(self):
        req = OpenAIRequest(
            model="downstream-model",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }],
        )
        responses_response = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "upstream-model",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            }],
            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        }

        async def run():
            upstream = {**UPSTREAM, "actual_format": RequestFormat.OPENAI_RESPONSES.value}
            with patch("app.proxy._post_json", return_value=responses_response) as post_json:
                result = await call_with_conversion(
                    req,
                    RequestFormat.OPENAI_CHAT_COMPLETIONS.value,
                    RequestFormat.OPENAI_RESPONSES.value,
                    upstream,
                )
                body = post_json.call_args.args[2]
                self.assertEqual(body["input"][0]["content"][0]["text"], "hi")
                self.assertEqual(body["tools"][0]["name"], "get_weather")

            self.assertEqual(result["object"], "chat.completion")
            self.assertEqual(result["choices"][0]["message"]["content"], "hello")

        asyncio.run(run())

    def test_chat_request_to_responses_preserves_tool_history(self):
        req = OpenAIRequest(
            model="downstream-model",
            messages=[
                {"role": "developer", "content": "You are concise."},
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Beijing\"}",
                        },
                    }],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ],
        )
        responses_response = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "upstream-model",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "It is sunny."}],
            }],
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }

        async def run():
            upstream = {**UPSTREAM, "actual_format": RequestFormat.OPENAI_RESPONSES.value}
            with patch("app.proxy._post_json", return_value=responses_response) as post_json:
                result = await call_with_conversion(
                    req,
                    RequestFormat.OPENAI_CHAT_COMPLETIONS.value,
                    RequestFormat.OPENAI_RESPONSES.value,
                    upstream,
                )
                body = post_json.call_args.args[2]

            self.assertEqual(body["instructions"], "You are concise.")
            self.assertEqual(body["input"][0]["type"], "message")
            self.assertEqual(body["input"][1]["type"], "function_call")
            self.assertEqual(body["input"][1]["call_id"], "call_1")
            self.assertEqual(body["input"][2]["type"], "function_call_output")
            self.assertEqual(body["input"][2]["call_id"], "call_1")
            self.assertEqual(result["usage"]["total_tokens"], 5)

        asyncio.run(run())

    def test_token_total_falls_back_to_input_plus_output(self):
        chat_as_responses = _chat_to_responses_response(
            {
                "id": "chatcmpl_1",
                "created": 1,
                "choices": [{
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 4, "completion_tokens": 6},
            },
            "downstream-model",
        )
        self.assertEqual(chat_as_responses["usage"]["total_tokens"], 10)

        responses_as_chat = _responses_to_chat_response(
            {
                "id": "resp_1",
                "created_at": 1,
                "output": [{
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hello"}],
                }],
                "usage": {"input_tokens": 3, "output_tokens": 7},
            },
            "downstream-model",
        )
        self.assertEqual(responses_as_chat["usage"]["total_tokens"], 10)

    def test_chat_tool_stream_to_responses_events(self):
        lines = [
            "data: " + json.dumps({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }]
                    },
                    "finish_reason": None,
                }]
            }),
            "data: " + json.dumps({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": "{\"city\""},
                        }]
                    },
                    "finish_reason": None,
                }]
            }),
            "data: " + json.dumps({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": ":\"Beijing\"}"},
                        }]
                    },
                    "finish_reason": "tool_calls",
                }]
            }),
            "data: [DONE]",
        ]

        chunks = asyncio.run(
            _collect_stream(
                lines,
                RequestFormat.OPENAI_CHAT_COMPLETIONS.value,
                RequestFormat.OPENAI_RESPONSES.value,
            )
        )
        payloads = _data_payloads(chunks)
        event_types = [payload["type"] for payload in payloads]

        self.assertIn("response.output_item.added", event_types)
        self.assertEqual(event_types.count("response.function_call_arguments.delta"), 2)
        self.assertIn("response.function_call_arguments.done", event_types)
        done = next(
            payload
            for payload in payloads
            if payload["type"] == "response.function_call_arguments.done"
        )
        self.assertEqual(done["arguments"], "{\"city\":\"Beijing\"}")
        item_done = next(
            payload
            for payload in payloads
            if payload["type"] == "response.output_item.done"
        )
        self.assertEqual(item_done["item"]["type"], "function_call")
        self.assertEqual(item_done["item"]["call_id"], "call_1")

    def test_anthropic_tool_stream_to_chat_events(self):
        lines = [
            "data: " + json.dumps({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {},
                },
            }),
            "data: " + json.dumps({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "{\"city\""},
            }),
            "data: " + json.dumps({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": ":\"Beijing\"}"},
            }),
            "data: " + json.dumps({"type": "content_block_stop", "index": 0}),
            "data: " + json.dumps({
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
            }),
            "data: " + json.dumps({"type": "message_stop"}),
        ]

        chunks = asyncio.run(
            _collect_stream(
                lines,
                RequestFormat.ANTHROPIC_MESSAGES.value,
                RequestFormat.OPENAI_CHAT_COMPLETIONS.value,
            )
        )
        payloads = _data_payloads(chunks)
        tool_chunks = [
            payload
            for payload in payloads
            if payload["choices"][0]["delta"].get("tool_calls")
        ]

        self.assertEqual(
            tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["id"], "toolu_1"
        )
        self.assertEqual(
            tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "get_weather",
        )
        argument_deltas = [
            chunk["choices"][0]["delta"]["tool_calls"][0]["function"].get(
                "arguments", ""
            )
            for chunk in tool_chunks[1:]
        ]
        self.assertEqual("".join(argument_deltas), "{\"city\":\"Beijing\"}")
        self.assertEqual(payloads[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_responses_tool_stream_to_anthropic_events(self):
        lines = [
            "data: " + json.dumps({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "",
                },
            }),
            "data: " + json.dumps({
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 0,
                "delta": "{\"city\"",
            }),
            "data: " + json.dumps({
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 0,
                "delta": ":\"Beijing\"}",
            }),
            "data: " + json.dumps({
                "type": "response.function_call_arguments.done",
                "item_id": "fc_1",
                "output_index": 0,
                "name": "get_weather",
                "arguments": "{\"city\":\"Beijing\"}",
            }),
            "data: " + json.dumps({"type": "response.completed", "response": {}}),
        ]

        chunks = asyncio.run(
            _collect_stream(
                lines,
                RequestFormat.OPENAI_RESPONSES.value,
                RequestFormat.ANTHROPIC_MESSAGES.value,
            )
        )
        payloads = _data_payloads(chunks)

        block_start = next(
            payload for payload in payloads if payload["type"] == "content_block_start"
        )
        deltas = [
            payload["delta"]["partial_json"]
            for payload in payloads
            if payload["type"] == "content_block_delta"
        ]
        message_delta = next(
            payload for payload in payloads if payload["type"] == "message_delta"
        )

        self.assertEqual(block_start["content_block"]["type"], "tool_use")
        self.assertEqual(block_start["content_block"]["id"], "call_1")
        self.assertEqual("".join(deltas), "{\"city\":\"Beijing\"}")
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")

    def test_responses_tool_stream_done_item_supplies_arguments(self):
        lines = [
            "data: " + json.dumps({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "",
                },
            }),
            "data: " + json.dumps({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{\"city\":\"Beijing\"}",
                },
            }),
            "data: " + json.dumps({"type": "response.completed", "response": {}}),
        ]

        chunks = asyncio.run(
            _collect_stream(
                lines,
                RequestFormat.OPENAI_RESPONSES.value,
                RequestFormat.OPENAI_CHAT_COMPLETIONS.value,
            )
        )
        payloads = _data_payloads(chunks)
        tool_chunks = [
            payload
            for payload in payloads
            if payload["choices"][0]["delta"].get("tool_calls")
        ]
        argument_deltas = [
            chunk["choices"][0]["delta"]["tool_calls"][0]["function"].get(
                "arguments", ""
            )
            for chunk in tool_chunks[1:]
        ]

        self.assertEqual("".join(argument_deltas), "{\"city\":\"Beijing\"}")
        self.assertEqual(payloads[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_responses_request_to_claude_and_back(self):
        req = ResponsesRequest(model="downstream-model", input="hi", max_output_tokens=64)
        claude_response = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "upstream-model",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }

        async def run():
            upstream = {**UPSTREAM, "actual_format": RequestFormat.ANTHROPIC_MESSAGES.value}
            with patch("app.proxy._post_json", return_value=claude_response) as post_json:
                result = await call_with_conversion(
                    req,
                    RequestFormat.OPENAI_RESPONSES.value,
                    RequestFormat.ANTHROPIC_MESSAGES.value,
                    upstream,
                )
                body = post_json.call_args.args[2]
                self.assertEqual(body["messages"][0]["content"], "hi")
                self.assertEqual(body["max_tokens"], 64)

            self.assertEqual(result["object"], "response")
            self.assertEqual(result["output"][0]["content"][0]["text"], "hello")

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
