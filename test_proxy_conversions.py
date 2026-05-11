import asyncio
import unittest
from unittest.mock import patch

from app.models import ClaudeRequest, OpenAIRequest, RequestFormat, ResponsesRequest
from app.proxy import call_with_conversion


UPSTREAM = {
    "actual_format": RequestFormat.OPENAI_CHAT_COMPLETIONS.value,
    "upstream_model": "upstream-model",
    "api_key": "test-key",
    "base_url": "http://upstream.test",
}


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
