from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class RequestFormat(str, Enum):
    ANTHROPIC_MESSAGES = "anthropic-messages"
    OPENAI_CHAT_COMPLETIONS = "openai-chat-completions"
    OPENAI_RESPONSES = "openai-responses"


class TextContent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["text"]
    text: str


class ThinkingContent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["thinking"]
    thinking: str
    signature: str | None = None


class ToolUseContent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ToolResultContent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, list[dict]] | None = None
    is_error: bool | None = None


class ImageContent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["image"]
    source: dict[str, Any]


ContentBlock = Union[TextContent, ThinkingContent, ToolUseContent, ToolResultContent, ImageContent]


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Union[str, list[ContentBlock]]


class ClaudeRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = ""
    messages: list[Message]
    max_tokens: int = 1024
    temperature: float = 1.0
    stream: bool = False
    system: Union[str, list[ContentBlock], None] = None
    stop_sequences: list[str] | None = None
    top_p: float = 1.0
    top_k: int | None = None
    tools: list[dict] | None = None
    tool_choice: dict | None = None
    metadata: dict | None = None
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    output_config: dict | None = None


class ClaudeResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock]
    model: str
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] = "end_turn"
    stop_sequence: str | None = None
    usage: dict = Field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Union[str, list[dict], None] = None


class OpenAIRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = ""
    messages: list[OpenAIMessage]
    max_tokens: int | None = None
    temperature: float = 1.0
    stream: bool = False
    top_p: float = 1.0
    stop: Union[str, list[str], None] = None
    tools: list[dict] | None = None
    tool_choice: Union[str, dict, None] = None  # Can be "auto", "none", "required", or dict
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None


class OpenAIResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: Usage | None = None


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = ""
    input: str | list[dict] = ""
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    reasoning_effort: str | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None


class ResponsesResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    object: str = "response"
    created: int = 0
    model: str = ""
    output: list[dict] = []
    usage: dict | None = None
