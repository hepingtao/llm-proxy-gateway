from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class RequestFormat(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ThinkingContent(BaseModel):
    type: Literal["thinking"]
    thinking: str
    signature: str | None = None


class ToolUseContent(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ToolResultContent(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, list[dict]] | None = None
    is_error: bool | None = None


class ImageContent(BaseModel):
    type: Literal["image"]
    source: dict[str, Any]


ContentBlock = Union[TextContent, ThinkingContent, ToolUseContent, ToolResultContent, ImageContent]


class Message(BaseModel):
    role: str
    content: Union[str, list[ContentBlock]]


class ClaudeRequest(BaseModel):
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
    role: str
    content: str


class OpenAIRequest(BaseModel):
    model: str = ""
    messages: list[OpenAIMessage]
    max_tokens: int = 1024
    temperature: float = 1.0
    stream: bool = False
    top_p: float = 1.0
    stop: list[str] | None = None
    reasoning_effort: str | None = None


class OpenAIResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: Usage | None = None
