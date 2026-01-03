"""Solace-AI Inference - Model inference utilities for structured outputs and prompt management."""
from __future__ import annotations
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, TypeVar
from pydantic import BaseModel, Field, ValidationError
import structlog
from solace_ml.llm_client import (
    LLMClient, LLMResponse, Message, MessageRole, ToolDefinition, ToolCall,
    FinishReason, build_messages,
)

logger = structlog.get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class OutputFormat(str, Enum):
    """Supported output formats."""
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


class ParseError(Exception):
    """Output parsing error."""
    def __init__(self, message: str, *, raw_output: str | None = None,
                 expected_format: str | None = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.expected_format = expected_format


class OutputParser(ABC, Generic[T]):
    """Abstract output parser."""

    @abstractmethod
    def parse(self, text: str) -> T:
        """Parse text into structured output."""

    @abstractmethod
    def get_format_instructions(self) -> str:
        """Get format instructions for the model."""


class JSONOutputParser(OutputParser[dict[str, Any]]):
    """Parse JSON output from LLM."""

    def __init__(self, schema: dict[str, Any] | None = None) -> None:
        self._schema = schema

    def parse(self, text: str) -> dict[str, Any]:
        """Parse JSON from text."""
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            text = json_match.group(1)
        text = text.strip()
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid JSON: {e}", raw_output=text, expected_format="json") from e

    def get_format_instructions(self) -> str:
        """Get JSON format instructions."""
        instructions = "Respond with valid JSON only, no additional text."
        if self._schema:
            instructions += f"\n\nExpected schema:\n```json\n{json.dumps(self._schema, indent=2)}\n```"
        return instructions


class PydanticOutputParser(OutputParser[T], Generic[T]):
    """Parse output into Pydantic model."""

    def __init__(self, model_class: type[T]) -> None:
        self._model_class = model_class

    def parse(self, text: str) -> T:
        """Parse text into Pydantic model."""
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            text = json_match.group(1)
        text = text.strip()
        try:
            data = json.loads(text)
            return self._model_class.model_validate(data)
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid JSON: {e}", raw_output=text,
                           expected_format=self._model_class.__name__) from e
        except ValidationError as e:
            raise ParseError(f"Validation error: {e}", raw_output=text,
                           expected_format=self._model_class.__name__) from e

    def get_format_instructions(self) -> str:
        """Get format instructions from Pydantic schema."""
        schema = self._model_class.model_json_schema()
        return f"Respond with valid JSON matching this schema:\n```json\n{json.dumps(schema, indent=2)}\n```"


class ListOutputParser(OutputParser[list[str]]):
    """Parse list output from LLM."""

    def __init__(self, separator: str = "\n") -> None:
        self._separator = separator

    def parse(self, text: str) -> list[str]:
        """Parse list from text."""
        lines = text.split(self._separator)
        items = []
        for line in lines:
            line = line.strip()
            line = re.sub(r"^[-*•]\s*", "", line)
            line = re.sub(r"^\d+[.)]\s*", "", line)
            if line:
                items.append(line)
        return items

    def get_format_instructions(self) -> str:
        """Get list format instructions."""
        return "Respond with a list, one item per line. Use - or numbers for formatting."


class PromptTemplate:
    """Template for prompt construction."""

    def __init__(self, template: str, *, input_variables: list[str] | None = None,
                 partial_variables: dict[str, Any] | None = None) -> None:
        self._template = template
        self._input_variables = input_variables or self._extract_variables(template)
        self._partial_variables = partial_variables or {}

    @staticmethod
    def _extract_variables(template: str) -> list[str]:
        """Extract variable names from template."""
        return list(set(re.findall(r"\{(\w+)\}", template)))

    def format(self, **kwargs: Any) -> str:
        """Format template with variables."""
        all_vars = {**self._partial_variables, **kwargs}
        missing = set(self._input_variables) - set(all_vars.keys())
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        return self._template.format(**all_vars)

    def partial(self, **kwargs: Any) -> PromptTemplate:
        """Create partial template with some variables filled."""
        new_partial = {**self._partial_variables, **kwargs}
        remaining = [v for v in self._input_variables if v not in new_partial]
        return PromptTemplate(self._template, input_variables=remaining, partial_variables=new_partial)

    @property
    def input_variables(self) -> list[str]:
        return self._input_variables


class ChatPromptTemplate:
    """Template for chat message construction."""

    def __init__(self, messages: list[tuple[MessageRole, str]]) -> None:
        self._message_templates = [(role, PromptTemplate(content)) for role, content in messages]

    def format_messages(self, **kwargs: Any) -> list[Message]:
        """Format all messages with variables."""
        return [Message(role=role, content=template.format(**kwargs))
                for role, template in self._message_templates]

    @classmethod
    def from_system_user(cls, system: str, user: str) -> ChatPromptTemplate:
        """Create template with system and user messages."""
        return cls([(MessageRole.SYSTEM, system), (MessageRole.USER, user)])


class ToolExecutor:
    """Execute tool calls from LLM responses."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}
        self._definitions: dict[str, ToolDefinition] = {}

    def register(self, name: str, func: Callable[..., Any],
                 definition: ToolDefinition) -> None:
        """Register a tool."""
        self._tools[name] = func
        self._definitions[name] = definition

    def get_definitions(self) -> list[ToolDefinition]:
        """Get all tool definitions."""
        return list(self._definitions.values())

    async def execute(self, tool_call: ToolCall) -> Any:
        """Execute a tool call."""
        if tool_call.name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_call.name}")
        func = self._tools[tool_call.name]
        result = func(**tool_call.arguments)
        if hasattr(result, "__await__"):
            result = await result
        return result

    async def execute_all(self, tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
        """Execute all tool calls and return results."""
        results = []
        for tc in tool_calls:
            try:
                result = await self.execute(tc)
                results.append({"tool_call_id": tc.id, "name": tc.name, "result": result, "error": None})
            except Exception as e:
                results.append({"tool_call_id": tc.id, "name": tc.name, "result": None, "error": str(e)})
        return results

    def create_tool_result_messages(self, results: list[dict[str, Any]]) -> list[Message]:
        """Create tool result messages for LLM."""
        return [Message(role=MessageRole.TOOL, content=json.dumps(r["result"]) if r["result"] else r["error"],
                       tool_call_id=r["tool_call_id"]) for r in results]


class InferenceChain:
    """Chain multiple inference steps."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self._steps: list[Callable[[Any], Any]] = []

    def add_step(self, step: Callable[[Any], Any]) -> InferenceChain:
        """Add processing step."""
        self._steps.append(step)
        return self

    async def run(self, initial_input: Any) -> Any:
        """Run chain with initial input."""
        result = initial_input
        for step in self._steps:
            result = step(result)
            if hasattr(result, "__await__"):
                result = await result
        return result


class StructuredInference(Generic[T]):
    """Generate structured output from LLM."""

    def __init__(self, client: LLMClient, output_parser: OutputParser[T]) -> None:
        self._client = client
        self._parser = output_parser

    async def generate(self, prompt: str, *, system: str | None = None,
                       max_retries: int = 2, **kwargs: Any) -> T:
        """Generate structured output."""
        format_instructions = self._parser.get_format_instructions()
        full_system = f"{system}\n\n{format_instructions}" if system else format_instructions
        messages = build_messages(prompt, system=full_system)
        for attempt in range(max_retries + 1):
            try:
                response = await self._client.complete_with_retry(messages, **kwargs)
                return self._parser.parse(response.content)
            except ParseError as e:
                if attempt >= max_retries:
                    raise
                logger.warning("parse_retry", attempt=attempt + 1, error=str(e))
                messages.append(Message(role=MessageRole.ASSISTANT, content=response.content))
                messages.append(Message(role=MessageRole.USER,
                                       content=f"Invalid format. Please try again.\n{format_instructions}"))
        raise ParseError("Max retries exceeded", expected_format=str(type(self._parser)))


class ConversationMemory:
    """Simple conversation memory."""

    def __init__(self, max_messages: int = 50) -> None:
        self._messages: list[Message] = []
        self._max_messages = max_messages

    def add(self, message: Message) -> None:
        """Add message to memory."""
        self._messages.append(message)
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages:]

    def add_user(self, content: str) -> None:
        """Add user message."""
        self.add(Message(role=MessageRole.USER, content=content))

    def add_assistant(self, content: str) -> None:
        """Add assistant message."""
        self.add(Message(role=MessageRole.ASSISTANT, content=content))

    def get_messages(self) -> list[Message]:
        """Get all messages."""
        return list(self._messages)

    def clear(self) -> None:
        """Clear memory."""
        self._messages.clear()


class InferenceLogger:
    """Log inference requests and responses."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def log_request(self, messages: list[Message], **kwargs: Any) -> str:
        """Log inference request."""
        entry_id = f"req_{len(self._entries)}"
        self._entries.append({
            "id": entry_id, "type": "request", "timestamp": datetime.now(timezone.utc).isoformat(),
            "messages": [{"role": m.role.value, "content": m.content[:200]} for m in messages],
            "kwargs": {k: str(v)[:100] for k, v in kwargs.items()}
        })
        return entry_id

    def log_response(self, request_id: str, response: LLMResponse) -> None:
        """Log inference response."""
        self._entries.append({
            "id": f"{request_id}_resp", "type": "response", "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_length": len(response.content), "finish_reason": response.finish_reason.value,
            "usage": response.usage.model_dump(), "latency_ms": response.latency_ms
        })

    def get_entries(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent log entries."""
        return self._entries[-limit:]


def create_json_parser(schema: dict[str, Any] | None = None) -> JSONOutputParser:
    """Create JSON output parser."""
    return JSONOutputParser(schema)


def create_pydantic_parser(model_class: type[T]) -> PydanticOutputParser[T]:
    """Create Pydantic output parser."""
    return PydanticOutputParser(model_class)


def create_prompt_template(template: str, **partial_vars: Any) -> PromptTemplate:
    """Create prompt template with optional partial variables."""
    return PromptTemplate(template, partial_variables=partial_vars or None)


async def generate_with_tools(client: LLMClient, messages: list[Message],
                              executor: ToolExecutor, *, max_iterations: int = 5,
                              system: str | None = None, **kwargs: Any) -> LLMResponse:
    """Generate response with automatic tool execution."""
    tools = executor.get_definitions()
    conversation = list(messages)
    for _ in range(max_iterations):
        response = await client.complete_with_retry(conversation, tools=tools,
                                                     system_prompt=system, **kwargs)
        if response.finish_reason != FinishReason.TOOL_CALL or not response.tool_calls:
            return response
        conversation.append(Message(role=MessageRole.ASSISTANT, content=response.content,
                                   tool_calls=response.tool_calls))
        results = await executor.execute_all(response.tool_calls)
        conversation.extend(executor.create_tool_result_messages(results))
    return response
