"""Unit tests for inference module."""
from __future__ import annotations
import pytest
from pydantic import BaseModel
from solace_ml.inference import (
    OutputFormat, ParseError, JSONOutputParser, PydanticOutputParser,
    ListOutputParser, PromptTemplate, ChatPromptTemplate, ToolExecutor,
    InferenceChain, ConversationMemory, InferenceLogger,
    create_json_parser, create_pydantic_parser, create_prompt_template,
)
from solace_ml.llm_client import Message, MessageRole, ToolDefinition, ToolCall


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_format_values(self):
        assert OutputFormat.TEXT.value == "text"
        assert OutputFormat.JSON.value == "json"
        assert OutputFormat.MARKDOWN.value == "markdown"


class TestParseError:
    """Tests for ParseError exception."""

    def test_create_error(self):
        error = ParseError("Invalid JSON", raw_output="not json", expected_format="json")
        assert str(error) == "Invalid JSON"
        assert error.raw_output == "not json"
        assert error.expected_format == "json"


class TestJSONOutputParser:
    """Tests for JSONOutputParser class."""

    @pytest.fixture
    def parser(self):
        return JSONOutputParser()

    def test_parse_simple_json(self, parser):
        result = parser.parse('{"key": "value"}')
        assert result["key"] == "value"

    def test_parse_json_in_markdown(self, parser):
        text = '```json\n{"key": "value"}\n```'
        result = parser.parse(text)
        assert result["key"] == "value"

    def test_parse_json_in_code_block(self, parser):
        text = '```\n{"key": "value"}\n```'
        result = parser.parse(text)
        assert result["key"] == "value"

    def test_parse_invalid_json(self, parser):
        with pytest.raises(ParseError):
            parser.parse("not json")

    def test_format_instructions(self, parser):
        instructions = parser.get_format_instructions()
        assert "JSON" in instructions

    def test_format_instructions_with_schema(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        parser = JSONOutputParser(schema=schema)
        instructions = parser.get_format_instructions()
        assert "schema" in instructions


class TestPydanticOutputParser:
    """Tests for PydanticOutputParser class."""

    class TestModel(BaseModel):
        name: str
        age: int

    @pytest.fixture
    def parser(self):
        return PydanticOutputParser(self.TestModel)

    def test_parse_valid_json(self, parser):
        result = parser.parse('{"name": "John", "age": 30}')
        assert result.name == "John"
        assert result.age == 30

    def test_parse_json_in_markdown(self, parser):
        text = '```json\n{"name": "Jane", "age": 25}\n```'
        result = parser.parse(text)
        assert result.name == "Jane"

    def test_parse_invalid_json(self, parser):
        with pytest.raises(ParseError):
            parser.parse("not json")

    def test_parse_validation_error(self, parser):
        with pytest.raises(ParseError):
            parser.parse('{"name": "John", "age": "not a number"}')

    def test_format_instructions(self, parser):
        instructions = parser.get_format_instructions()
        assert "JSON" in instructions
        assert "name" in instructions


class TestListOutputParser:
    """Tests for ListOutputParser class."""

    @pytest.fixture
    def parser(self):
        return ListOutputParser()

    def test_parse_simple_list(self, parser):
        text = "item1\nitem2\nitem3"
        result = parser.parse(text)
        assert len(result) == 3
        assert result[0] == "item1"

    def test_parse_bulleted_list(self, parser):
        text = "- item1\n- item2\n- item3"
        result = parser.parse(text)
        assert result[0] == "item1"
        assert result[1] == "item2"

    def test_parse_numbered_list(self, parser):
        text = "1. item1\n2. item2\n3. item3"
        result = parser.parse(text)
        assert result[0] == "item1"

    def test_parse_empty_lines(self, parser):
        text = "item1\n\nitem2\n\n\nitem3"
        result = parser.parse(text)
        assert len(result) == 3

    def test_format_instructions(self, parser):
        instructions = parser.get_format_instructions()
        assert "list" in instructions


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    def test_simple_template(self):
        template = PromptTemplate("Hello, {name}!")
        result = template.format(name="World")
        assert result == "Hello, World!"

    def test_multiple_variables(self):
        template = PromptTemplate("Hello, {name}! You are {age} years old.")
        result = template.format(name="John", age=30)
        assert result == "Hello, John! You are 30 years old."

    def test_extract_variables(self):
        template = PromptTemplate("Hello, {name}! You are {age} years old.")
        assert "name" in template.input_variables
        assert "age" in template.input_variables

    def test_missing_variable(self):
        template = PromptTemplate("Hello, {name}!")
        with pytest.raises(ValueError, match="Missing"):
            template.format()

    def test_partial_template(self):
        template = PromptTemplate("Hello, {name}! You are {age} years old.")
        partial = template.partial(name="John")
        assert "name" not in partial.input_variables
        assert "age" in partial.input_variables
        result = partial.format(age=30)
        assert result == "Hello, John! You are 30 years old."


class TestChatPromptTemplate:
    """Tests for ChatPromptTemplate class."""

    def test_format_messages(self):
        template = ChatPromptTemplate([
            (MessageRole.SYSTEM, "You are {role}."),
            (MessageRole.USER, "Hello, {name}!")
        ])
        messages = template.format_messages(role="helpful", name="Assistant")
        assert len(messages) == 2
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[0].content == "You are helpful."
        assert messages[1].content == "Hello, Assistant!"

    def test_from_system_user(self):
        template = ChatPromptTemplate.from_system_user(
            "You are {role}.",
            "What is {topic}?"
        )
        messages = template.format_messages(role="expert", topic="AI")
        assert len(messages) == 2
        assert messages[1].content == "What is AI?"


class TestToolExecutor:
    """Tests for ToolExecutor class."""

    @pytest.fixture
    def executor(self):
        exec = ToolExecutor()
        exec.register("add", lambda a, b: a + b,
                     ToolDefinition(name="add", description="Add two numbers"))
        exec.register("greet", lambda name: f"Hello, {name}!",
                     ToolDefinition(name="greet", description="Greet someone"))
        return exec

    def test_get_definitions(self, executor):
        defs = executor.get_definitions()
        assert len(defs) == 2
        names = [d.name for d in defs]
        assert "add" in names
        assert "greet" in names

    @pytest.mark.asyncio
    async def test_execute(self, executor):
        tc = ToolCall(id="tc_1", name="add", arguments={"a": 2, "b": 3})
        result = await executor.execute(tc)
        assert result == 5

    @pytest.mark.asyncio
    async def test_execute_all(self, executor):
        tool_calls = [
            ToolCall(id="tc_1", name="add", arguments={"a": 1, "b": 2}),
            ToolCall(id="tc_2", name="greet", arguments={"name": "World"})
        ]
        results = await executor.execute_all(tool_calls)
        assert len(results) == 2
        assert results[0]["result"] == 3
        assert results[1]["result"] == "Hello, World!"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, executor):
        tc = ToolCall(id="tc_1", name="unknown", arguments={})
        with pytest.raises(ValueError, match="Unknown tool"):
            await executor.execute(tc)

    @pytest.mark.asyncio
    async def test_execute_with_error(self, executor):
        def failing_func():
            raise ValueError("Intentional error")
        executor.register("fail", failing_func, ToolDefinition(name="fail", description="Fails"))
        tc = ToolCall(id="tc_1", name="fail", arguments={})
        results = await executor.execute_all([tc])
        assert results[0]["error"] is not None

    def test_create_tool_result_messages(self, executor):
        results = [
            {"tool_call_id": "tc_1", "name": "add", "result": 5, "error": None},
            {"tool_call_id": "tc_2", "name": "greet", "result": None, "error": "Failed"}
        ]
        messages = executor.create_tool_result_messages(results)
        assert len(messages) == 2
        assert messages[0].role == MessageRole.TOOL
        assert messages[0].tool_call_id == "tc_1"


class TestConversationMemory:
    """Tests for ConversationMemory class."""

    @pytest.fixture
    def memory(self):
        return ConversationMemory(max_messages=10)

    def test_add_message(self, memory):
        memory.add(Message(role=MessageRole.USER, content="Hello"))
        messages = memory.get_messages()
        assert len(messages) == 1
        assert messages[0].content == "Hello"

    def test_add_user(self, memory):
        memory.add_user("Hello")
        messages = memory.get_messages()
        assert messages[0].role == MessageRole.USER

    def test_add_assistant(self, memory):
        memory.add_assistant("Hi there!")
        messages = memory.get_messages()
        assert messages[0].role == MessageRole.ASSISTANT

    def test_max_messages(self):
        memory = ConversationMemory(max_messages=3)
        for i in range(5):
            memory.add_user(f"Message {i}")
        messages = memory.get_messages()
        assert len(messages) == 3
        assert messages[0].content == "Message 2"

    def test_clear(self, memory):
        memory.add_user("Hello")
        memory.clear()
        assert len(memory.get_messages()) == 0


class TestInferenceLogger:
    """Tests for InferenceLogger class."""

    @pytest.fixture
    def log(self):
        return InferenceLogger()

    def test_log_request(self, log):
        messages = [Message(role=MessageRole.USER, content="Hello")]
        request_id = log.log_request(messages, temperature=0.7)
        assert request_id.startswith("req_")

    def test_get_entries(self, log):
        messages = [Message(role=MessageRole.USER, content="Hello")]
        log.log_request(messages)
        entries = log.get_entries()
        assert len(entries) == 1
        assert entries[0]["type"] == "request"


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_json_parser(self):
        parser = create_json_parser()
        assert isinstance(parser, JSONOutputParser)

    def test_create_json_parser_with_schema(self):
        schema = {"type": "object"}
        parser = create_json_parser(schema)
        assert parser._schema == schema

    def test_create_pydantic_parser(self):
        class Model(BaseModel):
            name: str
        parser = create_pydantic_parser(Model)
        assert isinstance(parser, PydanticOutputParser)

    def test_create_prompt_template(self):
        template = create_prompt_template("Hello, {name}!")
        assert isinstance(template, PromptTemplate)

    def test_create_prompt_template_with_partial(self):
        template = create_prompt_template("Hello, {name}! Age: {age}", age=30)
        result = template.format(name="John")
        assert result == "Hello, John! Age: 30"
