import pytest

from satyarepro.client.mock import MockClient
from satyarepro.types import CompletionResponse, ToolCall, ToolSchema, Usage


@pytest.fixture
def client() -> MockClient:
    return MockClient()


async def test_complete_returns_default(client: MockClient) -> None:
    resp = await client.complete([{"role": "user", "content": "hello"}])
    assert isinstance(resp, CompletionResponse)
    assert resp.content
    assert client.calls[0]["type"] == "complete"


async def test_complete_with_tools_default(client: MockClient) -> None:
    tools = [ToolSchema("t", "A tool", {"type": "object", "properties": {}})]
    resp = await client.complete_with_tools([{"role": "user", "content": "hi"}], tools)
    assert resp.stop_reason == "end_turn"


async def test_enqueue_used_in_order(client: MockClient) -> None:
    first = CompletionResponse(
        content="first",
        raw_content=[{"type": "text", "text": "first"}],
        usage=Usage(10, 5),
    )
    second = CompletionResponse(
        content="second",
        raw_content=[{"type": "text", "text": "second"}],
        usage=Usage(20, 8),
    )
    client.enqueue(first)
    client.enqueue(second)

    r1 = await client.complete([{"role": "user", "content": "a"}])
    r2 = await client.complete([{"role": "user", "content": "b"}])
    assert r1.content == "first"
    assert r2.content == "second"


async def test_tool_use_flow(client: MockClient) -> None:
    tool_response = CompletionResponse(
        content="",
        raw_content=[
            {"type": "tool_use", "id": "tu_01", "name": "pubmed_search", "input": {"query": "x"}}
        ],
        tool_calls=[ToolCall(id="tu_01", name="pubmed_search", input={"query": "x"})],
        usage=Usage(50, 20),
        stop_reason="tool_use",
    )
    client.enqueue(tool_response)

    resp = await client.complete_with_tools(
        [{"role": "user", "content": "audit PMID 123"}],
        [ToolSchema("pubmed_search", "Search", {"type": "object", "properties": {}})],
    )
    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls[0].name == "pubmed_search"
