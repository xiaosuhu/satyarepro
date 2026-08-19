from __future__ import annotations

import json
import os
from typing import Any

from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client

# Verified against a live `list_tools()` call (scripts/verify_asta_connection.py)
# on 2026-08-18. See that script's output for the full tool list and schemas.
_DEFAULT_ASTA_MCP_URL = "https://asta-tools.allen.ai/mcp/v1"
_ASTA_MCP_URL_ENV = "ASTA_MCP_URL"

# Confirmed: `x-api-key` header, not a bearer token.
_ASTA_API_KEY_ENV = "ASTA_API_KEY"

# Both tool names confirmed present in list_tools(). Note: get_citations'
# real argument is `paper_id` (not `doi`) — it accepts a DOI via a "DOI:"
# prefix, e.g. "DOI:10.1016/j.artmed.2020.101987". AstaClient.get_citations()
# keeps its own `doi` parameter name and maps it internally so callers
# (e.g. outcome_distribution_checker.py) are unaffected.
_SNIPPET_SEARCH_TOOL = "snippet_search"
_CITATIONS_TOOL = "get_citations"


class AstaClient:
    """Thin async wrapper around the Asta MCP server (streamable HTTP transport).

    Modeled on satyarepro/client/claude.py: cheap to construct, lazy-created
    by callers that don't need to inject a mock. Each call opens its own
    MCP session — streamable_http has no long-lived state worth pooling here.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.url = url or os.environ.get(_ASTA_MCP_URL_ENV, _DEFAULT_ASTA_MCP_URL)
        self._api_key = api_key or os.environ.get(_ASTA_API_KEY_ENV, "")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            return {}
        return {"x-api-key": self._api_key}

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        async with streamablehttp_client(
            self.url, headers=self._headers(), timeout=self.timeout
        ) as (read_stream, write_stream, _get_session_id):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                return _unwrap_result(result)

    async def snippet_search(
        self,
        query: str,
        limit: int = 10,
        **extra: Any,
    ) -> list[dict[str, Any]]:
        """Search Asta for paper snippets relevant to `query`.

        Returns a list of result dicts as reported by the Asta MCP server
        (fields vary; callers should access via .get with defaults). Raises
        on transport/tool errors — callers decide how to surface failures.
        """
        payload = await self._call_tool(_SNIPPET_SEARCH_TOOL, {"query": query, "limit": limit, **extra})
        return _as_result_list(payload)

    async def get_citations(
        self,
        doi: str,
        limit: int = 20,
        **extra: Any,
    ) -> list[dict[str, Any]]:
        """Fetch papers citing `doi` via the Asta MCP server.

        The underlying MCP tool takes `paper_id` (accepting a "DOI:" prefixed
        id), not `doi` — mapped internally so this method's signature stays
        stable for callers.
        """
        payload = await self._call_tool(
            _CITATIONS_TOOL, {"paper_id": f"DOI:{doi}", "limit": limit, **extra}
        )
        return _as_result_list(payload)


def _unwrap_result(result: types.CallToolResult) -> Any:
    if result.isError:
        raise RuntimeError(f"Asta MCP tool call failed: {_first_text(result)}")
    if result.structuredContent is not None:
        return result.structuredContent
    text = _first_text(result)
    if text is None:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}


def _first_text(result: types.CallToolResult) -> str | None:
    for block in result.content:
        if isinstance(block, types.TextContent):
            return block.text
    return None


def _as_result_list(payload: Any) -> list[dict[str, Any]]:
    """Unwrap the Asta MCP response envelope down to the list of items.

    Confirmed shapes (scripts/verify_asta_connection.py, 2026-08-18):
    - snippet_search: {"result": {"data": [...], "retrievalVersion": ...}}
    - get_citations:  {"result": [...]}
    """
    if isinstance(payload, dict) and "result" in payload:
        payload = payload["result"]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return data
    return []
