"""One-off diagnostic: verify the real Asta MCP connection contract.

Not part of the satyarepro package — this is a throwaway script to confirm
the server URL, auth header, and actual MCP tool names before wiring
satyarepro/client/asta.py up for real. Prints raw, unformatted JSON so the
full field set of each response is visible.

Usage:
    python scripts/verify_asta_connection.py
"""
from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
import os

from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client

api_key = os.environ["ASTA_API_KEY"]

_URL = "https://asta-tools.allen.ai/mcp/v1"
_HEADERS = {"x-api-key": api_key}

_DIVIDER = "=" * 80


def _print_divider(title: str) -> None:
    print(f"\n{_DIVIDER}\n{title}\n{_DIVIDER}")


def _tool_result_to_json(result: types.CallToolResult):
    """Return the most structured representation available for raw printing."""
    if result.structuredContent is not None:
        return result.structuredContent
    texts = []
    for block in result.content:
        if isinstance(block, types.TextContent):
            try:
                texts.append(json.loads(block.text))
            except json.JSONDecodeError:
                texts.append(block.text)
    return texts if len(texts) != 1 else texts[0]


async def main() -> None:
    async with streamablehttp_client(_URL, headers=_HEADERS) as (
        read_stream,
        write_stream,
        _get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # (a) list_tools — confirm real tool names + input schemas
            _print_divider("(a) session.list_tools()")
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                print(f"\n--- tool: {tool.name} ---")
                print(f"description: {tool.description}")
                print("input_schema:")
                print(json.dumps(tool.inputSchema, indent=2))

            tool_names = {t.name for t in tools_result.tools}
            snippet_search_name = "snippet_search" if "snippet_search" in tool_names else None
            get_citations_name = "get_citations" if "get_citations" in tool_names else None
            if snippet_search_name is None:
                print(
                    "\n[!] 'snippet_search' not found in list_tools() output — "
                    "inspect the names above and rerun with the real name substituted."
                )
            if get_citations_name is None:
                print(
                    "\n[!] 'get_citations' not found in list_tools() output — "
                    "inspect the names above and rerun with the real name substituted."
                )

            # (b) snippet_search — raw result structure
            _print_divider("(b) call_tool(snippet_search-equivalent)")
            if snippet_search_name:
                search_result = await session.call_tool(
                    snippet_search_name,
                    {"query": "SMOTE oversampling AUC preterm birth", "limit": 5},
                )
                print(f"isError: {search_result.isError}")
                print(json.dumps(_tool_result_to_json(search_result), indent=2, default=str))
            else:
                print("Skipped — no snippet-search-like tool name confirmed above.")

            # (c) get_citations — raw result structure
            _print_divider("(c) call_tool(get_citations-equivalent)")
            if get_citations_name:
                citations_result = await session.call_tool(
                    get_citations_name,
                    {"paper_id": "DOI:10.1016/j.artmed.2020.101987", "limit": 20},
                )
                print(f"isError: {citations_result.isError}")
                print(json.dumps(_tool_result_to_json(citations_result), indent=2, default=str))
            else:
                print("Skipped — no citations-like tool name confirmed above.")


if __name__ == "__main__":
    asyncio.run(main())
