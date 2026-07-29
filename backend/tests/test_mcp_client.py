from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

os.environ["DATABASE_URL"] = "sqlite:////tmp/research_copilot_test.db"
os.environ["VECTOR_STORE_PROVIDER"] = "stub"
os.environ["EMBEDDING_PROVIDER"] = "stub"
os.environ["LLM_PROVIDER"] = "stub"
os.environ["LLM_API_KEY"] = ""

from app.mcp_client import (
    MCPClient,
    MCPProtocolError,
    MCPServerConfig,
    StdioMCPTransport,
    StreamableHTTPMCPTransport,
)


def write_fake_mcp_server(path: Path) -> None:
    path.write_text(
        """
import json
import sys

for raw in sys.stdin:
    message = json.loads(raw)
    method = message.get("method")
    request_id = message.get("id")
    if method == "notifications/initialized":
        continue
    if method == "initialize":
        result = {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
            "serverInfo": {"name": "fake", "version": "1.0"},
        }
    elif method == "tools/list":
        cursor = (message.get("params") or {}).get("cursor")
        if cursor:
            result = {"tools": [{"name": "create_issue", "description": "Create issue"}]}
        else:
            result = {
                "tools": [{"name": "list_issues", "description": "List issues", "annotations": {"readOnlyHint": True}}],
                "nextCursor": "page-2",
            }
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": "tool result"}]}
    elif method == "resources/list":
        result = {"resources": [{"uri": "repo://README.md", "name": "README"}]}
    elif method == "resources/read":
        result = {"contents": [{"uri": "repo://README.md", "text": "readme body"}]}
    elif method == "prompts/list":
        result = {"prompts": [{"name": "summarize", "description": "Summarize"}]}
    elif method == "prompts/get":
        result = {"messages": [{"role": "user", "content": {"type": "text", "text": "prompt body"}}]}
    elif method == "explode":
        print(json.dumps({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000, "message": "boom"}}), flush=True)
        continue
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}), flush=True)
""".strip(),
        encoding="utf-8",
    )


def test_stdio_mcp_client_supports_tools_resources_prompts(tmp_path: Path) -> None:
    server = tmp_path / "fake_mcp_server.py"
    write_fake_mcp_server(server)
    config = MCPServerConfig(
        name="fake",
        transport="stdio",
        command=[sys.executable, str(server)],
        timeout_seconds=2.0,
        initialize_timeout_seconds=2.0,
    )

    with MCPClient(config, StdioMCPTransport(config.command or [])) as client:
        tools = client.list_tools()
        assert [tool["name"] for tool in tools] == ["list_issues", "create_issue"]
        assert client.call_tool("list_issues")["content"][0]["text"] == "tool result"
        assert client.list_resources()[0]["uri"] == "repo://README.md"
        assert client.read_resource("repo://README.md")["contents"][0]["text"] == "readme body"
        assert client.list_prompts()[0]["name"] == "summarize"
        assert client.get_prompt("summarize")["messages"][0]["role"] == "user"
        with pytest.raises(MCPProtocolError):
            client.request("explode")


def test_streamable_http_transport_stores_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    class FakeResponse:
        headers = {"Mcp-Session-Id": "session-1", "content-type": "application/json"}
        content = b'{"jsonrpc":"2.0","id":1,"result":{"ok":true}}'
        text = content.decode()

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return json.loads(self.text)

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def post(self, url: str, json: dict, headers: dict) -> FakeResponse:
            calls.append({"url": url, "json": json, "headers": headers})
            return FakeResponse()

        def close(self) -> None:
            return None

    monkeypatch.setattr("app.mcp_client.httpx.Client", FakeClient)
    transport = StreamableHTTPMCPTransport("https://mcp.example.test", timeout_seconds=3.0)

    transport.send({"jsonrpc": "2.0", "id": 1, "method": "initialize"})

    assert transport.session_id == "session-1"
    assert transport.receive(1.0)["result"]["ok"] is True
    assert calls[0]["headers"]["Accept"] == "application/json, text/event-stream"
    assert calls[0]["headers"]["MCP-Protocol-Version"] == "2025-06-18"
