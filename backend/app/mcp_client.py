from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import queue
import shlex
import subprocess
import threading
from typing import Any

import httpx

from app.config import settings


logger = logging.getLogger("uvicorn.error")


class MCPError(RuntimeError):
    pass


class MCPTransportError(MCPError):
    pass


class MCPProtocolError(MCPError):
    pass


@dataclass
class MCPServerConfig:
    name: str
    transport: str
    command: list[str] | None = None
    url: str = ""
    env: dict[str, str] | None = None
    timeout_seconds: float = 30.0
    initialize_timeout_seconds: float = 20.0


class MCPTransport:
    def start(self) -> None:
        return None

    def send(self, message: dict[str, Any]) -> None:
        raise NotImplementedError

    def receive(self, timeout: float) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class StdioMCPTransport(MCPTransport):
    def __init__(self, command: list[str], *, env: dict[str, str] | None = None) -> None:
        self.command = command
        self.env = env or {}
        self.process: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[dict[str, Any] | Exception] = queue.Queue()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if self.process is not None:
            return
        merged_env = {**os.environ, **self.env}
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=merged_env,
        )
        self._threads = [
            threading.Thread(target=self._read_stdout, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                self._stdout_queue.put(json.loads(stripped))
            except json.JSONDecodeError as exc:
                self._stdout_queue.put(MCPProtocolError(f"Invalid MCP JSON line: {stripped[:200]} ({exc})"))
        if self.process.poll() is not None:
            self._stdout_queue.put(MCPTransportError(f"MCP stdio server exited with code {self.process.returncode}"))

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            stripped = line.strip()
            if stripped:
                logger.info("mcp_stdio_stderr command=%s line=%s", self.command[0], stripped[:500])

    def send(self, message: dict[str, Any]) -> None:
        if self.process is None:
            self.start()
        assert self.process is not None
        if self.process.poll() is not None:
            raise MCPTransportError(f"MCP stdio server already exited with code {self.process.returncode}")
        if self.process.stdin is None:
            raise MCPTransportError("MCP stdio server stdin is unavailable")
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def receive(self, timeout: float) -> dict[str, Any]:
        try:
            item = self._stdout_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise MCPTransportError("Timed out waiting for MCP stdio response") from exc
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
        self.process = None


class StreamableHTTPMCPTransport(MCPTransport):
    def __init__(self, url: str, *, headers: dict[str, str] | None = None, timeout_seconds: float = 30.0) -> None:
        self.url = url
        self.headers = headers or {}
        self.timeout_seconds = timeout_seconds
        self.session_id = ""
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self._client = httpx.Client(timeout=timeout_seconds)

    def send(self, message: dict[str, Any]) -> None:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": settings.mcp_protocol_version,
            **self.headers,
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        response = self._client.post(self.url, json=message, headers=headers)
        response.raise_for_status()
        self.session_id = response.headers.get("Mcp-Session-Id", self.session_id)
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            for line in response.text.splitlines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    self._responses.put(json.loads(payload))
        elif response.content:
            self._responses.put(response.json())

    def receive(self, timeout: float) -> dict[str, Any]:
        try:
            return self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            raise MCPTransportError("Timed out waiting for MCP HTTP response") from exc

    def close(self) -> None:
        self._client.close()


class MCPClient:
    def __init__(self, config: MCPServerConfig, transport: MCPTransport) -> None:
        self.config = config
        self.transport = transport
        self._next_id = 1
        self.initialized = False
        self.server_info: dict[str, Any] = {}
        self.server_capabilities: dict[str, Any] = {}

    def __enter__(self) -> MCPClient:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def start(self) -> None:
        self.transport.start()
        self.initialize()

    def close(self) -> None:
        self.transport.close()

    def _request_payload(self, method: str, params: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        request_id = self._next_id
        self._next_id += 1
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        return request_id, payload

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self.transport.send(payload)

    def request(self, method: str, params: dict[str, Any] | None = None, *, timeout: float | None = None) -> Any:
        request_id, payload = self._request_payload(method, params)
        self.transport.send(payload)
        deadline_timeout = timeout or self.config.timeout_seconds
        while True:
            response = self.transport.receive(deadline_timeout)
            if "method" in response and "id" not in response:
                continue
            if response.get("id") != request_id:
                continue
            if "error" in response:
                error = response["error"]
                raise MCPProtocolError(f"MCP {method} failed: {error}")
            return response.get("result", {})

    def initialize(self) -> None:
        if self.initialized:
            return
        result = self.request(
            "initialize",
            {
                "protocolVersion": settings.mcp_protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "research-copilot-runtime", "version": settings.app_version},
            },
            timeout=self.config.initialize_timeout_seconds,
        )
        if not isinstance(result, dict):
            raise MCPProtocolError("MCP initialize result must be an object")
        self.server_info = dict(result.get("serverInfo") or {})
        self.server_capabilities = dict(result.get("capabilities") or {})
        self.notify("notifications/initialized")
        self.initialized = True

    def list_tools(self) -> list[dict[str, Any]]:
        return self._list_paginated("tools/list", "tools")

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        result = self.request("tools/call", {"name": name, "arguments": arguments or {}})
        return dict(result or {})

    def list_resources(self) -> list[dict[str, Any]]:
        return self._list_paginated("resources/list", "resources")

    def read_resource(self, uri: str) -> dict[str, Any]:
        result = self.request("resources/read", {"uri": uri})
        return dict(result or {})

    def list_prompts(self) -> list[dict[str, Any]]:
        return self._list_paginated("prompts/list", "prompts")

    def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        result = self.request("prompts/get", {"name": name, "arguments": arguments or {}})
        return dict(result or {})

    def _list_paginated(self, method: str, key: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        cursor = ""
        while True:
            params = {"cursor": cursor} if cursor else None
            result = self.request(method, params)
            if not isinstance(result, dict):
                raise MCPProtocolError(f"MCP {method} result must be an object")
            page_items = result.get(key) or []
            if not isinstance(page_items, list):
                raise MCPProtocolError(f"MCP {method} field {key} must be a list")
            items.extend(dict(item) for item in page_items if isinstance(item, dict))
            cursor = str(result.get("nextCursor") or "")
            if not cursor:
                return items


def github_mcp_config() -> MCPServerConfig:
    if not settings.mcp_enabled:
        raise MCPTransportError("MCP is disabled. Set MCP_ENABLED=true to enable MCP execution.")
    if not settings.mcp_github_enabled:
        raise MCPTransportError("GitHub MCP is disabled. Set MCP_GITHUB_ENABLED=true to enable it.")
    if not settings.github_personal_access_token:
        raise MCPTransportError("GITHUB_PERSONAL_ACCESS_TOKEN is required for GitHub MCP execution.")
    transport = settings.mcp_github_transport.strip().lower()
    command = shlex.split(settings.mcp_github_command) if settings.mcp_github_command else []
    return MCPServerConfig(
        name="github",
        transport=transport,
        command=command,
        url=settings.mcp_github_url,
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": settings.github_personal_access_token},
        timeout_seconds=settings.mcp_request_timeout_seconds,
        initialize_timeout_seconds=settings.mcp_initialize_timeout_seconds,
    )


def create_github_mcp_client() -> MCPClient:
    config = github_mcp_config()
    if config.transport == "stdio":
        if not config.command:
            raise MCPTransportError("MCP_GITHUB_COMMAND is required when MCP_GITHUB_TRANSPORT=stdio")
        return MCPClient(config, StdioMCPTransport(config.command, env=config.env))
    if config.transport in {"streamable_http", "http"}:
        if not config.url:
            raise MCPTransportError("MCP_GITHUB_URL is required when MCP_GITHUB_TRANSPORT=streamable_http")
        headers = {"Authorization": f"Bearer {settings.github_personal_access_token}"}
        return MCPClient(
            config,
            StreamableHTTPMCPTransport(config.url, headers=headers, timeout_seconds=config.timeout_seconds),
        )
    raise MCPTransportError(f"Unsupported GitHub MCP transport: {config.transport}")
