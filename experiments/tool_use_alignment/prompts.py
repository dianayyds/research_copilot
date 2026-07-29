from __future__ import annotations

import json
from typing import Any


TOOL_CATALOG: list[dict[str, Any]] = [
    {
        "name": "local_rag_search",
        "description": "检索本项目或全局知识库里的论文、笔记、代码说明和项目资料。",
        "input_schema": {"query": "检索问题，字符串"},
        "read_only": True,
        "risk_level": "low",
    },
    {
        "name": "public_web_search",
        "description": "查询公开网络资料、官网、最新信息或用户明确要求联网搜索的问题。",
        "input_schema": {"query": "搜索问题，字符串"},
        "read_only": True,
        "risk_level": "medium",
    },
    {
        "name": "weather_lookup",
        "description": "查询实时天气、气温、降雨、风速或外出活动环境信息。",
        "input_schema": {"query": "包含城市或地点的天气问题，字符串"},
        "read_only": True,
        "risk_level": "low",
    },
    {
        "name": "memory_read",
        "description": "读取项目记忆，适用于刚才说过什么、项目记住了什么、用户偏好等问题。",
        "input_schema": {"query": "记忆检索问题，字符串，可省略"},
        "read_only": True,
        "risk_level": "low",
    },
    {
        "name": "memory_write",
        "description": "在用户明确要求记住某个偏好、约束或事实时写入短期工作记忆。",
        "input_schema": {"key": "记忆键", "content": "要记住的内容", "importance": "0-1，可选"},
        "read_only": False,
        "risk_level": "medium",
    },
    {
        "name": "todo_list",
        "description": "列出当前项目 TODO。",
        "input_schema": {"status": "可选，按状态过滤"},
        "read_only": True,
        "risk_level": "low",
    },
    {
        "name": "todo_create",
        "description": "在用户明确要求创建、添加或生成待办时，为当前项目创建 TODO。",
        "input_schema": {"title": "TODO 标题", "description": "描述，可选", "priority": "low|medium|high，可选"},
        "read_only": False,
        "risk_level": "medium",
    },
    {
        "name": "asset_list",
        "description": "列出知识库资产，适用于用户询问有哪些资料、文档、论文或资产。",
        "input_schema": {},
        "read_only": True,
        "risk_level": "low",
    },
    {
        "name": "calculator",
        "description": "执行简单算术和指标 sanity check。",
        "input_schema": {"expression": "算术表达式，字符串"},
        "read_only": True,
        "risk_level": "low",
    },
]

VALID_ACTIONS = tuple(tool["name"] for tool in TOOL_CATALOG) + ("final_answer",)
TOOL_ACTIONS = tuple(tool["name"] for tool in TOOL_CATALOG)

ACTION_ALIASES = {
    "answer": "final_answer",
    "final": "final_answer",
    "finish": "final_answer",
    "rag": "local_rag_search",
    "search_local": "local_rag_search",
    "local_search": "local_rag_search",
    "web_search": "public_web_search",
    "search_web": "public_web_search",
    "weather": "weather_lookup",
    "memory": "memory_read",
    "remember": "memory_write",
    "create_todo": "todo_create",
    "list_todo": "todo_list",
    "list_todos": "todo_list",
    "list_assets": "asset_list",
    "calculate": "calculator",
}

SYSTEM_INSTRUCTION = (
    "你是 Research Copilot 的工具决策器。你的任务是根据用户问题和工具目录选择恰当 action。"
    "只返回一个 JSON 对象，不要 Markdown，不要解释，不要输出 <think>。"
    "如果不需要工具，action 使用 final_answer，并在 arguments.answer 中给出简短回答。"
)


def decision_json(thought: str, action: str, arguments: dict[str, Any]) -> str:
    payload = {"thought": thought, "action": action, "arguments": arguments}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_prompt(user_query: str) -> str:
    catalog = json.dumps(TOOL_CATALOG, ensure_ascii=False, indent=2)
    schema = '{"thought":"简短思考","action":"工具名或 final_answer","arguments":{"参数名":"参数值"}}'
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"可用工具目录：\n{catalog}\n\n"
        f"输出 JSON schema：\n{schema}\n\n"
        "决策规则：\n"
        "- 天气、气温、降雨、风速、出行环境必须用 weather_lookup。\n"
        "- 最新信息、官网、联网、公开资料必须用 public_web_search。\n"
        "- 本项目资料、论文、代码、知识库证据必须用 local_rag_search。\n"
        "- 明确要求记住时用 memory_write；询问刚才或已记住内容时用 memory_read。\n"
        "- 明确要求创建待办时才用 todo_create；查看待办时用 todo_list。\n"
        "- 资料清单用 asset_list；算术表达式用 calculator。\n"
        "- 翻译、改写、解释常识且不需要外部资料时用 final_answer。\n\n"
        f"用户问题：{user_query}\n\n"
        "只返回 JSON："
    )
