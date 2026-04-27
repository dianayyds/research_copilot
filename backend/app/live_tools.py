from __future__ import annotations

from dataclasses import dataclass, field
import html
import re
from typing import Any
from urllib.parse import quote_plus

import httpx

from app.config import settings


WEATHER_TERMS = ("天气", "气温", "温度", "下雨", "降雨", "降水", "风力", "风速", "冷不冷", "热不热")
PUBLIC_WEB_TERMS = ("联网", "网上", "网络上", "公开资料", "搜索", "最新", "新闻", "官网", "实时")
GENERAL_PUBLIC_TERMS = ("是谁", "谁是", "是什么", "在哪里", "哪一年", "什么时候", "多少", "几岁")
LOCAL_CONTEXT_TERMS = (
    "项目",
    "本项目",
    "系统",
    "资产",
    "论文",
    "代码",
    "仓库",
    "刚才",
    "记录",
    "检索",
    "记忆",
    "TODO",
    "todo",
)

KNOWN_LOCATIONS: dict[str, tuple[float, float, str, str]] = {
    "北京": (39.9042, 116.4074, "Asia/Shanghai", "北京"),
    "北京市": (39.9042, 116.4074, "Asia/Shanghai", "北京"),
    "上海": (31.2304, 121.4737, "Asia/Shanghai", "上海"),
    "上海市": (31.2304, 121.4737, "Asia/Shanghai", "上海"),
    "广州": (23.1291, 113.2644, "Asia/Shanghai", "广州"),
    "深圳": (22.5431, 114.0579, "Asia/Shanghai", "深圳"),
    "杭州": (30.2741, 120.1551, "Asia/Shanghai", "杭州"),
    "南京": (32.0603, 118.7969, "Asia/Shanghai", "南京"),
    "成都": (30.5728, 104.0668, "Asia/Shanghai", "成都"),
    "武汉": (30.5928, 114.3055, "Asia/Shanghai", "武汉"),
    "西安": (34.3416, 108.9398, "Asia/Shanghai", "西安"),
    "重庆": (29.5630, 106.5516, "Asia/Shanghai", "重庆"),
    "天津": (39.3434, 117.3616, "Asia/Shanghai", "天津"),
    "香港": (22.3193, 114.1694, "Asia/Hong_Kong", "香港"),
    "台北": (25.0330, 121.5654, "Asia/Taipei", "台北"),
    "纽约": (40.7128, -74.0060, "America/New_York", "纽约"),
    "伦敦": (51.5072, -0.1276, "Europe/London", "伦敦"),
    "东京": (35.6764, 139.6500, "Asia/Tokyo", "东京"),
}

WEATHER_CODES = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "霜雾",
    51: "小毛毛雨",
    53: "中等毛毛雨",
    55: "较强毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "短时小阵雨",
    81: "短时中阵雨",
    82: "强阵雨",
    95: "雷暴",
}

@dataclass(frozen=True)
class LiveToolRoute:
    intent: str
    skill: str
    tool_name: str
    reason: str
    confidence: float
    planner_mode: str = "tool_router"


@dataclass
class LiveToolEvidence:
    title: str
    snippet: str
    source_path: str
    score: float = 1.0
    tags: list[str] = field(default_factory=list)


@dataclass
class LiveToolResult:
    answer: str
    evidence: list[LiveToolEvidence]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float
    timezone: str


def select_live_tool(query: str) -> LiveToolRoute | None:
    if not settings.live_tools_enabled:
        return None
    normalized = query.strip()
    if any(term in normalized for term in WEATHER_TERMS):
        return LiveToolRoute(
            intent="realtime_weather",
            skill="weather_qa",
            tool_name="weather_lookup",
            reason="用户询问天气/气温等实时信息，需要调用外部天气工具，而不是检索本地知识库。",
            confidence=0.92,
        )
    if settings.public_web_search_enabled and any(term in normalized for term in PUBLIC_WEB_TERMS):
        return LiveToolRoute(
            intent="public_web_lookup",
            skill="public_web_qa",
            tool_name="public_web_search",
            reason="用户显式询问公开网络资料或最新信息，需要调用公开搜索工具。",
            confidence=0.74,
        )
    if settings.public_web_search_enabled and looks_like_general_public_question(normalized):
        return LiveToolRoute(
            intent="public_web_lookup",
            skill="public_web_qa",
            tool_name="public_web_search",
            reason="用户询问通用公开事实，且问题没有明显绑定当前项目资料，优先调用公开搜索工具。",
            confidence=0.66,
        )
    return None


def looks_like_general_public_question(query: str) -> bool:
    if any(term in query for term in LOCAL_CONTEXT_TERMS):
        return False
    if "《" in query or "》" in query:
        return False
    return any(term in query for term in GENERAL_PUBLIC_TERMS)


def execute_live_tool(route: LiveToolRoute, query: str) -> LiveToolResult:
    if route.tool_name == "weather_lookup":
        return weather_lookup(query)
    if route.tool_name == "public_web_search":
        return public_web_search(query)
    raise ValueError(f"Unsupported live tool: {route.tool_name}")


def weather_lookup(query: str) -> LiveToolResult:
    location = resolve_location(query)
    if location is None:
        raise ValueError("没有识别出要查询天气的地点，请在问题里包含城市名。")
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "current": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
                "wind_direction_10m",
            ]
        ),
        "daily": ",".join(
            [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_probability_max",
            ]
        ),
        "timezone": location.timezone,
        "forecast_days": 3,
    }
    response = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params,
        timeout=settings.live_tool_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    current = payload.get("current", {})
    daily = payload.get("daily", {})
    current_code = int(current.get("weather_code", daily_value(daily, "weather_code", 0, 0)) or 0)
    daily_code = int(daily_value(daily, "weather_code", 0, current_code) or current_code)
    condition = WEATHER_CODES.get(current_code, WEATHER_CODES.get(daily_code, f"天气代码 {current_code}"))
    temp = current.get("temperature_2m")
    apparent = current.get("apparent_temperature")
    humidity = current.get("relative_humidity_2m")
    wind_speed = current.get("wind_speed_10m")
    precip = current.get("precipitation")
    high = daily_value(daily, "temperature_2m_max", 0, None)
    low = daily_value(daily, "temperature_2m_min", 0, None)
    rain_probability = daily_value(daily, "precipitation_probability_max", 0, None)
    observed_at = str(current.get("time") or "")
    answer_lines = [
        f"{location.name}当前天气：{condition}。",
        f"气温 {format_number(temp)}°C，体感 {format_number(apparent)}°C，湿度 {format_number(humidity)}%。",
        f"今日气温约 {format_number(low)}-{format_number(high)}°C，最高降水概率 {format_number(rain_probability)}%。",
        f"当前降水量 {format_number(precip)} mm，风速 {format_number(wind_speed)} km/h。",
    ]
    if observed_at:
        answer_lines.append(f"观测/预报时间：{observed_at}（{location.timezone}）。")
    answer_lines.append("数据来源：Open-Meteo forecast API。")
    snippet = "；".join(line.rstrip("。") for line in answer_lines[:4])
    return LiveToolResult(
        answer="\n".join(answer_lines),
        evidence=[
            LiveToolEvidence(
                title=f"{location.name}实时天气",
                snippet=snippet,
                source_path="https://open-meteo.com/",
                tags=["live", "weather", "open-meteo", location.name],
            )
        ],
        metadata={
            "location": location.name,
            "timezone": location.timezone,
            "observed_at": observed_at,
            "provider": "open-meteo",
            "weather": {
                "condition": condition,
                "weather_code": current_code,
                "temperature_2m_c": temp,
                "apparent_temperature_c": apparent,
                "relative_humidity_percent": humidity,
                "precipitation_mm": precip,
                "wind_speed_10m_kmh": wind_speed,
                "temperature_2m_max_c": high,
                "temperature_2m_min_c": low,
                "precipitation_probability_max_percent": rain_probability,
            },
        },
    )


def public_web_search(query: str) -> LiveToolResult:
    response = httpx.get(
        "https://api.duckduckgo.com/",
        params={
            "q": query,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
            "kl": "cn-zh",
        },
        timeout=settings.live_tool_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    evidence = duckduckgo_evidence(payload, query)
    if not evidence:
        raise ValueError("公开搜索没有返回可用摘要。")
    answer = "\n".join(
        [
            f"基于公开搜索结果，{query}",
            "",
            "可用信息：",
            *[f"- {item.title}: {item.snippet}" for item in evidence[:3]],
            "",
            "建议对需要强时效或权威性的结论继续打开原始来源核验。",
        ]
    )
    return LiveToolResult(
        answer=answer,
        evidence=evidence,
        metadata={"provider": "duckduckgo", "result_count": len(evidence)},
    )


def resolve_location(query: str) -> Location | None:
    for name, (latitude, longitude, timezone, display_name) in KNOWN_LOCATIONS.items():
        if name in query:
            return Location(display_name, latitude, longitude, timezone)
    match = re.search(r"(?:今天|明天|现在|当前)?([\u4e00-\u9fff]{2,8})(?:的)?(?:天气|气温|温度)", query)
    if not match:
        match = re.search(r"(?:天气|气温|温度).*?(?:在|查|看)([\u4e00-\u9fff]{2,8})", query)
    if not match:
        return None
    location_name = match.group(1).strip("的在查看看")
    if location_name in KNOWN_LOCATIONS:
        latitude, longitude, timezone, display_name = KNOWN_LOCATIONS[location_name]
        return Location(display_name, latitude, longitude, timezone)
    return geocode_location(location_name)


def geocode_location(location_name: str) -> Location | None:
    if not location_name:
        return None
    response = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location_name, "count": 1, "language": "zh", "format": "json"},
        timeout=settings.live_tool_timeout_seconds,
    )
    response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        return None
    item = results[0]
    return Location(
        str(item.get("name") or location_name),
        float(item["latitude"]),
        float(item["longitude"]),
        str(item.get("timezone") or "UTC"),
    )


def duckduckgo_evidence(payload: dict[str, Any], query: str) -> list[LiveToolEvidence]:
    evidence: list[LiveToolEvidence] = []
    abstract = clean_html(str(payload.get("AbstractText") or ""))
    abstract_url = str(payload.get("AbstractURL") or "")
    heading = str(payload.get("Heading") or query)
    if abstract:
        evidence.append(
            LiveToolEvidence(
                title=heading,
                snippet=abstract,
                source_path=abstract_url or "https://duckduckgo.com/?q=" + quote_plus(query),
                tags=["live", "web", "duckduckgo", "abstract"],
            )
        )
    for item in flatten_related_topics(payload.get("RelatedTopics") or []):
        text = clean_html(str(item.get("Text") or ""))
        if not text:
            continue
        evidence.append(
            LiveToolEvidence(
                title=str(item.get("FirstURL") or heading),
                snippet=text,
                source_path=str(item.get("FirstURL") or "https://duckduckgo.com/?q=" + quote_plus(query)),
                score=max(0.45, 0.9 - len(evidence) * 0.08),
                tags=["live", "web", "duckduckgo", "related"],
            )
        )
        if len(evidence) >= 5:
            break
    return evidence


def flatten_related_topics(items: list[Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "Topics" in item and isinstance(item["Topics"], list):
            flattened.extend(flatten_related_topics(item["Topics"]))
            continue
        flattened.append(item)
    return flattened


def clean_html(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def daily_value(daily: dict[str, Any], key: str, index: int, default: Any) -> Any:
    values = daily.get(key) or []
    if isinstance(values, list) and len(values) > index:
        return values[index]
    return default


def format_number(value: Any) -> str:
    if value is None:
        return "未知"
    if isinstance(value, float):
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return str(value)
