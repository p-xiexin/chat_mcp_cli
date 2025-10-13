import asyncio
import httpx
from mcp.server.fastmcp import FastMCP

# MCP 应用
mcp = FastMCP(name="Simple Weather", port=9099, host="0.0.0.0")

OPEN_METEO_API = "https://api.open-meteo.com/v1/forecast"

async def fetch_weather(lat: float, lon: float) -> str:
    """调用 Open-Meteo API 获取天气预报"""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto",
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(OPEN_METEO_API, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return f"获取天气数据失败: {e}"

    daily = data.get("daily", {})
    if not daily:
        return "没有可用的天气数据"

    dates = daily.get("time", [])
    max_t = daily.get("temperature_2m_max", [])
    min_t = daily.get("temperature_2m_min", [])
    prec = daily.get("precipitation_sum", [])

    lines = []
    for i, d in enumerate(dates[:5]):  # 只展示 5 天
        lines.append(
            f"{d}: 🌡 {min_t[i]}°C ~ {max_t[i]}°C | 🌧 {prec[i]} mm"
        )
    return "\n".join(lines)

# MCP 工具: 根据坐标获取天气
@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """获取天气预报 (全球通用)"""
    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        return "无效的坐标"
    return await fetch_weather(latitude, longitude)

# MCP 工具: 根据城市名获取经纬度
CITY_MAP = {
    "武汉": (30.5928, 114.3055),
    "北京": (39.9042, 116.4074),
    "上海": (31.2304, 121.4737),
}
@mcp.tool()
async def get_location(city_name: str) -> str:
    """获取武汉、北京或者上海的经纬度"""
    if city_name in CITY_MAP:
        lat, lon = CITY_MAP[city_name]
        return f"{city_name} 的经纬度: {lat}, {lon}"

# MCP 提示
@mcp.prompt("weather_help")
async def weather_help_prompt() -> str:
    return """
🌍 天气服务 Demo

使用方法:
- get_forecast(latitude, longitude) 获取全球天气预报
- get_location(city_name) 根据城市名获取经纬度

示例:
- get_location("San Francisco")
- get_forecast(37.7749, -122.4194)  # 旧金山
"""

# 主程序
async def main():
    await asyncio.gather(mcp.run_sse_async())

if __name__ == "__main__":
    asyncio.run(main())
