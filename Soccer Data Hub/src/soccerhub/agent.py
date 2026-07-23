"""Layer 3: Gemini agent over the soccerhub MCP tools.

``ask()`` spawns ``mcp_server.py`` as a stdio subprocess and lets Gemini call
the registered tools (hub_table, fbref_season, ...) via the SDK's built-in MCP
support — no hand-written tool schemas. Text in, text out.

Needs ``GEMINI_API_KEY`` in the environment (same convention as SUPABASE_*).
"""
import asyncio
import os
import sys

from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from soccerhub.errors import SoccerhubError

# Run the MCP server as a subprocess of the current interpreter/venv.
_SERVER = StdioServerParameters(
    command=sys.executable, args=["-m", "soccerhub.mcp_server"]
)


def ask(prompt: str, model: str = "gemini-3-flash-preview") -> str:
    """Ask the soccerhub agent a question. Calls MCP tools as needed; returns text."""
    try:
        return asyncio.run(_run(prompt, model))
    except SoccerhubError:
        raise
    except Exception as e:  # API / transport / unrecovered tool error
        raise SoccerhubError(str(e)) from e


async def _run(prompt: str, model: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    async with stdio_client(_SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(tools=[session]),
            )
    return resp.text
