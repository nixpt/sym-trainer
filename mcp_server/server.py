"""
sym-trainer MCP server — exposes RunPod tools to ml-intern via MCP protocol.

Run standalone:  python -m mcp_server.server
Used by ml-intern via stdio transport in agent config.
"""

from fastmcp import FastMCP

from mcp_server.runpod_exec import register_exec_tools
from mcp_server.runpod_pods import register_pod_tools
from mcp_server.runpod_sync import register_sync_tools
from mcp_server.training import register_training_tools

mcp = FastMCP("sym-trainer", instructions=(
    "RunPod GPU pod management and ML training tools for the Sym project. "
    "Use runpod_exec to run commands on pods, runpod_sync to transfer files, "
    "runpod_pods to manage pod lifecycle, and training tools for orchestration."
))

register_exec_tools(mcp)
register_sync_tools(mcp)
register_pod_tools(mcp)
register_training_tools(mcp)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
