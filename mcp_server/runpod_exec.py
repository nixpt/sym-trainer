"""
runpod_exec — execute commands on RunPod pods via SSH.
"""

from fastmcp import FastMCP

from mcp_server.ssh import resolve_connection, ssh_exec, DEFAULT_TIMEOUT, MAX_TIMEOUT


def register_exec_tools(mcp: FastMCP):

    @mcp.tool()
    def runpod_exec(
        command: str,
        profile: str | None = None,
        host: str | None = None,
        port: int = 22,
        user: str = "root",
        timeout: int = DEFAULT_TIMEOUT,
    ) -> str:
        """Execute a command on a RunPod GPU pod via SSH.

        Uses portal (secure-env SSH manager) when configured, otherwise raw SSH
        with ~/.ssh/runpod key. Once you specify a pod, subsequent calls remember it.

        For long-running commands (training), use tmux:
          tmux new-session -d -s train 'python train.py 2>&1 | tee train.log'
        Then monitor: tmux capture-pane -t train -p | tail -20

        Common operations:
          GPU: nvidia-smi
          Disk: df -h /workspace
          Logs: tail -50 /workspace/bodhi/train.log
          Tmux: tmux ls
        """
        try:
            conn = resolve_connection(profile, host, port, user)
        except ValueError as e:
            return str(e)

        timeout = min(timeout, MAX_TIMEOUT)
        output, success = ssh_exec(conn, command, timeout)

        if not success:
            return f"[FAILED]\n{output}"
        return output
