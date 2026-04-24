"""
runpod_sync — rsync files to/from RunPod pods.
"""

import subprocess

from fastmcp import FastMCP

from mcp_server.ssh import (
    DEFAULT_SSH_KEY,
    has_portal,
    resolve_connection,
    strip_ansi,
    truncate,
)


def _run_rsync(
    conn: dict,
    source: str,
    dest: str,
    direction: str,
    exclude: list[str] | None = None,
) -> tuple[str, bool]:
    host = conn["host"]
    port = str(conn["port"])
    user = conn["user"]
    remote_prefix = f"{user}@{host}:"

    ssh_cmd = f"ssh -i {DEFAULT_SSH_KEY} -o StrictHostKeyChecking=accept-new -p {port}"

    cmd = ["rsync", "-avz", "--progress", "-e", ssh_cmd]

    if exclude:
        for pat in exclude:
            cmd.extend(["--exclude", pat])

    if direction == "upload":
        cmd.extend([source, f"{remote_prefix}{dest}"])
    else:
        cmd.extend([f"{remote_prefix}{source}", dest])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = strip_ansi(result.stdout + result.stderr).strip()
        if not output:
            output = "(no output)"
        return truncate(output), result.returncode == 0
    except subprocess.TimeoutExpired:
        return "Rsync timed out after 600s.", False
    except Exception as e:
        return f"Rsync error: {e}", False


def register_sync_tools(mcp: FastMCP):

    @mcp.tool()
    def runpod_sync(
        source: str,
        dest: str,
        direction: str,
        profile: str | None = None,
        host: str | None = None,
        port: int = 22,
        exclude: list[str] | None = None,
    ) -> str:
        """Sync files between local machine and a RunPod pod using rsync.

        Args:
            source: Source path.
            dest: Destination path.
            direction: 'upload' (local→pod) or 'download' (pod→local).
            profile: Portal profile name (remembered for subsequent calls).
            host: Direct SSH host (if no portal profile).
            port: SSH port (default 22).
            exclude: Patterns to exclude (e.g. ['__pycache__', '.git']).

        Examples:
            Upload code: source='./sunaina/', dest='/workspace/sunaina/', direction='upload'
            Download ckpt: source='/workspace/bodhi/ckpt/', dest='./downloads/', direction='download'
        """
        if direction not in ("upload", "download"):
            return "direction must be 'upload' or 'download'."

        try:
            conn = resolve_connection(profile, host, port)
        except ValueError as e:
            return str(e)

        # Try portal sync first if profile + portal available
        if conn.get("profile") and has_portal():
            if direction == "upload":
                portal_cmd = ["portal", "sync", conn["profile"], source, f":{dest}"]
            else:
                portal_cmd = ["portal", "sync", conn["profile"], f":{source}", dest]
            try:
                result = subprocess.run(
                    portal_cmd, capture_output=True, text=True, timeout=600
                )
                output = strip_ansi(result.stdout + result.stderr).strip() or "(no output)"
                if result.returncode == 0:
                    return truncate(output)
                # Fall through to raw rsync on failure
            except Exception:
                pass

        output, success = _run_rsync(conn, source, dest, direction, exclude)
        if not success:
            return f"[FAILED]\n{output}"
        return output
