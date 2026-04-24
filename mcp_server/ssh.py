"""
Shared SSH execution layer — used by all RunPod tools.

Tries portal (secure-env SSH manager) first, falls back to raw SSH.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

MAX_OUTPUT_CHARS = 25_000
DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 3600
DEFAULT_SSH_KEY = os.path.expanduser("~/.ssh/runpod")
DEFAULT_USER = "root"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07")

# Module-level session memory — remembers the last-used pod connection
_active_connection: dict | None = None


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def truncate(output: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if len(output) <= max_chars:
        return output
    head_budget = int(max_chars * 0.25)
    tail_budget = max_chars - head_budget
    omitted = len(output) - max_chars
    return (
        output[:head_budget]
        + f"\n\n... ({omitted:,} chars omitted) ...\n\n"
        + output[-tail_budget:]
    )


def has_portal() -> bool:
    return shutil.which("portal") is not None and bool(os.environ.get("SECURE_ENV_PASSWORD"))


def has_runpodctl() -> bool:
    return shutil.which("runpodctl") is not None


def resolve_connection(
    profile: str | None = None,
    host: str | None = None,
    port: int = 22,
    user: str = DEFAULT_USER,
) -> dict:
    """Resolve SSH connection params, using session memory as fallback.

    Returns dict with keys: profile, host, port, user.
    Raises ValueError if no connection can be determined.
    """
    global _active_connection

    if not profile and not host and _active_connection:
        return _active_connection

    if not profile and not host:
        raise ValueError(
            "No pod specified. Provide a portal profile or host. "
            "Example: profile='runpod-bodhi' or host='194.68.245.29', port=22028"
        )

    conn = {"profile": profile, "host": host, "port": port, "user": user}
    _active_connection = conn
    return conn


def ssh_exec(
    conn: dict, command: str, timeout: int = DEFAULT_TIMEOUT
) -> tuple[str, bool]:
    """Execute a command on a remote pod via portal or raw SSH.

    Returns (output, success).
    """
    if conn.get("profile") and has_portal():
        cmd = ["portal", "connect", conn["profile"], "--command", command]
    else:
        host = conn["host"]
        port = str(conn["port"])
        user = conn["user"]
        cmd = [
            "ssh",
            "-i", DEFAULT_SSH_KEY,
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            "-p", port,
            f"{user}@{host}",
            command,
        ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=min(timeout, MAX_TIMEOUT),
        )
        output = strip_ansi(result.stdout + result.stderr).strip()
        if not output:
            output = "(no output)"
        return truncate(output), result.returncode == 0
    except subprocess.TimeoutExpired:
        return (
            f"Command timed out after {timeout}s. "
            "For long-running commands, use tmux:\n"
            "  tmux new-session -d -s train 'python train.py 2>&1 | tee train.log'\n"
            "Then check: tmux capture-pane -t train -p | tail -30"
        ), False
    except FileNotFoundError as e:
        return f"SSH client not found: {e}", False
    except Exception as e:
        return f"SSH error: {e}", False


def run_runpodctl(cmd_args: list[str], timeout: int = 30) -> tuple[str, bool]:
    """Run a runpodctl command and return (output, success)."""
    cmd = ["runpodctl"] + cmd_args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = (result.stdout + result.stderr).strip()
        if not output:
            output = "(no output)"
        return output, result.returncode == 0
    except FileNotFoundError:
        return "runpodctl not installed. Install from https://github.com/runpod/runpodctl", False
    except subprocess.TimeoutExpired:
        return f"runpodctl timed out after {timeout}s.", False
    except Exception as e:
        return f"runpodctl error: {e}", False
