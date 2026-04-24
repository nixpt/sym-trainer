"""
runpod_pods — manage RunPod pod lifecycle via runpodctl CLI.
"""

import json

from fastmcp import FastMCP

from mcp_server.ssh import has_runpodctl, resolve_connection, run_runpodctl, ssh_exec


def _format_pod_list(raw_json: str) -> str:
    try:
        pods = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return raw_json

    if not pods:
        return "No pods found. Use runpod_pod_create to spin one up."

    lines = [f"{'ID':<14} {'Name':<25} {'GPU':<25} {'Status':<12} {'Cost/hr':>8}"]
    lines.append("-" * 86)
    for pod in pods:
        pod_id = pod.get("id", "?")
        name = pod.get("name", "?")[:24]
        gpu = pod.get("machine", {}).get("gpuDisplayName", pod.get("gpuType", "?"))[:24]
        status = pod.get("desiredStatus", pod.get("status", "?"))
        cost = pod.get("costPerHr", 0)
        lines.append(f"{pod_id:<14} {name:<25} {gpu:<25} {status:<12} ${cost:>7.3f}")
    return "\n".join(lines)


def _format_balance(raw_json: str) -> str:
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return raw_json

    balance = data.get("clientBalance", 0)
    spend = data.get("currentSpendPerHr", 0)
    email = data.get("email", "?")

    lines = [f"Account: {email}", f"Balance: ${balance:.2f}"]
    if spend > 0:
        lines.append(f"Current spend: ${spend:.3f}/hr")
        lines.append(f"Estimated runway: {balance / spend:.0f}h")
    else:
        lines.append("No active spend.")
    return "\n".join(lines)


def register_pod_tools(mcp: FastMCP):

    @mcp.tool()
    def runpod_pod_list() -> str:
        """List all RunPod pods with status, GPU type, and cost per hour."""
        if not has_runpodctl():
            return "runpodctl not installed."
        output, success = run_runpodctl(["pod", "list"])
        if success:
            return _format_pod_list(output)
        return output

    @mcp.tool()
    def runpod_balance() -> str:
        """Show RunPod account balance, current spend rate, and estimated runway."""
        if not has_runpodctl():
            return "runpodctl not installed."
        output, success = run_runpodctl(["user"])
        if success:
            return _format_balance(output)
        return output

    @mcp.tool()
    def runpod_gpu_list() -> str:
        """List available RunPod GPU types and pricing."""
        if not has_runpodctl():
            return "runpodctl not installed."
        output, _success = run_runpodctl(["gpu", "list"])
        return output

    @mcp.tool()
    def runpod_pod_create(
        gpu_type: str,
        template_id: str | None = None,
        image: str | None = None,
        name: str | None = None,
        gpu_count: int = 1,
        volume_size: int | None = None,
        cloud_type: str = "COMMUNITY",
    ) -> str:
        """Create a new RunPod GPU pod.

        Args:
            gpu_type: GPU type (e.g. 'NVIDIA GeForce RTX 4090', 'NVIDIA A40').
            template_id: RunPod template ID (find via runpodctl template search).
            image: Docker image (alternative to template_id).
            name: Pod name.
            gpu_count: Number of GPUs (default 1).
            volume_size: Persistent volume size in GB.
            cloud_type: SECURE or COMMUNITY (default COMMUNITY).
        """
        if not has_runpodctl():
            return "runpodctl not installed."
        if not template_id and not image:
            return "Either template_id or image is required."

        cmd = ["pod", "create", "--gpu-id", gpu_type, "--ssh"]
        if template_id:
            cmd.extend(["--template-id", template_id])
        if image:
            cmd.extend(["--image", image])
        if name:
            cmd.extend(["--name", name])
        if gpu_count != 1:
            cmd.extend(["--gpu-count", str(gpu_count)])
        if volume_size:
            cmd.extend(["--volume-in-gb", str(volume_size)])
        cmd.extend(["--cloud-type", cloud_type])

        output, _success = run_runpodctl(cmd, timeout=60)
        return output

    @mcp.tool()
    def runpod_pod_action(pod_id: str, action: str) -> str:
        """Start, stop, restart, or delete a RunPod pod.

        Args:
            pod_id: Pod ID (from runpod_pod_list).
            action: One of 'start', 'stop', 'restart', 'delete'.
        """
        if action not in ("start", "stop", "restart", "delete"):
            return f"Invalid action: {action}. Must be start/stop/restart/delete."
        if not has_runpodctl():
            return "runpodctl not installed."
        output, _success = run_runpodctl(["pod", action, pod_id])
        return output

    @mcp.tool()
    def runpod_ssh_info(pod_id: str) -> str:
        """Show SSH connection details for a RunPod pod (host, port, key).

        Use this after creating a pod to get the SSH info needed for runpod_exec.
        """
        if not has_runpodctl():
            return "runpodctl not installed."
        output, _success = run_runpodctl(["ssh", "info", pod_id, "-v"])
        return output

    @mcp.tool()
    def runpod_pod_status(
        profile: str | None = None,
        host: str | None = None,
        port: int = 22,
    ) -> str:
        """Quick health check on a running pod: GPU, disk, tmux sessions, training processes.

        Run this before starting new work to check pod state.
        """
        try:
            conn = resolve_connection(profile, host, port)
        except ValueError as e:
            return str(e)

        status_cmd = (
            "echo '=== GPU ==='; "
            "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu "
            "--format=csv,noheader 2>/dev/null || echo 'No GPU'; "
            "echo ''; echo '=== DISK ==='; df -h /workspace 2>/dev/null || df -h /; "
            "echo ''; echo '=== TMUX ==='; tmux ls 2>/dev/null || echo 'No tmux sessions'; "
            "echo ''; echo '=== TRAINING ==='; "
            "ps aux | grep -E 'python|train' | grep -v grep | head -10 || echo 'None'"
        )
        output, success = ssh_exec(conn, status_cmd, 30)
        if not success:
            return f"[FAILED]\n{output}"
        return output
