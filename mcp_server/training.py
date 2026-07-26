"""
training — higher-level training orchestration tools.

These combine multiple SSH operations into workflow-oriented actions:
launch training, monitor progress, run eval, push to HF.
"""

import re

from fastmcp import FastMCP

from mcp_server.ssh import resolve_connection, ssh_exec


def register_training_tools(mcp: FastMCP):

    @mcp.tool()
    def training_launch(
        script_path: str | None = None,
        session_name: str = "train",
        log_file: str | None = None,
        work_dir: str = "/workspace",
        command: str | None = None,
        profile: str | None = None,
        host: str | None = None,
        port: int = 22,
    ) -> str:
        """Launch a training run in a tmux session on a RunPod pod.

        Starts the run in the background via tmux so it survives SSH disconnects.
        Automatically tees output to a log file. Engine-agnostic: torch scripts,
        zazen/cargo binaries, anything.

        Args:
            script_path: Path to a training script on the pod. Runner inferred
                from extension: .py -> python3, .sh -> bash, anything else is
                executed directly (e.g. a prebuilt zazen binary).
            command: Full shell command to run instead of script_path (e.g.
                'cargo run --release --example train_avatar_l1_tied -- ...').
                Exactly one of script_path / command must be given.
            session_name: Tmux session name (default 'train').
            log_file: Log file path. Defaults to script_path with .log extension,
                or /workspace/<session_name>.log for command launches.
            work_dir: Working directory on the pod (default '/workspace').
            profile: Portal profile name.
            host: Direct SSH host.
            port: SSH port.
        """
        if (script_path is None) == (command is None):
            return "Give exactly one of script_path or command."
        try:
            conn = resolve_connection(profile, host, port)
        except ValueError as e:
            return str(e)

        if command is not None:
            run_cmd = command
            if not log_file:
                log_file = f"/workspace/{session_name}.log"
        else:
            if script_path.endswith(".py"):
                run_cmd = f"python3 {script_path}"
            elif script_path.endswith(".sh"):
                run_cmd = f"bash {script_path}"
            else:
                run_cmd = script_path
            if not log_file:
                log_file = re.sub(r"\.(py|sh)$", ".log", script_path)
                if log_file == script_path:
                    log_file = script_path + ".log"

        # Check if tmux session already exists
        check_output, _ = ssh_exec(conn, f"tmux has-session -t {session_name} 2>/dev/null && echo EXISTS || echo FREE", 10)

        if "EXISTS" in check_output:
            return (
                f"Tmux session '{session_name}' already exists. "
                f"Check it with: tmux capture-pane -t {session_name} -p | tail -20\n"
                f"Kill it first with: tmux kill-session -t {session_name}"
            )

        # Single-quote-escape for the tmux wrapper ('...' -> '\'' quoting).
        tmux_payload = f"{run_cmd} 2>&1 | tee {log_file}".replace("'", "'\\''")
        launch_cmd = (
            f"cd {work_dir} && "
            f"tmux new-session -d -s {session_name} "
            f"'{tmux_payload}'"
        )

        output, success = ssh_exec(conn, launch_cmd, 30)
        if not success:
            return f"Failed to launch training:\n{output}"

        # Verify it started
        verify_output, _ = ssh_exec(conn, f"tmux has-session -t {session_name} 2>/dev/null && echo RUNNING || echo FAILED", 10)

        if "RUNNING" in verify_output:
            return (
                f"Training launched in tmux session '{session_name}'.\n"
                f"Run: {run_cmd}\n"
                f"Log: {log_file}\n\n"
                f"Monitor with:\n"
                f"  tail -30 {log_file}\n"
                f"  tmux capture-pane -t {session_name} -p | tail -20"
            )
        return f"Launch command ran but session not found. Output:\n{output}"

    @mcp.tool()
    def training_monitor(
        log_file: str | None = None,
        session_name: str = "train",
        lines: int = 30,
        profile: str | None = None,
        host: str | None = None,
        port: int = 22,
    ) -> str:
        """Monitor a running training job — shows recent log output and loss/accuracy metrics.

        Args:
            log_file: Path to training log file. If not set, uses tmux capture.
            session_name: Tmux session to capture from (default 'train').
            lines: Number of lines to show (default 30).
            profile: Portal profile name.
            host: Direct SSH host.
            port: SSH port.
        """
        try:
            conn = resolve_connection(profile, host, port)
        except ValueError as e:
            return str(e)

        parts = []

        # Get recent log output
        if log_file:
            log_output, log_ok = ssh_exec(conn, f"tail -{lines} {log_file} 2>/dev/null || echo 'Log file not found'", 15)
            parts.append(f"=== Log ({log_file}) ===\n{log_output}")
        else:
            tmux_output, tmux_ok = ssh_exec(
                conn,
                f"tmux capture-pane -t {session_name} -p 2>/dev/null | tail -{lines} || echo 'Session not found'",
                15,
            )
            parts.append(f"=== Tmux ({session_name}) ===\n{tmux_output}")

        # Extract loss/accuracy metrics from recent output
        if log_file:
            metrics_cmd = (
                f"grep -oE '(loss|train_loss|eval_loss|accuracy|token_acc)[\":]?\\s*[=:]?\\s*[0-9]+\\.[0-9]+' "
                f"{log_file} 2>/dev/null | tail -5 || echo 'No metrics found'"
            )
            metrics_output, _ = ssh_exec(conn, metrics_cmd, 10)
            parts.append(f"\n=== Recent Metrics ===\n{metrics_output}")

        # Check if training is still running
        running_check, _ = ssh_exec(
            conn,
            f"tmux has-session -t {session_name} 2>/dev/null && echo 'RUNNING' || echo 'STOPPED'",
            10,
        )
        status = "RUNNING" if "RUNNING" in running_check else "STOPPED"
        parts.append(f"\nStatus: {status}")

        return "\n".join(parts)

    @mcp.tool()
    def training_eval(
        eval_script: str,
        work_dir: str = "/workspace",
        timeout: int = 300,
        profile: str | None = None,
        host: str | None = None,
        port: int = 22,
    ) -> str:
        """Run an evaluation script on the pod and return results.

        Unlike training_launch, this runs synchronously and returns the output
        directly. Use for identity eval, capability tests, etc.

        Args:
            eval_script: Path to eval script on the pod, or inline bash command.
            work_dir: Working directory (default '/workspace').
            timeout: Max seconds to wait (default 300).
            profile: Portal profile name.
            host: Direct SSH host.
            port: SSH port.
        """
        try:
            conn = resolve_connection(profile, host, port)
        except ValueError as e:
            return str(e)

        cmd = f"cd {work_dir} && bash {eval_script}"
        if eval_script.endswith(".py"):
            cmd = f"cd {work_dir} && python3 {eval_script}"

        output, success = ssh_exec(conn, cmd, timeout)
        if not success:
            return f"[EVAL FAILED]\n{output}"
        return output

    @mcp.tool()
    def training_push_to_hf(
        source_path: str,
        hf_repo: str,
        commit_message: str = "Upload from sym-trainer",
        repo_type: str = "model",
        profile: str | None = None,
        host: str | None = None,
        port: int = 22,
    ) -> str:
        """Push a checkpoint or adapter directory from a pod to HuggingFace Hub.

        Runs huggingface-cli upload on the pod. Requires HF_TOKEN to be set on the pod.

        Args:
            source_path: Path on the pod to upload (e.g. '/workspace/bodhi/bodhi-v2-dpo').
            hf_repo: HF repo ID (e.g. 'nixprabin/bodhi').
            commit_message: Commit message for the upload.
            repo_type: 'model', 'dataset', or 'space' (default 'model').
            profile: Portal profile name.
            host: Direct SSH host.
            port: SSH port.
        """
        try:
            conn = resolve_connection(profile, host, port)
        except ValueError as e:
            return str(e)

        cmd = (
            f"huggingface-cli upload {hf_repo} {source_path} "
            f"--repo-type {repo_type} "
            f"--commit-message '{commit_message}'"
        )
        output, success = ssh_exec(conn, cmd, 600)
        if not success:
            return f"[PUSH FAILED]\n{output}"
        return f"Pushed to https://huggingface.co/{hf_repo}\n\n{output}"
