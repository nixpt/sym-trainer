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
        tmux_payload = f"({run_cmd}) 2>&1 | tee {log_file}".replace("'", "'\\''")
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
    def training_resume(
        run_cmd: str,
        ckpt_dir: str,
        prev_final_val: float | None = None,
        session_name: str = "train",
        log_file: str | None = None,
        profile: str | None = None,
        host: str | None = None,
        port: int = 22,
    ) -> str:
        """Resume a training run from the latest checkpoint, with a load-verification gate.

        Why this exists: a "continuation" that silently starts from random init looks
        exactly like a normal run for the first few hundred steps, and you only discover
        it hours later. The tell is the FIRST eval: a genuine resume reports an `init val`
        matching the previous run's FINAL val; a failed load reports the from-scratch
        value (e.g. ~11.2 for a 64k-vocab LM). This tool finds the newest checkpoint,
        injects --init-from/--step-offset, and -- when given prev_final_val -- refuses to
        leave the run going if `init val` does not match.

        Also note: a checkpoint that does NOT carry optimizer state restarts Adam's
        moments from zero, costing a transient loss bump. Use a short --warmup to absorb it.

        Args:
            run_cmd: The training command WITHOUT --init-from/--step-offset (they are added).
            ckpt_dir: Directory holding step_*.pt checkpoints.
            prev_final_val: Previous run's FINAL val loss. If set, the resume is verified
                against the first `init val` and killed on mismatch (tolerance 0.05).
            session_name: tmux session name.
            log_file: Log path (defaults to /workspace/<session_name>.log).
            profile: Portal profile name.
            host: Direct SSH host.
            port: SSH port.
        """
        try:
            conn = resolve_connection(profile, host, port)
        except ValueError as e:
            return str(e)

        log_file = log_file or f"/workspace/{session_name}.log"

        # newest checkpoint by step number, not mtime (mtime lies after a restore/rsync)
        find_cmd = (
            f"ls {ckpt_dir}/step_*.pt 2>/dev/null | "
            + r"sed 's/.*step_0*//; s/\.pt//'" + " | sort -n | tail -1"
        )
        step_out, ok = ssh_exec(conn, find_cmd, 20)
        step_str = step_out.strip().splitlines()[-1].strip() if step_out.strip() else ""
        if not ok or not step_str.isdigit():
            return (f"No step_*.pt checkpoints found in {ckpt_dir} -- nothing to resume "
                    f"from.\n{step_out}")
        step = int(step_str)
        ckpt = f"{ckpt_dir}/step_{step:06d}.pt"

        has_opt, _ = ssh_exec(
            conn,
            f"python3 -c \"import torch,sys; d=torch.load('{ckpt}',map_location='cpu',"
            f"weights_only=False); print('OPT' if 'opt' in d else 'NOOPT')\"",
            300,
        )
        opt_note = ("optimizer state present -- clean resume"
                    if "OPT" in has_opt and "NOOPT" not in has_opt
                    else "NO optimizer state in checkpoint: Adam moments restart from zero, "
                         "expect a transient loss bump (use --warmup to absorb it)")

        resumed_cmd = f"{run_cmd} --init-from {ckpt} --step-offset {step}"
        launched = training_launch(
            run_cmd=resumed_cmd, session_name=session_name, log_file=log_file,
            profile=profile, host=host, port=port,
        )
        if "Failed" in launched or "already exists" in launched:
            return launched

        result = [f"Resuming from {ckpt} (step {step}).", opt_note, "", launched]

        if prev_final_val is not None:
            import time as _t
            init_val = None
            for _ in range(60):                       # up to ~10 min for the first eval
                _t.sleep(10)
                out, _ = ssh_exec(conn, f"grep -m1 'init val' {log_file} 2>/dev/null", 20)
                if "init val" in out:
                    try:
                        init_val = float(out.split("init val")[1].split()[0])
                    except (IndexError, ValueError):
                        pass
                    break
            if init_val is None:
                result.append("\nWARNING: no 'init val' line seen yet -- verify by hand "
                              f"that it matches {prev_final_val:.4f}.")
            elif abs(init_val - prev_final_val) > 0.05:
                ssh_exec(conn, f"tmux kill-session -t {session_name}", 20)
                result.append(
                    f"\nRESUME VERIFICATION FAILED -- KILLED.\n"
                    f"  init val {init_val:.4f} vs previous FINAL {prev_final_val:.4f}.\n"
                    f"  The checkpoint did not load; this would have trained a randomly "
                    f"initialised model. Check that the model class matches the saved config."
                )
            else:
                result.append(f"\nRESUME VERIFIED: init val {init_val:.4f} matches previous "
                              f"FINAL {prev_final_val:.4f}.")
        return "\n".join(result)

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

        # An upload that exits 0 is NOT a verified upload. huggingface-cli reports
        # success on exit code alone; a truncated or partially-resumed LFS object still
        # exits 0. Verify by reading the remote objects back and comparing each one's
        # LFS sha256 against a locally computed sha256 on the pod. Size alone is not
        # enough -- it will not catch a truncated-then-padded object.
        verify_py = f"""
import hashlib, os, subprocess, sys
from huggingface_hub import HfApi
src, repo, rtype = {source_path!r}, {hf_repo!r}, {repo_type!r}
r = subprocess.run(["huggingface-cli","upload",repo,src,"--repo-type",rtype,
                    "--commit-message",{commit_message!r}],
                   capture_output=True, text=True)
print(r.stdout[-2000:]); print(r.stderr[-2000:], file=sys.stderr)
if r.returncode != 0:
    print("PUSH_FAILED"); sys.exit(1)

def sha256(p):
    h = hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(8<<20), b""): h.update(b)
    return h.hexdigest()

local = {{}}
if os.path.isdir(src):
    for root,_,files in os.walk(src):
        for fn in files:
            fp = os.path.join(root,fn)
            local[os.path.relpath(fp,src)] = fp
else:
    local[os.path.basename(src)] = src

info = HfApi().repo_info(repo, repo_type=rtype, files_metadata=True)
remote = {{f.rfilename: f for f in info.siblings}}
ok = True
for rel, fp in sorted(local.items()):
    m = remote.get(rel)
    if m is None:
        print(f"[MISSING] {{rel}}"); ok = False; continue
    size = os.path.getsize(fp)
    line = f"{{rel}}: remote={{m.size}} local={{size}}"
    if m.size != size:
        line += " SIZE-MISMATCH"; ok = False
    lfs = getattr(m, "lfs", None)
    oid = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
    if oid:
        good = (oid == sha256(fp))
        line += f" | sha256 {{'OK' if good else 'MISMATCH'}}"
        ok = ok and good
    print(line)
print("UPLOAD_VERIFIED_MARKER" if ok else "UPLOAD_VERIFY_FAILED")
sys.exit(0 if ok else 1)
"""
        import shlex
        cmd = f"python3 -c {shlex.quote(verify_py)}"
        output, success = ssh_exec(conn, cmd, 1800)
        if not success or "UPLOAD_VERIFIED_MARKER" not in output:
            return f"[PUSH FAILED OR UNVERIFIED]\n{output}"
        return (f"Pushed AND VERIFIED (remote sha256 == local) to "
                f"https://huggingface.co/{hf_repo}\n\n{output}")
