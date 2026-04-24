# sym-trainer

ML training orchestrator for the Sym project. Wraps [ml-intern](https://github.com/huggingface/ml-intern) with a RunPod-centric MCP server and a Sym-specific system prompt.

## What it does

- Reuses ml-intern's autonomous agent loop + research sub-agents (HF papers, GitHub examples, dataset inspection).
- Replaces the default prompt with `prompts/system_prompt.yaml` — Sym project context, bodhi/sunaina/seahorse conventions, `nixprabin/*` HF repo conventions, `/workspace/...` pod paths.
- Adds 12 MCP tools over RunPod pods:
  - `runpod_pod_list`, `runpod_balance`, `runpod_gpu_list`, `runpod_pod_create`, `runpod_pod_action`, `runpod_ssh_info`, `runpod_pod_status`
  - `runpod_exec`, `runpod_sync`
  - `training_launch` (tmux-based, survives disconnect), `training_monitor`, `training_eval`, `training_push_to_hf`
- Uses [portal](https://github.com/nixpt/secure-env) (secure-env SSH manager) when available; falls back to raw SSH with `~/.ssh/runpod`.

## Install

```bash
# 1. Install ml-intern (must be installed first — sym-trainer is a thin wrapper)
cd ~/ml-intern
uv sync
uv tool install -e .

# 2. Install sym-trainer
cd ~/sym-trainer
uv sync
uv tool install -e .

# Verify
which ml-intern sym-trainer
```

## Environment

Create `~/sym-trainer/.env` (or export in your shell):

```bash
# Required for ml-intern
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...
GITHUB_TOKEN=ghp_...

# Required for RunPod MCP tools
RUNPOD_API_KEY=...                  # used by runpodctl (install from https://github.com/runpod/runpodctl)

# Optional: portal SSH (preferred over raw SSH)
SECURE_ENV_PASSWORD=...              # unlocks portal profiles
```

Also ensure you have:
- `runpodctl` on PATH (from https://github.com/runpod/runpodctl)
- `~/.ssh/runpod` SSH key (if not using portal)
- `portal` CLI installed (optional — only if using secure-env profiles)

## Usage

```bash
# Interactive
sym-trainer

# Headless (auto-approve, single prompt)
sym-trainer "train bodhi v3 SFT on nixprabin/bodhi-sft-v3 with Qwen2.5-7B-Instruct base"

# Override model
sym-trainer --model anthropic/claude-opus-4-6 "check pod status and training progress"
```

## How the wrapper works

`sym_trainer.py` at import time does three things, then delegates to ml-intern's CLI:

1. **Config swap** — monkey-patches `agent.config.load_config` to use `configs/agent_config.json` (sets the MCP transport to `python -m mcp_server.server`).
2. **Prompt swap** — copies `prompts/system_prompt.yaml` into ml-intern's installed `agent/prompts/` dir as `sym_prompt.yaml` (creates the dir if missing — ml-intern's setuptools config ships `.py` files only), then monkey-patches `ContextManager.__init__` to load that suffix instead of `system_prompt_v3.yaml`. Re-copied on every launch so source edits take effect without reinstall.
3. **PYTHONPATH** — prepends project root so `mcp_server.server` resolves when ml-intern spawns it over stdio.

This keeps ml-intern unforked. When ml-intern updates upstream, `uv sync` picks it up.

## Architecture

```
  user prompt
      │
      ▼
  sym-trainer (thin CLI shim)
      │  patches config + prompt
      ▼
  ml-intern (agent loop, LLM, research tools, HF tools)
      │
      │  stdio MCP
      ▼
  mcp_server.server (FastMCP)
      │
      ▼
  RunPod GPU pod (portal or raw SSH)
```

## Files

```
sym_trainer.py             # CLI entrypoint; patches + delegates
configs/
  agent_config.json        # ml-intern config (MCP transport points at mcp_server.server)
prompts/
  system_prompt.yaml       # Sym-specific system prompt (~150 lines)
mcp_server/
  server.py                # FastMCP server, registers all 12 tools
  ssh.py                   # shared SSH/portal/runpodctl layer, session memory, output truncation
  runpod_exec.py           # 1 tool
  runpod_pods.py           # 7 tools
  runpod_sync.py           # 1 tool (rsync-based)
  training.py              # 4 tools
tests/                     # TODO
scripts/                   # TODO
```

## Upstream potential

The `mcp_server/` is self-contained — no sym-specific code, only RunPod operations. Candidates for upstreaming:

- Submit as a standalone `ml-intern-runpod` plugin package (fastmcp stdio server).
- Or PR directly into `ml-intern/backend/tools` if HF wants first-party RunPod support.

The Sym-specific bits (`sym_trainer.py`, `prompts/system_prompt.yaml`, `configs/agent_config.json`) stay local.

## TODO

- [ ] Tests for MCP tools (mock `subprocess.run`, assert argv structure)
- [ ] `scripts/bootstrap_pod.sh` — one-shot pod provisioner
- [ ] CI smoke test (fastmcp server starts, tools register, no import errors)
