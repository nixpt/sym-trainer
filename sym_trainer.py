"""
sym-trainer CLI — launches ml-intern with sym-trainer's config and MCP server.

Usage:
    sym-trainer                              # interactive mode
    sym-trainer "check my pod status"        # headless mode
    sym-trainer --model anthropic/claude-opus-4-6 "train bodhi v3"

How it works:
    1. Patches ml-intern's config loader to use our configs/agent_config.json
    2. Symlinks our system prompt into ml-intern's prompts dir
    3. Delegates to ml-intern's CLI
    4. ml-intern connects to our MCP server (mcp_server.server) via stdio transport
    5. Our MCP server exposes runpod_exec, runpod_sync, runpod_pods, training_* tools
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent


def _patch_ml_intern():
    """Patch ml-intern to use our config and prompt."""
    # Ensure our project root is in PYTHONPATH for the MCP server subprocess
    root_str = str(_PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    pythonpath = os.environ.get("PYTHONPATH", "")
    if root_str not in pythonpath:
        os.environ["PYTHONPATH"] = f"{root_str}:{pythonpath}"

    # 1. Patch config loading to use our config
    our_config = str(_PROJECT_ROOT / "configs" / "agent_config.json")

    from agent import config as agent_config
    _orig_load = agent_config.load_config

    def _load_our_config(path=None):
        return _orig_load(our_config)

    agent_config.load_config = _load_our_config

    # 2. Install our system prompt into ml-intern's prompts directory
    our_prompt = _PROJECT_ROOT / "prompts" / "system_prompt.yaml"
    if our_prompt.exists():
        try:
            import agent.context_manager.manager as cm
            ml_prompts = Path(cm.__file__).parent.parent / "prompts"
            # ml-intern packages prompts/ as bare yaml files without __init__.py,
            # so the dir may not be shipped with the installed wheel (setuptools
            # packages.find() drops it). Recreate it on-demand.
            ml_prompts.mkdir(parents=True, exist_ok=True)
            target = ml_prompts / "sym_prompt.yaml"
            # Refresh the file each run so edits to our source prompt take effect
            # without needing a reinstall. Write instead of symlink — readonly
            # installs (cached wheels, sandboxed PATH) can reject symlink creation.
            target.write_text(our_prompt.read_text())

            # Patch ContextManager.__init__ to use our prompt
            _orig_init = cm.ContextManager.__init__

            def _patched_init(self, model_max_tokens=180_000, compact_size=0.1,
                              untouched_messages=5, tool_specs=None,
                              prompt_file_suffix="system_prompt_v3.yaml",
                              hf_token=None, local_mode=False):
                # Replace __init__ body to override the hardcoded prompt path
                self.system_prompt = self._load_system_prompt(
                    tool_specs or [],
                    prompt_file_suffix="sym_prompt.yaml",
                    hf_token=hf_token,
                    local_mode=local_mode,
                )
                self.model_max_tokens = model_max_tokens
                self.compact_size = int(model_max_tokens * compact_size)
                self.running_context_usage = 0
                self.untouched_messages = untouched_messages
                from litellm import Message
                self.items = [Message(role="system", content=self.system_prompt)]

            cm.ContextManager.__init__ = _patched_init
        except Exception as e:
            print(f"Warning: could not install custom prompt: {e}", file=sys.stderr)


def cli():
    _patch_ml_intern()
    from agent.main import cli as ml_intern_cli
    ml_intern_cli()


if __name__ == "__main__":
    cli()
