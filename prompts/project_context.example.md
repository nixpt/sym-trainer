# Example project context for sym-trainer

Copy this file to `~/.config/sym-trainer/project_context.md` (or point
`SYM_TRAINER_PROJECT_CONTEXT` at your own file, or create the gitignored
`prompts/project_context.local.md`) and replace it with your project's details.
It is spliced into the system prompt at launch and never committed.

The project trains and ships the following artifacts:
- **my-model** — the main model (SFT then DPO). Checkpoints land at
  `/workspace/my-model/` on pods; HF repo is `<hf-user>/my-model`.
- **my-tokenizer** — the tokenizer (`tokenizer.json`, byte-level BPE).

Standard local paths: `~/projects/my-project/`. Standard pod paths:
`/workspace/<component>/`.

HF repo conventions: `<hf-user>/<component>[-<stage>]`.
