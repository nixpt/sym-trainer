# sym-runner — design sketch

**Status:** parked / pre-build · captured to FOREMAN_THREADS parking lot 2026-05-09 (session 142)
**Pairs with:** `sym-trainer` (existing) — closes the upstream/downstream loop
**Author:** foreman (claude-opus) · captain ratified
**Blocking:** rama foundation live-smoke (sudo session — `./install.sh && systemctl enable --now ravan`, exercise socket + arena end-to-end). See FOREMAN_THREADS rama section.

---

## Why this exists

`sym-trainer` is the **upstream** half of the model loop: HF pull → fine-tune on RunPod → push back to HF. The **downstream** half — pull a trained checkpoint and actually SERVE it for agent use — is currently DIY each time. Captain spends recurring effort spinning pods, picking backends, configuring inference, port-forwarding. This is the same problem `sym-trainer` solved for training, just shifted one step right.

`sym-runner` is the symmetric MCP wrapper: same RunPod orchestration, same portal-for-SSH layer, same Sym-aware system prompt — but for **inference**.

```
                          ┌─────────────────────────────────┐
                          │   Hugging Face (model store)    │
                          └─────────────────────────────────┘
                                 ▲                  │
                       push      │                  │ pull
                                 │                  ▼
              ┌──────────────────┴──┐   ┌──────────────────────┐
              │     sym-trainer     │   │      sym-runner      │  ← this design
              │  (upstream loop)    │   │   (downstream loop)  │
              └──────────────────────┘   └──────────────────────┘
                       │                            │
              fine-tune│                       serve│
                       ▼                            ▼
                  ┌─────────────────────────────────────┐
                  │         RunPod GPU pods             │
                  │   (orchestrated via runpodctl +     │
                  │    portal-for-SSH + tmux survives)  │
                  └─────────────────────────────────────┘
```

---

## Captain's load-bearing directive

**`seahorse` + `rama` are the architectural priority. Any new backend goes THROUGH seahorse, not parallel to it.**

`seahorse` (in `libs/seahorse/` and `crates/ai/capabilities/seahorse/` in exosphere) is already the Sym inference abstraction layer. It wraps:

- **rama-zpu native** — Linux + zram-backed weights, ZPU compute fabric
- **llama.cpp** — GGUF inference, CPU + GPU paths

The principle: `sym-runner` exposes ONE backend surface — seahorse. New engines (transformers, vllm, SGLang, MLC, etc.) are added INSIDE seahorse, and `sym-runner` picks them up for free. **Do not let `sym-runner` grow a parallel backend taxonomy.**

```
sym-runner serve <hf_repo> --engine <native|llamacpp>
                         │
                         ▼
                   seahorse adapter
                  ┌─────┴─────┬─────────────────────────────┐
                  │           │                             │
            rama-zpu       llama.cpp                future engines
            (native)        (GGUF)                   (added INSIDE
            zram-backed                              seahorse, not
            ZPU lanes                                in sym-runner)
```

If `sym-runner` ever needs a backend seahorse doesn't have, **the right move is to add it to seahorse, not work around it in sym-runner.**

---

## Goals

- **Pull a trained model from HF** to a RunPod-attached network volume (or local cache) without manual download steps.
- **Spin a serving pod** sized correctly for the model + chosen seahorse engine, in the right datacenter (volume-pinned).
- **Start a serve process** via seahorse, expose it as a reachable endpoint (port forward through portal tunnel).
- **Run agent eval** against the served endpoint — same rubrics as `training_eval` (eval is symmetric).
- **Tear down cleanly** when done — no orphan pods, no idling network volumes.
- **MCP-shaped** — identical pattern to `sym-trainer`'s `training_*` tools so an agent can chain `pull → serve → eval → stop` autonomously.

## Non-goals

- **Multi-tenant production serving.** This is for Sym-internal use (foreman dispatch, captain-driven eval, agent integration). vllm/SGLang come if/when multi-tenant serving becomes a real need — and they go inside seahorse first.
- **Model conversion / quantization tooling.** GGUF conversion stays in seahorse / llama.cpp's own toolchain. `sym-runner` consumes already-quantized artifacts; if a conversion is needed, it's a separate `model_convert` MCP tool (future), not blended into `runtime_serve`.
- **Replacing `sym-trainer`.** Trainer keeps owning training; runner keeps owning serving. They share infra (pod/portal/HF), not responsibilities.
- **Custom inference UI.** Endpoint is OpenAI-compatible (or whatever seahorse exposes); existing tooling (joker-mcp providers, agent-launch `--provider`, opencode model registry) consumes it.

---

## MCP tool surface

Mirrors `sym-trainer`'s `training_*` tools:

| Tool | Purpose | Sketch args |
|---|---|---|
| `runtime_pull` | Pull HF repo to local cache or RunPod volume; convert if needed | `hf_repo: str`, `target_volume: str = "auto"`, `quant: Optional[str]` (GGUF Q4_0 / Q8_0 / fp16 / int4) |
| `runtime_serve` | Spin pod + start seahorse serve process; return endpoint | `model_path: str`, `engine: str = "native"` (routes inside seahorse), `pod_gpu: str = "auto"`, `port: int = 8000`, `tunnel: bool = True` |
| `runtime_status` | Query running serve process: endpoint URL, latency, throughput, pod cost | `pod_id: Optional[str] = None` (lists all if None) |
| `runtime_stop` | Tear down: kill serve process, optionally delete pod | `pod_id: str`, `keep_pod: bool = False` |
| `runtime_eval` | Run eval rubric against running endpoint | `pod_id: str`, `rubric: str` (shared registry with `training_eval`) |

**`engine` arg routes inside seahorse:**
- `native` → rama-zpu path (Linux only, requires rama foundation)
- `llamacpp` → seahorse's llama.cpp adapter (cross-platform within seahorse's portability)
- `auto` → seahorse picks based on model format + pod GPU (GGUF → llamacpp; safetensors → native if Linux + ZPU available, else error)

**Future engines** (don't add to sym-runner; add to seahorse):
- `transformers` — HF native, fp16/int4-bnb (dev/eval path; not prod-cheap)
- `vllm` — paged KV, high throughput (multi-tenant; needs design)
- `sglang` — structured output, similar to vllm

---

## Lifecycle

```
captain/foreman                sym-runner MCP                seahorse                 RunPod pod
─────────────                  ──────────────                ────────                 ──────────
                runtime_pull
                ─────────────►   pull HF → volume
                                (huggingface-cli login,
                                 hf-transfer, optional
                                 GGUF conversion)
                ◄──────────── volume_path

                runtime_serve(engine=native)
                ─────────────►   pod create
                                (volume-pinned DC, GPU
                                 sized for model+engine)
                                ──────────────────────────────────────────────►  pod boots
                                                                                  (ssh up,
                                                                                   volume mounted)
                                                                                       │
                                via portal exec:                                      │
                                  install seahorse + engine deps via uv               │
                                  start `seahorse serve --engine native --port 8000` │
                                  (in tmux so survives ssh disconnect)                │
                                                                                       │
                                portal tunnel: localhost:8000 ◄──────────────────────┘
                                  ↳ local agent can call http://localhost:8000/v1/...
                ◄──────────── endpoint_url, pod_id

                runtime_eval(rubric="bodhi_eval")
                ─────────────►   POST eval prompts to endpoint
                                tally rubric scores
                ◄──────────── eval_report

                runtime_stop(pod_id, keep_pod=False)
                ─────────────►   kill tmux session                                    │
                                pod remove ──────────────────────────────────────────►  pod gone
                ◄──────────── stopped
```

Implementation: each MCP tool is a thin Python function over `runpodctl` + `portal exec/sync` + a small set of seahorse-specific helpers. Same shape as the existing `training_launch` / `training_monitor` / `training_eval` / `training_push_to_hf` tools in `mcp_server/server.py`.

---

## Sequencing (what blocks what)

```
[rama foundation live-smoke]   ← BLOCKS native engine path
        │
        ▼
[seahorse engine API stabilized]   ← needs to expose `serve(model, engine, port)` cleanly
        │
        ▼
[sym-runner skeleton: pull + serve + stop via llamacpp engine]   ← MVP
        │
        ▼
[sym-runner: native engine via seahorse rama-zpu adapter]   ← real loop closure
        │
        ▼
[sym-runner: eval integration shared with training_eval]   ← rubric registry
        │
        ▼
[future engines added to seahorse: transformers, vllm, ...]
        ↑
        └── these do NOT change sym-runner; sym-runner picks them up automatically
```

**The rama foundation live-smoke is the gating dependency.** s109 EXO/RAMA integration audit closed 3 deployment-blocking defects (model_downloader empty source, pool path canon, ravan systemd unit) but has not been live-exercised end-to-end against a real socket + arena. Until that's done:
1. We don't know if seahorse's `with_ravan_check` path actually works in production.
2. The native engine path is theoretical.
3. Building `sym-runner` on top would compound the integration debt.

**Foreman recommendation:** do not start `sym-runner` skeleton until rama foundation is live-smoked. Once that's green, the llamacpp path is buildable in 1-2 days, native path follows in 1-2 more.

---

## Open design decisions

### 1. Trainer-as-superset vs separate crate

**Option A — Extend `sym-trainer`** with `runtime_*` tools alongside `training_*`. One wrapper, one MCP server, one prompt, one repo. Most of the pod-orchestration code is shared.

**Option B — New `sym-runner` crate** under `libs/sym-runner`. Clean separation of concerns; if runner grows large (multi-engine support, stress tests, monitoring), the boundary keeps it manageable.

**Lean: A.** Trainer-as-superset is the right call until scope explodes. Common code is real (pod create + portal SSH + HF auth + tmux session management), and an MCP server with both `training_*` and `runtime_*` tools is what the captain or an agent actually wants to call. The downside is a slightly bloated `sym_trainer.py` — easy to refactor later by splitting if needed.

**Decision rule:** revisit if `runtime_*` tools start needing materially different prompt context, agent personas, or scheduling semantics from `training_*`.

### 2. `runtime_pull` placement: HF helper vs orchestrated step

Should `runtime_pull` always run on a pod (target_volume must be a RunPod volume), or can it run locally (cache to `~/.cache/huggingface/`)?

**Sketch:** support both. `target_volume="local"` pulls locally for tiny models / testing. `target_volume=<volume_id>` provisions a small CPU pod with the volume attached, runs `huggingface-cli download` there, tears down. The latter saves local disk and pre-positions weights in the right DC.

### 3. Engine arg taxonomy

Currently sketched: `engine: "native"|"llamacpp"|"auto"`. Should we expose seahorse's deeper config (CUDA vs CPU vs ZPU lanes, KV cache size, quant level)? Or keep `sym-runner`'s engine arg simple and let seahorse own all engine-specific knobs via a config file the user mounts?

**Lean:** keep sym-runner's surface dumb. Expose `engine` (which engine) + `pod_gpu` (which hardware) + `port` (where it listens). Anything more granular goes through a `seahorse_config: Optional[dict]` passthrough that gets serialized and dropped on the pod, NOT exposed as separate MCP args. Keeps the tool surface stable as seahorse evolves.

### 4. Tunnel: portal vs reverse-ssh vs runpod public port

**Portal tunnel (preferred):** `portal tunnel <profile> 8000:localhost:8000` opens a background SSH port-forward; `localhost:8000` on captain's box hits the pod's served endpoint. Survives reconnects via portal's auto-reconnect. Cleanest for agent integration since OpenAI clients can target `http://localhost:8000/v1/...` without TLS or auth headaches.

**Reverse-ssh (alternative):** if portal isn't available or breaks, a plain `ssh -L 8000:localhost:8000 root@<pod>` works. Less robust on disconnect.

**RunPod public port (last resort):** RunPod can expose pod ports publicly via `--ports "8000/http"` but that needs auth/TLS handling that we'd rather avoid for internal use.

**Decision:** portal tunnel as primary, reverse-ssh fallback. Public port is a non-goal.

### 5. Error / cost containment

`sym-runner` MUST: hard-cap session pod time (default 1 hour, configurable), idle-detect (no requests in 15 min → auto-stop), surface cost-to-date in `runtime_status`. Captain's pod-spend-by-accident risk is real (s142 lost ~$0.04 to runpodctl creating pods despite returning 500 errors); the wrapper should catch this class.

---

## Implementation phases

### Phase 0 — prerequisites (foreman, captain, agents involved)

- [ ] Live-smoke rama foundation (sudo session: `./install.sh && systemctl enable --now ravan`, exercise socket + arena, verify seahorse `with_ravan_check` works end-to-end). **BLOCKER.**
- [ ] Verify seahorse exposes a clean `serve(model, engine, port)` API or document the equivalent CLI. If gaps, file follow-up tickets.

### Phase 1 — MVP (~1-2 days)

- [ ] Decide trainer-as-superset (option A) — extend `sym-trainer` repo with `runtime_*` tools.
- [ ] Add `runtime_pull` (local + pod-targeted modes).
- [ ] Add `runtime_serve` (llamacpp engine only initially) + `runtime_status` + `runtime_stop`.
- [ ] Portal tunnel integration (`portal tunnel` background command, life cycle managed by `runtime_serve`/`runtime_stop`).
- [ ] Cost cap + idle-stop wired into `runtime_serve`.
- [ ] Update `sym-trainer` README + system prompt to document the new tools.
- [ ] Smoke test: pull `nixprabin/bodhi` (whatever GGUF we have), serve via llamacpp engine, hit endpoint with one prompt, stop.

### Phase 2 — native engine (~1-2 days, after Phase 0 complete)

- [ ] Wire `engine="native"` to seahorse's rama-zpu adapter.
- [ ] Verify zram-backed weights actually stay hot during serve (this is what TASK-70 Phase 1 was about — bench validation in s142 found we don't have a zram-backed test setup; sym-runner needs one).
- [ ] Smoke test: serve a small model via native engine, confirm endpoint behavior matches llamacpp engine.

### Phase 3 — eval shared with trainer (~1 day)

- [ ] Lift the eval rubric registry from `training_eval` into a shared module both `training_eval` and `runtime_eval` import.
- [ ] Add `runtime_eval` MCP tool that posts rubric prompts to a running serve endpoint, tallies scores, returns the same shape as `training_eval`.
- [ ] Document the rubric format + add 2-3 starter rubrics (bodhi_eval, sunaina_eval, instruction_following).

### Phase 4 — operational polish (~1-2 days)

- [ ] `runtime_status` includes pod-cost-to-date, GPU utilization, request rate.
- [ ] Log forwarding (pod's seahorse logs → local file via portal sync).
- [ ] Auto-resume on pod migration (RunPod community pods can get migrated; portal's host-resolver pattern handles SSH reconnect, but in-flight HTTP requests need handling).

### Future / out-of-scope-for-MVP

- Multi-engine routing / A/B testing (engine="native" vs "llamacpp" simultaneously)
- Multi-model serving (one pod, multiple models — needs vllm-class engine)
- Streaming + tool-use API parity with anthropic/openai

---

## Effort estimate (post-Phase-0)

| Phase | Effort | Cumulative |
|---|---|---|
| 1 — MVP llamacpp path | 1-2 days | 1-2d |
| 2 — native engine | 1-2 days | 2-4d |
| 3 — eval shared | 1 day | 3-5d |
| 4 — polish | 1-2 days | 4-7d |

**~1 working week to "trainer + runner closes the loop end-to-end" assuming Phase 0 is unblocked.** Each phase is independently shippable; no Phase 1+ work locks scope on later phases.

---

## Risks & open questions

- **Seahorse divergence (s109 finding).** `crates/ai/capabilities/seahorse/bindings/rust/src/` has 9 source files diverged from upstream `libs/seahorse/bindings/rust/src/` plus an extra `zpu.rs`. Captain noted "I'll remind you later about seahorse divergence" but it's not closed. `sym-runner` building on seahorse means picking which seahorse — exosphere's vendored fork or the libs/ original. Foreman recommends: clarify the canonical seahorse before sym-runner starts.
- **Eval rubric portability.** `training_eval` rubrics may be tied to specific dataset shapes (Bodhi's identity rubric, Sunaina's RLM rubric). Lifting them to a shared registry needs a clean abstraction over "rubric = (prompt-set, scoring-fn)". Possibly file as a separate spec before Phase 3.
- **Engine config drift.** If seahorse changes its engine API between Phase 1 and Phase 2, sym-runner's serve command needs a versioning story. Could pin seahorse via uv lock or HF release tags; needs a decision before Phase 2 ships.
- **CUDA driver mismatch (recurring).** s142 hit it: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` vs pip-default torch cu130. seahorse's deps need to either pin torch / pin a custom container image / detect-and-warn. File as Phase 0 sub-item.

---

## Linked threads

- FOREMAN_THREADS parking lot: `💡 sym-runner — inference orchestrator paired with sym-trainer`
- FOREMAN_THREADS rama section: `🟡 ZPU Compute Fabric — Phase 1: EXO/RAMA integration smoke` (BLOCKER)
- FOREMAN_THREADS exosphere section: `🟡 libs/seahorse upstream/fork divergence`
- Research: `.squad/research/sym-trainer/2026-05-09-bodhi-v2-salvage-probe.md` (s142, motivation for closing the inference half of the loop — currently DIY each time)
- Sym-trainer README: `libs/sym-trainer/README.md` (architecture this design parallels)
