<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Agent Skills

This directory is a collection of [Agent Skills](https://code.claude.com/docs/en/skills):
self-contained guides that capture knowledge about
working in this codebase, packaged so that coding agents (and humans) can
load them on demand.

Each skill lives in its own subdirectory containing a `SKILL.md` file with
YAML frontmatter (`name` + `description`) followed by the guide itself. The
`description` is what an agent reads to decide whether a skill is relevant
to the task at hand.

We welcome contributions, additions, and amendments to existing skills.

## Available skills

Porting a model to AIE/NPU generally moves through these phases in order;
each skill below covers one:

| Skill | What it covers |
|-------|----------------|
| [`aie-model-baseline`](aie-model-baseline/SKILL.md) | Phase 1 — preparing a model for deployment: locking a quantization scheme, exporting ONNX, extracting a deployment manifest, and building a bit-exact numeric oracle. |
| [`aie-dataflow-presim`](aie-dataflow-presim/SKILL.md) | Phase 2 — validating an IRON dataflow design in software before hardware: a threaded ObjectFifo mock for deadlock/depth bugs, bit-exact validation against the oracle, probing novel mechanisms in isolation, and capacity/regime modeling. |
| [`aie-hw-bringup`](aie-hw-bringup/SKILL.md) | Phase 3 — bringing up a design on real hardware for the first time: sequential block-by-block bring-up against the reference, methodical bisection, and memory-budget tile splits. |
| [`aie-kernel-opt`](aie-kernel-opt/SKILL.md) | Phase 4 (micro) — optimizing AIE / Peano-compiled kernels — measure-first methodology plus a priority-ordered catalog of concrete levers and the codegen traps each one trips. |
| [`aie-dataflow-opt`](aie-dataflow-opt/SKILL.md) | Phase 4 (macro) — optimizing the dataflow around already-correct kernels: NOOP-ablation-driven prioritization, regime-aware placement/overlays, and DMA bandwidth/compression modeling. |

## Using a skill

**With Claude Code.** Symlink this whole `skills/` directory into a
discovered skills location once, and every skill here (current and future)
auto-loads — no per-skill step:

```shell
# Project-scoped (this repo only):
mkdir -p .claude
ln -s ../skills .claude/skills

# Or personal (all your projects):
ln -s "$(pwd)/skills" ~/.claude/skills
```

Claude Code scans `.claude/skills/` (walking up to the repo root) and
`~/.claude/skills/` at startup and loads each `SKILL.md` it finds. To pull
in just one skill instead of the whole collection, symlink that single
subdirectory (e.g. `mkdir -p .claude/skills && ln -s ../../skills/aie-kernel-opt .claude/skills/aie-kernel-opt`).

**With any other agent, or by hand.** A `SKILL.md` is plain Markdown — read
it directly, or point your agent's instructions file at the relevant skill.

## Contributing a skill

- One directory per skill; the directory name is the skill identifier.
- Start every `SKILL.md` with `---` frontmatter containing at least `name`
  and `description`, then the guide body.
- Write a `description` that states both *what* the skill does and *when*
  to use it — that line is the only thing an agent sees before deciding to
  load it.
- Keep skills grounded in verifiable specifics (commands, intrinsics,
  measured deltas) rather than generic advice.
- Include the standard license header (see the existing skills).
