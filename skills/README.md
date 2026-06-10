<!--
This file is licensed under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Copyright (C) 2026, Advanced Micro Devices, Inc.
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

| Skill | What it covers |
|-------|----------------|
| [`aie-kernel-opt`](aie-kernel-opt/SKILL.md) | Optimizing AIE / Peano-compiled kernels — measure-first methodology plus a priority-ordered catalog of concrete levers and the codegen traps each one trips. |

## Using a skill

**With Claude Code.** Symlink or copy a skill into a discovered skills
location so it auto-loads:

```shell
# Project-scoped (this repo only):
mkdir -p .claude/skills
ln -s ../../skills/aie-kernel-opt .claude/skills/aie-kernel-opt

# Or personal (all your projects):
ln -s "$(pwd)/skills/aie-kernel-opt" ~/.claude/skills/aie-kernel-opt
```

Claude Code scans `.claude/skills/` (walking up to the repo root) and
`~/.claude/skills/` at startup and loads each `SKILL.md` it finds.

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
