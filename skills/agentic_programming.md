<!---//===- agentic_programming.md --------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Agentic Programming

**Agentic programming** here means working with a coding agent — [Claude
Code](https://code.claude.com/docs/en/skills) or any other agent that
supports the Agent Skills format — to port and optimize models on AIE/NPU
hardware using IRON. The agent does the typing; the
[Agent Skills](./README.md) in this repo give it the project-specific
knowledge it would otherwise have to rediscover on every task: which
mechanism to reach for, which pitfalls are known traps, and which
methodology (measure first, verify against an oracle, bisect on failure)
actually works in this codebase.

This isn't a replacement for the [Programming Guide](../programming_guide/) —
it's a second way to use the same knowledge. The Programming Guide teaches
*you* to write IRON. The skills teach an *agent* to write IRON with you,
grounded in the same dialect, the same DMA/ObjectFifo model, and the same
hardware constraints.

## The workflow, phase by phase

Porting a model to AIE/NPU moves through four phases in order. Each has a
skill:

1. **Prepare the model** ([`aie-model-baseline`](aie-model-baseline/SKILL.md)) —
   lock a quantization scheme, export ONNX, extract a deployment manifest,
   build a bit-exact numeric oracle to check everything else against.
2. **Validate the dataflow before hardware** ([`aie-dataflow-presim`](aie-dataflow-presim/SKILL.md)) —
   a threaded ObjectFifo mock catches deadlocks and depth bugs, bit-exact
   checks against the oracle catch logic bugs, all before touching a device.
3. **Bring it up on real hardware** ([`aie-hw-bringup`](aie-hw-bringup/SKILL.md)) —
   sequential block-by-block bring-up against the reference, methodical
   bisection when something hangs or mismatches.
4. **Optimize** — once correct, go faster at either level:
   [`aie-kernel-opt`](aie-kernel-opt/SKILL.md) (micro: make one compiled
   kernel faster) or [`aie-dataflow-opt`](aie-dataflow-opt/SKILL.md) (macro:
   fix tile placement, overlays, and DMA bandwidth around already-correct
   kernels).

An agent with these skills loaded picks the relevant one(s) for the task at
hand automatically — you don't need to name a phase or a skill explicitly.

## Getting started

See [Using a skill](./README.md#using-a-skill) for how to load these skills
into Claude Code (or point any other agent's instructions file at a
`SKILL.md` directly). The [skills README](./README.md) also covers the
format and how to contribute a new skill or amend an existing one.
