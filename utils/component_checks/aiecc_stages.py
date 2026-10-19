# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Group an aiecc ``--profile-json`` file's edges into compile stages.

The JIT writes one beside every build it runs aiecc for. Stages:

- ``place``: tile placement (``placed.mlir``, ``aie-place-tiles``, which is
  where the SA placer runs).
- ``route``: routing the placed design's flows (``input_physical.mlir``).
- ``lower``: the rest of getting from the parsed input to a placed, routed,
  addressed module (objectFifo lowering, buffer addresses, each core's
  module), and the runtime-sequence lowering.
- ``kernels``: aiecc blocked until the external kernel objects the cores
  link are compiled (``awaitLinkFiles``); compiling them overlaps the lowering
  before it, so only the wait shows here.
- ``codegen``: each core's program to an object (LLVM IR, Peano).
- ``link``: core ELFs, linker scripts, and the stack/data sizes and bank
  checks read back from them, plus the device cache that restores them.
- ``package``: CDO, PDI, xclbin and instruction stream.
- ``other``: an edge no rule names, so a new one shows up rather than
  vanishing.

Edges run in parallel, so a stage's time (its edges' summed time) can add up,
over the stages, to more than ``wall_s``.
"""

import json
import os
import re

STAGES = ("place", "route", "lower", "kernels", "codegen", "link", "package", "other")
# What the JIT names the file, in each build's ``<xclbin stem>.prj`` dir.
PROFILE_NAME = "aiecc_profile.json"
# A compile's rows: each stage's time, aiecc's wall time, and the rest of the
# compile's (the Python trace, the external kernels' compile it overlapped
# with, and anything driving it).
STAGE_METRICS = (
    *(f"compile_{stage}_s" for stage in STAGES),
    "compile_aiecc_wall_s",
    "compile_outside_aiecc_s",
)

# First match wins; output names chosen on aiecc's command line (xclbin, pdi,
# instructions) are matched by extension.
_RULES = [
    ("place", r"placed\.mlir"),
    ("route", r"input_physical\.mlir"),
    ("kernels", r"awaitLinkFiles"),
    ("link", r"deviceCacheLookup|preBakedCores|placedCoreCompile"),
    (
        "link",
        r"(probeElfs|elfs|preBakedElfs)_\{0\}\.elf|(probeScripts|ldScripts)_\{0\}\.ld\.script"
        r"|placedCore_\{0\}\.mlir|measured_(stack|data)_sizes\.mlir"
        r"|checked_(bank_placement|lut_banks)\.mlir|physical_with_elfs\.mlir|image_check\.txt"
        r"|linkwith_\{0\}\.txt|\{0\}\.bcf|xchesscc_wrapper",
    ),
    (
        "codegen",
        r"perCoreInDevice|perCoreCompile|(llvmIR|peano-compat|peano-linked|chess-compat|chesslinked|opted)_\{0\}\.ll"
        r"|objects_\{0\}\.o",
    ),
    (
        "lower",
        r"(input|default_stack_size|traced|input_with_symbols|input_with_addresses)\.mlir"
        r"|(perCore|lowered|perDevice|npu_seq|ctrlpkt_lowered|ctrlpkt_expanded_seq)_\{0\}\.mlir"
        r"|perCore(StackSpace|Arches|IRLinkFiles)_\{0\}\.txt|npu_[a-z_]+\.mlir|ctrlpkt_[a-z_]+\.mlir"
        r"|perDeviceMatching|perSeqMatching|sequencePlacement",
    ),
    (
        "package",
        r"cdo_\{0\}|.*\.(bif|pdi|xclbin|bin|elf|cpp|h)|(memTopology|kernels|partition|input_aie_partition_parsed"
        r"|merged_partition|ctrlpkt_extbuf)_\{0\}\.json|full_elf.*",
    ),
]
_COMPILED = [(stage, re.compile(pattern)) for stage, pattern in _RULES]


def stage_of(edge: str) -> str:
    """Return the stage of the aiecc edge named ``edge``."""
    for stage, pattern in _COMPILED:
        if pattern.fullmatch(edge):
            return stage
    return "other"


def stage_seconds(profile_path: str) -> dict | None:
    """Return seconds per stage plus ``wall_s``, or None without a profile.

    Args:
        profile_path: an aiecc ``--profile-json`` file.

    Returns:
        ``{stage: seconds for each of STAGES, "wall_s": seconds}``, or None
        when the file does not exist (an aiecc without ``--profile-json``).
    """
    if not os.path.exists(profile_path):
        return None
    with open(profile_path) as f:
        profile = json.load(f)
    out: dict[str, float] = dict.fromkeys(STAGES, 0.0)
    for edge in profile["edges"]:
        out[stage_of(edge["name"])] += edge["ms"] / 1000.0
    out["wall_s"] = profile["wall_ms"] / 1000.0
    return out


def stage_rows(profile_path: str, compile_s: float | None, suffix: str) -> dict:
    """Return one compile's STAGE_METRICS values, each name ending in ``suffix``.

    Args:
        profile_path: the compile's aiecc ``--profile-json`` file.
        compile_s: the whole compile's wall time, or None if it failed.
        suffix: ``""`` for a cold compile, ``"_warm"`` for a warm one.

    Returns:
        ``{metric + suffix: seconds}``, empty when the compile failed or
        wrote no profile.
    """
    stages = stage_seconds(profile_path)
    if stages is None or compile_s is None:
        return {}
    seconds = {f"compile_{stage}_s": stages[stage] for stage in STAGES}
    seconds["compile_aiecc_wall_s"] = stages["wall_s"]
    seconds["compile_outside_aiecc_s"] = max(compile_s - stages["wall_s"], 0.0)
    return {f"{k}{suffix}": round(v, 2) for k, v in seconds.items()}
