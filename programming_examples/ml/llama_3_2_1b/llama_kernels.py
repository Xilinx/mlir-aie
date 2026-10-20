#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The example's AIE kernels (kernels/*.cc) as IRON ExternalFunctions.

Each function returns one translation unit with its defines for the requested
shape, and `KernelSource.bind` declares one of its entry points. Entry points
of one unit share one object, so a design links each unit once; a different
shape is a different recipe, which the JIT compiles and caches as its own
object.
"""

import hashlib
import os
from pathlib import Path

from aie.iron.kernel import ExternalFunction
from aie.utils.compile import NPU_CACHE_HOME
from gen_exp_lut import exp_lut_header
from gen_silu_lut import silu_lut_header
from llama_config import EXP_QUANT_SCALE

KERNELS_DIR = Path(__file__).parent / "kernels"


def generated_include_dir(headers: dict[str, str]) -> str:
    """Write `headers` (file name -> text) into a directory named by their content.

    Args:
        headers: The generated headers a kernel includes.

    Returns:
        The directory, for an ExternalFunction's `include_dirs`.
    """
    digest = hashlib.sha256(repr(sorted(headers.items())).encode()).hexdigest()[:16]
    out = NPU_CACHE_HOME / "llama_3_2_1b" / "include" / digest
    out.mkdir(parents=True, exist_ok=True)
    for name, text in headers.items():
        path = out / name
        if not path.exists():
            staged = path.with_name(f".{name}.{os.getpid()}")
            staged.write_text(text)
            staged.replace(path)
    return str(out)


class KernelSource:
    """One kernels/*.cc translation unit and the flags it is compiled with."""

    def __init__(self, source: str, compile_flags: list[str], include_dirs=()):
        self.source = source
        self.compile_flags = compile_flags
        self.include_dirs = list(include_dirs)

    def bind(self, symbol: str, arg_types: list) -> ExternalFunction:
        """Declare entry point `symbol` of this unit with `arg_types`."""
        return ExternalFunction(
            symbol,
            object_file_name=f"{self.source}.o",
            source_file=str(KERNELS_DIR / self.source),
            arg_types=arg_types,
            compile_flags=self.compile_flags,
            include_dirs=self.include_dirs,
        )


def rmsnorm(*, cols=2048):
    return KernelSource(
        "llama_rmsnorm_int8.cc",
        ["-ffp-contract=off", f"-DLLAMA_RMSNORM_COLS={cols}"],
    )


def silu_mul(*, cols=8192, gate_scale=0.05):
    return KernelSource(
        "llama_silu_mul_int8.cc",
        ["-ffp-contract=off", f"-DLLAMA_SILU_MUL_COLS={cols}"],
        [generated_include_dir({"silu_lut.h": silu_lut_header(gate_scale)})],
    )


def attn_block(*, tb=64):
    return KernelSource(
        "llama_attn_block.cc",
        [
            "-ffp-contract=off",
            "-DLLAMA_ATTN_HEAD_DIM=64",
            f"-DLLAMA_ATTN_TB={tb}",
            "-DLLAMA_ATTN_REP=4",
        ],
    )


def lmhead(*, kset=16):
    return KernelSource(
        "llama_lmhead_topk.cc",
        ["-ffp-contract=off", f"-DLLAMA_LMHEAD_KSET={kset}"],
    )


def gemv_array(*, kset=8):
    return KernelSource(
        "llama_gemv_array.cc",
        ["-ffp-contract=off", f"-DLLAMA_GEMV_ARRAY_KSET={kset}"],
    )


def gemv_mma():
    return KernelSource("llama_gemv_mma.cc", ["-ffp-contract=off"])


def gqa_glue():
    return KernelSource(
        "llama_gqa_glue.cc",
        [
            "-DLLAMA_GQA_HEAD_DIM=64",
            "-DLLAMA_GQA_N_HEADS_Q=32",
            "-DLLAMA_GQA_N_HEADS_KV=8",
        ],
    )


def topk_sample(*, dim=2048, kset=8):
    return KernelSource(
        "llama_topk_sample.cc",
        [
            "-ffp-contract=off",
            f"-DLLAMA_TOPK_D={dim}",
            "-DLLAMA_TOPK_NTILE=4",
            f"-DLLAMA_TOPK_KSET={kset}",
        ],
        [generated_include_dir({"exp_lut.h": exp_lut_header(EXP_QUANT_SCALE)})],
    )
