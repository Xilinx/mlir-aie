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

from pathlib import Path

from aie.iron.kernel import ExternalFunction
from llama_config import HD, HEAD_D, KV_TB, N_HEADS_KV, N_HEADS_Q, N_TILE, REP, D

KERNELS_DIR = Path(__file__).parent / "kernels"


class KernelSource:
    """One kernels/*.cc translation unit and the flags it is compiled with."""

    def __init__(self, source: str, compile_flags: list[str]):
        self.source = source
        self.compile_flags = compile_flags

    def bind(self, symbol: str, arg_types: list) -> ExternalFunction:
        """Declare entry point `symbol` of this unit with `arg_types`."""
        return ExternalFunction(
            symbol,
            object_file_name=f"{self.source}.o",
            source_file=str(KERNELS_DIR / self.source),
            arg_types=arg_types,
            compile_flags=self.compile_flags,
        )


def rmsnorm(*, cols=D):
    return KernelSource(
        "llama_rmsnorm_int8.cc",
        ["-ffp-contract=off", f"-DLLAMA_RMSNORM_COLS={cols}"],
    )


def silu_mul(*, cols=HD, chunk=1024):
    return KernelSource(
        "llama_silu_mul_int8.cc",
        [
            "-ffp-contract=off",
            f"-DLLAMA_SILU_MUL_COLS={cols}",
            f"-DLLAMA_SILU_CHUNK={chunk}",
        ],
    )


def attn_block(*, tb=KV_TB):
    return KernelSource(
        "llama_attn_block.cc",
        [
            "-ffp-contract=off",
            f"-DLLAMA_ATTN_HEAD_DIM={HEAD_D}",
            f"-DLLAMA_ATTN_TB={tb}",
            f"-DLLAMA_ATTN_REP={REP}",
        ],
    )


def lmhead(*, kset=8):
    return KernelSource(
        "llama_lmhead_topk.cc",
        ["-ffp-contract=off", f"-DLLAMA_LMHEAD_KSET={kset}"],
    )


def gemv_array(*, m_max):
    return KernelSource(
        "llama_gemv_array.cc", ["-ffp-contract=off", f"-DLLAMA_GEMV_M_MAX={m_max}"]
    )


def gemv_mma(*, m_max, out_bytes):
    return KernelSource(
        "llama_gemv_mma.cc",
        [
            "-ffp-contract=off",
            f"-DLLAMA_GEMV_M_MAX={m_max}",
            f"-DLLAMA_GEMV_OUT_BYTES={out_bytes}",
        ],
    )


def gemm_mma():
    return KernelSource("llama_gemm_mma.cc", ["-ffp-contract=off"])


def prefill_glue():
    return KernelSource("llama_prefill_glue.cc", ["-ffp-contract=off"])


def gqa_glue():
    return KernelSource(
        "llama_gqa_glue.cc",
        [
            "-ffp-contract=off",
            f"-DLLAMA_GQA_HEAD_DIM={HEAD_D}",
            f"-DLLAMA_GQA_N_HEADS_Q={N_HEADS_Q}",
            f"-DLLAMA_GQA_N_HEADS_KV={N_HEADS_KV}",
        ],
    )


def topk_sample(*, kset=8):
    return KernelSource(
        "llama_topk_sample.cc",
        [
            "-ffp-contract=off",
            f"-DLLAMA_TOPK_D={D}",
            f"-DLLAMA_TOPK_NTILE={N_TILE}",
            f"-DLLAMA_TOPK_KSET={kset}",
        ],
    )
