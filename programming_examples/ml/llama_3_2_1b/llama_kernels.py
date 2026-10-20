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

KERNELS_DIR = Path(__file__).parent / "kernels"

# The softmax/sampler exp LUT resolution; the numpy references use the same.
EXP_QUANT_SCALE = 0.05


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


def rope(*, head_dim=64, n_heads=8):
    return KernelSource(
        "llama_rope_int8.cc",
        [f"-DLLAMA_ROPE_HEAD_DIM={head_dim}", f"-DLLAMA_ROPE_N_HEADS={n_heads}"],
    )


def flowkv(*, head_dim=64, T=16):
    return KernelSource(
        "llama_flowkv.cc",
        [
            f"-DLLAMA_FLOWKV_HEAD_DIM={head_dim}",
            f"-DLLAMA_FLOWKV_T={T}",
            f"-DLLAMA_FLOWKV_EXP_QUANT_SCALE={EXP_QUANT_SCALE}f",
        ],
        [generated_include_dir({"exp_lut.h": exp_lut_header(EXP_QUANT_SCALE)})],
    )


def gemm_layer(*, kset=16):
    return KernelSource(
        "llama_gemm_int8_srs_tiled_layer.cc",
        ["-ffp-contract=off", f"-DLLAMA_LMHEAD_KSET={kset}"],
    )


def gemm_layer_mh():
    return KernelSource(
        "llama_gemm_int8_srs_tiled_layer_mh.cc",
        ["-ffp-contract=off"],
    )


def gemv_array(*, kset=8):
    return KernelSource(
        "llama_gemv_array.cc",
        ["-ffp-contract=off", f"-DLLAMA_GEMV_ARRAY_KSET={kset}"],
    )


def kv_append(*, T=128):
    return KernelSource(
        "llama_kv_append.cc",
        ["-ffp-contract=off", "-DLLAMA_KVA_HEAD_DIM=64", f"-DLLAMA_KVA_T={T}"],
    )


def rope_mh():
    return KernelSource(
        "llama_rope_int8_mh.cc",
        [
            "-ffp-contract=off",
            "-DLLAMA_ROPE_MH_HEAD_DIM=64",
            "-DLLAMA_ROPE_MH_N_HEADS=32",
        ],
    )


def flowkv_mh(*, T=128):
    return KernelSource(
        "llama_flowkv_mh.cc",
        [
            "-ffp-contract=off",
            "-DLLAMA_FLOWKV_HEAD_DIM=64",
            f"-DLLAMA_FLOWKV_T={T}",
            "-DLLAMA_FLOWKV_REP=4",
            f"-DLLAMA_FLOWKV_EXP_QUANT_SCALE={EXP_QUANT_SCALE}f",
        ],
        [generated_include_dir({"exp_lut.h": exp_lut_header(EXP_QUANT_SCALE)})],
    )


def gqa_glue():
    return KernelSource(
        "llama_gqa_glue.cc",
        [
            "-DLLAMA_GQA_HEAD_DIM=64",
            "-DLLAMA_GQA_N_HEADS_Q=32",
            "-DLLAMA_GQA_N_HEADS_KV=8",
        ],
    )


def rescale_add():
    return KernelSource(
        "llama_rescale_add.cc",
        ["-ffp-contract=off", "-DLLAMA_RESCALE_D=2048"],
    )


def sample_streamed():
    return KernelSource(
        "llama_sample_streamed.cc",
        [
            "-ffp-contract=off",
            f"-DLLAMA_SAMPLE_EXP_QUANT_SCALE={EXP_QUANT_SCALE}f",
            "-DLLAMA_SAMPLE_VOCAB=128256",
            "-DLLAMA_SAMPLE_CHUNK_N=2004",
        ],
        [generated_include_dir({"exp_lut.h": exp_lut_header(EXP_QUANT_SCALE)})],
    )


def embed_select():
    return KernelSource(
        "llama_embed_select.cc",
        [
            "-ffp-contract=off",
            "-DLLAMA_EMBED_D=2048",
            "-DLLAMA_EMBED_NTILE=4",
            "-DLLAMA_EMBED_PREFIX=64",
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
