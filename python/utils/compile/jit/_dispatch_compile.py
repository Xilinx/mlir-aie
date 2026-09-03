# _dispatch_compile.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Compile the per-design "dispatch bridge" for ``DispatchTime[T]`` designs.

A ``DispatchTime[T]`` parameter used dynamically (driving a real ``scf.for``,
not folded to a constant) cannot go through aiecc's static ``--get-npu-insts``
emitter, which requires every patched value to be a compile-time constant.

So instead: take the ``input_with_addresses.mlir`` aiecc already writes into
every JIT kernel_dir, lower it, translate it to a C++ instruction-stream
builder (``aie-translate --aie-npu-to-cpp``), and compile that to
``kernel_dir/dispatch.so`` with a HOST compiler -- not Peano, which only
targets ``aie2*-none-unknown-elf``. ``DispatchBridge`` loads the ``.so`` via
``ctypes`` at dispatch time.
"""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

from aie.utils import config

from ._dispatch_bridge import (
    C_TYPE_BY_NP_TYPE,
    EMIT_DISPATCH_SHIM_FLAG,
    read_dispatch_abi,
)

# Resolves to the same builder aiecc calls in-process for the static path, so a
# dynamic design cannot lower differently. See AIEXNpuPipelines.cpp.
_DYNAMIC_LOWERING_PASSES = ["--aie-npu-dma-lowering"]


class DispatchCompileError(RuntimeError):
    """Raised when the dynamic dispatch bridge cannot be built for a design."""


def _run(cmd: list[str], step: str) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error_msg = result.stderr if result.stderr else result.stdout
        raise DispatchCompileError(
            f"[{step}] failed with exit code {result.returncode}:\n{error_msg}"
        )
    return result.stdout


def _lower_dynamic_runtime_sequence(kernel_dir: Path) -> Path:
    """Run the dynamic-lowering aie-opt pipeline; return the lowered .mlir path."""
    input_mlir = kernel_dir / "input_with_addresses.mlir"
    if not input_mlir.is_file():
        raise DispatchCompileError(
            f"{input_mlir} does not exist; expected aiecc's "
            "--get-input-with-addresses output."
        )
    lowered_mlir = kernel_dir / "dispatch_lowered.mlir"
    cmd = (
        [config.aie_opt_path()]
        + _DYNAMIC_LOWERING_PASSES
        + [str(input_mlir), "-o", str(lowered_mlir)]
    )
    _run(cmd, "aie-opt")
    return lowered_mlir


def _translate_to_cpp(
    lowered_mlir: Path, kernel_dir: Path, fold_ddr_addr_offset: bool
) -> Path:
    """Run aie-translate --aie-npu-to-cpp; return the generated .cpp path.

    ``EMIT_DISPATCH_SHIM_FLAG`` emits the ``extern "C"`` ``dispatch_generate``
    / ``dispatch_abi`` entry points alongside the builder, making this file the
    whole translation unit.

    ``fold_ddr_addr_offset`` is the active backend's DDR-patch ABI, resolved by
    ``CompilableDesign._resolve_fold_ddr_addr_offset()`` and baked in here
    exactly as the static insts.bin bakes it -- not a per-call option.
    """
    gen_cpp = kernel_dir / "dispatch_gen.cpp"
    fold_flag = (
        f"--aie-npu-fold-ddr-addr-offset={'true' if fold_ddr_addr_offset else 'false'}"
    )
    stdout = _run(
        [
            config.aie_translate_path(),
            "--aie-npu-to-cpp",
            fold_flag,
            EMIT_DISPATCH_SHIM_FLAG,
            str(lowered_mlir),
        ],
        "aie-translate",
    )
    gen_cpp.write_text(stdout)
    return gen_cpp


def _compile_so(gen_cpp: Path, kernel_dir: Path) -> Path:
    so_path = kernel_dir / "dispatch.so"
    cmd = [
        config.host_cxx_path(),
        "-shared",
        "-fPIC",
        "-O2",
        # Matches the project-wide CMAKE_CXX_STANDARD; the generated builder
        # needs nothing newer than std::optional/std::vector.
        "-std=c++17",
        f"-I{config.runtime_header_path()}",
        str(gen_cpp),
        "-o",
        str(so_path),
    ]
    _run(cmd, "host C++ compile")
    return so_path


def _check_built_abi(
    so_path: Path,
    dispatch_params: list[str],
    dispatch_param_types: list,
) -> None:
    """Check the built ``.so``'s own ABI against what the design declares.

    The generated parameter order is the hand-written ``Runtime(inputs=[...])``
    order; the declared order is the Python signature. Nothing ties the two
    together, so threading scalars in a different order than they are declared
    silently transposes values at every call. Comparing type sequences catches
    that whenever the types differ; two same-typed parameters stay
    indistinguishable. A declared type absent from ``C_TYPE_BY_NP_TYPE`` is
    left unchecked rather than guessed at.
    """
    try:
        c_types = read_dispatch_abi(ctypes.CDLL(str(so_path)), so_path)
    except (OSError, ValueError) as e:
        raise DispatchCompileError(f"dispatch bridge: {e}") from None

    if len(c_types) != len(dispatch_params):
        raise DispatchCompileError(
            f"the generated builder takes {len(c_types)} scalar parameter(s) "
            f"but the design declares {len(dispatch_params)} DispatchTime[T] "
            f"param(s) ({dispatch_params!r}). Check that every DispatchTime[T] "
            f"value is threaded into Runtime(inputs=[...]) in declaration order."
        )
    for c_type, name, declared in zip(c_types, dispatch_params, dispatch_param_types):
        expected = C_TYPE_BY_NP_TYPE.get(declared)
        if expected is not None and expected != c_type:
            raise DispatchCompileError(
                f"DispatchTime[T] parameter {name!r} is declared as "
                f"{getattr(declared, '__name__', declared)} (C {expected}) but "
                f"the generated builder takes {c_type} in that position. The "
                f"order values are threaded into Runtime(inputs=[...]) must "
                f"match the order they are declared in the signature "
                f"({dispatch_params!r})."
            )


def compile_dispatch_bridge(
    kernel_dir: Path,
    dispatch_params: list[str],
    fold_ddr_addr_offset: bool,
    dispatch_param_types: list,
) -> Path:
    """Build ``kernel_dir/dispatch.so`` for a design with DispatchTime[T] params.

    Must be called after the xclbin build, which writes
    ``input_with_addresses.mlir`` into ``kernel_dir``, and only on a cache
    miss: this does no idempotency checking of its own.
    """
    lowered_mlir = _lower_dynamic_runtime_sequence(kernel_dir)
    gen_cpp = _translate_to_cpp(lowered_mlir, kernel_dir, fold_ddr_addr_offset)
    so_path = _compile_so(gen_cpp, kernel_dir)
    _check_built_abi(so_path, dispatch_params, dispatch_param_types)
    return so_path
