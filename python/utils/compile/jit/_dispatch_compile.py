# _dispatch_compile.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Compile the per-design "dispatch bridge" for ``DispatchTime[T]`` designs.

A design with a ``DispatchTime[T]`` parameter used dynamically (an
``aie.runtime_sequence`` argument driving a real ``scf.for``, not folded to a
constant) cannot go through aiecc's static ``--get-npu-insts`` binary emitter
at all -- ``AIETargetNPU.cpp`` requires every patched value to resolve to a
compile-time constant. The dynamic path instead lowers the runtime sequence
through a separate pipeline and translates it to a C++ function
(``generate_txn_<device>_<sequence>``, via ``aie-translate --aie-npu-to-cpp``)
that rebuilds the whole instruction stream from scratch given the runtime
scalar value(s) -- proven end-to-end only by hand-written lit tests today
(``test/npu-xrt/dynamic_pingpong_passthrough/``).

This module is the Python entry point for that pipeline: given the
``input_with_addresses.mlir`` aiecc already writes into every JIT kernel_dir,
it lowers the runtime sequence and translates it to C++, then compiles the
result with a HOST compiler (not Peano -- Peano only targets
``aie2*-none-unknown-elf``) into ``kernel_dir/dispatch.so``, which
``DispatchBridge`` (_dispatch_bridge.py) loads via ``ctypes`` at dispatch time.

``ConvertAIEXToEmitC`` emits the ``extern "C"`` entry points from the
``aie.runtime_sequence`` argument types, so the parameter types cross into
Python through the built ``.so``'s own ``dispatch_abi()``.
"""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

from aie.utils import config

from ._dispatch_abi import (
    C_TYPE_BY_NP_TYPE,
    EMIT_DISPATCH_SHIM_FLAG,
    read_dispatch_abi,
)

# The same pipeline aiecc runs in-process for the static path (its
# getNpuDmaLoweringPipeline calls this exact builder), so a dynamic design
# cannot lower differently. See lib/Dialect/AIEX/Transforms/AIEXNpuPipelines.cpp.
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

    ``EMIT_DISPATCH_SHIM_FLAG`` makes the pass emit the ``extern "C"``
    ``dispatch_generate`` / ``dispatch_abi`` entry points alongside the builder,
    so this file is the whole translation unit and its parameter types are the
    ones MLIR resolved.

    ``fold_ddr_addr_offset`` mirrors the static path's flag of the same name
    (``compilabledesign.py``'s ``_resolve_fold_ddr_addr_offset()``): the
    firmware auto-translates host buffer addresses for only the first 5
    arguments, so a DDR address_patch for a later argument must fold the AIE
    DDR aperture offset into ``arg_plus`` itself -- required for XRT/HSA
    (which consume the folded firmware ABI), and must be off for HRX (which
    translates every host buffer address itself; folding here too would
    double-translate it). This is a compile-time choice baked into the
    generated function, exactly like the static insts.bin is compiled once
    per fold choice -- not a per-call runtime option.
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

    The generated parameter order is the ``Runtime(inputs=[...])`` order the
    author wrote by hand; the declared order is the Python signature. Nothing
    ties the two together, so a design that threads its scalars in a different
    order than it declares them silently transposes values at every call.
    Comparing the type sequences catches that whenever the types differ; two
    same-typed parameters stay indistinguishable, as nothing distinguishes them.

    A declared type absent from ``C_TYPE_BY_NP_TYPE`` is left unchecked rather
    than guessed at, so a new scalar type cannot fail the build spuriously.
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

    Must be called after the xclbin build (which writes
    ``input_with_addresses.mlir`` into ``kernel_dir`` as a side effect) and
    only on a cache miss -- this function does no idempotency checking of its
    own; the caller's cache-hit gate is the single source of truth for
    whether recompilation is needed.

    ``fold_ddr_addr_offset`` must match the value the caller resolved for the
    active backend (``CompilableDesign._resolve_fold_ddr_addr_offset()``) --
    see ``_translate_to_cpp``'s docstring. Already part of the on-disk cache
    key via ``_hash.py``'s unconditional ``fold_ddr_addr_offset`` hash input,
    so an XRT/HSA compile and an HRX compile of the same design never share a
    cache entry (matching the static insts.bin path).
    """
    lowered_mlir = _lower_dynamic_runtime_sequence(kernel_dir)
    gen_cpp = _translate_to_cpp(lowered_mlir, kernel_dir, fold_ddr_addr_offset)
    so_path = _compile_so(gen_cpp, kernel_dir)
    _check_built_abi(so_path, dispatch_params, dispatch_param_types)
    return so_path
