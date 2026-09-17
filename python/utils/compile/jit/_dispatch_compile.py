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
an immutable ``kernel_dir/dispatch-<digest>.so`` (``.dll`` on Windows) with a
HOST compiler -- not Peano, which only targets ``aie2*-none-unknown-elf``.
``DispatchBridge`` loads the selected generation via ``ctypes`` at dispatch time.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import get_args

from aie.helpers.util import NpuDType, try_convert_np_type_to_mlir_type
from aie.ir import (  # pyright: ignore[reportMissingImports]
    Context,  # pyright: ignore[reportAttributeAccessIssue]
    IndexType,  # pyright: ignore[reportAttributeAccessIssue]
    IntegerType,  # pyright: ignore[reportAttributeAccessIssue]
)
from aie.utils import config

from . import _manifest
from ._dispatch_bridge import EMIT_DISPATCH_SHIM_FLAG

# Resolves to the same builder aiecc calls in-process for the static path, so a
# dynamic design cannot lower differently. See AIEXNpuPipelines.cpp.
_DYNAMIC_LOWERING_PASSES = ["--aie-npu-dma-lowering"]

_WINDOWS = sys.platform == "win32"

# Windows loads DLLs, and -fPIC is a hard driver error on the MSVC target
# (position-independent code is the only mode there) rather than a no-op.
SHARED_LIB_SUFFIX = ".dll" if _WINDOWS else ".so"
SHARED_LIB_FLAGS = ["-shared"] if _WINDOWS else ["-shared", "-fPIC"]


def host_shared_lib_cmd(src: Path, out: Path, *, opt: str, includes=()) -> list[str]:
    """Build the host-compiler command that turns *src* into a shared library.

    Shared with the dispatch-bridge tests so the flag set they exercise is the
    one the product actually uses.
    """
    return [
        config.host_cxx_path(),
        *SHARED_LIB_FLAGS,
        opt,
        # Matches the project-wide CMAKE_CXX_STANDARD; the generated builder
        # needs nothing newer than std::optional/std::vector.
        "-std=c++17",
        *(f"-I{inc}" for inc in includes),
        str(src),
        "-o",
        str(out),
    ]


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


def _compile_so(gen_cpp: Path, so_path: Path) -> None:
    cmd = host_shared_lib_cmd(
        gen_cpp, so_path, opt="-O2", includes=[config.runtime_header_path()]
    )
    _run(cmd, "host C++ compile")


def dispatch_scalar_c_type(declared) -> str:
    """Return the scalar's C ABI type using Runtime's NumPy-to-MLIR conversion."""
    if declared in get_args(NpuDType):
        with Context():
            mlir_type = try_convert_np_type_to_mlir_type(declared)
            if isinstance(mlir_type, IndexType):
                return "size_t"
            if isinstance(mlir_type, IntegerType):
                integer_type = IntegerType(mlir_type)
                prefix = "u" if integer_type.is_unsigned else ""
                return f"{prefix}int{integer_type.width}_t"
    raise TypeError(
        f"Unsupported DispatchTime[{getattr(declared, '__name__', declared)}]: "
        "use a NumPy integer scalar type supported by Runtime, such as np.int32."
    )


def _check_built_abi(
    so_path: Path,
    dispatch_params: list[str],
    dispatch_param_types: list,
) -> None:
    """Check the built library's own ABI against what the design declares.

    The generated parameter order is the hand-written ``Runtime(inputs=[...])``
    order; the declared order is the Python signature. Nothing ties the two
    together, so threading scalars in a different order than they are declared
    silently transposes values at every call. Comparing type sequences catches
    that whenever the types differ; two same-typed parameters stay
    indistinguishable.
    """
    try:
        # Loading in the compiler pins DLLs on Windows and caches pathnames on
        # POSIX. The probe owns the only staging-library handle and exits before
        # publication (or removal on failure).
        c_types = json.loads(
            _run(
                [
                    sys.executable,
                    "-c",
                    "import ctypes, json, sys; "
                    "from aie.utils.compile.jit._dispatch_bridge import read_dispatch_abi; "
                    "print(json.dumps(read_dispatch_abi(ctypes.CDLL(sys.argv[1]), sys.argv[1])))",
                    str(so_path.resolve()),
                ],
                "dispatch ABI probe",
            )
        )
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
        expected = dispatch_scalar_c_type(declared)
        if expected != c_type:
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
    """Build an immutable dispatch library for a design with DispatchTime[T] params.

    Must be called after the xclbin build, which writes
    ``input_with_addresses.mlir`` into ``kernel_dir``, and only on a cache
    miss, under the kernel-directory lock.
    """
    lowered_mlir = _lower_dynamic_runtime_sequence(kernel_dir)
    gen_cpp = _translate_to_cpp(lowered_mlir, kernel_dir, fold_ddr_addr_offset)
    staging = kernel_dir / f"dispatch.staging{SHARED_LIB_SUFFIX}"
    try:
        _compile_so(gen_cpp, staging)
        _check_built_abi(staging, dispatch_params, dispatch_param_types)
        published = (
            kernel_dir / f"dispatch-{_manifest._digest(staging)}{SHARED_LIB_SUFFIX}"
        )
        if not published.exists():
            staging.replace(published)
        return published
    finally:
        staging.unlink(missing_ok=True)
