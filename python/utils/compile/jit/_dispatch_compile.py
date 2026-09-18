# _dispatch_compile.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Lower and translate dynamic runtime sequences in process, then build a host DLL.

Only the host C++ compiler is an external tool. Libraries are published under
immutable content-addressed names and never loaded by the compiler, so rebuilds
cannot replace a mapped generation or pin a staging DLL on Windows.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import get_args

from aie.helpers.util import NpuDType, try_convert_np_type_to_mlir_type
from aie.ir import (  # pyright: ignore[reportMissingImports]
    Context,  # pyright: ignore[reportAttributeAccessIssue]
    IndexType,  # pyright: ignore[reportAttributeAccessIssue]
    IntegerType,  # pyright: ignore[reportAttributeAccessIssue]
    MemRefType,  # pyright: ignore[reportAttributeAccessIssue]
    MLIRError,  # pyright: ignore[reportAttributeAccessIssue]
    Module,  # pyright: ignore[reportAttributeAccessIssue]
    UnrankedMemRefType,  # pyright: ignore[reportAttributeAccessIssue]
    WalkResult,  # pyright: ignore[reportAttributeAccessIssue]
)
from aie.passmanager import (  # pyright: ignore[reportMissingImports]
    PassManager,  # pyright: ignore[reportAttributeAccessIssue]
)
from aie.utils import config
from aie.utils.compile.utils import SHARED_LIB_SUFFIX, host_shared_lib_cmd

from . import _manifest


class DispatchCompileError(RuntimeError):
    """Raised when the dynamic dispatch bridge cannot be built for a design."""


def _scalar_c_type(mlir_type) -> str:
    """C spelling used by the EmitC translator for a runtime scalar."""
    if isinstance(mlir_type, IndexType):
        return "size_t"
    if isinstance(mlir_type, IntegerType):
        integer = IntegerType(mlir_type)
        if integer.width == 1:
            return "bool"
        if integer.width in (8, 16, 32, 64):
            prefix = "u" if integer.is_unsigned else ""
            return f"{prefix}int{integer.width}_t"
    raise TypeError(f"Unsupported dispatch scalar MLIR type: {mlir_type}")


def dispatch_scalar_c_type(declared) -> str:
    """Return the scalar's C ABI type using Runtime's NumPy-to-MLIR conversion."""
    if declared in get_args(NpuDType):
        with Context():
            mlir_type = try_convert_np_type_to_mlir_type(declared)
            if isinstance(mlir_type, (IndexType, IntegerType)):
                return _scalar_c_type(mlir_type)
    raise TypeError(
        f"Unsupported DispatchTime[{getattr(declared, '__name__', declared)}]: "
        "use a NumPy integer scalar type supported by Runtime, such as np.int32."
    )


def _check_runtime_sequence_abi(
    module: Module, dispatch_params: list[str], dispatch_param_types: list
) -> None:
    """Validate the canonical scalar ABI before translation consumes it.

    Memrefs become address patches, not C parameters. Every other argument
    must have a supported scalar ABI. IRON Runtime assigns dispatch block
    arguments in signature order, independently of callback argument order.
    """
    sequences = []

    def collect(op):
        if op.name == "aie.runtime_sequence":
            sequences.append(op)
        return WalkResult.ADVANCE

    module.operation.walk(collect)
    if len(sequences) != 1:
        raise DispatchCompileError(
            "dispatch bridge requires exactly one runtime_sequence; "
            f"found {len(sequences)}."
        )
    if len(dispatch_param_types) != len(dispatch_params):
        raise DispatchCompileError(
            "Every DispatchTime[T] parameter must have a declared scalar type."
        )
    c_types = []
    for arg in sequences[0].regions[0].blocks[0].arguments:
        if isinstance(arg.type, (MemRefType, UnrankedMemRefType)):
            continue
        try:
            c_types.append(_scalar_c_type(arg.type))
        except TypeError as e:
            raise DispatchCompileError(str(e)) from None
    if len(c_types) != len(dispatch_params):
        raise DispatchCompileError(
            f"the generated builder takes {len(c_types)} scalar parameter(s) "
            f"but the design declares {len(dispatch_params)} DispatchTime[T] "
            f"param(s) ({dispatch_params!r}). Check that every DispatchTime[T] "
            "value is forwarded once into Runtime(seq, fn_args=[...])."
        )
    for c_type, name, declared in zip(c_types, dispatch_params, dispatch_param_types):
        expected = dispatch_scalar_c_type(declared)
        if expected != c_type:
            raise DispatchCompileError(
                f"DispatchTime[T] parameter {name!r} is declared as "
                f"{getattr(declared, '__name__', declared)} (C {expected}) but "
                f"the generated builder takes {c_type} in that position. The "
                "scalar block-argument ABI must match signature order "
                f"({dispatch_params!r})."
            )


def compile_dispatch_bridge(
    kernel_dir: Path,
    dispatch_params: list[str],
    fold_ddr_addr_offset: bool,
    dispatch_param_types: list,
) -> Path:
    """Build an immutable dispatch library under the kernel-directory lock."""
    input_mlir = kernel_dir / "input_with_addresses.mlir"
    if not input_mlir.is_file():
        raise DispatchCompileError(
            f"{input_mlir} does not exist; expected aiecc's "
            "--get-input-with-addresses output."
        )
    gen_cpp = kernel_dir / "dispatch_gen.cpp"
    with Context():
        try:
            module = Module.parse(input_mlir.read_text())
            # The same registered pipeline aiecc uses, not a copied pass list.
            PassManager.parse("builtin.module(aie-npu-dma-lowering)").run(
                module.operation
            )
            (kernel_dir / "dispatch_lowered.mlir").write_text(str(module))
            _check_runtime_sequence_abi(module, dispatch_params, dispatch_param_types)
            from aie.dialects.aie import translate_npu_to_cpp

            gen_cpp.write_text(
                translate_npu_to_cpp(
                    module.operation,
                    fold_ddr_addr_offset=fold_ddr_addr_offset,
                    emit_dispatch_shim=True,
                )
            )
        except (MLIRError, RuntimeError) as e:
            raise DispatchCompileError(
                f"dispatch bridge lowering/translation: {e}"
            ) from e
    staging = kernel_dir / f"dispatch.staging{SHARED_LIB_SUFFIX}"
    try:
        cmd = host_shared_lib_cmd(
            gen_cpp, staging, opt="-O2", includes=[config.runtime_header_path()]
        )
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as e:
            detail = (
                (e.stderr or e.stdout)
                if isinstance(e, subprocess.CalledProcessError)
                else str(e)
            )
            raise DispatchCompileError(f"host C++ compile failed: {detail}") from e
        published = (
            kernel_dir / f"dispatch-{_manifest._digest(staging)}{SHARED_LIB_SUFFIX}"
        )
        if not published.exists():
            staging.replace(published)
        return published
    finally:
        staging.unlink(missing_ok=True)
