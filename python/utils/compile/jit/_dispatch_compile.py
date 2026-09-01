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

This module is the first Python entry point for that pipeline: given the
``input_with_addresses.mlir`` aiecc already writes into every JIT kernel_dir,
it lowers the runtime sequence, translates it to C++, wraps the generated
function in a small ``extern "C"`` shim, and compiles that shim with a HOST
compiler (not Peano -- Peano only targets ``aie2*-none-unknown-elf``) into
``kernel_dir/dispatch.so``, which ``DispatchBridge`` (_dispatch_bridge.py)
loads via ``ctypes`` at dispatch time.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
from aie.utils import config

from ._dispatch_artifacts import write_dispatch_abi

# aie-opt spelling of aiecc's getNpuDmaLoweringPipeline (tools/aiecc/IRTransforms.h),
# which the static path runs in-process. Same passes in the same order -- a dynamic
# design must lower identically. test_dispatch_bridge.py pins the two together.
_DYNAMIC_LOWERING_PASSES = [
    "--aie-materialize-bd-chains",
    "--aie-substitute-shim-dma-allocations",
    "--aie-unroll-runtime-sequence-loops",
    "--canonicalize",
    "--aie-decompose-large-dma-bd",
    "--aie-lower-dynamic-bd-pool",
    "--canonicalize",
    "--aie-assign-runtime-sequence-bd-ids",
    "--aie-dma-tasks-to-npu",
    "--aie-lower-dma-channel-reset",
    "--aie-dma-to-npu",
    "--aie-lower-set-lock",
    "--aie-lower-core-reset",
]

_GENERATE_TXN_SIGNATURE_RE = re.compile(
    r"inline\s+std::optional<std::vector<uint32_t>>\s+"
    r"(generate_txn_\w+)\s*\(([^)]*)\)"
)


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
    """Run aie-translate --aie-npu-to-cpp; return the generated header path.

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
    header_path = kernel_dir / "dispatch_gen.h"
    fold_flag = (
        f"--aie-npu-fold-ddr-addr-offset={'true' if fold_ddr_addr_offset else 'false'}"
    )
    stdout = _run(
        [
            config.aie_translate_path(),
            "--aie-npu-to-cpp",
            fold_flag,
            str(lowered_mlir),
        ],
        "aie-translate",
    )
    header_path.write_text(stdout)
    return header_path


# DispatchTime[T] wrapped type -> the C spelling aie-translate emits for it.
# Only unambiguous mappings: anything absent here is left unchecked rather
# than guessed at, so a new scalar type cannot fail the build spuriously.
_C_TYPE_BY_NP_TYPE = {
    np.int8: "int8_t",
    np.uint8: "uint8_t",
    np.int16: "int16_t",
    np.uint16: "uint16_t",
    np.int32: "int32_t",
    np.uint32: "uint32_t",
    np.int64: "int64_t",
    np.uint64: "uint64_t",
}


def _check_param_types(
    params: list[tuple[str, str]],
    dispatch_params: list[str],
    dispatch_param_types: list | None,
) -> None:
    """Check the generated C parameters against the declared DispatchTime[T] types.

    The order of the generated parameters is the ``Runtime(inputs=[...])``
    order the author wrote by hand; the declared order is the Python
    signature. Nothing ties the two together, so a design that threads its
    scalars in a different order than it declares them silently transposes
    values at every call. Comparing the type sequences catches that whenever
    the types differ.

    Same-typed parameters stay indistinguishable here -- see the note on
    ``compile_dispatch_bridge``.
    """
    if not dispatch_param_types:
        return
    for (c_type, _c_name), name, declared in zip(
        params, dispatch_params, dispatch_param_types
    ):
        expected = _C_TYPE_BY_NP_TYPE.get(declared)
        if expected is not None and expected != c_type:
            raise DispatchCompileError(
                f"DispatchTime[T] parameter {name!r} is declared as "
                f"{getattr(declared, '__name__', declared)} (C {expected}) but "
                f"the generated builder takes {c_type} in that position. The "
                f"order values are threaded into Runtime(inputs=[...]) must "
                f"match the order they are declared in the signature "
                f"({dispatch_params!r})."
            )


def _parse_signature(
    header_text: str,
    dispatch_params: list[str],
    dispatch_param_types: list | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    """Return ``(func_name, [(ctype, name), ...])`` for the sole ``generate_txn_*``.

    Raises if none or more than one such function is present, if the parameter
    count doesn't match ``len(dispatch_params)``, or if a parameter's C type
    contradicts the declared ``DispatchTime[T]`` type in that position (see
    ``_check_param_types``). Two same-typed parameters threaded in the wrong
    order remain undetectable -- nothing distinguishes them.

    The "exactly one" requirement is not a gap in AIEXToEmitC.cpp -- it
    already emits one correctly-named generate_txn_<device>_<sequence>
    function per (aie.device, aie.runtime_sequence) pair in the module. It
    reflects a real, current limit one level up: `iron.Program` takes exactly
    one `Device` and one `Runtime`, so no design built through the normal
    @iron.jit/Program/Runtime API can ever produce more than one. Revisit
    only if Program itself grows multi-device support -- a separate, larger
    feature than this one.
    """
    matches = _GENERATE_TXN_SIGNATURE_RE.findall(header_text)
    if len(matches) != 1:
        raise DispatchCompileError(
            f"expected exactly one generate_txn_* function in the generated "
            f"header, found {len(matches)}. A multi-device or multi-runtime-"
            f"sequence module cannot be built through iron.Program today "
            f"(it takes exactly one Device and one Runtime), so this isn't "
            f"reachable via @iron.jit; wiring it up needs Program itself to "
            f"support multiple devices/sequences first."
        )
    func_name, params_str = matches[0]
    params_str = params_str.strip()
    if not params_str:
        params: list[tuple[str, str]] = []
    else:
        params = []
        for param in params_str.split(","):
            param = param.strip()
            ctype, name = param.rsplit(" ", 1)
            params.append((ctype.strip(), name.strip()))

    if len(params) != len(dispatch_params):
        raise DispatchCompileError(
            f"generate_txn_* declares {len(params)} scalar parameter(s) but "
            f"the design declares {len(dispatch_params)} DispatchTime[T] "
            f"param(s) ({dispatch_params!r}). Check that every DispatchTime[T] "
            f"value is threaded into Runtime(inputs=[...]) in declaration "
            f"order."
        )
    _check_param_types(params, dispatch_params, dispatch_param_types)
    return func_name, params


def _write_wrapper_cpp(
    kernel_dir: Path, func_name: str, params: list[tuple[str, str]]
) -> Path:
    """Emit the extern "C" ctypes-callable shim around ``func_name``.

    ``generate_txn_*`` already builds a complete ``std::vector<uint32_t>``
    before returning, so its exact size is known on the C++ side -- the
    wrapper keeps owning that vector (in thread-local storage, overwritten by
    each call) and hands Python a pointer + the exact word count, instead of
    Python guessing a buffer capacity. ``DispatchBridge`` copies the words out
    immediately, before making any further call on this thread.

    Sentinel (matching DispatchBridge's expectations): ``-2`` means the
    generator itself returned ``std::nullopt`` (a runtime scalar overflowed a
    hardware BD field). Any non-negative return is the actual word count.
    """
    param_decls = ", ".join(f"{ctype} {name}" for ctype, name in params)
    if param_decls:
        param_decls += ", "
    arg_names = ", ".join(name for _ctype, name in params)

    wrapper_cpp = f"""\
#include "dispatch_gen.h"
#include <cstddef>
#include <cstdint>
#include <vector>

thread_local static std::vector<uint32_t> g_dispatch_result;

extern "C" int64_t dispatch_generate({param_decls}uint32_t **out_ptr) {{
  auto result = {func_name}({arg_names});
  if (!result) return -2;
  g_dispatch_result = std::move(*result);
  *out_ptr = g_dispatch_result.data();
  return static_cast<int64_t>(g_dispatch_result.size());
}}
"""
    wrapper_path = kernel_dir / "dispatch_wrapper.cpp"
    wrapper_path.write_text(wrapper_cpp)
    return wrapper_path


def _compile_wrapper(wrapper_cpp: Path, kernel_dir: Path) -> Path:
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
        f"-I{kernel_dir}",
        str(wrapper_cpp),
        "-o",
        str(so_path),
    ]
    _run(cmd, "host C++ compile")
    return so_path


def compile_dispatch_bridge(
    kernel_dir: Path,
    dispatch_params: list[str],
    fold_ddr_addr_offset: bool = True,
    dispatch_param_types: list | None = None,
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
    header_path = _translate_to_cpp(lowered_mlir, kernel_dir, fold_ddr_addr_offset)
    header_text = header_path.read_text()
    func_name, params = _parse_signature(
        header_text, dispatch_params, dispatch_param_types
    )
    wrapper_cpp = _write_wrapper_cpp(kernel_dir, func_name, params)
    so_path = _compile_wrapper(wrapper_cpp, kernel_dir)
    # Record the ABI now, while it is known for certain. Dispatch then needs
    # only the .so and this sidecar -- never the generated C++ header, whose
    # formatting would otherwise have to stay parseable for the life of the
    # cache entry.
    write_dispatch_abi(
        kernel_dir, func_name, dispatch_params, [ctype for ctype, _name in params]
    )
    return so_path
