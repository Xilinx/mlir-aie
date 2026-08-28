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

from aie.utils import config

# aie-opt pass list that lowers a materialized runtime sequence to npu ops,
# mirroring test/npu-xrt/dynamic_pingpong_passthrough/run.lit verbatim (not
# re-derived) -- this is the one place in the repo this pipeline is proven
# end-to-end on real hardware.
_DYNAMIC_LOWERING_PASSES = [
    "--aie-materialize-bd-chains",
    "--aie-substitute-shim-dma-allocations",
    "--aie-unroll-runtime-sequence-loops",
    "--canonicalize",
    "--aie-lower-dynamic-bd-pool",
    "--canonicalize",
    "--aie-assign-runtime-sequence-bd-ids",
    "--aie-dma-tasks-to-npu",
    "--aie-dma-to-npu",
    "--aie-lower-set-lock",
]

# The NPU firmware ABI (xclbin + insts.bin, "kernel(opcode, insts_bo, ninsts,
# host_bo0, ...)") auto-translates host buffer addresses for only the first
# this-many arguments; a DDR address_patch for any later argument must fold
# in the AIE DDR aperture offset (0x80000000) to land correctly. Confirmed at
# lib/Targets/AIETargetNPU.cpp:127-145. AIEXToEmitC.cpp's NpuAddressPatchOp
# conversion (the dynamic path) has no equivalent fold -- confirmed by reading
# AIEXToEmitC.cpp:197-208, which passes arg_plus straight through. Until that
# C++ gap is fixed, a dispatch design needs at most this many host tensor
# buffers (arg_idx 0-2 are the fixed opcode/insts_bo/ninsts slots, so this
# allows 2 tensor buffers: arg_idx 3 and 4).
_NUM_FIRMWARE_TRANSLATED_ARGS = 5

_GENERATE_TXN_SIGNATURE_RE = re.compile(
    r"inline\s+std::optional<std::vector<uint32_t>>\s+"
    r"(generate_txn_\w+)\s*\(([^)]*)\)"
)
_ADDRESS_PATCH_CALL_RE = re.compile(
    r"aie_runtime::txn_append_address_patch\([^,]+,[^,]+,\s*(\d+)\s*[,)]"
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


def _translate_to_cpp(lowered_mlir: Path, kernel_dir: Path) -> Path:
    """Run aie-translate --aie-npu-to-cpp; return the generated header path."""
    header_path = kernel_dir / "dispatch_gen.h"
    stdout = _run(
        [config.aie_translate_path(), "--aie-npu-to-cpp", str(lowered_mlir)],
        "aie-translate",
    )
    header_path.write_text(stdout)
    return header_path


def _parse_signature(
    header_text: str, dispatch_params: list[str]
) -> tuple[str, list[tuple[str, str]]]:
    """Return ``(func_name, [(ctype, name), ...])`` for the sole ``generate_txn_*``.

    Raises if none or more than one such function is present (v1 supports
    only a single-device, single-runtime-sequence design), or if the
    parameter count doesn't match ``len(dispatch_params)`` -- the generator
    author is responsible for passing DispatchTime[T] values into
    ``Runtime(inputs=[...])`` in declaration order (this only catches a count
    mismatch, e.g. a forgotten or extra scalar, not a silent reordering).
    """
    matches = _GENERATE_TXN_SIGNATURE_RE.findall(header_text)
    if len(matches) != 1:
        raise DispatchCompileError(
            f"expected exactly one generate_txn_* function in the generated "
            f"header, found {len(matches)}. Multi-device / multi-runtime-"
            f"sequence DispatchTime[T] designs are not supported yet."
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
    return func_name, params


def _check_address_patch_fold_gap(header_text: str) -> None:
    """Raise if any address_patch targets a host buffer beyond the firmware-translated range.

    AIEXToEmitC.cpp does not fold the DDR aperture offset the way
    AIETargetNPU.cpp does for the static path (see module docstring and
    _NUM_FIRMWARE_TRANSLATED_ARGS above). Unverified without a >2-buffer
    dynamic design to test against, so this is rejected outright rather than
    silently producing wrong DMA addresses.
    """
    bad_indices = sorted(
        {
            int(idx)
            for idx in _ADDRESS_PATCH_CALL_RE.findall(header_text)
            if int(idx) >= _NUM_FIRMWARE_TRANSLATED_ARGS
        }
    )
    if bad_indices:
        raise DispatchCompileError(
            f"design has host buffer(s) at arg_idx {bad_indices}, beyond the "
            f"firmware-translated range (0-{_NUM_FIRMWARE_TRANSLATED_ARGS - 1}, "
            f"i.e. at most 2 host tensor buffers). The dynamic C++ TXN "
            f"builder does not fold the AIE DDR aperture offset for these "
            f"(a known gap in AIEXToEmitC.cpp, unlike the static path's "
            f"AIETargetNPU.cpp), so DispatchTime[T] designs are limited to 2 "
            f"host tensor buffers until that is fixed."
        )


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
        "-std=c++20",
        f"-I{config.cxx_header_path()}",
        f"-I{kernel_dir}",
        str(wrapper_cpp),
        "-o",
        str(so_path),
    ]
    _run(cmd, "host C++ compile")
    return so_path


def compile_dispatch_bridge(kernel_dir: Path, dispatch_params: list[str]) -> Path:
    """Build ``kernel_dir/dispatch.so`` for a design with DispatchTime[T] params.

    Must be called after the xclbin build (which writes
    ``input_with_addresses.mlir`` into ``kernel_dir`` as a side effect) and
    only on a cache miss -- this function does no idempotency checking of its
    own; the caller's cache-hit gate is the single source of truth for
    whether recompilation is needed.
    """
    lowered_mlir = _lower_dynamic_runtime_sequence(kernel_dir)
    header_path = _translate_to_cpp(lowered_mlir, kernel_dir)
    header_text = header_path.read_text()
    func_name, params = _parse_signature(header_text, dispatch_params)
    _check_address_patch_fold_gap(header_text)
    wrapper_cpp = _write_wrapper_cpp(kernel_dir, func_name, params)
    return _compile_wrapper(wrapper_cpp, kernel_dir)
