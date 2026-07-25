# hostruntime.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""HSA/ROCR implementation of the HostRuntime.

Consumes the aiecc artifacts ``insts.bin`` + ``main.pdi`` (the xclbin is
ignored on this path) and dispatches them as one AIE AQL packet:

    insts.bin + main.pdi -> HSA region allocations (hsa_memory_allocate)
    I/O tensors          -> kernarg buffer of 2*N uint64 (VAs then sizes)
    fill AQL packet, ring doorbell, wait on completion signal
"""

import atexit
import ctypes
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

from ..hostruntime import HostRuntime, HostRuntimeError, KernelHandle, KernelResult
from .tensor import HSATensor
from . import HSAContext

if TYPE_CHECKING:
    from aie.iron.device import Device

_TRACE_UNSUPPORTED_MSG = (
    "Trace capture is not supported on the HSA backend. Re-run without a "
    "trace_config, or use the XRT backend (IRON_RUNTIME=xrt) for trace-enabled "
    "designs."
)


class HSAKernelHandle(KernelHandle):
    """Handle for a loaded HSA kernel (PDI + insts in region memory)."""

    def __init__(self, pdi_ptr, pdi_size, insts_ptr, insts_size, kernel_name):
        self.pdi_ptr = pdi_ptr
        self.pdi_size = pdi_size
        self.insts_ptr = insts_ptr
        self.insts_size = insts_size
        self.kernel_name = kernel_name


class HSAKernelResult(KernelResult):
    """Result wrapper for an HSA dispatch (raises on failure, so success here)."""

    def __init__(self, npu_time, success=True, trace_config=None):
        super().__init__(npu_time, trace_config)
        self._success = success

    def is_success(self) -> bool:
        return self._success


class HSAHostRuntime(HostRuntime):
    """HostRuntime that dispatches IRON designs through HSA/ROCR."""

    _tensor_class = HSATensor

    def __init__(self):
        self._ctx = HSAContext.get()
        self._exe_cache = OrderedDict()
        self._cache_size = int(os.environ.get("HSA_EXE_CACHE_SIZE", "32"))

    def _find_pdi(self, xclbin_path: Path) -> Path:
        kernel_dir = xclbin_path.parent
        main_pdi = kernel_dir / "main.pdi"
        if main_pdi.is_file():
            return main_pdi
        pdis = sorted(kernel_dir.glob("*.pdi"))
        if pdis:
            return pdis[0]
        raise HostRuntimeError(
            f"No PDI (main.pdi) found in {kernel_dir}. The HSA backend needs a "
            f"PDI; ensure aiecc emitted one alongside the xclbin."
        )

    def load(self, npu_kernel, **kwargs) -> HSAKernelHandle:
        self.check_device_consistency()
        xclbin_path = Path(npu_kernel.xclbin_path).resolve()
        insts_path = Path(npu_kernel.insts_path).resolve()
        kernel_name = npu_kernel.kernel_name or "MLIR_AIE"

        if not insts_path.exists() or not insts_path.is_file():
            raise HostRuntimeError(f"insts {insts_path} does not exist or is not a file.")
        pdi_path = self._find_pdi(xclbin_path)

        key = (
            str(insts_path), insts_path.stat().st_mtime,
            str(pdi_path), pdi_path.stat().st_mtime,
            kernel_name,
        )
        if key in self._exe_cache:
            self._exe_cache.move_to_end(key)
            return self._exe_cache[key]

        insts_bytes = insts_path.read_bytes()
        if len(insts_bytes) % 4 != 0:
            raise HostRuntimeError("insts.bin length is not a multiple of 4 bytes")
        pdi_bytes = pdi_path.read_bytes()

        insts_ptr = self._ctx.alloc_region(len(insts_bytes))
        ctypes.memmove(insts_ptr, insts_bytes, len(insts_bytes))
        try:
            pdi_ptr = self._ctx.alloc_region(len(pdi_bytes))
            ctypes.memmove(pdi_ptr, pdi_bytes, len(pdi_bytes))
        except BaseException:
            self._ctx.free_region(insts_ptr)
            raise

        handle = HSAKernelHandle(
            pdi_ptr, len(pdi_bytes), insts_ptr, len(insts_bytes), kernel_name
        )

        if self._cache_size > 0 and len(self._exe_cache) >= self._cache_size:
            _, old = self._exe_cache.popitem(last=False)
            self._ctx.free_region(old.pdi_ptr)
            self._ctx.free_region(old.insts_ptr)
        self._exe_cache[key] = handle
        return handle

    def run(self, kernel_handle, args, trace_config=None, fail_on_error=True,
            only_if_loaded=False, **kwargs) -> HSAKernelResult:
        assert isinstance(kernel_handle, HSAKernelHandle)
        if trace_config is not None:
            raise HostRuntimeError(_TRACE_UNSUPPORTED_MSG)
        self.check_device_consistency()

        kept = [a for a in args if not callable(a)]
        if not all(isinstance(a, self._tensor_class) for a in kept):
            raise HostRuntimeError(
                f"The {self.__class__.__name__} can only take "
                f"{self._tensor_class.__name__} as arguments, but got: {kept}"
            )
        n = len(kept)

        # kernarg buffer: 2*N uint64 (N VAs then N sizes)
        ka_handle, ka_va, ka_size = self._ctx.vmem_alloc(2 * n * 8)
        try:
            ka = (ctypes.c_uint64 * (2 * n)).from_address(ka_va)
            for i, t in enumerate(kept):
                ka[i] = t.buffer_object()
                # Logical byte size (matching dispatch.cc), not the padded/granule-rounded alloc size.
                ka[n + i] = t.nbytes

            signal = self._ctx.create_signal(1)
            try:
                start = time.time_ns()
                self._ctx.dispatch(
                    kernel_handle.pdi_ptr, kernel_handle.insts_ptr,
                    kernel_handle.insts_size, ka_va, n, signal,
                )
                self._ctx.wait(signal)
                stop = time.time_ns()
            finally:
                self._ctx.destroy_signal(signal)
        finally:
            self._ctx.vmem_free(ka_handle, ka_va, ka_size)

        return HSAKernelResult(stop - start, success=True)

    def device(self) -> "Device":
        from aie.iron.device import from_name

        return from_name(self._ctx.device_gen, n_cols=None)

    def cleanup(self) -> None:
        cache = getattr(self, "_exe_cache", None)
        if not cache:
            return
        while cache:
            _, handle = cache.popitem(last=False)
            self._ctx.free_region(handle.pdi_ptr)
            self._ctx.free_region(handle.insts_ptr)


class CachedHSAHostRuntime(HSAHostRuntime):
    """Cache-by-default entry point matching CachedXRTRuntime naming."""

    def __init__(self):
        super().__init__()
        atexit.register(self.cleanup)
