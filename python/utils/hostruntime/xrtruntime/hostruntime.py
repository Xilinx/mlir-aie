# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""XRT-based implementation of the HostRuntime."""

import atexit
import ctypes
import gc
import logging
import os
import re
import stat
import subprocess
import sys
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyxrt  # pyright: ignore[reportMissingImports]

from ..hostruntime import HostRuntime, HostRuntimeError, KernelHandle, KernelResult
from ..tensor_class import TensorRegion
from .device import acquire_device, xrt_smi_path
from .parameter_scratchpad import ParameterScratchpad

if TYPE_CHECKING:
    from aie.iron.device import Device

    from ...trace import TraceConfig
from .tensor import XRTTensor, XrtTransport

if sys.platform == "linux":
    import fcntl  # amdxdna's ioctls; Windows has neither


# amdxdna_accel.h: struct amdxdna_drm_get_resource_info, read by
# DRM_IOCTL_AMDXDNA_GET_INFO (DRM_IOWR(DRM_COMMAND_BASE + 7, struct
# amdxdna_drm_get_info)) with param DRM_AMDXDNA_QUERY_RESOURCE_INFO.
class _ResourceInfo(ctypes.Structure):
    _fields_ = [
        ("npu_clk_max", ctypes.c_uint64),
        ("npu_tops_max", ctypes.c_uint64),
        ("npu_task_max", ctypes.c_uint64),
        ("npu_tops_curr", ctypes.c_uint64),
        ("npu_task_curr", ctypes.c_uint64),
    ]


class _GetInfo(ctypes.Structure):
    _fields_ = [
        ("param", ctypes.c_uint32),
        ("buffer_size", ctypes.c_uint32),
        ("buffer", ctypes.c_uint64),
    ]


_GET_INFO = 0xC0106447
_QUERY_RESOURCE_INFO = 12

# aiecc's kEmptySequenceName: the runtime sequence that does nothing, which a
# full ELF carries on each device with a runtime sequence.
EMPTY_SEQUENCE = "aiecc_empty"

logger = logging.getLogger(__name__)


class _KernelRun:
    """The one `pyxrt.run` a loaded kernel is dispatched through.

    A fresh `pyxrt.run` reconfigures a full ELF's device on its first start;
    a run rebound with `set_arg`, even to other buffers, does not. Every
    handle of the kernel shares the run, so `dispatch` holds a lock from the
    first `set_arg` to the end of the wait, and threads dispatching one kernel
    take turns. The run keeps the last dispatch's buffers bound, holding
    their device memory, until the kernel's next dispatch or until the run is
    dropped with its handle.

    Args:
        kernel: The loaded kernel.
        params_path: The lowering's ``params.txt``, for a kernel whose design
            declares ScratchpadParameters; None for one that declares none.
        command: The ``device:sequence`` the kernel runs, whose parameters
            its scratchpad holds.
        device: The device the kernel's hardware context is on.
        empty: For a full ELF's kernel, the run of the empty sequence aiecc
            puts on the same device of the ELF (`EMPTY_SEQUENCE`); None
            otherwise, or for an ELF without one.
    """

    def __init__(
        self,
        kernel,
        params_path: Path | None = None,
        command: str = "",
        device: "pyxrt.device | None" = None,
        empty: "_KernelRun | None" = None,
    ):
        self.kernel = kernel
        self.run = None
        self.lock = threading.Lock()
        self.params_path = params_path
        self.command = command
        self.device = device
        self.empty = empty
        self._scratchpad: ParameterScratchpad | None = None
        # The 4-word header of the last transaction dispatched through the run.
        self.txn_header: np.ndarray | None = None
        self._empty_txn: XrtTransport | None = None

    def dispatch(
        self, args, parameters: Mapping[str, Any] | None = None
    ) -> tuple["pyxrt.ert_cmd_state", int]:
        """Bind `args`, write `parameters` and run the kernel to completion.

        Args:
            args: The kernel's arguments in order, `pyxrt.bo`s and ints. A
                `None` leaves its slot unbound.
            parameters: ScratchpadParameter values by name, written to the
                run's ctrl scratchpad before it starts.

        Returns:
            The run's final command state, and the nanoseconds from its start
            to the end of the wait. A run that does not complete, or raises,
            is dropped so the next dispatch starts a fresh one.

        Raises:
            HostRuntimeError: `parameters` for a kernel that declares no
                ScratchpadParameter.
        """
        if parameters and self.params_path is None:
            raise HostRuntimeError(
                f"{self.command or 'this kernel'} declares no ScratchpadParameter "
                f"to write {sorted(parameters)} to"
            )
        with self.lock:
            if self.run is None:
                self.run = pyxrt.run(self.kernel)
                self._scratchpad = None
            try:
                for i, arg in enumerate(args):
                    if arg is not None:
                        self.run.set_arg(i, arg)
                if parameters:
                    if self._scratchpad is None:
                        assert self.params_path is not None
                        self._scratchpad = ParameterScratchpad(
                            self.run, self.params_path, self.command, self.device
                        )
                    for name, value in parameters.items():
                        self._scratchpad.write(name, value)
                    self._scratchpad.sync()
                start = time.perf_counter_ns()
                self.run.start()
                state = self.run.wait()
                stop = time.perf_counter_ns()
            except BaseException:
                self.run = None
                raise
            if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                self.run = None
            return state, stop - start

    def empty_dispatch_ns(self) -> int | None:
        """Time one dispatch that does nothing, on this kernel's hardware context.

        For a full ELF's kernel it is the ELF's empty sequence (`empty`).
        Otherwise it is a transaction with the header of the last one
        dispatched and no operations, through this run, which keeps that
        call's buffers bound.

        Returns:
            Nanoseconds from submit to completion, timed as a call is; None
            for a full ELF without an empty sequence, and before a transaction
            has been dispatched, as for an ELF module, which carries its own
            control code.

        Raises:
            HostRuntimeError: The empty dispatch did not complete.
        """
        if self.empty is not None:
            with self.lock:
                state, ns = self.empty.dispatch([])
            if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise HostRuntimeError(f"An empty dispatch returned {state}")
            return ns
        if self.txn_header is None:
            return None
        if self._empty_txn is None:
            words = self.txn_header.copy()
            words[2:4] = (0, words.nbytes)
            self._empty_txn = XrtTransport(
                acquire_device(),
                words.nbytes,
                pyxrt.bo.cacheable,
                self.kernel.group_id(1),
            )
            self._empty_txn.host_bytes[:] = words.view(np.uint8)
            self._empty_txn.to_device(0, words.nbytes)
        nbytes = self._empty_txn.nbytes
        state, ns = self.dispatch([3, self._empty_txn.handle(0, nbytes), nbytes])
        if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise HostRuntimeError(f"An empty dispatch returned {state}")
        return ns


def _full_elf_empty_run(context, kernel_name: str) -> _KernelRun | None:
    """Return the run of the empty sequence on `kernel_name`'s device of a full ELF.

    Args:
        context: The ELF's hardware context.
        kernel_name: The ``"<device>:<sequence>"`` kernel loaded from the ELF.

    Returns:
        The run, or None for an ELF with no empty sequence on that device.
    """
    name = f"{kernel_name.partition(':')[0]}:{EMPTY_SEQUENCE}"
    try:
        return _KernelRun(pyxrt.ext.kernel(context, name), command=name)
    except RuntimeError:
        return None


# XRTKernelHandle(kernel, xclbin, context, insts_path)
class XRTKernelHandle(KernelHandle):
    """Handle for a loaded XRT kernel."""

    def __init__(
        self,
        kernel,
        xclbin,
        context,
        insts,
        insts_bo=None,
        name=None,
        is_full_elf=False,
        num_args=None,
        init_name=None,
        kernel_run=None,
    ):
        """Initialize the XRTKernelHandle.

        Args:
            kernel: The XRT kernel object.
            xclbin: The XRT xclbin object.  ``None`` on the full-ELF path.
            context: The XRT context object.
            insts: The instructions for the kernel.  ``None`` on the full-ELF
                path (the ELF carries its own control code).
            insts_bo (optional): The instruction buffer object. Defaults to None.
            name (optional): The resolved name of the loaded kernel.
                Defaults to None.
            is_full_elf (bool, optional): True when the kernel was loaded from a
                self-contained full ELF; selects the ``run.set_arg`` +
                ``run.start`` execution path in ``run()``.  Defaults to False.
            num_args (optional): The number of arguments the xclbin declares
                for the loaded kernel, opcode and instructions included;
                ``run()`` refuses more host buffers than they leave. ``None``
                skips the check. Defaults to None.
            init_name (optional): The full ELF's init entry, which ``run()``
                runs first whenever the kernel's context is not the one it last
                dispatched. Defaults to None.
            kernel_run (optional): The run slot `kernel` is dispatched
                through, shared by every handle of the same kernel. Defaults
                to a slot of this handle's own.
        """
        super().__init__(needs_dispatch_insts=insts is None and not is_full_elf)
        self.kernel = kernel
        self.xclbin = xclbin
        self.context = context
        self.insts = insts
        self.insts_bo = insts_bo
        self.name = name
        self.is_full_elf = is_full_elf
        self.num_args = num_args
        self.init_name = init_name
        self.kernel_run = kernel_run if kernel_run is not None else _KernelRun(kernel)

    def empty_dispatch_ns(self) -> int | None:
        # An invalidated handle has no context left, e.g. after Phoenix's post-trace drain.
        if self.kernel_run is None:
            return None
        return self.kernel_run.empty_dispatch_ns()


class XRTKernelResult(KernelResult):
    """A wrapper around data produced as the result of running a kernel with the PyXRT runtime."""

    def __init__(
        self,
        ret: pyxrt.ert_cmd_state,
        npu_time: int,
        trace_config: "TraceConfig | None" = None,
    ):
        super().__init__(npu_time, trace_config)
        self.ret = ret

    def is_success(self) -> bool:
        return self.ret == pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED


class XRTHostRuntime(HostRuntime):
    """Singleton manager for AIE XRT resources."""

    # TODO: this is duplicated from the LIT helpers.
    # NPU Model mappings - centralized for easy updates
    # Maps generation name to list of model strings that may appear in xrt-smi
    NPU_MODELS = {
        "npu1": ["npu1", "Phoenix"],
        "npu2": [
            "npu4",
            "Strix",
            "npu5",
            "Strix Halo",
            "npu6",
            "Krackan",
            "Gorgon Point",
        ],
    }
    _tensor_class = XRTTensor

    def __init__(self):
        """Initialize the XRTHostRuntime."""
        self._device = acquire_device()

        self._device_type_str = self._device.get_info(pyxrt.xrt_info_device.name)

        self.npu_str: str | None = None
        for key, value in self.NPU_MODELS.items():
            if any([model in self._device_type_str for model in self.NPU_MODELS[key]]):
                self.npu_str = key
                break
        if not self.npu_str:
            raise RuntimeError(f"Unknown device type: {self._device_type_str}")
        # The hardware context of the last dispatch; see `run`.
        self._dispatched_context = None

    def provenance(self) -> dict[str, str]:
        """Add XRT's version from the shared, cached stack probe."""
        from ...probe import _examine_field

        fields = super().provenance()
        version = _examine_field("Version", section="XRT")
        if version:
            fields["xrt"] = version.split()[0]
        return fields

    def power_mode(self) -> str | None:
        """Return the power mode ``xrt-smi`` reports for this device, or ``None``.

        Scraped from ``xrt-smi examine --report platform`` for the device's
        BDF because XRT exposes no API for it. Best effort: a missing or
        failing ``xrt-smi``, or an unexpected report, gives ``None`` rather
        than an error, and a run that requires a mode then refuses to measure.
        """
        xrt_smi = xrt_smi_path()
        if xrt_smi is None:
            return None
        try:
            bdf = self._device.get_info(pyxrt.xrt_info_device.bdf)
            out = subprocess.run(
                [xrt_smi, "examine", "-d", bdf, "--report", "platform"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout
        except Exception:  # noqa: BLE001 - informational only
            return None
        m = re.search(r"(?i)\b(?:performance|power)\s*mode\s*:\s*(\S+)", out)
        return m.group(1).lower() if m else None

    def device_name(self) -> str | None:
        return self._device_type_str

    @property
    def core_mhz(self) -> int:
        """The highest AIE core (H) clock this runtime's NPU runs at, in MHz.

        Read from amdxdna's resource info on the device's accel node. The
        current H clock drops with load (792 MHz idle against 1800 under load
        on a Strix), so a cycles-to-time bound wants this maximum.

        Raises:
            NotImplementedError: The host is not Linux, where amdxdna runs.
            OSError: XRT holds no descriptor for the accel node, or the
                driver refuses the query.
        """
        if sys.platform != "linux":
            raise NotImplementedError("The NPU clock is read from amdxdna, on Linux.")
        bdf = self._device.get_info(pyxrt.xrt_info_device.bdf)
        (node,) = os.listdir(f"/sys/bus/pci/devices/{bdf}/accel")
        accel = f"/dev/accel/{node}"
        # amdxdna can refuse a second open of the node (EBUSY) in a process that
        # already has it open, so query through the descriptor XRT holds.
        held = []
        for fd in os.listdir("/proc/self/fd"):
            try:
                if os.readlink(f"/proc/self/fd/{fd}") == accel:
                    held.append(int(fd))
            except OSError:
                continue
        if not held:
            raise OSError(f"XRT holds no open descriptor for {accel}")
        resources = _ResourceInfo()
        request = _GetInfo(
            _QUERY_RESOURCE_INFO,
            ctypes.sizeof(resources),
            ctypes.addressof(resources),
        )
        fcntl.ioctl(held[0], _GET_INFO, request)
        return resources.npu_clk_max

    @classmethod
    def read_insts(cls, insts_path: Path):
        """Read instructions from the given file, with XRT-specific handling for ELF files.

        Args:
            insts_path (Path): Path to the instruction file.

        Returns:
            The instructions (either as bytes/array or XRT module).
        """
        # Overload the function in the generic class so we can use xrt-specific handling of elf files.
        ext = insts_path.suffix.lower()
        if ext == ".elf":
            elf = pyxrt.elf(str(insts_path))
            return pyxrt.module(elf)
        else:
            return super().read_insts(insts_path)

    def load(
        self,
        npu_kernel,
        **kwargs,
    ) -> XRTKernelHandle:
        """Load an NPU kernel into the XRT runtime.

        Args:
            npu_kernel: The NPU kernel to load.
            **kwargs: Additional arguments for loading.

        Returns:
            XRTKernelHandle: A handle to the loaded kernel.

        Raises:
            HostRuntimeError: If xclbin or insts files do not exist, or if kernel is not found.
        """
        self.check_device_consistency()

        if getattr(npu_kernel, "elf_path", None) is not None:
            return self._load_full_elf(npu_kernel)

        # Not the full-ELF path, so the xclbin is always populated. insts_path
        # is None for a DispatchTime[T] design, which synthesizes a fresh
        # stream per call instead (see run()'s dispatch_insts handling).
        assert npu_kernel.xclbin_path is not None
        xclbin_path = Path(npu_kernel.xclbin_path).resolve()
        insts_path = self._resolve_insts_path(npu_kernel)
        kernel_name = npu_kernel.kernel_name

        if not xclbin_path.exists() or not xclbin_path.is_file():
            raise HostRuntimeError(
                f"xclbin {xclbin_path} does not exist or is not a file."
            )

        xclbin = pyxrt.xclbin(str(xclbin_path))
        self._device.register_xclbin(xclbin)
        xclbin_uuid = xclbin.get_uuid()
        context = pyxrt.hw_context(self._device, xclbin_uuid)

        kernel_args = {k.get_name(): k.get_num_args() for k in xclbin.get_kernels()}
        kernel_name = self._kernel_name(kernel_name, kernel_args)

        insts = self.read_insts(insts_path) if insts_path is not None else None
        if isinstance(insts, pyxrt.module):
            kernel = pyxrt.ext.kernel(context, insts, kernel_name)
        else:
            kernel = pyxrt.kernel(context, kernel_name)

        kernel_handle = XRTKernelHandle(
            kernel,
            xclbin,
            context,
            insts,
            name=kernel_name,
            num_args=kernel_args[kernel_name],
        )
        return kernel_handle

    @staticmethod
    def _kernel_name(kernel_name: str | None, kernel_args: dict[str, int]) -> str:
        """Pick the kernel a load names, or the xclbin's first when it names none.

        Args:
            kernel_name: The name the NPU kernel gives, or None.
            kernel_args: The xclbin's kernels, by name, in its order.

        Returns:
            The name of a kernel in `kernel_args`.

        Raises:
            RuntimeError: If the xclbin has no kernels.
            HostRuntimeError: If `kernel_name` is not one of them.
        """
        if kernel_name is None:
            if not kernel_args:
                raise RuntimeError("No kernels found in xclbin")
            return next(iter(kernel_args))
        if kernel_name not in kernel_args:
            raise HostRuntimeError(
                f"Kernel {kernel_name} not found in xclbin (kernels found: {list(kernel_args)})"
            )
        return kernel_name

    def _load_full_elf(self, npu_kernel) -> XRTKernelHandle:
        """Load a kernel from a self-contained full ELF.

        The full ELF bundles the PDIs + TXN control code, so the hardware
        context is created directly from ``pyxrt.elf(path)`` (no xclbin), and
        the kernel is addressed by its ``"<device>:<sequence>"`` name.
        """
        elf_path = Path(npu_kernel.elf_path).resolve()
        kernel_name = npu_kernel.kernel_name

        if not elf_path.exists() or not elf_path.is_file():
            raise HostRuntimeError(
                f"full ELF {elf_path} does not exist or is not a file."
            )

        elf = pyxrt.elf(str(elf_path))
        context = pyxrt.hw_context(self._device, elf)
        kernel = pyxrt.ext.kernel(context, kernel_name)

        return XRTKernelHandle(
            kernel,
            None,
            context,
            None,
            name=kernel_name,
            is_full_elf=True,
            init_name=getattr(npu_kernel, "init_kernel_name", None),
            kernel_run=_KernelRun(
                kernel,
                npu_kernel.params_path,
                kernel_name,
                self._device,
                empty=_full_elf_empty_run(context, kernel_name),
            ),
        )

    def _run_init_entry(self, context, init_name: str | None, kernel_name: str) -> None:
        """Run the init entry ``init_name``, if any, before ``kernel_name``.

        Args:
            context: The hardware context about to run ``kernel_name``.
            init_name: ``NPUKernel.init_kernel_name``.
            kernel_name: The ``"<device>:<sequence>"`` kernel about to run.

        Raises:
            HostRuntimeError: If the init entry does not complete.
        """
        if init_name is None or init_name == kernel_name:
            return
        run = pyxrt.run(pyxrt.ext.kernel(context, init_name))
        run.start()
        r = run.wait()
        if r != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise HostRuntimeError(f"{init_name} returned {r}")

    def run(
        self,
        kernel_handle: KernelHandle,
        args,
        trace_config=None,
        fail_on_error: bool = True,
        only_if_loaded: bool = False,
        dispatch_insts=None,
        parameters: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> XRTKernelResult:
        """Run a loaded XRT kernel.

        Args:
            kernel_handle (XRTKernelHandle): The handle to the loaded kernel.
            args: Arguments to pass to the kernel.
            trace_config (optional): Configuration for tracing. Defaults to None.
            fail_on_error (bool, optional): Whether to raise an exception on kernel failure. Defaults to True.
            only_if_loaded (bool, optional): Accepted for API compatibility with the runtime base class.
            dispatch_insts (np.ndarray | None, optional): Per-call instruction
                words, submitted in a fresh BO instead of static instructions.
            parameters (Mapping[str, Any] | None, optional): ScratchpadParameter
                values by name, written to the run's ctrl scratchpad before it
                starts; a full ELF's kernel takes them.
            **kwargs: Additional arguments.

        Returns:
            XRTKernelResult: The result of the kernel execution.

        Raises:
            HostRuntimeError: If arguments are invalid or kernel execution fails (and fail_on_error is True).
        """
        assert isinstance(kernel_handle, XRTKernelHandle)
        self._require_dispatch_insts(kernel_handle, dispatch_insts)
        self.check_device_consistency()
        # Filter out callable functions and check arg types
        args = [a for a in args if not callable(a)]
        owners = [a.tensor if isinstance(a, TensorRegion) else a for a in args]
        if not all(isinstance(owner, self._tensor_class) for owner in owners):
            raise HostRuntimeError(
                f"The {self.__class__.__name__} can only take {self._tensor_class.__name__} "
                f"arguments, or regions of them, but got: {args}"
            )
        for owner in owners:
            owner.to("npu")
        buffers = [
            (
                a.tensor.storage.binding_handle(a.storage_offset, a.nbytes)
                if isinstance(a, TensorRegion)
                else a.buffer_object()
            )
            for a in args
        ]

        # Another context's dispatch may have reset the array since this
        # context last ran, so a full ELF's init entry runs again.
        if kernel_handle.context is not self._dispatched_context:
            if kernel_handle.is_full_elf:
                self._run_init_entry(
                    kernel_handle.context, kernel_handle.init_name, kernel_handle.name
                )
            self._dispatched_context = kernel_handle.context

        if kernel_handle.is_full_elf:
            return self._run_full_elf(kernel_handle, buffers, fail_on_error, parameters)

        # Validate BO count against xclbin metadata before calling into XRT.
        # XRT's validate_bo_at_index segfaults if the index exceeds the count
        # declared in kernels.json. The fixed args (opcode, instr, ninstr) are
        # the first 3; the rest are host BOs. Passing fewer BOs than declared is
        # fine (the kernel may declare a minimum ABI width); passing more is the
        # fatal case.
        if kernel_handle.num_args is not None:
            declared_bo_count = kernel_handle.num_args - 3
            if len(buffers) > declared_bo_count:
                raise HostRuntimeError(
                    f"The xclbin declares {declared_bo_count} host buffer "
                    f"argument(s) but {len(buffers)} were passed. Passing more "
                    f"host buffers than the kernel ABI declares would segfault "
                    f"in XRT argument setup."
                )

        insts_bo = None
        insts_bytes = 0
        txn = None
        try:
            if dispatch_insts is not None:
                txn = dispatch_insts
                insts_bytes = dispatch_insts.nbytes
                # Keep the owning transport alive through submission and wait.
                dispatch_storage = XrtTransport(
                    self._device,
                    insts_bytes,
                    pyxrt.bo.cacheable,
                    kernel_handle.kernel.group_id(1),
                )
                dispatch_storage.host_bytes[:] = dispatch_insts.view(np.uint8).reshape(
                    -1
                )
                dispatch_storage.to_device(0, insts_bytes)
                insts_bo = dispatch_storage.handle(0, insts_bytes)
            else:
                is_module = isinstance(kernel_handle.insts, pyxrt.module)
                if not is_module:
                    assert kernel_handle.insts is not None
                    txn = kernel_handle.insts
                    insts_bytes = kernel_handle.insts.nbytes
                    if kernel_handle.insts_bo:
                        insts_bo = kernel_handle.insts_bo
                    else:
                        insts_bo = self._tensor_class(
                            kernel_handle.insts,
                            flags=pyxrt.bo.cacheable,
                            group_id=kernel_handle.kernel.group_id(1),
                            xrt_device=self._device,
                        ).buffer_object()

            r, ns = kernel_handle.kernel_run.dispatch(
                [3, insts_bo, insts_bytes, *buffers], parameters
            )

            if fail_on_error and r != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise HostRuntimeError(f"Kernel returned {str(r)}")
            if txn is not None and kernel_handle.kernel_run.txn_header is None:
                kernel_handle.kernel_run.txn_header = (
                    np.ascontiguousarray(txn).view(np.uint32)[:4].copy()
                )
        finally:
            # delete insts buffer if it was created locally
            if insts_bo and not kernel_handle.insts_bo:
                del insts_bo

        return XRTKernelResult(r, ns)

    def _run_full_elf(
        self,
        kernel_handle: "XRTKernelHandle",
        buffers,
        fail_on_error: bool,
        parameters: Mapping[str, Any] | None,
    ) -> XRTKernelResult:
        """Execute a full-ELF kernel via the ``run.set_arg`` / ``start`` path.

        Full-ELF kernels carry their own control code in the ELF, so there is
        no instruction buffer to pass: each host buffer (including any trace
        buffer the trace lowering appended) is bound positionally with
        ``set_arg``.
        """
        r, ns = kernel_handle.kernel_run.dispatch(buffers, parameters)

        if fail_on_error and r != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise HostRuntimeError(f"Kernel returned {str(r)}")

        return XRTKernelResult(r, ns)

    def device(self) -> "Device":
        """Get the device associated with this runtime.

        Returns:
            Device: The device object (NPU1 or NPU2).

        Raises:
            HostRuntimeError: If the device string is unknown.
        """
        from aie.iron.device import from_name

        if self.npu_str in ("npu1", "npu2"):
            return from_name(self.npu_str, n_cols=None)
        else:
            raise HostRuntimeError(
                f"Unknown device string: {self.npu_str}: expected npu1 or npu2"
            )


class CachedXRTKernelHandle(XRTKernelHandle):
    """A cached handle for a loaded XRT kernel."""

    def __init__(
        self,
        kernel,
        xclbin,
        context,
        insts,
        insts_bo=None,
        name=None,
        is_full_elf=False,
        num_args=None,
        init_name=None,
        kernel_run=None,
    ):
        """Initialize the CachedXRTKernelHandle.

        Args:
            kernel: The XRT kernel object.
            xclbin: The XRT xclbin object.  ``None`` on the full-ELF path.
            context: The XRT context object.
            insts: The instructions for the kernel.  ``None`` on the full-ELF path.
            insts_bo (optional): The instruction buffer object. Defaults to None.
            name (optional): The resolved kernel name. Defaults to None.
            is_full_elf (bool, optional): True when loaded from a full ELF.
                Defaults to False.
            num_args (optional): See `XRTKernelHandle`. Defaults to None.
            init_name (optional): See `XRTKernelHandle`. Defaults to None.
            kernel_run (optional): See `XRTKernelHandle`. Defaults to None.
        """
        super().__init__(
            kernel,
            xclbin,
            context,
            insts,
            insts_bo,
            name=name,
            is_full_elf=is_full_elf,
            num_args=num_args,
            init_name=init_name,
            kernel_run=kernel_run,
        )
        self._is_valid = True

    def invalidate(self):
        """Invalidate the handle and release resources in dependency order."""
        self._is_valid = False
        # Instruction BOs and kernels depend on the hardware context. Those must
        # be released before dropping the handle's context reference.
        self.kernel_run = None
        if hasattr(self, "insts_bo"):
            del self.insts_bo
        if hasattr(self, "kernel"):
            del self.kernel
        if hasattr(self, "context"):
            del self.context
        if hasattr(self, "xclbin"):
            del self.xclbin
        if hasattr(self, "insts"):
            del self.insts


class CachedXRTRuntime(XRTHostRuntime):
    """A cached version of XRTHostRuntime that caches up to n contexts.

    The number of cached contexts depends on the type of NPU. It reuses
    contexts for the same xclbin (identified by device, inode and mtime).
    """

    # I got these values through experimentation on two machines
    # These values are primarily determined by the hardware/driver, and could change
    # in the future. But currently, if you exceed these sizes, you will fail to be
    # able to create a new context. At the driver level, the cached contexts are
    # a system-wide constrained resource, so caching on systems with many concurrent
    # processes trying to create contexts (as in parallel CI jobs) can be flaky.
    # TODO: use some sort of file system artifact or figure out how to query the driver
    # for the state of the cache, and how to make loading operations atomic between processes.
    # I got these values through experimentation on two machines
    # These values are primarily determined by the hardware/driver, and could change
    # in the future. But currently, if you exceed these sizes, you will fail to be
    # able to create a new context. At the driver level, the cached contexts are
    # a system-wide constrained resource, so caching on systems with many concurrent
    # processes trying to create contexts (as in parallel CI jobs) can be flaky.
    #
    # The values mirror the driver's per-device hwctx_limit (amdxdna npu*_regs.c):
    # npu1 (Phoenix) = 6, npu4/npu5/npu6 (Strix / Strix Halo / Krackan) = 16. They are
    # also reported live by the driver via QUERY_RESOURCE_INFO (npu_task_max).
    # Exceeding the pool makes CREATE_HWCTX fail with MGMT_ERT_NOAVAIL (0x2000003);
    # the retry-eviction loop cannot recover when other processes hold the remaining
    # slots, so on machines where several processes create contexts concurrently,
    # set XRT_CONTEXT_CACHE_SIZE to a per-process share (e.g. 2) instead.
    # TODO: use some sort of file system artifact or figure out how to query the driver
    # for the state of the cache, and how to make loading operations atomic between processes.
    NPU_CONTEXT_CACHE_SIZE = {
        "npu1": 6,
        "npu2": 16,
    }

    # Instruction streams are not a driver-limited resource, and one xclbin
    # can carry many (one per kernel or runtime sequence), so they are bounded
    # apart from the contexts: at the context limit, a dispatch loop over more
    # streams than that misses on every call.
    INSTS_CACHE_SIZE = 256

    def __init__(self):
        """Initialize the CachedXRTRuntime."""
        super().__init__()
        # We use OrderedDict so that we can use Fifo behavior for LRU eviction policies
        self._context_cache = OrderedDict()
        self._insts_cache = OrderedDict()
        # Memoised read_insts() output keyed by the file's _file_key.
        # Skips the file open+read on every load() call when content is
        # unchanged.  Cheap (insts files are ~hundreds of bytes); the win
        # shows up when NPU_CACHE_HOME lives on a slow/networked filesystem
        # — open+read can be ~290us there vs ~10us on local FS.
        self._insts_content_cache = OrderedDict()

        # Set default from dict if present
        cache_size: int | None = None
        if self.npu_str is not None and self.npu_str in self.NPU_CONTEXT_CACHE_SIZE:
            cache_size = self.NPU_CONTEXT_CACHE_SIZE[self.npu_str]

        # Environment variable always override default values
        # TODO: should probably emit warning if exceeds recorded max size.
        _env_cache_size = os.environ.get("XRT_CONTEXT_CACHE_SIZE")
        if _env_cache_size is not None:
            cache_size = int(_env_cache_size)

        # Error if no default and no env var
        if cache_size is None:
            raise HostRuntimeError(f"No known cache size for {self.npu_str}")
        self._cache_size: int = cache_size
        self._insts_cache_size: int = int(
            os.environ.get("XRT_INSTS_CACHE_SIZE", self.INSTS_CACHE_SIZE)
        )
        for name, size in (
            ("XRT_CONTEXT_CACHE_SIZE", self._cache_size),
            ("XRT_INSTS_CACHE_SIZE", self._insts_cache_size),
        ):
            if size < 1:
                raise HostRuntimeError(f"{name} must be at least 1, got {size}")

        atexit.register(self.cleanup)

    def cleanup(self):
        """Clean up cached XRT resources in dependency order."""
        while self._insts_cache:
            self._evict_insts()
        while self._context_cache:
            self._evict()
        self._insts_content_cache.clear()
        gc.collect()  # Make sure contexts are garbage collected.

    def evict_context(self, xclbin_path: Path) -> None:
        """Evict a stale cached hw_context (after an IOCTL EINVAL).

        Ensures the next load rebuilds a fresh one keyed by the same xclbin.
        """
        try:
            entry = self._context_cache.pop(self._file_key(xclbin_path, "xclbin"), None)
            if entry is not None:
                self._cleanup_entry(entry)
        except Exception:
            # Recovery path: must not raise, but log loudly -- silent failure
            # would keep recycling a broken _context_cache into every retry.
            logger.warning(
                "evict_context: failed to evict %s; retry may reuse a stale "
                "hardware context",
                xclbin_path,
                exc_info=True,
            )

    def _cleanup_entry_insts(self, entry):
        """Release instruction BOs owned by a cached context entry."""
        for insts_key in list(entry.get("insts_keys", ())):
            insts_entry = self._insts_cache.pop(insts_key, None)
            if insts_entry is not None:
                self._cleanup_insts_entry(insts_key, insts_entry)

    def _cleanup_entry(self, entry):
        handles = entry["handles"]

        # Invalidate all handles
        for ref in handles:
            handle = ref()
            if handle:
                handle.invalidate()

        self._cleanup_entry_insts(entry)

        # Clear kernel cache so pyxrt.kernel objects are released with the context
        entry["runs"].clear()
        entry["kernels"].clear()
        if self._dispatched_context is entry["context"]:
            self._dispatched_context = None

        # Release the hw_context by removing its strong reference from the entry dict.
        # Simply assigning a local `context = entry["context"]` and then `del context`
        # only removes the local name — entry["context"] would keep the object alive for
        # as long as any caller holds a reference to the entry dict (e.g. tests, or the
        # exception handler).  Deleting the key guarantees the refcount drops here.
        del entry["context"]

    def _evict(self):
        # Pop the oldest item
        key, entry = self._context_cache.popitem(last=False)
        self._cleanup_entry(entry)
        gc.collect()

    def _cleanup_insts_entry(self, insts_key, entry):
        owner_entry = entry.get("owner_entry")
        if owner_entry is not None:
            owner_entry.get("insts_keys", set()).discard(insts_key)
        # Delete the key (not a local copy) so the refcount drops here.
        del entry["insts_bo"]

    def _evict_insts(self):
        key, entry = self._insts_cache.popitem(last=False)
        self._cleanup_insts_entry(key, entry)

    @staticmethod
    def _file_key(path, what: str) -> tuple[int, int, int]:
        """Return the cache identity of `path`: device, inode and mtime.

        One stat and no path resolution, which walks every component of a
        path (about 40 us on NFS, per file per dispatch). Aliases of a file
        share its key, and a rebuild changes its mtime or inode.

        Args:
            path: The file.
            what: What the file is, for the error.

        Returns:
            `(st_dev, st_ino, st_mtime_ns)` of the file.

        Raises:
            HostRuntimeError: If `path` is not a regular file.
        """
        try:
            st = os.stat(path)
        except OSError:
            st = None
        if st is None or not stat.S_ISREG(st.st_mode):
            raise HostRuntimeError(f"{what} {path} does not exist or is not a file.")
        return (st.st_dev, st.st_ino, st.st_mtime_ns)

    def _read_insts_cached(self, insts_path, key):
        """``read_insts(insts_path)`` memoised by the file's `_file_key`.

        See ``__init__`` comment: shaves ~280us per load() when insts.bin
        lives on a networked filesystem (e.g. ``$HOME`` on NFS).  Stale
        files are detected via mtime so the cache is correct across rebuilds.
        """
        cached = self._insts_content_cache.get(key)
        if cached is not None:
            self._insts_content_cache.move_to_end(key)
            return cached
        # read_insts() picks the format by suffix, so decode the target, not an alias.
        insts = self.read_insts(insts_path.resolve())
        if len(self._insts_content_cache) >= self._insts_cache_size:
            self._insts_content_cache.popitem(last=False)
        self._insts_content_cache[key] = insts
        return insts

    def run(
        self,
        kernel_handle: KernelHandle,
        args,
        trace_config=None,
        fail_on_error: bool = True,
        only_if_loaded: bool = False,
        dispatch_insts=None,
        parameters: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> XRTKernelResult:
        """Run a loaded XRT kernel.

        Args:
            kernel_handle (XRTKernelHandle): The handle to the loaded kernel.
            args: Arguments to pass to the kernel.
            trace_config (optional): Configuration for tracing. Defaults to None.
            fail_on_error (bool, optional): Whether to raise an exception on kernel failure. Defaults to True.
            only_if_loaded (bool, optional): If True, only run if the kernel is currently loaded in the cache. Defaults to False.
            dispatch_insts (np.ndarray | None, optional): See
                ``XRTHostRuntime.run()``.
            parameters (Mapping[str, Any] | None, optional): See
                ``XRTHostRuntime.run()``.
            **kwargs: Additional arguments.

        Returns:
            XRTKernelResult: The result of the kernel execution.

        Raises:
            HostRuntimeError: If arguments are invalid, kernel execution fails, or kernel is not loaded (if only_if_loaded=True).
        """
        if only_if_loaded:
            if (
                isinstance(kernel_handle, CachedXRTKernelHandle)
                and not kernel_handle._is_valid
            ):
                raise HostRuntimeError("Kernel not loaded (evicted from cache)")

        return super().run(
            kernel_handle,
            args,
            trace_config,
            fail_on_error,
            dispatch_insts=dispatch_insts,
            parameters=parameters,
            **kwargs,
        )

    def load_and_run(
        self, npu_kernel, run_args, dispatch_scalars=None, parameters=None, **kwargs
    ):
        """Wrap the base implementation to paper over a Phoenix firmware-state quirk.

        A trace-on run leaves the amdxdna firmware in
        a state where the next submit on a *different* cached context fails
        with ``DRM_IOCTL_AMDXDNA_EXEC_CMD IOCTL failed (err=2): No such file
        or directory`` (EXEC_CMD ENOENT) — even if we evict the failing
        context and re-create a fresh hw_context for the same xclbin.  This
        is the same failure mode the partial-eviction workaround at the top
        of ``load()`` already drains the whole cache to dodge (see comment
        on ``NPU_CONTEXT_CACHE_SIZE``).

        Workaround: after a trace-on run completes (and the base
        ``load_and_run`` has already extracted the trace BOs via
        ``process_trace``), drain the entire context cache on Phoenix so
        the next call rebuilds from scratch.  Strix (npu2) handles
        single-entry eviction correctly and isn't affected, so we leave it
        alone.

        Same-kernel re-runs after their own trace work fine; the bug only
        bites when switching to a different kernel after a trace, but
        draining unconditionally on Phoenix is simpler than tracking which
        contexts are about to be touched next.
        """
        handle, ret = super().load_and_run(
            npu_kernel, run_args, dispatch_scalars, parameters, **kwargs
        )
        if (
            npu_kernel.trace_config is not None
            and getattr(self, "npu_str", None) == "npu1"
        ):
            self._drain_for_phoenix_trace_quirk()
        return handle, ret

    def _drain_for_phoenix_trace_quirk(self):
        """Drain the full context + insts caches.

        Mirrors the partial-eviction workaround at the top of ``load()`` — see
        comment on ``NPU_CONTEXT_CACHE_SIZE`` for why a single-entry evict isn't
        enough on Phoenix.
        """
        while self._context_cache:
            self._evict()
        while self._insts_cache:
            self._evict_insts()
        gc.collect()

    def _run_init_entry(self, context, init_name: str | None, kernel_name: str) -> None:
        """Evict the context if its init entry fails, as the array state is unknown."""
        try:
            super()._run_init_entry(context, init_name, kernel_name)
        except Exception:
            for key, entry in list(self._context_cache.items()):
                if entry["context"] is context:
                    del self._context_cache[key]
                    self._cleanup_entry(entry)
            raise

    def _load_full_elf_cached(
        self, npu_kernel, retry: bool = True
    ) -> CachedXRTKernelHandle:
        """Load a full ELF, reusing a cached hw_context keyed on the ELF's `_file_key`.

        Full ELFs carry their own PDIs + control code, so the context is built
        from ``pyxrt.elf`` (no xclbin, no instruction BO cache).  Context reuse
        follows the same LRU + Phoenix-drain policy as the xclbin path.  A new
        context first runs the kernel's init entry, if it has one.
        """
        elf_path = Path(npu_kernel.elf_path)
        kernel_name = npu_kernel.kernel_name
        context_key = self._file_key(elf_path, "full ELF")

        try:
            if context_key in self._context_cache:
                entry = self._context_cache[context_key]
                self._context_cache.move_to_end(context_key)
                context = entry["context"]
                entry["handles"] = [
                    ref for ref in entry["handles"] if ref() is not None
                ]
            else:
                if len(self._context_cache) >= self._cache_size:
                    if self.npu_str == "npu1":
                        while self._context_cache:
                            self._evict()
                    else:
                        self._evict()

                elf = pyxrt.elf(str(elf_path))

                context = None
                retries = 0
                max_retries = len(self._context_cache) if retry else 0
                while context is None:
                    try:
                        context = pyxrt.hw_context(self._device, elf)
                    except RuntimeError as e:
                        if self._context_cache and retries < max_retries:
                            self._evict()
                            retries += 1
                        else:
                            raise e

                entry = {
                    "context": context,
                    "xclbin": None,
                    "kernels": {},
                    "runs": {},
                    "handles": [],
                    "insts_keys": set(),
                    "uuid": None,
                }
                self._context_cache[context_key] = entry

            if kernel_name not in entry["kernels"]:
                kernel = pyxrt.ext.kernel(context, kernel_name)
                entry["kernels"][kernel_name] = kernel
                entry["runs"][kernel_name] = _KernelRun(
                    kernel,
                    npu_kernel.params_path,
                    kernel_name,
                    self._device,
                    empty=_full_elf_empty_run(context, kernel_name),
                )
            kernel = entry["kernels"][kernel_name]

            kernel_handle = CachedXRTKernelHandle(
                kernel,
                None,
                context,
                None,
                name=kernel_name,
                is_full_elf=True,
                init_name=getattr(npu_kernel, "init_kernel_name", None),
                kernel_run=entry["runs"][kernel_name],
            )
            entry["handles"].append(weakref.ref(kernel_handle))
            return kernel_handle

        except Exception:
            if context_key in self._context_cache:
                entry = self._context_cache[context_key]
                entry["handles"] = [
                    ref for ref in entry["handles"] if ref() is not None
                ]
                if not entry["handles"]:
                    del self._context_cache[context_key]
                    self._cleanup_entry(entry)
            raise

    def load(
        self,
        npu_kernel,
        retry: bool = True,
        **kwargs,
    ) -> XRTKernelHandle:
        """Load an NPU kernel into the cached XRT runtime.

        Args:
            npu_kernel: The NPU kernel to load.
            retry (bool, optional): Whether to retry loading if context creation fails due to resource limits. Defaults to True.
            **kwargs: Additional arguments for loading.

        Returns:
            XRTKernelHandle: A handle to the loaded kernel.

        Raises:
            HostRuntimeError: If xclbin or insts files do not exist, or if kernel is not found.
        """
        self.check_device_consistency()

        if getattr(npu_kernel, "elf_path", None) is not None:
            return self._load_full_elf_cached(npu_kernel, retry=retry)

        xclbin_path = Path(npu_kernel.xclbin_path)
        # insts_path is None for a DispatchTime[T] design: its stream is
        # synthesized per call, so there is nothing to cache. Skip the
        # insts.bin read and always load a plain (non-ext) kernel below.
        insts_path = Path(npu_kernel.insts_path) if npu_kernel.insts_path else None
        kernel_name = npu_kernel.kernel_name

        context_key = self._file_key(xclbin_path, "xclbin")
        insts_file = (
            self._file_key(insts_path, "insts") if insts_path is not None else None
        )

        try:
            if context_key in self._context_cache:
                entry = self._context_cache[context_key]
                self._context_cache.move_to_end(context_key)
                context = entry["context"]
                xclbin = entry["xclbin"]
                # Clean up dead handles
                entry["handles"] = [
                    ref for ref in entry["handles"] if ref() is not None
                ]
            else:
                xclbin = pyxrt.xclbin(str(xclbin_path))
                xclbin_uuid = xclbin.get_uuid()

                if len(self._context_cache) >= self._cache_size:
                    if self.npu_str == "npu1":
                        # Phoenix-only workaround: single-entry LRU eviction
                        # leaves the firmware in a state where the next submit
                        # on a freshly-created context fails with EXEC_CMD
                        # ENOENT. Even retaining one old entry reproduces it;
                        # only a full drain works. Strix (npu2) handles
                        # single-entry eviction correctly.
                        while self._context_cache:
                            self._evict()
                    else:
                        self._evict()

                self._device.register_xclbin(xclbin)

                # Try to create context, evicting if necessary
                context = None
                retries = 0
                max_retries = len(self._context_cache) if retry else 0
                while context is None:
                    try:
                        context = pyxrt.hw_context(self._device, xclbin_uuid)
                    except RuntimeError as e:
                        # Context-slot exhaustion is reported differently across XRT backends.
                        # Evict cached contexts and retry, but only while cached entries remain.
                        if self._context_cache and retries < max_retries:
                            self._evict()
                            retries += 1
                        else:
                            raise e

                entry = {
                    "context": context,
                    "xclbin": xclbin,
                    "kernels": {},  # kernel_name -> pyxrt.kernel (strong ref, tied to context)
                    "runs": {},
                    "kernel_args": {
                        k.get_name(): k.get_num_args() for k in xclbin.get_kernels()
                    },
                    "handles": [],
                    "insts_keys": set(),
                    "uuid": xclbin_uuid,
                }
                self._context_cache[context_key] = entry

            kernel_name = self._kernel_name(kernel_name, entry["kernel_args"])
            num_args = entry["kernel_args"][kernel_name]

            insts = (
                self._read_insts_cached(insts_path, insts_file)
                if insts_file is not None
                else None
            )
            insts_bo = None
            is_module = isinstance(insts, pyxrt.module)
            kernel_key = (kernel_name, insts_file) if is_module else kernel_name
            if kernel_key not in entry["kernels"]:
                kernel = (
                    pyxrt.ext.kernel(context, insts, kernel_name)
                    if is_module
                    else pyxrt.kernel(context, kernel_name)
                )
                entry["kernels"][kernel_key] = kernel
                entry["runs"][kernel_key] = _KernelRun(kernel)
            kernel = entry["kernels"][kernel_key]
            kernel_run = entry["runs"][kernel_key]
            if not is_module:
                if insts_file is None:
                    # DispatchTime[T] design: no static insts to cache --
                    # run() allocates a per-call BO from dispatch_insts.
                    kernel_handle = CachedXRTKernelHandle(
                        kernel,
                        xclbin,
                        context,
                        None,
                        None,
                        name=kernel_name,
                        num_args=num_args,
                        kernel_run=kernel_run,
                    )
                    entry["handles"].append(weakref.ref(kernel_handle))
                    return kernel_handle

                # Magic number for RyzenAI group id that will be fixed in the future. See same code at XRT:
                # https://github.com/Xilinx/XRT/blob/56222ed5cfd119dff0d5bd920735b87024e8c829/src/runtime_src/core/common/api/xrt_module.cpp#L1621
                group_id = kernel.group_id(1)
                insts_key = (*insts_file, group_id)

                if insts_key in self._insts_cache:
                    insts_entry = self._insts_cache[insts_key]
                    self._insts_cache.move_to_end(insts_key)
                    insts_bo = insts_entry["insts_bo"]
                else:
                    if len(self._insts_cache) >= self._insts_cache_size:
                        self._evict_insts()

                    insts_bo = self._tensor_class(
                        insts,
                        flags=pyxrt.bo.cacheable,
                        group_id=group_id,
                        xrt_device=self._device,
                    ).buffer_object()

                    insts_entry = {
                        "insts_bo": insts_bo,
                        "owner_entry": entry,
                    }
                    self._insts_cache[insts_key] = insts_entry
                    entry["insts_keys"].add(insts_key)

            kernel_handle = CachedXRTKernelHandle(
                kernel,
                xclbin,
                context,
                insts,
                insts_bo,
                name=kernel_name,
                num_args=num_args,
                kernel_run=kernel_run,
            )
            entry["handles"].append(weakref.ref(kernel_handle))

            return kernel_handle

        except Exception:
            if context_key in self._context_cache:
                entry = self._context_cache[context_key]
                # Clean up dead handles
                entry["handles"] = [
                    ref for ref in entry["handles"] if ref() is not None
                ]
                if not entry["handles"]:
                    del self._context_cache[context_key]
                    self._cleanup_entry(entry)
            raise
