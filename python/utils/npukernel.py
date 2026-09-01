# npukernel.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
from pathlib import Path

from .trace import TraceConfig


class NPUKernel:
    """Represents a compiled NPU kernel."""

    def __init__(
        self,
        xclbin_path=None,
        insts_path=None,
        device_index=0,
        kernel_name="MLIR_AIE",
        trace_config: TraceConfig | None = None,
        num_host_bos: int | None = None,
        elf_path=None,
        dispatch_params: list[str] | None = None,
        dispatch_lib_path=None,
    ):
        """Initialize the NPUKernel.

        Args:
            xclbin_path (str | Path | None): Path to the xclbin file. ``None``
                on the full-ELF path (see ``elf_path``).
            insts_path (str | Path | None): Path to the instructions file.
                ``None`` on the full-ELF path, and on a DispatchTime[T]
                design (see ``dispatch_params``) -- both synthesize their
                instruction stream some other way instead of reading a
                static ``insts.bin``.
            device_index (int, optional): Device index. Defaults to 0.
            kernel_name (str, optional): Name of the kernel. Defaults to
                "MLIR_AIE". On the full-ELF path this is the
                ``"<device>:<sequence>"`` XRT kernel name.
            trace_config (TraceConfig | None, optional): Trace configuration. Defaults to None.
            elf_path (str | Path | None, optional): Path to a single
                self-contained full ELF (PDIs + TXN control code).  When set,
                the kernel is loaded standalone via
                ``pyxrt.hw_context(dev, pyxrt.elf(path))`` and ``xclbin_path`` /
                ``insts_path`` are unused.  Defaults to None.
            num_host_bos (int | None, optional): The compiled design's true
                host-buffer count -- the number of ``aie.runtime_sequence``
                operands, including any trace/ctrl-packet buffer the lowering
                appended. This is floor-independent (unlike the kernels.json
                ``boN`` slot count, which aiecc floors to the firmware
                command-chain minimum), so it is the correct value to validate
                host buffer counts against. ``None`` when it could not be
                determined (validation is then skipped).
            dispatch_params (list[str] | None, optional): Declared
                ``DispatchTime[T]`` parameter names for this design, in
                declaration order. ``None``/empty for a design with no
                DispatchTime[T] parameters (the ordinary static/cached path).
            dispatch_lib_path (str | Path | None, optional): Path to the
                compiled dispatch bridge (``dispatch.so``) for a
                DispatchTime[T] design. Required (non-None) whenever
                ``dispatch_params`` is non-empty.
        """
        self._xclbin_path = xclbin_path
        self._insts_path = insts_path
        self._elf_path = elf_path
        self._kernel_name = kernel_name
        self._trace_config = trace_config
        self._device_index = device_index
        self._num_host_bos = num_host_bos
        self._dispatch_params = list(dispatch_params) if dispatch_params else []
        self._dispatch_lib_path = dispatch_lib_path
        self._dispatch_bridge = None

    @property
    def trace_config(self) -> TraceConfig | None:
        """Get the trace configuration.

        Returns:
            TraceConfig | None: The trace configuration.
        """
        return self._trace_config

    @property
    def xclbin_path(self):
        """Get the path to the xclbin file.

        Returns:
            str | Path: The xclbin path.
        """
        return self._xclbin_path

    @property
    def insts_path(self):
        """Get the path to the instructions file.

        Returns:
            str | Path: The instructions path.
        """
        return self._insts_path

    @property
    def elf_path(self):
        """Get the path to the full ELF file, or ``None`` on the xclbin path.

        Returns:
            str | Path | None: The full-ELF path.
        """
        return self._elf_path

    @property
    def kernel_name(self):
        """Get the kernel name.

        Returns:
            str: The kernel name.
        """
        return self._kernel_name

    @property
    def num_host_bos(self) -> int | None:
        """Get the compiled design's true host-buffer count.

        Returns:
            int | None: The number of ``aie.runtime_sequence`` operands the
            design was compiled with (including any appended trace buffer), or
            ``None`` if it could not be determined.
        """
        return self._num_host_bos

    @property
    def dispatch_params(self) -> list[str]:
        """Get the declared ``DispatchTime[T]`` parameter names, in order.

        Returns:
            list[str]: Empty for a design with no DispatchTime[T] parameters.
        """
        return self._dispatch_params

    def _get_dispatch_bridge(self):
        """Return this kernel's ``DispatchBridge``, constructing it once.

        Deferred (function-local) imports: ``compile.jit`` is not imported at
        ``aie.utils`` load time, and ``_dispatch_bridge`` itself imports back
        into ``aie.utils.hostruntime`` (for ``HostRuntimeError``), which
        imports ``NPUKernel`` from this module -- a module-level import here
        would be circular.
        """
        if self._dispatch_bridge is None:
            from .compile.jit._dispatch_bridge import DispatchBridge
            from .compile.jit._dispatch_compile import _parse_signature

            # Non-None whenever dispatch_params is non-empty (see __init__).
            assert self._dispatch_lib_path is not None
            lib_path = Path(self._dispatch_lib_path)
            header_path = lib_path.parent / "dispatch_gen.h"
            header_text = header_path.read_text()
            _func_name, params = _parse_signature(header_text, self._dispatch_params)
            param_ctypes = [ctype for ctype, _name in params]
            self._dispatch_bridge = DispatchBridge(
                lib_path, self._dispatch_params, param_ctypes
            )
        return self._dispatch_bridge

    # Blocking call.
    def __call__(self, *args, **kwargs):
        """Run the kernel with the given arguments.

        This is a blocking call.

        Args:
            *args: Arguments passed to the kernel.
            **kwargs: Additional arguments passed to the runtime load_and_run method.

        Returns:
            The result returned by the runtime ``load_and_run`` call.
        """
        from . import DefaultNPURuntime

        if DefaultNPURuntime is None:
            raise Exception("Cannot run kernel; DefaultNPURuntime not set.")

        dispatch_names = set(self._dispatch_params)
        dispatch_scalars = {k: v for k, v in kwargs.items() if k in dispatch_names}
        other_kwargs = {k: v for k, v in kwargs.items() if k not in dispatch_names}
        return DefaultNPURuntime.load_and_run(
            self,
            list(args),
            dispatch_scalars=dispatch_scalars or None,
            **other_kwargs,
        )
