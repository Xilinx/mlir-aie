#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""One hardware context that dispatches the decode's full ELF."""

import time

import pyxrt  # pyright: ignore[reportMissingImports]
from aie.utils.hostruntime.xrtruntime.device import acquire_device
from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
    ParameterScratchpad,
)


class DispatchSession:
    """A design compiled to a full ELF, dispatched with per-call ScratchpadParameters.

    One hardware context and one run serve every dispatch: persistent workers
    keep their state from one dispatch to the next, and the device is not
    reconfigured per dispatch, which a fresh run would do.
    """

    def __init__(self, design, full_elf_path: str | None = None, **compile_kwargs):
        """Compile `design` at `compile_kwargs`, or reuse its build.

        Args:
            `design`: The `@iron.jit` design.
            `full_elf_path`: Where the full ELF goes, or where an up-to-date one
                built at the same `compile_kwargs` already is. `None` keeps it
                in the JIT cache.
            `**compile_kwargs`: The design's CompileTime values.
        """
        specialized = design.specialize(full_elf=True, **compile_kwargs)
        specialized.compile(full_elf_path=full_elf_path)
        entry = specialized.compilable.get_cache_entry()
        assert entry is not None and entry.elf is not None
        self._context = pyxrt.hw_context(acquire_device(), pyxrt.elf(str(entry.elf)))
        self._run = pyxrt.run(pyxrt.ext.kernel(self._context, entry.full_elf_kernel))
        self._scratchpad = (
            ParameterScratchpad(self._run, entry.params)
            if entry.params is not None
            else None
        )
        self._bound = []

    def __call__(self, tensors, **params) -> float:
        """Run one dispatch and wait for it, without a timeout.

        Args:
            `tensors`: The sequence's host tensors, in argument order. They stay
                bound, and alive, until a later call replaces them.
            `**params`: ScratchpadParameter values by name.

        Returns:
            The dispatch's wall time, start to completion, in ms.

        Raises:
            RuntimeError: The dispatch did not complete.
        """
        bos = []
        for i, tensor in enumerate(tensors):
            tensor.to("npu")
            bo = tensor.buffer_object()
            if i >= len(self._bound) or self._bound[i] is not bo:
                self._run.set_arg(i, bo)
            bos.append(bo)
        self._bound = bos
        if params:
            assert self._scratchpad is not None
            for name, value in params.items():
                self._scratchpad.write(name, value)
            self._scratchpad.sync()
        start = time.perf_counter_ns()
        self._run.start()
        state = self._run.wait()
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"dispatch ended in {state}")
        return elapsed_ms
