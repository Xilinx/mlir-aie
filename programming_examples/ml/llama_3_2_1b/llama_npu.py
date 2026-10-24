#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""One hardware context that dispatches a design's full ELF, its commands
queued back to back."""

import time

import pyxrt  # pyright: ignore[reportMissingImports]
from aie.utils.hostruntime.xrtruntime.device import acquire_device
from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
    ParameterScratchpad,
)

# Commands a hardware context holds started; a further start blocks until
# the oldest completes (amdxdna, measured on Strix).
QUEUE_DEPTH = 4


class _PooledRun:
    """A command's run, its scratchpad and the buffers bound to it."""

    def __init__(self, kernel, params_path, command):
        self.run = pyxrt.run(kernel)
        self.scratchpad = (
            ParameterScratchpad(self.run, params_path, command)
            if params_path is not None
            else None
        )
        self.bound = []


class Dispatch:
    """A started command. Wait for it before its tensors change or go away.

    Args:
        `session`: The DispatchSession it runs in.
        `command`: Its ``device:sequence``.
        `pooled`: The run it started.
    """

    def __init__(self, session, command, pooled):
        self._session = session
        self._command = command
        self._pooled = pooled
        self._start = time.perf_counter_ns()

    def wait(self) -> float:
        """Wait for the command, without a timeout, and free its run.

        Returns:
            Its wall time, start to completion, in ms.

        Raises:
            RuntimeError: The command did not complete.
        """
        state = self._pooled.run.wait()
        elapsed_ms = (time.perf_counter_ns() - self._start) / 1e6
        if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"{self._command} ended in {state}")
        self._session._idle[self._command].append(self._pooled)
        return elapsed_ms


class DispatchSession:
    """A design compiled to a full ELF, its commands dispatched with
    per-call ScratchpadParameters.

    One hardware context serves every command, and each command reuses its
    runs: persistent workers keep their state from one command to the next,
    and the device is not reconfigured, which a fresh run of the ELF's entry
    may do. Commands start in the order they are submitted and run in that
    order; a command may be submitted while earlier ones run, each queued
    command taking a run of its own. Past QUEUE_DEPTH started commands,
    submit blocks until the oldest completes.

    Args:
        `design`: The `@iron.jit` design.
        `full_elf_path`: Where the full ELF goes, or where an up-to-date one
            built at the same `compile_kwargs` already is. `None` keeps it
            in the JIT cache.
        `**compile_kwargs`: The design's CompileTime values.
    """

    def __init__(self, design, full_elf_path: str | None = None, **compile_kwargs):
        specialized = design.specialize(full_elf=True, **compile_kwargs)
        specialized.compile(full_elf_path=full_elf_path)
        entry = specialized.compilable.get_cache_entry()
        assert entry is not None and entry.elf is not None
        self.entry = entry.full_elf_kernel
        self._context = pyxrt.hw_context(acquire_device(), pyxrt.elf(str(entry.elf)))
        self._params_path = entry.params
        self._kernels = {}
        self._idle = {}

    def submit(self, tensors, command: str | None = None, **params) -> Dispatch:
        """Start one command behind those already started.

        Args:
            `tensors`: The command's host tensors, in argument order. They stay
                bound, and must stay alive and unchanged, until the command's
                wait.
            `command`: Its ``device:sequence``; None for the ELF's entry.
            `**params`: Its ScratchpadParameter values by name.

        Returns:
            The started Dispatch.
        """
        command = self.entry if command is None else command
        if command not in self._kernels:
            self._kernels[command] = pyxrt.ext.kernel(self._context, command)
            self._idle[command] = []
        idle = self._idle[command]
        pooled = (
            idle.pop()
            if idle
            else _PooledRun(self._kernels[command], self._params_path, command)
        )
        bos = []
        for i, tensor in enumerate(tensors):
            tensor.to("npu")
            bo = tensor.buffer_object()
            if i >= len(pooled.bound) or pooled.bound[i] is not bo:
                pooled.run.set_arg(i, bo)
            bos.append(bo)
        pooled.bound = bos
        if params:
            assert pooled.scratchpad is not None
            for name, value in params.items():
                pooled.scratchpad.write(name, value)
            pooled.scratchpad.sync()
        pooled.run.start()
        return Dispatch(self, command, pooled)

    def __call__(self, tensors, command: str | None = None, **params) -> float:
        """Run one command and wait for it, without a timeout.

        Args:
            `tensors`, `command`, `**params`: As submit's.

        Returns:
            The command's wall time, start to completion, in ms.

        Raises:
            RuntimeError: The command did not complete.
        """
        return self.submit(tensors, command, **params).wait()
