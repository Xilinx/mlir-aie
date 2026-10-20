#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""How the example's tests get the NPU kernel for the design they check."""

import argparse
import sys
import time

import pyxrt  # pyright: ignore[reportMissingImports]
from aie.utils import NPUKernel
from aie.utils.hostruntime.xrtruntime.device import acquire_device
from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
    ParameterScratchpad,
)


def add_artifact_args(parser: argparse.ArgumentParser) -> None:
    """Add `--xclbin`/`--instr`, a prebuilt design to run, and `-v`."""
    parser.add_argument(
        "--xclbin",
        help="run this prebuilt xclbin instead of compiling the design",
    )
    parser.add_argument("--instr", help="the prebuilt xclbin's instructions")
    parser.add_argument("-v", "--verbosity", type=int, default=0)


def design_kernel(design, opts: argparse.Namespace, **compile_kwargs) -> NPUKernel:
    """The NPU kernel for `design` at `compile_kwargs`.

    Args:
        design: The `@iron.jit` design the test checks.
        opts: Parsed flags from `add_artifact_args`.
        **compile_kwargs: The design's CompileTime values, which must match the
            prebuilt artifacts when `opts.xclbin` names them.

    Returns:
        A kernel over `opts.xclbin`/`opts.instr` if given, else over the design
        compiled (or found in the JIT cache) at `compile_kwargs`.
    """
    if opts.xclbin is None:
        xclbin, instr = design.specialize(**compile_kwargs).compile()
    elif opts.instr is None:
        sys.exit("--xclbin needs --instr")
    else:
        xclbin, instr = opts.xclbin, opts.instr
    return NPUKernel(
        xclbin_path=str(xclbin), insts_path=str(instr), kernel_name="MLIR_AIE"
    )


class DispatchSession:
    """A design compiled to a full ELF, dispatched with per-call ScratchpadParameters.

    One hardware context serves every dispatch, so persistent workers keep
    their state from one dispatch to the next.
    """

    def __init__(self, design, full_elf_path: str | None = None, **compile_kwargs):
        """Compile `design` at `compile_kwargs`, or reuse its build.

        Args:
            design: The `@iron.jit` design.
            full_elf_path: Where the full ELF goes, or where an up-to-date one
                built at the same `compile_kwargs` already is. `None` keeps it
                in the JIT cache.
            **compile_kwargs: The design's CompileTime values.
        """
        specialized = design.specialize(full_elf=True, **compile_kwargs)
        specialized.compile(full_elf_path=full_elf_path)
        entry = specialized.compilable.get_cache_entry()
        assert entry is not None and entry.elf is not None
        self._params = entry.params
        self._context = pyxrt.hw_context(acquire_device(), pyxrt.elf(str(entry.elf)))
        self._kernel = pyxrt.ext.kernel(self._context, entry.full_elf_kernel)

    def __call__(self, tensors, **params) -> float:
        """Run one dispatch and wait for it, without a timeout.

        Args:
            tensors: The sequence's host tensors, in argument order.
            **params: ScratchpadParameter values by name.

        Returns:
            The dispatch's wall time, start to completion, in ms.

        Raises:
            RuntimeError: The dispatch did not complete.
        """
        run = pyxrt.run(self._kernel)
        for i, tensor in enumerate(tensors):
            tensor.to("npu")
            run.set_arg(i, tensor.buffer_object())
        if params:
            assert self._params is not None
            scratchpad = ParameterScratchpad(run, self._params)
            for name, value in params.items():
                scratchpad.write(name, value)
            scratchpad.sync()
        start = time.perf_counter_ns()
        run.start()
        state = run.wait()
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"dispatch ended in {state}")
        return elapsed_ms
