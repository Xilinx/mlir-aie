# harness.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Shared pyxrt.runlist dispatch for the ctrlpkt reconfiguration device tests.

Not a test itself (excluded in lit.local.cfg). A folded full ELF from
``iron.Reconfiguration.compile()`` exposes one entrypoint per design (plus a
leading ``main:init`` for the ctrlpkt overlay). ``dispatch_runlist`` submits
them in one batched, ordered ``pyxrt.runlist``; ``read_i32`` reads a result
buffer back with an explicit device->host sync (raw-pyxrt dispatch bypasses the
iron runtime's device-dirty marking, so a lazy ``.to("cpu")`` would see stale
host bytes).
"""

import numpy as np
import pyxrt  # pyright: ignore[reportMissingImports]

import aie.iron as iron
from aie.utils.hostruntime.xrtruntime.device import acquire_device


def dispatch_runlist(elf, per_ep):
    """Dispatch every entrypoint of ``elf`` in one runlist.

    ``per_ep`` maps each name in ``elf.entrypoints`` to its ordered iron
    tensors (``"main:init"`` -> ``()``). A ctrlpkt fold (``elf.needs_ctrl_bo``)
    additionally gives every run one inert control-packet buffer; the runlist
    itself is method-blind.
    """
    dev = acquire_device()
    ctx = pyxrt.hw_context(dev, pyxrt.elf(str(elf.path)))
    dummy = (
        iron.zeros(1024, dtype=np.int32, device="npu") if elf.needs_ctrl_bo else None
    )

    def _bos(*tensors):
        bos = [t.buffer_object() for t in tensors]
        if elf.needs_ctrl_bo:
            bos.append(dummy.buffer_object())
        return bos

    runlist = pyxrt.runlist(ctx)
    keep = []  # keep kernels + runs alive until wait() returns
    for name in elf.entrypoints:
        kernel = pyxrt.ext.kernel(ctx, name)
        run = pyxrt.run(kernel)
        for i, bo in enumerate(_bos(*per_ep[name])):
            run.set_arg(i, bo)
        runlist.add(run)  # NOT run.start() -- UB for a run inside a runlist
        keep.append((kernel, run))
    runlist.execute()
    runlist.wait()
    del runlist, keep, ctx


def read_i32(tensor):
    """Read an int32 result buffer back with an explicit device->host sync."""
    bo = tensor.buffer_object()
    bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    return np.frombuffer(bo.map(), dtype=np.int32)
