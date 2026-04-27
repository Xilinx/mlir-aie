# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""IRON: High-level Python API for programming AMD Ryzen AI NPUs.

Provides the primary abstractions for describing NPU designs:

- :class:`Buffer` -- named memory region shared between Workers and the Runtime
- :class:`ObjectFifo` -- synchronized dataflow channel between program components
- :class:`Worker` -- a task running on an AIE compute core
- :class:`Runtime` -- host-side orchestration of data movement and worker execution
- :class:`Program` -- top-level container that compiles a design to MLIR
- :class:`Kernel` / :class:`ExternalFunction` -- pre-compiled or C++ kernel functions
- :class:`WorkerRuntimeBarrier` -- synchronization primitive between workers and runtime
- Tensor utilities (:func:`arange`, :func:`zeros`, :func:`ones`, etc.) for NPU-accessible buffers

Specialized FIFO subclasses (composable over :class:`ObjectFifo`):

- :class:`CascadeFifo` -- first-class cascade-stream ObjectFifo subclass.
- :class:`PacketFifo` -- variable-rate ObjectFifo + pktMerge / TLAST / OoO BD.
- :class:`AccumFifo` -- FP32 accumulator inter-tile state passing.
- :class:`SparseFifo` -- on-the-fly N:M sparsity decompression on S2MM.
- :class:`MemtileAggregator` -- memtile-mediated fan-in helper.

These are reservation stubs raising :class:`NotImplementedError` until the
matching primitive lands; subsequent commits in this PR replace each stub
with the real class.
"""

from .buffer import Buffer
from .kernel import ExternalFunction, Kernel
from .program import Program
from .worker import Worker, WorkerRuntimeBarrier
from .runtime import Runtime
from .dataflow import ObjectFifo
from .cascade import CascadeFifo
from .memtile import MemtileAggregator
from .sparse import SparseFifo
from .dtype import str_to_dtype, dtype_to_str
from aie.utils.jit import jit
from aie.utils import (
    tensor,
    ones,
    zeros,
    randint,
    rand,
    arange,
    zeros_like,
    set_tensor_class,
    get_current_device,
)


# ---------------------------------------------------------------------------
# T1.2: pre-staged Wave 2 primitive export stubs.
#
# Each stub raises NotImplementedError until its owning Wave 2 task replaces
# it with a real implementation. The reservation slots exist so that Wave 2
# tasks (T2.1 .. T2.6) executing in parallel never collide on this file:
# each task swaps in ONE class definition + one `from .module import Class`
# line at its reserved slot, never editing another task's slot.
#
# Per the Phase 2 plan's "Wave 2 fork-task serialization rule": adding the
# real class is one substitution; adding the import alongside it is the
# other. Conflicts on this file are mechanical (different lines).
# ---------------------------------------------------------------------------


# T2.1: CascadeFifo reservation slot replaced by real implementation
# (`from .cascade import CascadeFifo` above). The class lives in
# `python/iron/cascade.py`. See its module docstring for the AM020
# Ch. 4 p. 67 cascade-stream architectural reference.


# T2.2: PacketFifo reservation slot replaced by real implementation.
# The class lives in `python/iron/packet.py` (sibling to `accum.py` /
# `dataflow/objectfifo.py`). Variable-rate packet-switched stream
# primitive exposing pktMerge N:1 (AM020 Ch. 2 Figure 17),
# finish-on-TLAST (Ch. 2 p. 27), and out-of-order BD processing
# (Ch. 5 p. 74). Closes G-T6.2-001 + G-T6.4-101 + G-T7.4-200.
from .packet import PacketFifo, PacketFifoHandle  # noqa: E402  (reserved slot)


# T2.3: AccumFifo reservation slot replaced by real implementation.
# The class lives in `python/iron/accum.py` (sibling to `dataflow/objectfifo.py`).
# Persists 512-bit BM (accumulator) state across timesteps within a tile
# (BM-to-BM register move; AM020 Ch. 4 p. 67) AND across tiles
# (cascade-stream BM transfer). Closes G-T6.4-100.
from .accum import AccumFifo, AccumFifoHandle  # noqa: E402  (reserved slot)




# G-T3.2-007: VariableRateFifo — producer-side conditional-forward
# FIFO. Closes the single-producer / conditional-forward half of
# G-T6.2-001 + G-T7.4-200 (the N:1 multi-producer fan-in half is
# closed by PacketFifo above). The class lives in
# ``python/iron/variable_rate.py`` (sibling to ``packet.py`` /
# ``sparse.py``); uses the same discardable-attr-on-ObjectFifo
# pattern SparseFifo uses, plus a corresponding lowering-pass change
# in ``AIEObjectFifoStatefulTransform.cpp`` to skip variable-rate
# fifos from LCM-based loop unrolling.
from .variable_rate import (  # noqa: E402
    VariableRateFifo,
    VariableRateFifoHandle,
)


__all__ = [
    # Existing IRON primitives.
    "Buffer",
    "ExternalFunction",
    "Kernel",
    "Program",
    "Worker",
    "WorkerRuntimeBarrier",
    "Runtime",
    "ObjectFifo",
    "str_to_dtype",
    "dtype_to_str",
    "jit",
    "tensor",
    "ones",
    "zeros",
    "randint",
    "rand",
    "arange",
    "zeros_like",
    "set_tensor_class",
    "get_current_device",
    # Wave 2 reservation slots (T1.2 stubs, replaced by T2.1..T2.6).
    "CascadeFifo",
    "PacketFifo",
    "PacketFifoHandle",  # T2.2
    "AccumFifo",
    "AccumFifoHandle",  # T2.3
    "SparseFifo",
    "MemtileAggregator",
    # G-T3.2-007: VariableRateFifo (producer-side conditional forward;
    # sibling to PacketFifo for the single-producer half of
    # G-T6.2-001 + G-T7.4-200).
    "VariableRateFifo",
    "VariableRateFifoHandle",
]
