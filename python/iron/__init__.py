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
from .cascade import CascadeFifo  # T2.1: cascade-stream first-class primitive
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


class SparseFifo:
    """T2.5 reservation slot -- on-the-fly N:M sparsity decompression on S2MM.

    Producer accepts compressed weights; consumer receives dense data
    transparently. Closes G-T5.1-005; makes T6.4-C compressed-weight LSTM
    idiomatic.

    Raises :class:`NotImplementedError` until T2.5 lands.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "SparseFifo: T2.5 not yet landed (T1.2 reservation slot)"
        )


class MemtileAggregator:
    """T2.6 reservation slot -- memtile-mediated fan-in helper.

    Encapsulates 4-into-1 memtile fan-in via S2MM ch 0..3, memtile-side
    layout reorganization (5D address generation), and fan-out via memtile
    MM2S. Promotes Phase 1's hand-rolled T5.3-memtile retrofit topology.

    Raises :class:`NotImplementedError` until T2.6 lands.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MemtileAggregator: T2.6 not yet landed (T1.2 reservation slot)"
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
]
