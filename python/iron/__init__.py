# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""IRON: High-level Python API for programming AMD Ryzen™ AI NPUs.

Provides the primary abstractions for describing NPU designs:

- [`Buffer`][iron.Buffer] — named memory region shared between Workers and the Runtime
- [`ObjectFifo`][iron.ObjectFifo] — synchronized dataflow channel between program components
- [`CascadeFlow`][iron.CascadeFlow] — directed cascade stream connection between two Workers
- [`Worker`][iron.Worker] — a task running on an AIE compute core
- [`Runtime`][iron.Runtime] — host-side orchestration of data movement and worker execution
- [`Program`][iron.Program] — top-level container that compiles a design to MLIR
- [`Kernel`][iron.Kernel] / [`ExternalFunction`][iron.ExternalFunction] — pre-compiled or C++ kernel functions
- [`WorkerRuntimeBarrier`][iron.WorkerRuntimeBarrier] — synchronization primitive between workers and runtime
- Tensor utilities ([`arange`][iron.arange], [`zeros`][iron.zeros], [`ones`][iron.ones], etc.) for NPU-accessible buffers
- dtype helpers ([`str_to_dtype`][iron.str_to_dtype], [`dtype_to_str`][iron.dtype_to_str])
- [`CompilableDesign`][iron.CompilableDesign] / [`compileconfig`][iron.compileconfig] — bundle a generator with compile-time config
- [`CallableDesign`][iron.CallableDesign] / [`jit`][iron.jit] — JIT-compile and run on the NPU (Triton-style)
- [`CompileTime`][iron.CompileTime] / [`In`][iron.In] / [`Out`][iron.Out] / [`InOut`][iron.InOut] — type-annotation markers
- [`get_compile_arg`][iron.get_compile_arg] — dynamic compile-time injection (advanced)

!!! note "Implicit MLIR context"
    IRON designs decorated with `@iron.jit` execute inside an implicit
    MLIR context (thread-local `Location` / `InsertionPoint`). See the
    [Implicit MLIR Context](../../programming_guide/implicit_mlir_context.md)
    guide for the mental model and how to read "no active location" errors.

!!! note "Automatic shape collapse"
    `Kernel.__call__` / `ExternalFunction.__call__` silently insert a
    `memref.collapse_shape` when an N-D contiguous memref arg is fed
    into a 1-D kernel signature with the same element count and dtype
    (typical case: IRON L1 ObjectFifo elements are 2-D, `kernels.X`
    helpers declare flat 1-D args). Real shape / dtype mismatches still
    surface at MLIR verification time.
"""

from .buffer import Buffer
from .kernel import ExternalFunction, Kernel
from .lock import Lock
from .scratchpad_parameter import ScratchpadParameter
from .program import Program
from .worker import Worker, WorkerRuntimeBarrier
from .runtime import Runtime, TaskGroup, RuntimeData, Task
from .dataflow import (
    Acquire,
    Bd,
    CascadeFlow,
    DmaChannel,
    Flow,
    ObjectFifo,
    PacketDest,
    PacketFlow,
    PadDims,
    Release,
    StreamDims,
    TileDma,
)
from .dtype import str_to_dtype, dtype_to_str
from aie.utils.compile.jit import (
    CompilableDesign,
    compile_context,
    CompileTime,
    In,
    InOut,
    Out,
    compileconfig,
    get_compile_arg,
)
from aie.utils.jit import jit
from aie.utils.callabledesign import CallableDesign
from . import kernels
from . import algorithms
from aie.utils import (
    tensor,
    ones,
    zeros,
    full,
    randint,
    rand,
    arange,
    zeros_like,
    ceildiv,
    set_tensor_class,
    set_current_device,
    get_current_device,
    ensure_current_device,
)

__all__ = [
    # Core design abstractions
    "Buffer",
    "ExternalFunction",
    "Kernel",
    "Program",
    "Worker",
    "WorkerRuntimeBarrier",
    "Runtime",
    "TaskGroup",
    "RuntimeData",
    "Task",
    "ObjectFifo",
    "PadDims",
    "StreamDims",
    # Lower-level explicit-routing primitives
    "Acquire",
    "Bd",
    "CascadeFlow",
    "DmaChannel",
    "Flow",
    "Lock",
    "ScratchpadParameter",
    "PacketDest",
    "PacketFlow",
    "Release",
    "TileDma",
    # Compile-time / JIT API
    "CompileTime",
    "In",
    "Out",
    "InOut",
    "CompilableDesign",
    "CallableDesign",
    "compileconfig",
    "jit",
    "compile_context",
    "get_compile_arg",
    # Tensor factories
    "tensor",
    "ones",
    "zeros",
    "full",
    "randint",
    "rand",
    "arange",
    "zeros_like",
    "ceildiv",
    "set_tensor_class",
    "set_current_device",
    "get_current_device",
    "ensure_current_device",
    # dtype helpers
    "str_to_dtype",
    "dtype_to_str",
    # Submodules
    "kernels",
    "algorithms",
]
