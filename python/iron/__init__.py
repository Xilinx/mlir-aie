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
- :class:`CompilableDesign` / :func:`compileconfig` -- bundle a generator with compile-time config
- :class:`CallableDesign` / :func:`jit` -- JIT-compile and run on the NPU (Triton-style)
- :class:`Compile` / :class:`In` / :class:`Out` / :class:`InOut` -- type-annotation markers
- :func:`get_compile_arg` -- dynamic compile-time injection (advanced)
"""

from .buffer import Buffer
from .kernel import ExternalFunction, Kernel
from .program import Program
from .worker import Worker, WorkerRuntimeBarrier
from .runtime import Runtime
from .dataflow import ObjectFifo
from .dtype import str_to_dtype, dtype_to_str
from aie.utils.compile.jit import (
    CompilableDesign,
    compile_context,
    Compile,
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
    randint,
    rand,
    arange,
    zeros_like,
    set_tensor_class,
    get_current_device,
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
    "ObjectFifo",
    # Compile-time / JIT API
    "Compile",
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
    "randint",
    "rand",
    "arange",
    "zeros_like",
    "set_tensor_class",
    "get_current_device",
    # dtype helpers
    "str_to_dtype",
    "dtype_to_str",
    # Submodules
    "kernels",
    "algorithms",
]
