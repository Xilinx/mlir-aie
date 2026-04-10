# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""High-level host-runtime layer: CallableDesign and jit decorator."""

from .callabledesign import CallableDesign
from .jit import jit

__all__ = [
    "CallableDesign",
    "jit",
]
