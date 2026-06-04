# parameter_scratchpad.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.
"""Host-side runtime for writing named parameters to AIE cores via the
scratchpad mechanism.

Thin Python wrapper around the C++ ``test_utils::ParameterScratchpad``
class (exposed via pybind11).  The encoding logic (shift-2, delta) lives
entirely in C++; this layer only manages the ``pyxrt.bo`` handle for
``sync()``.

Usage::

    import pyxrt
    from aie.utils.parameter_scratchpad import ParameterScratchpad

    run = pyxrt.run(kernel)
    params = ParameterScratchpad(run, "params.txt")
    params.write("seq_len", 42)
    params.sync()
    run.start()
"""

import struct
from pathlib import Path

import pyxrt

from aie._mlir_libs._parameter_scratchpad import (
    ParameterScratchpad as _ParameterScratchpadImpl,
)


def _to_bytes(value) -> bytes:
    """Convert any scalar to its little-endian in-memory bytes."""
    if isinstance(value, int):
        return struct.pack("<I", value & 0xFFFFFFFF)
    if isinstance(value, float):
        return struct.pack("<f", value)
    if hasattr(value, "tobytes"):
        return value.tobytes()
    raise TypeError(f"unsupported parameter type: {type(value)}")


class ParameterScratchpad:
    """Write named runtime parameters to the NPU scratchpad buffer."""

    def __init__(self, run, params_path: str | Path):
        self._bo = run.get_ctrl_scratchpad_bo()
        self._mv = self._bo.map()
        self._impl = _ParameterScratchpadImpl(self._mv, str(params_path))

    def write(self, name: str, value) -> None:
        """Write a parameter value to the scratchpad.

        Args:
            name: The parameter name (must match a name in the params file).
            value: A scalar value — ``int``, ``float``, or any type with
                   a ``tobytes()`` method (``np.float32``, ``bfloat16``, etc.).
        """
        self._impl.write_bytes(name, _to_bytes(value))

    def sync(self) -> None:
        """Sync the scratchpad buffer to device."""
        self._bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def read(self, name: str) -> int:
        """Read back a parameter's current decoded value (for debugging)."""
        return self._impl.read(name)
