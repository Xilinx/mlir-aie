# parameter_scratchpad.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Host-side runtime for writing named parameters to AIE cores.

Writes named parameters to AIE cores via the scratchpad mechanism.
Thin Python wrapper around the C++ ``test_utils::ParameterScratchpad``
class (exposed via pybind11).

Usage:

```python
import pyxrt
from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import ParameterScratchpad

# ... get kernel from ELF, etc., then:
run = pyxrt.run(kernel)
params = ParameterScratchpad(run, "params.txt")
params.write("seq_len", 42)
params.sync()
run.start()
```
"""

import ctypes
import struct
from pathlib import Path

import numpy as np
import pyxrt  # pyright: ignore[reportMissingImports]
from aie._mlir_libs._parameter_scratchpad import (  # pyright: ignore[reportMissingImports]
    ParameterScratchpad as _ParameterScratchpadImpl,
)

# PyCapsule_New(pointer, name, destructor): pyxrt.ext.bo takes a host pointer
# only wrapped in a capsule.
_pointer_capsule = ctypes.PYFUNCTYPE(
    ctypes.py_object, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p
)(("PyCapsule_New", ctypes.pythonapi))


def _to_bytes(value) -> bytes:
    """Convert any scalar to its little-endian in-memory bytes.

    This helper is required to support native Python `int` and convert them to the 4-byte little-endian format assumed on the cores.
    """
    if isinstance(value, int):
        assert -0x80000000 <= value <= 0xFFFFFFFF
        return value.to_bytes(4, "little", signed=value < 0)
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
        self._alias = None

    def write(self, name: str, value) -> None:
        """Write a parameter value to the scratchpad.

        Args:
            name: The parameter name (must match a name in the params file).
            value: A scalar value — ``int``, or any type with a ``tobytes()``
                   method (``np.int32``, ``bfloat16``, etc.).
        """
        self._impl.write_bytes(name, _to_bytes(value))

    def sync(self) -> None:
        """Sync the scratchpad buffer to device."""
        self._bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def sync_from_device(self) -> None:
        """Sync the scratchpad buffer from the device.

        What a device transfer into ``alias()`` wrote is then seen by ``read()``.
        """
        self._bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    def read(self, name: str) -> int:
        """Read back a parameter's current decoded value (for debugging)."""
        return self._impl.read(name)

    def alias(self, device) -> "pyxrt.bo":
        """Return a buffer object over this scratchpad's host mapping.

        It is on ``device``, and a kernel argument can be bound to it.
        The scratchpad's own buffer object is on the device heap, at an
        address the shim DMA does not reach: a transfer into it silently goes
        nowhere. This user-pointer buffer is reached like any host buffer, so
        another run can drain into it and set this run's parameters with no
        host step between. The alias points into this scratchpad's mapping,
        which this object keeps alive; keep it for as long as the alias.
        """
        if self._alias is None:
            pointer = np.frombuffer(self._mv, dtype=np.uint8).ctypes.data
            self._alias = pyxrt.ext.bo(
                device, _pointer_capsule(pointer, None, None), self._bo.size()
            )
        return self._alias
