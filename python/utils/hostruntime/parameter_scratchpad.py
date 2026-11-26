# parameter_scratchpad.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""A command's parameter scratchpad as host words, for any host runtime.

Encodes named runtime parameters into the state table words firmware reads,
with the checks ``params.txt`` asks for. Thin Python wrapper around the C++
``test_utils::ParameterScratchpad`` class (exposed via nanobind); it needs no
device. XRT's scratchpad
(``aie.utils.hostruntime.xrtruntime.parameter_scratchpad``) writes the same
words into a run's ctrl scratchpad; a runtime that stages the table itself
dispatches ``table``, and fills a slot it sources elsewhere (a buffer's
device address, say) as ``slot`` describes.

Usage:

```python
from aie.utils.hostruntime.parameter_scratchpad import ParameterScratchpad

params = ParameterScratchpad("params.txt", "main:sequence")
params.write("seq_len", 42)
params.validate()
words = params.table  # numpy uint32, one word per slot
```
"""

import struct
from pathlib import Path

import numpy as np
from aie._mlir_libs._parameter_scratchpad import (  # pyright: ignore[reportMissingImports]
    ParameterScratchpad as _ParameterScratchpadImpl,
)
from aie._mlir_libs._parameter_scratchpad import (  # pyright: ignore[reportMissingImports]
    Slot as Slot,
)


def _to_bytes(value) -> bytes:
    """Convert any scalar to its little-endian in-memory bytes.

    This helper is required to support native Python `int` and convert them to the 4-byte little-endian format assumed on the cores.
    """
    if isinstance(value, int):
        assert -0x80000000 <= value <= 0xFFFFFFFF
        return value.to_bytes(4, "little", signed=value < 0)
    if isinstance(value, float):
        return struct.pack("<f", value)
    if isinstance(value, (np.generic, np.ndarray)):
        return value.tobytes()
    raise TypeError(f"unsupported parameter type: {type(value)}")


class ParameterScratchpad:
    """Encode named runtime parameters into a command's scratchpad words."""

    def __init__(self, params_path: str | Path, command: str = "", buffer=None):
        """Hold ``command``'s parameters of ``params_path``, cleared.

        Args:
            params_path: The lowering's ``params.txt``.
            command: The ``device:sequence`` whose parameters this holds;
                empty for the module's one command.
            buffer: The writable host memory to encode into, of at least
                ``slot_count`` words; a new table when None.
        """
        if buffer is None:
            buffer = np.zeros(slot_count(params_path), dtype=np.uint32)
        self._table = np.frombuffer(buffer, dtype=np.uint32)
        self._impl = _ParameterScratchpadImpl(self._table, str(params_path), command)

    @property
    def table(self) -> np.ndarray:
        """The scratchpad's words as firmware reads them, a view of the buffer."""
        return self._table

    @property
    def names(self) -> frozenset[str]:
        """The parameters this scratchpad holds: its command's."""
        return frozenset(self._impl.names())

    def slot(self, name: str) -> Slot:
        """Return where firmware reads ``name``.

        Its ``index`` in the table, the left ``shift`` ``write`` applies to
        its value's bits for its ``kind`` (2 for ``core``, 16 for
        ``repeat``, 0 for ``addr``), and its scalar ``type``.

        Raises:
            RuntimeError: No command reads ``name``, or there is none.
        """
        return self._impl.slot(name)

    def write(self, name: str, value) -> None:
        """Write a parameter value to the scratchpad.

        Args:
            name: The parameter name (must match a name in the params file).
            value: A scalar value — ``int``, or any type with a ``tobytes()``
                   method (``np.int32``, ``bfloat16``, etc.).

        Raises:
            ValueError: `name` is a DMA offset or length parameter and `value`
                is outside the range that keeps its transfers within their
                buffers, or an offset into elements narrower than 4 bytes
                that is not a multiple of the elements in a word.
            RuntimeError: `name` is not one of ``names``.
        """
        self._impl.write_bytes(name, _to_bytes(value))

    def validate(self) -> None:
        """Check the offset and length parameters each DMA transfer uses together.

        Raises:
            ValueError: An offset and a length parameter that one DMA transfer
                uses together take it past the end of its buffer.
        """
        self._impl.validate()

    def read(self, name: str) -> int:
        """Read back a parameter's current encoded word (for debugging)."""
        return self._impl.read(name)


def slot_count(params_path: str | Path) -> int:
    """Return the number of 32-bit words a scratchpad of ``params_path`` holds."""
    return _ParameterScratchpadImpl.slot_count(str(params_path))
