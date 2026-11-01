# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

# An error raised before an object is named names it after the variable its
# statement assigns it to, or else where it was declared.

import inspect

import numpy as np

from aie.dialects.aie import Port
from aie.iron import ObjectFifo

line_type = np.ndarray[(16,), np.dtype[np.int32]]


def repin_message(fifo):
    fifo.prod(port=Port.dma(0))
    try:
        fifo.prod(port=Port.dma(1))
    except ValueError as exc:
        return str(exc)
    raise AssertionError("expected re-pinning the producer to be rejected")


act_in = ObjectFifo(line_type)
message = repin_message(act_in)
assert message.startswith("Producer handle for act_in already pinned"), message
print("PASS: assigned fifo named after its variable")

line = inspect.currentframe().f_lineno + 1
message = repin_message(ObjectFifo(line_type))
expected = f"Producer handle for ObjectFifo at unnamed_object_errors.py:{line} "
assert message.startswith(expected), message
print("PASS: unassigned fifo named after its declaration")

explicit = ObjectFifo(line_type, name="by_hand")
message = repin_message(explicit)
assert message.startswith("Producer handle for by_hand already pinned"), message
assert "name='by_hand'" in str(explicit), str(explicit)
print("PASS: explicit name kept")
