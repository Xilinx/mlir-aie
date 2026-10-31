# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s
# UNSUPPORTED: peano

# Without Peano a kernel's frame reads as 0 bytes, and the host says so once.

import warnings

import numpy as np
from aie.iron.kernel import ExternalFunction

kernels = [
    ExternalFunction(
        name,
        source_string=f'extern "C" void {name}(int *a) {{}}',
        arg_types=[np.ndarray[(64,), np.dtype[np.int32]]],
    )
    for name in ("a", "b")
]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("default")
    frames = [k.frame_bytes("aie2p") for k in kernels]
assert frames == [0, 0], frames
messages = [str(w.message) for w in caught if w.category is RuntimeWarning]
assert len(messages) == 1 and "Peano is not installed" in messages[0], messages
print("PASS: one warning for two kernels read without Peano")
