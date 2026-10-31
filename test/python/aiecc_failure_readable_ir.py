# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s
# REQUIRES: peano

# When aiecc rejects a design compiled into a work_dir, the design's IR is
# written beside it with each op's source as a comment, and the error says
# where.

import tempfile
from pathlib import Path

from aie.utils.compile.utils import compile_mlir_module

IR = """
module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2) loc("design.py":3:0)
    %t03 = aie.tile(0, 3) loc("design.py":4:0)
    aie.objectfifo @act_in(%t02, {%t03}, 2 : i32) : !aie.objectfifo<memref<16xi32>> loc(#loc2)
  } loc(#loc)
} loc(#loc)
#loc = loc("design.py":1:0)
#loc1 = loc("design.py":12:4)
#loc2 = loc("iron:act_in"(#loc1))
"""

with tempfile.TemporaryDirectory() as work:
    try:
        compile_mlir_module(
            IR,
            insts_path=f"{work}/insts.bin",
            xclbin_path=f"{work}/final.xclbin",
            work_dir=work,
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected aiecc to reject a fifo with no drainer")
    path = Path(work) / "aie.readable.mlir"
    assert message.endswith(f"each op noted with its source: {path}"), message
    lines = path.read_text().splitlines()
    fifo = next(line for line in lines if "aie.objectfifo @act_in" in line)
    assert fifo.endswith("  // act_in · design.py:12"), fifo
    assert not any("#loc" in line for line in lines), lines
print("PASS: readable IR written beside a failed compile")

# A readable dump that cannot be written still reports aiecc's error.
with tempfile.TemporaryDirectory() as work:
    (Path(work) / "aie.readable.mlir").mkdir()
    try:
        compile_mlir_module(
            IR,
            insts_path=f"{work}/insts.bin",
            xclbin_path=f"{work}/final.xclbin",
            work_dir=work,
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected aiecc to reject a fifo with no drainer")
    assert message.startswith("[aiecc] Compilation failed"), message
    assert "Could not write the design's readable IR: IsADirectoryError" in message
print("PASS: aiecc's error kept when the readable IR cannot be written")
