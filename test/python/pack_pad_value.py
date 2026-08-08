# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s

# Unit test for pack_pad_value: a per-element pad value is packed into the raw
# 32-bit CONSTANT_PAD_VALUE stream word by replicating it across the word for
# sub-32-bit element types, passed through for 32-bit, and rejected beyond that.

from aie.helpers.util import pack_pad_value

# 1-byte elements: replicated 4x across the word.
assert pack_pad_value(0, 1) == 0
assert pack_pad_value(7, 1) == 0x07070707
assert pack_pad_value(0xAB, 1) == 0xABABABAB

# 2-byte elements: replicated 2x; value masked to the element width.
assert pack_pad_value(7, 2) == 0x00070007
assert pack_pad_value(0x1234, 2) == 0x12341234
assert pack_pad_value(0x1FFFF, 2) == 0xFFFFFFFF  # masked to 16 bits, then packed

# 4-byte elements: passed through (masked to 32 bits).
assert pack_pad_value(1000, 4) == 1000
assert pack_pad_value(-1, 4) == 0xFFFFFFFF

# Wider than the 32-bit register: rejected.
for elem_bytes in (8, 16):
    try:
        pack_pad_value(1, elem_bytes)
        raise AssertionError(f"expected ValueError for {elem_bytes}-byte elements")
    except ValueError:
        pass

print("PASS")
