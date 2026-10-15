-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/-!
The hand-written specification whole_array is checked against: which tile of A, B and C each core
works on, and the operand layouts its matmul kernel expects (m = 64, k = 64, n = 32 with
r = s = 4, t = 8 blocking, i16). Nothing here is derived from the design.
-/

namespace IronVerify.Spec.WholeArray

/-- Slot of row `a`, column `b` of the 64 × 64 A tile: (4 × 4) blocks, row-major. -/
def layoutA (a b : Nat) : Nat := (a / 4) * 256 + (b / 4) * 16 + (a % 4) * 4 + b % 4

/-- Slot of row `a`, column `b` of the 64 × 32 B tile: (4 × 8) blocks, row-major. -/
def layoutB (a b : Nat) : Nat := (a / 4) * 128 + (b / 8) * 32 + (a % 4) * 8 + b % 8

/-- Slot of row `a`, column `b` of the 64 × 32 C tile: (4 × 8) blocks, row-major. -/
def layoutC (a b : Nat) : Nat := (a / 4) * 128 + (b / 8) * 32 + (a % 4) * 8 + b % 8

/-- `cols` columns of 4 cores, N = 32·cols·Nc: at its `τ`-th output tile, the core in row `row`
and column `col` computes the 64 × 32 tile of C at rows `i0 Nc row τ ..` and columns
`j0 cols Nc col τ ..`. Its tiles go row block by row block (256 rows), and within a block one
column group (32·cols columns) at a time. -/
def i0 (Nc row τ : Nat) : Nat := (τ / Nc) * 256 + row * 64

def j0 (cols Nc col τ : Nat) : Nat := (τ % Nc) * (32 * cols) + col * 32

end IronVerify.Spec.WholeArray
