-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Mathlib
import IronVerify.Schema.WholeArray

/-!
The hand-written specification the parametric theorems check `schema p` against, and the
kernel/core contracts, all as functions of the parameters. Nothing here is derived from the
design: the tile each core owns, the kernel's operand layouts (`mm_stream_dims` blocking), the
host index of each logical element, and what one kernel call computes.
-/

namespace IronVerify.Schema.WholeArray

open IronVerify

namespace Params

variable (p : Params)

/-- Host index of `A[i, x]` (row-major `M × K`). -/
def aIdx (_p : Params) (K i x : Nat) : Nat := i * K + x

/-- Core slot of A tile element `(a, b)` (`a < m`, `b < k`): `(r × s)` blocks, row-major. -/
def layoutA (a b : Nat) : Nat :=
  (a / p.r) * (p.k / p.s) * (p.r * p.s) + (b / p.s) * (p.r * p.s) + (a % p.r) * p.s + b % p.s

/-- Core slot of B tile element `(a, b)` (`a < k`, `b < n`): `(s × t)` blocks of B, or
`(t × s)` blocks of Bᵀ with `bColMaj`. -/
def layoutB (a b : Nat) : Nat :=
  if p.bColMaj then
    (b / p.t) * (p.k / p.s) * (p.t * p.s) + (a / p.s) * (p.t * p.s) + (b % p.t) * p.s + a % p.s
  else
    (a / p.s) * (p.n / p.t) * (p.s * p.t) + (b / p.t) * (p.s * p.t) + (a % p.s) * p.t + b % p.t

/-- Core slot of C tile element `(a, b)` (`a < m`, `b < n`): `(r × t)` blocks of C, or
`(t × r)` blocks of Cᵀ with `cColMaj`. -/
def layoutC (a b : Nat) : Nat :=
  if p.cColMaj then
    (b / p.t) * (p.m / p.r) * (p.t * p.r) + (a / p.r) * (p.t * p.r) + (b % p.t) * p.r + a % p.r
  else
    (a / p.r) * (p.n / p.t) * (p.r * p.t) + (b / p.t) * (p.r * p.t) + (a % p.r) * p.t + b % p.t


/-- Host index of `B[x, j]`: row-major `K × N`, or `Bᵀ` (`N × K`) with `bColMaj`. -/
def bIdx (K N x j : Nat) : Nat := if p.bColMaj then j * K + x else x * N + j

/-- Host index of `C[i, j]`: row-major `M × N`, or `Cᵀ` (`N × M`) with `cColMaj`. -/
def cIdx (M N i j : Nat) : Nat := if p.cColMaj then j * M + i else i * N + j

/-- At its `τ`-th output tile, core `(row, col)` computes the `m × n` tile of C at rows
`i0 Nt row τ ..` and columns `j0 Nt col τ ..` (`Nt = N / (n · cols)` column tiles per core
column): row block by row block (`rows · m` rows), one column group (`n · cols`) at a time. -/
def i0 (Nt row τ : Nat) : Nat := (τ / Nt) * (p.rows * p.m) + row * p.m

def j0 (Nt col τ : Nat) : Nat := (τ % Nt) * (p.n * p.cols) + col * p.n

/-- The core row, core column and output tile that own `C[i, j]`. -/
def rowC (i : Nat) : Nat := (i % (p.rows * p.m)) / p.m
def colC (j : Nat) : Nat := (j % (p.n * p.cols)) / p.n
def tauC (Nt i j : Nat) : Nat := (i / (p.rows * p.m)) * Nt + j / (p.n * p.cols)

/-- Inverse of `layoutA`/`layoutB`/`layoutC` on a tile: the logical `(row, col)` of a slot. -/
def unlayoutA (q : Nat) : Nat × Nat :=
  let blk := q / (p.r * p.s)
  ((blk / (p.k / p.s)) * p.r + (q % (p.r * p.s)) / p.s,
   (blk % (p.k / p.s)) * p.s + q % p.s)

def unlayoutB (q : Nat) : Nat × Nat :=
  let blk := q / (p.s * p.t)
  if p.bColMaj then
    ((blk % (p.k / p.s)) * p.s + q % p.s, (blk / (p.k / p.s)) * p.t + (q % (p.s * p.t)) / p.s)
  else
    ((blk / (p.n / p.t)) * p.s + (q % (p.s * p.t)) / p.t, (blk % (p.n / p.t)) * p.t + q % p.t)

def unlayoutC (q : Nat) : Nat × Nat :=
  let blk := q / (p.r * p.t)
  if p.cColMaj then
    ((blk % (p.m / p.r)) * p.r + q % p.r, (blk / (p.m / p.r)) * p.t + (q % (p.r * p.t)) / p.r)
  else
    ((blk / (p.n / p.t)) * p.r + (q % (p.r * p.t)) / p.t, (blk % (p.n / p.t)) * p.t + q % p.t)

end Params

/-! ### Kernel and core contracts

One kernel call maps the core's A and B buffers (slot → value) and its C tile (logical
`(row, col)` → value) to the new C tile. `KernelCall` leaves that arbitrary: whatever the
call computes, the data movement fixes *which* operands each call sees and in *which* order
(`orderedAcc`). The exact, wrapping and saturating integer kernels and the floating-point ones
are instances. -/

abbrev KernelCall (R : Type*) := (Nat → R) → (Nat → R) → (Nat → Nat → R) → (Nat → Nat → R)

/-- The core contract (whole_array's `core_fn`): output tile `τ` starts from `zero` and runs
the kernel on the core's A and B objects `τ · Kt + kk`, for `kk = 0 … Kt - 1`, in order. -/
def coreTile {R : Type*} (call : KernelCall R) (zero : Nat → Nat → R)
    (opA opB : Nat → Nat → R) (Kt τ : Nat) : Nat → Nat → R :=
  (List.range Kt).foldl (fun C kk => call (opA (τ * Kt + kk)) (opB (τ * Kt + kk)) C) zero

namespace Params

variable (p : Params)

/-- The exact kernel over a commutative semiring:
`C'[a, b] = C[a, b] + Σ_{c < k} A[a, c] · B[c, b]`, reading A and B at their layouts. -/
def matmul {R : Type*} [CommSemiring R] : KernelCall R := fun A B C a b =>
  C a b + ∑ c ∈ Finset.range p.k, A (p.layoutA a c) * B (p.layoutB c b)

/-- The integer kernel when the core stores C with saturation on (`to_vector<T_out>` clamps
to `[lo, hi]` after every call). -/
def matmulSat (lo hi : ℤ) : KernelCall ℤ := fun A B C a b =>
  max lo (min hi (p.matmul A B C a b))

end Params

/-- Integer element widths: `C` stored as a `w`-bit integer. With saturation off at kernel entry
(`mm.cc` never sets the mode; whole_array never changes it), `to_vector<T_out>` truncates, and
the accumulator (acc32 / acc64) is at least `w` bits wide, so every call is `matmul` in
`ZMod (2 ^ w)`: the wrapping instance. -/
abbrev Wrap (w : Nat) := ZMod (2 ^ w)

end IronVerify.Schema.WholeArray
