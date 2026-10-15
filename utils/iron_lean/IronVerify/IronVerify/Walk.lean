-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Mathlib.Tactic

/-!
A `Walk` is a tensor access pattern: an element offset plus `(size, stride)` dims,
outermost first. Stream position `p` (row-major over the sizes) reads element
`offset + Σ i_d * stride_d`, as taplib's `TensorAccessPattern._walk` does.
Values are `Int` so exported MLIR arithmetic keeps its meaning; sizes are read
through `Int.toNat`.
-/

namespace IronVerify

structure Walk where
  offset : Int
  dims : List (Int × Int)
  deriving Repr, DecidableEq

def dimsLen : List (Int × Int) → Nat
  | [] => 1
  | (s, _) :: ds => s.toNat * dimsLen ds

def walkAt : List (Int × Int) → Nat → Int
  | [], _ => 0
  | (_, t) :: ds, p => ((p / dimsLen ds : Nat) : Int) * t + walkAt ds (p % dimsLen ds)

namespace Walk

def len (w : Walk) : Nat := dimsLen w.dims

def idx (w : Walk) (p : Nat) : Int := w.offset + walkAt w.dims p

def toList (w : Walk) : List Int := (List.range w.len).map w.idx

end Walk

end IronVerify
