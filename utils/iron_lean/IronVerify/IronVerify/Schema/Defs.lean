-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Schema.Spec
import IronVerify.Schema.Overflow

/-!
Definitions the schema theorems and their proofs share: the shape `dims p Mt Kt Nt`
(`M = rows·m·Mt`, `K = k·Kt`, `N = n·cols·Nt`) and the values cores read and drains write.
-/

namespace IronVerify.Schema.WholeArray

open IronVerify

namespace Params

variable (p : Params)

def Mn (Mt : Nat) : Nat := p.rows * p.m * Mt
def Kn (Kt : Nat) : Nat := p.k * Kt
def Nn (Nt : Nat) : Nat := p.n * p.cols * Nt

/-- The dispatch values of a shape given in tiles. -/
def dims (Mt Kt Nt : Nat) : Int × Int × Int := (p.Mn Mt, p.Kn Kt, p.Nn Nt)

end Params

variable (p : Params)

/-! ### C = A @ B

The core operands as values: a delivered slot holds the host value of its provenance. The
kernel reads only its tiles, so slots outside them read `0`. -/

section Values

variable {R : Type*} [Zero R] (hA hB : Nat → R) (Mt Kt Nt : Nat)

def hostVal : Option Src → R
  | some (.host 0 idx) => hA idx.toNat
  | some (.host 1 idx) => hB idx.toNat
  | _ => 0

/-- Worker `w`'s `j`-th A and B objects, restricted to the tile. -/
def opA (w j : Nat) : Nat → R := fun s =>
  if s < p.m * p.k then
    hostVal hA hB ((schema p).deliver (p.dims Mt Kt Nt) (aL2Fifo p (w / p.cols)) w j s)
  else 0

def opB (w j : Nat) : Nat → R := fun s =>
  if s < p.k * p.n then
    hostVal hA hB ((schema p).deliver (p.dims Mt Kt Nt) (bL2Fifo p (w % p.cols)) w j s)
  else 0

/-- The value a drain writes at stream position `q`, given what one kernel call computes. -/
def drainVal (call : KernelCall R) (zero : Nat → Nat → R) (c q : Nat) : R :=
  match (schema p).stream (p.dims Mt Kt Nt) (cL3Fifo p c) q with
  | some (.core w _ τ s) =>
    coreTile call zero (opA p hA hB Mt Kt Nt w) (opB p hA hB Mt Kt Nt w) Kt τ
      (p.unlayoutC s).1 (p.unlayoutC s).2
  | _ => 0

/-- The operands the spec says the `kk`-th call on tile `(i0, j0)` must see. -/
def specA (i0 kk : Nat) : Nat → R := fun s =>
  if s < p.m * p.k then
    hA (p.aIdx (p.Kn Kt) (i0 + (p.unlayoutA s).1) (kk * p.k + (p.unlayoutA s).2))
  else 0

def specB (j0 kk : Nat) : Nat → R := fun s =>
  if s < p.k * p.n then
    hB (p.bIdx (p.Kn Kt) (p.Nn Nt) (kk * p.k + (p.unlayoutB s).1) (j0 + (p.unlayoutB s).2))
  else 0

end Values

end IronVerify.Schema.WholeArray
