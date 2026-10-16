-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Net

/-!
A kernel's contract as the exporter reads it off `KernelContract`: per argument its element type,
role and `TensorLayout`. `Layout.slotElem` rebuilds `_tile_layout`'s codec from the layout's
fields: buffer slot `q` holds logical element `slotElem q` (row-major over the tile).
-/

namespace IronVerify

/-- A 2-D operand layout: the logical `rows × cols` tile, whether it is transposed before
streaming (`axes = (1, 0)`), the `to_stream` walk, and whether that walk drains the buffer
(`inverse`) instead of filling it. -/
structure Layout where
  rows : Nat
  cols : Nat
  transposed : Bool
  stream : Option Walk
  inverse : Bool
  deriving Repr, DecidableEq

namespace Layout

variable (L : Layout)

/-- The logical flat index at position `x` of the (possibly transposed) streamed view. -/
def untranspose (x : Nat) : Nat :=
  if L.transposed then (x % L.rows) * L.cols + x / L.rows else x

/-- The logical flat index buffer slot `q` holds. -/
def slotElem (q : Nat) : Option Nat :=
  match L.stream with
  | none => some (L.untranspose q)
  | some w =>
    if L.inverse then (scatterPos w q).map L.untranspose
    else some (L.untranspose (w.idx q).toNat)

/-- The logical `(row, col)` buffer slot `q` holds. -/
def decode (q : Nat) : Option (Nat × Nat) :=
  (L.slotElem q).map fun e => (e / L.cols, e % L.cols)

end Layout

/-- One kernel argument: element type (`i8`, `i16`, `i32`, `bf16`, `f32`), role (`In`, `Out`,
`InOut`, `Param`) and layout (`none`: as stored). -/
structure KernelArg where
  dtype : String
  role : String
  layout : Option Layout
  deriving Repr, DecidableEq

/-- A kernel a worker runs, with its contract. `declared`: the factory attaches a
`KernelContract`. `saturation`: the mode the kernel narrows in when it sets one itself
(`none` = the core's current mode). `leavesSaturation`: the mode it leaves set on the core
(`none` = the core keeps its mode). Modes are `aie::saturation_mode` names. -/
structure KernelD where
  name : String
  args : List KernelArg
  accDtype : Option String
  reduction : Option Nat
  setup : Option String
  declared : Bool
  saturation : Option String
  leavesSaturation : Option String
  deriving Repr, DecidableEq

/-! ### Value domains

What a stored output element is: the output dtype, and the core saturation mode the kernel
narrows in. That mode is the kernel's own when its contract sets one; otherwise it is the
core's, which boots in `none` (measured: whole_array's i16 → i16 matmul wraps element-exact on
Strix Halo) and changes only through kernels that leave a mode set. -/

/-- An `aie::saturation_mode`. -/
inductive Narrowing where
  | wrap
  | saturate
  | symmetric
  deriving Repr, DecidableEq

def Narrowing.ofName? : String → Option Narrowing
  | "none" => some .wrap
  | "saturate" => some .saturate
  | "symmetric" => some .symmetric
  | _ => none

/-- What one call's stored output is. `wrap w`: the exact integer result read as a `w`-bit
two's-complement value (arithmetic in `ZMod (2 ^ w)`). `saturate lo hi`: clamped to
`[lo, hi]` after every call. `float`: rounded in a way the contract leaves abstract, so only
the order of accumulation is fixed. -/
inductive ValueDomain where
  | wrap (w : Nat)
  | saturate (lo hi : Int)
  | float (dtype : String)
  deriving Repr, DecidableEq

/-- Bit width of an integer dtype name, and whether it is signed. -/
def intWidth? : String → Option (Nat × Bool)
  | "i8" => some (8, true)
  | "i16" => some (16, true)
  | "i32" => some (32, true)
  | "i64" => some (64, true)
  | "ui8" => some (8, false)
  | "ui16" => some (16, false)
  | _ => none

def isFloatDtype (dtype : String) : Bool := dtype == "bf16" || dtype == "f32"

namespace ValueDomain

/-- The stored integer an exact result `v` becomes (floats keep `v`). -/
def read : ValueDomain → Int → Int
  | wrap w, v => Int.bmod v (2 ^ w)
  | saturate lo hi, v => max lo (min hi v)
  | float _, v => v

/-- The domain of a `w`-bit integer stored in mode `mode`. -/
def ofInt (w : Nat) (signed : Bool) : Narrowing → ValueDomain
  | .wrap => wrap w
  | .saturate =>
    if signed then saturate (-2 ^ (w - 1)) (2 ^ (w - 1) - 1) else saturate 0 (2 ^ w - 1)
  | .symmetric =>
    if signed then saturate (-(2 ^ (w - 1) - 1)) (2 ^ (w - 1) - 1) else saturate 0 (2 ^ w - 1)

end ValueDomain

namespace KernelD

variable (kd : KernelD)

/-- The argument the kernel writes (`Out` or `InOut`). -/
def output? : Option KernelArg := kd.args.find? fun a => a.role == "Out" || a.role == "InOut"

/-- The mode the kernel narrows in on a core whose other kernels are `onCore`, called in any
order: its own declared mode; else the boot mode (`wrap`) when every kernel on the core has a
contract and none leaves a mode set; else unknown. -/
def narrowingOn (onCore : List KernelD) : Option Narrowing :=
  match kd.saturation with
  | some m => Narrowing.ofName? m
  | none =>
    if kd.declared && onCore.all (fun k => k.declared && k.leavesSaturation.isNone) then
      some .wrap
    else none

/-- The value domain of the kernel's output on a core whose kernels are `onCore`. -/
def domainOn (onCore : List KernelD) : Option ValueDomain := do
  let o ← kd.output?
  match intWidth? o.dtype with
  | some (w, signed) => (kd.narrowingOn onCore).map (ValueDomain.ofInt w signed)
  | none => if isFloatDtype o.dtype then some (.float o.dtype) else none

/-- The wrap model needs the accumulator at least as wide as the stored output, so the
accumulator itself never wraps below the output width. -/
def accCovers : Bool :=
  match kd.output?, kd.accDtype with
  | some o, some acc =>
    match intWidth? o.dtype, intWidth? acc with
    | some (w, _), some (wa, _) => w ≤ wa
    | none, none => isFloatDtype o.dtype && isFloatDtype acc
    | _, _ => false
  | _, _ => false

end KernelD

end IronVerify
