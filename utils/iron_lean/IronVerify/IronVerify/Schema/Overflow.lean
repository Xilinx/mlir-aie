-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Seq

/-!
Fixed-width vocabulary. Exported terms are exact `Int`s (the exporter drops casts); these say
when that is the machine's answer. `fitsS w x`: `x` is an `iw` value. `Seq.Every`: a property of
every transfer a sequence issues (taken branches, every iteration). `Seq.HostFits cap` is the
compiler's runtime host-buffer guard (`guardWithinHostBuffer`): a transfer's offset is
non-negative and the furthest element it touches is below its buffer's element count.
-/

namespace IronVerify

/-- `x` is representable as a signed `w`-bit integer. `index` is 64 bits. -/
def fitsS (w : Nat) (x : Int) : Bool := decide (-(2 ^ (w - 1) : Int) ≤ x ∧ x < 2 ^ (w - 1))

/-- The furthest element a walk with non-negative strides touches. -/
def Walk.maxIdx (w : Walk) : Int :=
  w.offset + (w.dims.map fun (s, t) => (s - 1) * t).sum

namespace Seq

/-- Every transfer the sequence issues satisfies `P chan buf walk`. -/
def Every (P : String → Nat → Walk → Prop) : Seq → Prop
  | xfer c b w => P c b w
  | skip => True
  | seq a b => Every P a ∧ Every P b
  | ite c t e => if c then Every P t else Every P e
  | loop lo hi body => ∀ i < (hi - lo).toNat, Every P (body (lo + i))
  | check _ => True

/-- The compiler's host-buffer guard passes for every transfer: buffer `b` holds `cap b`
elements, every stride and offset is non-negative and the walk stays inside it. -/
def HostFits (cap : Nat → Nat) (s : Seq) : Prop :=
  s.Every fun _ b w => 0 ≤ w.offset ∧ (∀ d ∈ w.dims, 0 ≤ d.1 ∧ 0 ≤ d.2) ∧ w.maxIdx < cap b

end Seq

theorem bmod_of_fitsS {w : Nat} {x : Int} (hw : 0 < w) (h : fitsS w x = true) :
    x.bmod (2 ^ w) = x := by
  simp only [fitsS, decide_eq_true_eq] at h
  have h2 : (2 : Int) ^ w = 2 * 2 ^ (w - 1) := by
    rw [← pow_succ', Nat.sub_add_cancel hw]
  apply Int.bmod_eq_of_le_mul_two <;> push_cast <;> rw [h2] <;> omega

/-! ### Fixed-width `arith`

The exporter writes each tap and bound as an exact `Int` term (casts dropped). `IExpr` is that
term with its widths: `exact` is what the exporter writes, `machine` what the core computes
(two's complement at every result's width, as `BitVec` does, see `IExpr.bv_toInt`), and
`Fits` is what `export.py --wraps` checks: every intermediate fits its width. Then the two
agree (`IExpr.machine_eq_exact`). -/

inductive BinOp where
  | add | sub | mul
  /-- `arith.divsi` -/
  | tdiv
  /-- `arith.floordivsi` -/
  | fdiv
  /-- `arith.remsi` -/
  | tmod
  deriving DecidableEq, Repr

def BinOp.int : BinOp → Int → Int → Int
  | .add, a, b => a + b
  | .sub, a, b => a - b
  | .mul, a, b => a * b
  | .tdiv, a, b => a.tdiv b
  | .fdiv, a, b => a.fdiv b
  | .tmod, a, b => a.tmod b

/-- An integer `arith` expression; every node carries its result width (`index` is 64). -/
inductive IExpr where
  | lit (w : Nat) (v : Int)
  | var (w : Nat) (i : Nat)
  | bin (op : BinOp) (w : Nat) (a b : IExpr)
  /-- `extsi`, `trunci`, `index_cast` to width `w`. -/
  | cast (w : Nat) (a : IExpr)

namespace IExpr

variable (env : Nat → Int)

def width : IExpr → Nat
  | lit w _ | var w _ | bin _ w _ _ | cast w _ => w

/-- The exporter's term: casts dropped, exact integers. -/
def exact : IExpr → Int
  | lit _ v => v
  | var _ i => env i
  | bin op _ a b => op.int (exact a) (exact b)
  | cast _ a => exact a

/-- What the machine computes: every result wrapped to its width. -/
def machine : IExpr → Int
  | lit w v => v.bmod (2 ^ w)
  | var w i => (env i).bmod (2 ^ w)
  | bin op w a b => (op.int (machine a) (machine b)).bmod (2 ^ w)
  | cast w a => (machine a).bmod (2 ^ w)

/-- The `--wraps` obligations: every node's exact value fits its (positive) width. -/
def Fits : IExpr → Prop
  | e@(lit w _) | e@(var w _) => 0 < w ∧ fitsS w (exact env e) = true
  | e@(bin _ w a b) => Fits a ∧ Fits b ∧ 0 < w ∧ fitsS w (exact env e) = true
  | e@(cast w a) => Fits a ∧ 0 < w ∧ fitsS w (exact env e) = true

/-- With every intermediate in range, the machine's answer is the exported exact term. -/
theorem machine_eq_exact : ∀ e : IExpr, e.Fits env → e.machine env = e.exact env
  | lit _ _, ⟨hw, h⟩ => bmod_of_fitsS hw h
  | var _ _, ⟨hw, h⟩ => bmod_of_fitsS hw h
  | bin op _ a b, ⟨ha, hb, hw, h⟩ => by
    simp only [machine, machine_eq_exact a ha, machine_eq_exact b hb]
    exact bmod_of_fitsS hw h
  | cast _ a, ⟨ha, hw, h⟩ => by
    simp only [machine, machine_eq_exact a ha]
    exact bmod_of_fitsS hw h

/-- Operands of a binary op have the op's width, as `arith` requires. -/
def WellTyped : IExpr → Prop
  | lit _ _ | var _ _ => True
  | bin _ w a b => a.WellTyped ∧ b.WellTyped ∧ a.width = w ∧ b.width = w
  | cast _ a => a.WellTyped

/-- The `BitVec` evaluation, with its width: `add`/`sub`/`mul` wrap, `divsi` is `sdiv`, `remsi` is
`srem`, casts sign-extend or truncate. `floordivsi` has no `BitVec` primitive; it is the floor of
the operands' quotient at width `w`, as `arith` defines it. -/
def bv : IExpr → (w : Nat) × BitVec w
  | lit w v => ⟨w, BitVec.ofInt w v⟩
  | var w i => ⟨w, BitVec.ofInt w (env i)⟩
  | bin op w a b =>
    let x := (bv a).2.signExtend w
    let y := (bv b).2.signExtend w
    ⟨w, match op with
      | .add => x + y
      | .sub => x - y
      | .mul => x * y
      | .tdiv => x.sdiv y
      | .fdiv => BitVec.ofInt w (x.toInt.fdiv y.toInt)
      | .tmod => x.srem y⟩
  | cast w a => ⟨w, (bv a).2.signExtend w⟩

theorem bv_fst : ∀ e : IExpr, (e.bv env).1 = e.width
  | lit _ _ | var _ _ | bin _ _ _ _ | cast _ _ => rfl

theorem bmod_toInt {w : Nat} (x : BitVec w) : x.toInt.bmod (2 ^ w) = x.toInt := by
  rw [← BitVec.toInt_ofInt, BitVec.ofInt_toInt]

theorem signExtend_toInt {v : Nat} (x : BitVec v) (w : Nat) :
    (x.signExtend w).toInt = x.toInt.bmod (2 ^ w) := by
  rcases le_total w v with h | h
  · rw [BitVec.toInt_signExtend, Nat.min_eq_left h]
  · rw [BitVec.toInt_signExtend_of_le h]
    rcases Nat.eq_zero_or_pos v with hv | hv
    · subst hv
      rw [BitVec.eq_nil x]
      simp
    · have hle := BitVec.le_toInt x
      have hlt := BitVec.toInt_lt (x := x)
      have hvw : (2 : Int) ^ v ≤ 2 ^ w := pow_le_pow_right₀ (by norm_num) h
      have e : (2 : Int) ^ v = 2 * 2 ^ (v - 1) := by rw [← pow_succ', Nat.sub_add_cancel hv]
      symm
      apply Int.bmod_eq_of_le_mul_two <;> push_cast <;> omega

/-- On well-typed expressions the machine semantics is `BitVec` arithmetic. -/
theorem bv_toInt : ∀ e : IExpr, e.WellTyped → (e.bv env).2.toInt = e.machine env
  | lit _ _, _ => BitVec.toInt_ofInt _
  | var _ _, _ => BitVec.toInt_ofInt _
  | bin op w a b, ⟨ha, hb, haw, hbw⟩ => by
    have ea : ((a.bv env).2.signExtend w).toInt = a.machine env := by
      rw [BitVec.toInt_signExtend_of_le (by rw [bv_fst, haw]), bv_toInt a ha]
    have eb : ((b.bv env).2.signExtend w).toInt = b.machine env := by
      rw [BitVec.toInt_signExtend_of_le (by rw [bv_fst, hbw]), bv_toInt b hb]
    cases op
    case tmod =>
      simp only [bv, machine, BinOp.int]
      rw [← ea, ← eb, ← BitVec.toInt_srem, bmod_toInt]
    all_goals
      simp only [bv, machine, BinOp.int, BitVec.toInt_add, BitVec.toInt_sub, BitVec.toInt_mul,
        BitVec.toInt_sdiv, BitVec.toInt_ofInt, ea, eb]
  | cast w a, ha => by
    simp only [bv, machine, signExtend_toInt, bv_toInt a ha]

end IExpr

end IronVerify
