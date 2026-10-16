-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Lemmas
import IronVerify.Schema.Spec

/-!
Index arithmetic over symbolic radices for the parametric schema: the kernel's blocked operand
layouts (`layoutA/B/C`, `unlayoutA/B/C`), the `mm_stream_dims` walks that produce and consume
them (`walkA/B/C`), tile ownership (`i0`, `j0`, `rowC`, `colC`, `tauC`), host indices and the
k-block reindexing of a dot product. Every lemma holds for any valid `p`.

All three layouts are one blocked layout `blk`, with the operands swapped for the column-major
variants; the generic lemmas are proved once for `blk` and transferred.
-/

namespace IronVerify.Schema.WholeArray

open IronVerify

namespace Arith

theorem mix_lt {x y X Y : Nat} (hx : x < X) (hy : y < Y) : x * Y + y < X * Y := by
  have : (x + 1) * Y ≤ X * Y := Nat.mul_le_mul_right _ hx
  nlinarith

theorem mix_div {x y Y : Nat} (hy : y < Y) : (x * Y + y) / Y = x := by
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hy, Nat.zero_add]

theorem mix_mod {x y Y : Nat} (hy : y < Y) : (x * Y + y) % Y = y := by
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hy]

/-! ### The blocked layout -/

/-- Slot of `(x, y)` in a tile `nb · bc` columns wide stored as `(br × bc)` blocks, row-major
blocks, row-major within a block. -/
def blk (nb br bc x y : Nat) : Nat := ((x / br * nb + y / bc) * br + x % br) * bc + y % bc

/-- The `(x, y)` that `blk` puts at slot `q`. -/
def unblk (nb br bc q : Nat) : Nat × Nat :=
  ((q / (br * bc) / nb) * br + q % (br * bc) / bc, (q / (br * bc) % nb) * bc + q % bc)

theorem blk_eq_split (nb br bc x y : Nat) :
    blk nb br bc x y = (x / br * nb + y / bc) * (br * bc) + (x % br * bc + y % bc) := by
  unfold blk; ring

theorem blk_lt {nr nb br bc x y : Nat} (hbr : 0 < br) (hbc : 0 < bc) (hx : x < nr * br)
    (hy : y < nb * bc) : blk nb br bc x y < nr * br * (nb * bc) := by
  have h1 : x / br < nr := (Nat.div_lt_iff_lt_mul hbr).2 hx
  have h2 : y / bc < nb := (Nat.div_lt_iff_lt_mul hbc).2 hy
  have h3 := mix_lt (mix_lt (mix_lt h1 h2) (Nat.mod_lt x hbr)) (Nat.mod_lt y hbc)
  unfold blk
  calc _ < nr * nb * br * bc := h3
    _ = nr * br * (nb * bc) := by ring

theorem unblk_blk {nb br bc x y : Nat} (hbr : 0 < br) (hbc : 0 < bc) (hy : y < nb * bc) :
    unblk nb br bc (blk nb br bc x y) = (x, y) := by
  have h2 : y / bc < nb := (Nat.div_lt_iff_lt_mul hbc).2 hy
  have hin : x % br * bc + y % bc < br * bc := by
    have := mix_lt (Nat.mod_lt x hbr) (Nat.mod_lt y hbc); simpa [Nat.mul_comm] using this
  have hq := blk_eq_split nb br bc x y
  have hdiv : blk nb br bc x y / (br * bc) = x / br * nb + y / bc := by rw [hq, mix_div hin]
  have hmod : blk nb br bc x y % (br * bc) = x % br * bc + y % bc := by rw [hq, mix_mod hin]
  have hbcmod : blk nb br bc x y % bc = y % bc := by
    unfold blk; exact mix_mod (Nat.mod_lt y hbc)
  unfold unblk
  rw [hdiv, hmod, hbcmod, mix_div h2, mix_mod h2, mix_div (Nat.mod_lt y hbc)]
  simp only [Prod.mk.injEq]
  constructor
  · exact Nat.div_add_mod' x br
  · exact Nat.div_add_mod' y bc

theorem unblk_fst_lt {nr nb br bc q : Nat} (hbr : 0 < br) (hbc : 0 < bc)
    (hq : q < nr * br * (nb * bc)) : (unblk nb br bc q).1 < nr * br := by
  have hB : q / (br * bc) < nr * nb := by
    rw [Nat.div_lt_iff_lt_mul (Nat.mul_pos hbr hbc)]
    calc q < nr * br * (nb * bc) := hq
      _ = nr * nb * (br * bc) := by ring
  have hnb : 0 < nb := by
    rcases Nat.eq_zero_or_pos nb with h | h
    · subst h; simp at hq
    · exact h
  have h1 : q / (br * bc) / nb < nr := (Nat.div_lt_iff_lt_mul hnb).2 hB
  have h2 : q % (br * bc) / bc < br :=
    (Nat.div_lt_iff_lt_mul hbc).2 (Nat.mod_lt q (Nat.mul_pos hbr hbc))
  exact mix_lt h1 h2

theorem unblk_snd_lt {nr nb br bc q : Nat} (hbc : 0 < bc) (hq : q < nr * br * (nb * bc)) :
    (unblk nb br bc q).2 < nb * bc := by
  have hnb : 0 < nb := by
    rcases Nat.eq_zero_or_pos nb with h | h
    · subst h; simp at hq
    · exact h
  exact mix_lt (Nat.mod_lt _ hnb) (Nat.mod_lt q hbc)

theorem blk_unblk {nr nb br bc q : Nat} (hbr : 0 < br) (hbc : 0 < bc)
    (hq : q < nr * br * (nb * bc)) :
    blk nb br bc (unblk nb br bc q).1 (unblk nb br bc q).2 = q := by
  have hnb : 0 < nb := by
    rcases Nat.eq_zero_or_pos nb with h | h
    · subst h; simp at hq
    · exact h
  set B := q / (br * bc)
  set R := q % (br * bc)
  have hR : R / bc < br := (Nat.div_lt_iff_lt_mul hbc).2 (Nat.mod_lt q (Nat.mul_pos hbr hbc))
  have hqb : q % bc < bc := Nat.mod_lt q hbc
  have hRmod : R % bc = q % bc := Nat.mod_mul_left_mod q br bc
  unfold unblk blk
  rw [show (B / nb * br + R / bc) / br = B / nb from mix_div hR,
    show (B / nb * br + R / bc) % br = R / bc from mix_mod hR,
    show (B % nb * bc + q % bc) / bc = B % nb from mix_div hqb,
    show (B % nb * bc + q % bc) % bc = q % bc from mix_mod hqb]
  have e1 : B / nb * nb + B % nb = B := Nat.div_add_mod' B nb
  have e2 : R / bc * bc + R % bc = R := Nat.div_add_mod' R bc
  have e3 : B * (br * bc) + R = q := Nat.div_add_mod' q (br * bc)
  rw [e1, ← hRmod]
  calc (B * br + R / bc) * bc + R % bc = B * (br * bc) + (R / bc * bc + R % bc) := by ring
    _ = q := by rw [e2, e3]

theorem blk_inj {nb br bc x y x' y' : Nat} (hbr : 0 < br) (hbc : 0 < bc) (hy : y < nb * bc)
    (hy' : y' < nb * bc) (h : blk nb br bc x y = blk nb br bc x' y') : x = x' ∧ y = y' := by
  have := congrArg (unblk nb br bc) h
  rw [unblk_blk hbr hbc hy, unblk_blk hbr hbc hy'] at this
  simpa using this

/-! ### The walks that make and read the blocked layout -/

/-- Reads a row-major tile `W` wide (`W = nb · bc`) as `(br × bc)` blocks: stream position
`blk nb br bc x y` reads element `x · W + y`. `mm_stream_dims` A and B. -/
def walkBlk (nr nb br bc W : Nat) : Walk :=
  { offset := 0,
    dims := [((nr : Nat), (br * W : Nat)), ((nb : Nat), (bc : Nat)), ((br : Nat), (W : Nat)),
             ((bc : Nat), 1)] }

/-- Reads a `(br × bc)`-blocked buffer in row-major order: stream position `x · W + y` reads
slot `blk nb br bc x y`. `mm_stream_dims` C. -/
def walkUnblk (nr nb br bc W : Nat) : Walk :=
  { offset := 0,
    dims := [((nr : Nat), (br * W : Nat)), ((br : Nat), (bc : Nat)),
             ((nb : Nat), (br * bc : Nat)), ((bc : Nat), 1)] }

theorem walkBlk_len (nr nb br bc W : Nat) :
    (walkBlk nr nb br bc W).len = nr * br * (nb * bc) := by
  simp [walkBlk, Walk.len, dimsLen]; ring

theorem walkUnblk_len (nr nb br bc W : Nat) :
    (walkUnblk nr nb br bc W).len = nr * br * (nb * bc) := by
  simp [walkUnblk, Walk.len, dimsLen]; ring

theorem walkBlk_idx {nr nb br bc W x y : Nat} (hbr : 0 < br) (hbc : 0 < bc)
    (hx : x < nr * br) (hy : y < nb * bc) :
    (walkBlk nr nb br bc W).idx (blk nb br bc x y) = ((x * W + y : Nat) : Int) := by
  have h1 : x / br < nr := (Nat.div_lt_iff_lt_mul hbr).2 hx
  have h2 : y / bc < nb := (Nat.div_lt_iff_lt_mul hbc).2 hy
  have hfit : Fits (walkBlk nr nb br bc W).dims [x / br, y / bc, x % br, y % bc] := by
    simp [walkBlk, Fits, h1, h2, Nat.mod_lt x hbr, Nat.mod_lt y hbc]
  have henc : encode (walkBlk nr nb br bc W).dims [x / br, y / bc, x % br, y % bc] =
      blk nb br bc x y := by
    simp [walkBlk, encode, dimsLen, blk]; ring
  rw [← henc, Walk.idx_encode _ hfit]
  have hn : x / br * (br * W) + (y / bc * bc + (x % br * W + y % bc)) = x * W + y := by
    have ex := Nat.div_add_mod' x br
    have ey := Nat.div_add_mod' y bc
    calc _ = (x / br * br + x % br) * W + (y / bc * bc + y % bc) := by ring
      _ = x * W + y := by rw [ex, ey]
  simp only [walkBlk, dot]
  rw [← hn]
  simp only [Nat.cast_add, Nat.cast_mul]
  ring

theorem walkUnblk_idx {nr nb br bc W x y : Nat} (hbr : 0 < br) (hbc : 0 < bc)
    (hW : W = nb * bc) (hx : x < nr * br) (hy : y < nb * bc) :
    (walkUnblk nr nb br bc W).idx (x * W + y) = ((blk nb br bc x y : Nat) : Int) := by
  have h1 : x / br < nr := (Nat.div_lt_iff_lt_mul hbr).2 hx
  have h2 : y / bc < nb := (Nat.div_lt_iff_lt_mul hbc).2 hy
  have hfit : Fits (walkUnblk nr nb br bc W).dims [x / br, x % br, y / bc, y % bc] := by
    simp [walkUnblk, Fits, h1, h2, Nat.mod_lt x hbr, Nat.mod_lt y hbc]
  have henc : encode (walkUnblk nr nb br bc W).dims [x / br, x % br, y / bc, y % bc] =
      x * W + y := by
    simp only [walkUnblk, encode, dimsLen, Int.toNat_natCast]
    subst hW
    have ex : x / br * br + x % br = x := Nat.div_add_mod' x br
    have ey : y / bc * bc + y % bc = y := Nat.div_add_mod' y bc
    calc x / br * (br * (nb * (bc * 1))) + (x % br * (nb * (bc * 1)) + (y / bc * (bc * 1) +
          (y % bc * 1 + 0)))
        = (x / br * br + x % br) * (nb * bc) + (y / bc * bc + y % bc) := by ring
      _ = x * (nb * bc) + y := by rw [ex, ey]
  rw [← henc, Walk.idx_encode _ hfit]
  simp only [walkUnblk, dot, blk]
  subst hW
  simp only [Nat.cast_add, Nat.cast_mul]
  ring

/-! ### The schema's layouts are `blk` -/

variable (p : Params)

theorem layoutA_eq (a b : Nat) : p.layoutA a b = blk (p.k / p.s) p.r p.s a b := by
  unfold Params.layoutA blk; ring

theorem layoutB_eq (a b : Nat) :
    p.layoutB a b = if p.bColMaj then blk (p.k / p.s) p.t p.s b a
      else blk (p.n / p.t) p.s p.t a b := by
  unfold Params.layoutB blk; split <;> ring

theorem layoutC_eq (a b : Nat) :
    p.layoutC a b = if p.cColMaj then blk (p.m / p.r) p.t p.r b a
      else blk (p.n / p.t) p.r p.t a b := by
  unfold Params.layoutC blk; split <;> ring

theorem unlayoutA_eq (q : Nat) : p.unlayoutA q = unblk (p.k / p.s) p.r p.s q := rfl

theorem unlayoutB_eq (q : Nat) :
    p.unlayoutB q = if p.bColMaj then ((unblk (p.k / p.s) p.t p.s q).2, (unblk (p.k / p.s) p.t p.s q).1)
      else unblk (p.n / p.t) p.s p.t q := by
  unfold Params.unlayoutB unblk; split <;> simp [Nat.mul_comm p.s p.t]

theorem unlayoutC_eq (q : Nat) :
    p.unlayoutC q = if p.cColMaj then ((unblk (p.m / p.r) p.t p.r q).2, (unblk (p.m / p.r) p.t p.r q).1)
      else unblk (p.n / p.t) p.r p.t q := by
  unfold Params.unlayoutC unblk; split <;> simp [Nat.mul_comm p.r p.t]

theorem walkA_eq : walkA p = walkBlk (p.m / p.r) (p.k / p.s) p.r p.s p.k := rfl

theorem walkB_eq : walkB p = if p.bColMaj then walkBlk (p.n / p.t) (p.k / p.s) p.t p.s p.k
    else walkBlk (p.k / p.s) (p.n / p.t) p.s p.t p.n := by
  unfold walkB; split <;> rfl

theorem walkC_eq : walkC p = if p.cColMaj then walkUnblk (p.n / p.t) (p.m / p.r) p.t p.r p.m
    else walkUnblk (p.m / p.r) (p.n / p.t) p.r p.t p.n := by
  unfold walkC; split <;> rfl

end Arith

open Arith

namespace Params

variable (p : Params)

/-- The tile position C drains element `(a, b)` at: row-major, or column-major with `cColMaj`. -/
def cPos (a b : Nat) : Nat := if p.cColMaj then b * p.m + a else a * p.n + b

/-- The element of the tile the C drain emits at tile position `q`. -/
def cPosRow (q : Nat) : Nat := if p.cColMaj then q % p.m else q / p.n
def cPosCol (q : Nat) : Nat := if p.cColMaj then q / p.m else q % p.n

theorem cPos_cPosRow (q : Nat) : p.cPos (p.cPosRow q) (p.cPosCol q) = q := by
  unfold cPos cPosRow cPosCol; split
  · exact Nat.div_add_mod' q p.m
  · exact Nat.div_add_mod' q p.n

end Params

namespace Params.Valid

variable {p : Params} (hp : p.Valid)
include hp

theorem m_eq : p.m / p.r * p.r = p.m := Nat.div_mul_cancel hp.r_dvd
theorem k_eq : p.k / p.s * p.s = p.k := Nat.div_mul_cancel hp.s_dvd
theorem n_eq : p.n / p.t * p.t = p.n := Nat.div_mul_cancel hp.t_dvd

/-! ### Layouts: in range, inverted by `unlayout`, onto the tile -/

theorem layoutA_lt {a b : Nat} (ha : a < p.m) (hb : b < p.k) : p.layoutA a b < p.m * p.k := by
  rw [layoutA_eq]
  have := blk_lt (nr := p.m / p.r) hp.r_pos hp.s_pos (by rw [hp.m_eq]; exact ha)
    (by rw [hp.k_eq]; exact hb)
  rwa [hp.m_eq, hp.k_eq] at this

theorem unlayoutA_layoutA {a b : Nat} (hb : b < p.k) : p.unlayoutA (p.layoutA a b) = (a, b) := by
  rw [layoutA_eq, unlayoutA_eq]
  exact unblk_blk hp.r_pos hp.s_pos (by rw [hp.k_eq]; exact hb)

theorem unlayoutA_fst_lt {q : Nat} (hq : q < p.m * p.k) : (p.unlayoutA q).1 < p.m := by
  rw [unlayoutA_eq]
  have := unblk_fst_lt (nr := p.m / p.r) (nb := p.k / p.s) hp.r_pos hp.s_pos
    (by rw [hp.m_eq, hp.k_eq]; exact hq)
  rwa [hp.m_eq] at this

theorem unlayoutA_snd_lt {q : Nat} (hq : q < p.m * p.k) : (p.unlayoutA q).2 < p.k := by
  rw [unlayoutA_eq]
  have := unblk_snd_lt (nr := p.m / p.r) (br := p.r) (nb := p.k / p.s) hp.s_pos
    (by rw [hp.m_eq, hp.k_eq]; exact hq)
  rwa [hp.k_eq] at this

theorem layoutA_unlayoutA {q : Nat} (hq : q < p.m * p.k) :
    p.layoutA (p.unlayoutA q).1 (p.unlayoutA q).2 = q := by
  rw [layoutA_eq, unlayoutA_eq]
  exact blk_unblk (nr := p.m / p.r) hp.r_pos hp.s_pos (by rw [hp.m_eq, hp.k_eq]; exact hq)

theorem layoutA_inj {a b a' b' : Nat} (hb : b < p.k) (hb' : b' < p.k)
    (h : p.layoutA a b = p.layoutA a' b') : a = a' ∧ b = b' := by
  have := congrArg p.unlayoutA h
  rw [hp.unlayoutA_layoutA hb, hp.unlayoutA_layoutA hb'] at this
  simpa using this

theorem layoutB_lt {a b : Nat} (ha : a < p.k) (hb : b < p.n) : p.layoutB a b < p.k * p.n := by
  rw [layoutB_eq]
  split
  · have := blk_lt (nr := p.n / p.t) hp.t_pos hp.s_pos (by rw [hp.n_eq]; exact hb)
      (by rw [hp.k_eq]; exact ha)
    rw [hp.n_eq, hp.k_eq] at this; rw [Nat.mul_comm]; exact this
  · have := blk_lt (nr := p.k / p.s) hp.s_pos hp.t_pos (by rw [hp.k_eq]; exact ha)
      (by rw [hp.n_eq]; exact hb)
    rwa [hp.k_eq, hp.n_eq] at this

theorem unlayoutB_layoutB {a b : Nat} (ha : a < p.k) (hb : b < p.n) :
    p.unlayoutB (p.layoutB a b) = (a, b) := by
  rw [layoutB_eq, unlayoutB_eq]
  split
  · rw [unblk_blk hp.t_pos hp.s_pos (by rw [hp.k_eq]; exact ha)]
  · exact unblk_blk hp.s_pos hp.t_pos (by rw [hp.n_eq]; exact hb)

theorem unlayoutB_fst_lt {q : Nat} (hq : q < p.k * p.n) : (p.unlayoutB q).1 < p.k := by
  rw [unlayoutB_eq]
  split
  · have := unblk_snd_lt (nr := p.n / p.t) (br := p.t) (nb := p.k / p.s) hp.s_pos
      (by rw [hp.n_eq, hp.k_eq, Nat.mul_comm]; exact hq)
    rwa [hp.k_eq] at this
  · have := unblk_fst_lt (nr := p.k / p.s) (nb := p.n / p.t) hp.s_pos hp.t_pos
      (by rw [hp.k_eq, hp.n_eq]; exact hq)
    rwa [hp.k_eq] at this

theorem unlayoutB_snd_lt {q : Nat} (hq : q < p.k * p.n) : (p.unlayoutB q).2 < p.n := by
  rw [unlayoutB_eq]
  split
  · have := unblk_fst_lt (nr := p.n / p.t) (nb := p.k / p.s) hp.t_pos hp.s_pos
      (by rw [hp.n_eq, hp.k_eq, Nat.mul_comm]; exact hq)
    rwa [hp.n_eq] at this
  · have := unblk_snd_lt (nr := p.k / p.s) (br := p.s) (nb := p.n / p.t) hp.t_pos
      (by rw [hp.k_eq, hp.n_eq]; exact hq)
    rwa [hp.n_eq] at this

theorem layoutB_unlayoutB {q : Nat} (hq : q < p.k * p.n) :
    p.layoutB (p.unlayoutB q).1 (p.unlayoutB q).2 = q := by
  rw [layoutB_eq, unlayoutB_eq]
  split
  · exact blk_unblk (nr := p.n / p.t) hp.t_pos hp.s_pos
      (by rw [hp.n_eq, hp.k_eq, Nat.mul_comm]; exact hq)
  · exact blk_unblk (nr := p.k / p.s) hp.s_pos hp.t_pos (by rw [hp.k_eq, hp.n_eq]; exact hq)

theorem layoutB_inj {a b a' b' : Nat} (ha : a < p.k) (hb : b < p.n) (ha' : a' < p.k)
    (hb' : b' < p.n) (h : p.layoutB a b = p.layoutB a' b') : a = a' ∧ b = b' := by
  have := congrArg p.unlayoutB h
  rw [hp.unlayoutB_layoutB ha hb, hp.unlayoutB_layoutB ha' hb'] at this
  simpa using this

theorem layoutC_lt {a b : Nat} (ha : a < p.m) (hb : b < p.n) : p.layoutC a b < p.m * p.n := by
  rw [layoutC_eq]
  split
  · have := blk_lt (nr := p.n / p.t) hp.t_pos hp.r_pos (by rw [hp.n_eq]; exact hb)
      (by rw [hp.m_eq]; exact ha)
    rw [hp.n_eq, hp.m_eq] at this; rw [Nat.mul_comm]; exact this
  · have := blk_lt (nr := p.m / p.r) hp.r_pos hp.t_pos (by rw [hp.m_eq]; exact ha)
      (by rw [hp.n_eq]; exact hb)
    rwa [hp.m_eq, hp.n_eq] at this

theorem unlayoutC_layoutC {a b : Nat} (ha : a < p.m) (hb : b < p.n) :
    p.unlayoutC (p.layoutC a b) = (a, b) := by
  rw [layoutC_eq, unlayoutC_eq]
  split
  · rw [unblk_blk hp.t_pos hp.r_pos (by rw [hp.m_eq]; exact ha)]
  · exact unblk_blk hp.r_pos hp.t_pos (by rw [hp.n_eq]; exact hb)

theorem unlayoutC_fst_lt {q : Nat} (hq : q < p.m * p.n) : (p.unlayoutC q).1 < p.m := by
  rw [unlayoutC_eq]
  split
  · have := unblk_snd_lt (nr := p.n / p.t) (br := p.t) (nb := p.m / p.r) hp.r_pos
      (by rw [hp.n_eq, hp.m_eq, Nat.mul_comm]; exact hq)
    rwa [hp.m_eq] at this
  · have := unblk_fst_lt (nr := p.m / p.r) (nb := p.n / p.t) hp.r_pos hp.t_pos
      (by rw [hp.m_eq, hp.n_eq]; exact hq)
    rwa [hp.m_eq] at this

theorem unlayoutC_snd_lt {q : Nat} (hq : q < p.m * p.n) : (p.unlayoutC q).2 < p.n := by
  rw [unlayoutC_eq]
  split
  · have := unblk_fst_lt (nr := p.n / p.t) (nb := p.m / p.r) hp.t_pos hp.r_pos
      (by rw [hp.n_eq, hp.m_eq, Nat.mul_comm]; exact hq)
    rwa [hp.n_eq] at this
  · have := unblk_snd_lt (nr := p.m / p.r) (br := p.r) (nb := p.n / p.t) hp.t_pos
      (by rw [hp.m_eq, hp.n_eq]; exact hq)
    rwa [hp.n_eq] at this

theorem layoutC_unlayoutC {q : Nat} (hq : q < p.m * p.n) :
    p.layoutC (p.unlayoutC q).1 (p.unlayoutC q).2 = q := by
  rw [layoutC_eq, unlayoutC_eq]
  split
  · exact blk_unblk (nr := p.n / p.t) hp.t_pos hp.r_pos
      (by rw [hp.n_eq, hp.m_eq, Nat.mul_comm]; exact hq)
  · exact blk_unblk (nr := p.m / p.r) hp.r_pos hp.t_pos (by rw [hp.m_eq, hp.n_eq]; exact hq)

theorem layoutC_inj {a b a' b' : Nat} (ha : a < p.m) (hb : b < p.n) (ha' : a' < p.m)
    (hb' : b' < p.n) (h : p.layoutC a b = p.layoutC a' b') : a = a' ∧ b = b' := by
  have := congrArg p.unlayoutC h
  rw [hp.unlayoutC_layoutC ha hb, hp.unlayoutC_layoutC ha' hb'] at this
  simpa using this

/-! ### Walks: the to_stream walks invert the layouts -/

theorem walkA_len : (walkA p).len = p.m * p.k := by
  rw [walkA_eq, walkBlk_len, hp.m_eq, hp.k_eq]

/-- Core A slot `layoutA a b` holds element `(a, b)` of the row-major `m × k` tile. -/
theorem walkA_idx {a b : Nat} (ha : a < p.m) (hb : b < p.k) :
    (walkA p).idx (p.layoutA a b) = ((a * p.k + b : Nat) : Int) := by
  rw [walkA_eq, layoutA_eq]
  exact walkBlk_idx hp.r_pos hp.s_pos (by rw [hp.m_eq]; exact ha)
    (by rw [hp.k_eq]; exact hb)

theorem walkB_len : (walkB p).len = p.k * p.n := by
  rw [walkB_eq]; split
  · rw [walkBlk_len, hp.n_eq, hp.k_eq, Nat.mul_comm]
  · rw [walkBlk_len, hp.k_eq, hp.n_eq]

/-- Core B slot `layoutB a b` holds element `(a, b)` of the `k × n` tile: row-major, or read
from the `n × k` transpose with `bColMaj`. -/
theorem walkB_idx {a b : Nat} (ha : a < p.k) (hb : b < p.n) :
    (walkB p).idx (p.layoutB a b) =
      (((if p.bColMaj then b * p.k + a else a * p.n + b) : Nat) : Int) := by
  rw [walkB_eq, layoutB_eq]
  split
  · exact walkBlk_idx hp.t_pos hp.s_pos (by rw [hp.n_eq]; exact hb)
      (by rw [hp.k_eq]; exact ha)
  · exact walkBlk_idx hp.s_pos hp.t_pos (by rw [hp.k_eq]; exact ha)
      (by rw [hp.n_eq]; exact hb)

theorem walkC_len : (walkC p).len = p.m * p.n := by
  rw [walkC_eq]; split
  · rw [walkUnblk_len, hp.n_eq, hp.m_eq, Nat.mul_comm]
  · rw [walkUnblk_len, hp.m_eq, hp.n_eq]

/-- The C drain walk reads element `(a, b)` of the tile from core slot `layoutC a b`. -/
theorem walkC_idx {a b : Nat} (ha : a < p.m) (hb : b < p.n) :
    (walkC p).idx (p.cPos a b) = ((p.layoutC a b : Nat) : Int) := by
  rw [walkC_eq, layoutC_eq, Params.cPos]
  split
  · exact walkUnblk_idx hp.t_pos hp.r_pos hp.m_eq.symm (by rw [hp.n_eq]; exact hb)
      (by rw [hp.m_eq]; exact ha)
  · exact walkUnblk_idx hp.r_pos hp.t_pos hp.n_eq.symm (by rw [hp.m_eq]; exact ha)
      (by rw [hp.n_eq]; exact hb)

theorem cPosRow_lt {q : Nat} (hq : q < p.m * p.n) : p.cPosRow q < p.m := by
  unfold Params.cPosRow; split
  · exact Nat.mod_lt _ hp.m_pos
  · exact (Nat.div_lt_iff_lt_mul hp.n_pos).2 hq

theorem cPosCol_lt {q : Nat} (hq : q < p.m * p.n) : p.cPosCol q < p.n := by
  unfold Params.cPosCol; split
  · exact (Nat.div_lt_iff_lt_mul hp.m_pos).2 (by rw [Nat.mul_comm]; exact hq)
  · exact Nat.mod_lt _ hp.n_pos

/-- The drain at tile position `q` reads core slot `layoutC` of the element `cPos` puts there. -/
theorem walkC_idx_pos {q : Nat} (hq : q < p.m * p.n) :
    (walkC p).idx q = ((p.layoutC (p.cPosRow q) (p.cPosCol q) : Nat) : Int) := by
  conv_lhs => rw [← p.cPos_cPosRow q]
  exact hp.walkC_idx (hp.cPosRow_lt hq) (hp.cPosCol_lt hq)

/-! ### Tile ownership -/

theorem rowC_lt (i : Nat) : p.rowC i < p.rows := by
  unfold Params.rowC
  exact (Nat.div_lt_iff_lt_mul hp.m_pos).2 (Nat.mod_lt _ (Nat.mul_pos hp.rows_pos hp.m_pos))

theorem colC_lt (j : Nat) : p.colC j < p.cols := by
  unfold Params.colC
  exact (Nat.div_lt_iff_lt_mul hp.n_pos).2
    ((Nat.mod_lt _ (Nat.mul_pos hp.n_pos hp.cols_pos)).trans_eq (Nat.mul_comm _ _))

theorem tauC_lt {Mt Nt i j : Nat} (hi : i < p.rows * p.m * Mt) (hj : j < p.n * p.cols * Nt) :
    p.tauC Nt i j < Mt * Nt := by
  unfold Params.tauC
  exact mix_lt ((Nat.div_lt_iff_lt_mul (Nat.mul_pos hp.rows_pos hp.m_pos)).2
      (by rw [Nat.mul_comm]; exact hi))
    ((Nat.div_lt_iff_lt_mul (Nat.mul_pos hp.n_pos hp.cols_pos)).2
      (by rw [Nat.mul_comm]; exact hj))

theorem tauC_div {Nt i j : Nat} (hj : j < p.n * p.cols * Nt) :
    p.tauC Nt i j / Nt = i / (p.rows * p.m) ∧ p.tauC Nt i j % Nt = j / (p.n * p.cols) := by
  have h : j / (p.n * p.cols) < Nt :=
    (Nat.div_lt_iff_lt_mul (Nat.mul_pos hp.n_pos hp.cols_pos)).2 (by rw [Nat.mul_comm]; exact hj)
  unfold Params.tauC
  exact ⟨mix_div h, mix_mod h⟩

/-- `C[i, j]` sits at `(i % m, j % n)` of its owner's tile. -/
theorem i_decomp {Nt i j : Nat} (hj : j < p.n * p.cols * Nt) :
    p.i0 Nt (p.rowC i) (p.tauC Nt i j) + i % p.m = i := by
  unfold Params.i0 Params.rowC
  rw [(hp.tauC_div hj).1]
  have e1 := Nat.div_add_mod' i (p.rows * p.m)
  have e2 := Nat.div_add_mod' (i % (p.rows * p.m)) p.m
  have e3 : i % (p.rows * p.m) % p.m = i % p.m := Nat.mod_mul_left_mod i p.rows p.m
  calc i / (p.rows * p.m) * (p.rows * p.m) + i % (p.rows * p.m) / p.m * p.m + i % p.m
      = i / (p.rows * p.m) * (p.rows * p.m) +
          (i % (p.rows * p.m) / p.m * p.m + i % (p.rows * p.m) % p.m) := by rw [e3]; ring
    _ = i := by rw [e2, e1]

theorem j_decomp {Nt i j : Nat} (hj : j < p.n * p.cols * Nt) :
    p.j0 Nt (p.colC j) (p.tauC Nt i j) + j % p.n = j := by
  unfold Params.j0 Params.colC
  rw [(hp.tauC_div hj).2]
  have e1 := Nat.div_add_mod' j (p.n * p.cols)
  have e2 := Nat.div_add_mod' (j % (p.n * p.cols)) p.n
  have e3 : j % (p.n * p.cols) % p.n = j % p.n := by
    rw [Nat.mul_comm]; exact Nat.mod_mul_left_mod j p.cols p.n
  calc j / (p.n * p.cols) * (p.n * p.cols) + j % (p.n * p.cols) / p.n * p.n + j % p.n
      = j / (p.n * p.cols) * (p.n * p.cols) +
          (j % (p.n * p.cols) / p.n * p.n + j % (p.n * p.cols) % p.n) := by rw [e3]; ring
    _ = j := by rw [e2, e1]

end Params.Valid

namespace Params

variable (p : Params)

/-! ### Tile ownership: the converse direction, and drain positions -/

theorem cPos_lt {a b : Nat} (ha : a < p.m) (hb : b < p.n) : p.cPos a b < p.m * p.n := by
  unfold Params.cPos; split
  · exact (mix_lt hb ha).trans_eq (Nat.mul_comm _ _)
  · exact mix_lt ha hb

theorem cPosRow_cPos {a b : Nat} (ha : a < p.m) (hb : b < p.n) :
    p.cPosRow (p.cPos a b) = a ∧ p.cPosCol (p.cPos a b) = b := by
  unfold Params.cPos Params.cPosRow Params.cPosCol; split
  · exact ⟨mix_mod ha, mix_div ha⟩
  · exact ⟨mix_div hb, mix_mod hb⟩

/-- The converse: element `(a, b)` of core `(row, col)`'s `τ`-th tile is owned by that core. -/
theorem own_i0 {Nt row τ a : Nat} (hrow : row < p.rows) (ha : a < p.m) :
    p.rowC (p.i0 Nt row τ + a) = row ∧ (p.i0 Nt row τ + a) % p.m = a ∧
      (p.i0 Nt row τ + a) / (p.rows * p.m) = τ / Nt := by
  have hin : row * p.m + a < p.rows * p.m := by
    calc row * p.m + a < (row + 1) * p.m := by nlinarith
      _ ≤ p.rows * p.m := Nat.mul_le_mul_right _ hrow
  have e : p.i0 Nt row τ + a = τ / Nt * (p.rows * p.m) + (row * p.m + a) := by
    unfold Params.i0; ring
  refine ⟨?_, ?_, ?_⟩
  · unfold Params.rowC; rw [e, mix_mod hin, mix_div ha]
  · rw [e, show τ / Nt * (p.rows * p.m) + (row * p.m + a) = (τ / Nt * p.rows + row) * p.m + a by
      ring, mix_mod ha]
  · rw [e, mix_div hin]

theorem own_j0 {Nt col τ b : Nat} (hcol : col < p.cols) (hb : b < p.n) :
    p.colC (p.j0 Nt col τ + b) = col ∧ (p.j0 Nt col τ + b) % p.n = b ∧
      (p.j0 Nt col τ + b) / (p.n * p.cols) = τ % Nt := by
  have hin : col * p.n + b < p.n * p.cols := by
    calc col * p.n + b < (col + 1) * p.n := by nlinarith
      _ ≤ p.cols * p.n := Nat.mul_le_mul_right _ hcol
      _ = p.n * p.cols := Nat.mul_comm _ _
  have e : p.j0 Nt col τ + b = τ % Nt * (p.n * p.cols) + (col * p.n + b) := by
    unfold Params.j0; ring
  refine ⟨?_, ?_, ?_⟩
  · unfold Params.colC; rw [e, mix_mod hin, mix_div hb]
  · rw [e, show τ % Nt * (p.n * p.cols) + (col * p.n + b) = (τ % Nt * p.cols + col) * p.n + b by
      ring, mix_mod hb]
  · rw [e, mix_div hin]

theorem own_tauC {Nt row col τ a b : Nat} (hrow : row < p.rows)
    (hcol : col < p.cols) (ha : a < p.m) (hb : b < p.n) :
    p.tauC Nt (p.i0 Nt row τ + a) (p.j0 Nt col τ + b) = τ := by
  unfold Params.tauC
  rw [(p.own_i0 hrow ha).2.2, (p.own_j0 hcol hb).2.2]
  exact Nat.div_add_mod' τ Nt

theorem i0_lt {Mt Nt row τ a : Nat} (hrow : row < p.rows) (hτ : τ < Mt * Nt) (ha : a < p.m) :
    p.i0 Nt row τ + a < p.rows * p.m * Mt := by
  have hNt : 0 < Nt := by
    rcases Nat.eq_zero_or_pos Nt with h | h
    · subst h; simp at hτ
    · exact h
  have h1 : τ / Nt < Mt := (Nat.div_lt_iff_lt_mul hNt).2 hτ
  have hin : row * p.m + a < p.rows * p.m := by
    calc row * p.m + a < (row + 1) * p.m := by nlinarith
      _ ≤ p.rows * p.m := Nat.mul_le_mul_right _ hrow
  unfold Params.i0
  calc τ / Nt * (p.rows * p.m) + row * p.m + a = τ / Nt * (p.rows * p.m) + (row * p.m + a) := by
        ring
    _ < Mt * (p.rows * p.m) := mix_lt h1 hin
    _ = p.rows * p.m * Mt := by ring

theorem j0_lt {Nt col τ b : Nat} (hNt : 0 < Nt) (hcol : col < p.cols) (hb : b < p.n) :
    p.j0 Nt col τ + b < p.n * p.cols * Nt := by
  have h1 : τ % Nt < Nt := Nat.mod_lt _ hNt
  have hin : col * p.n + b < p.n * p.cols := by
    calc col * p.n + b < (col + 1) * p.n := by nlinarith
      _ ≤ p.cols * p.n := Nat.mul_le_mul_right _ hcol
      _ = p.n * p.cols := Nat.mul_comm _ _
  unfold Params.j0
  calc τ % Nt * (p.n * p.cols) + col * p.n + b = τ % Nt * (p.n * p.cols) + (col * p.n + b) := by
        ring
    _ < Nt * (p.n * p.cols) := mix_lt h1 hin
    _ = p.n * p.cols * Nt := by ring

end Params

/-! ### Host indices -/

namespace Params

variable (p : Params)

theorem aIdx_lt {M K i x : Nat} (hi : i < M) (hx : x < K) : p.aIdx K i x < M * K :=
  mix_lt hi hx

theorem bIdx_lt {K N x j : Nat} (hx : x < K) (hj : j < N) : p.bIdx K N x j < K * N := by
  unfold bIdx; split
  · exact (mix_lt hj hx).trans_eq (Nat.mul_comm _ _)
  · exact mix_lt hx hj

theorem cIdx_lt {M N i j : Nat} (hi : i < M) (hj : j < N) : p.cIdx M N i j < M * N := by
  unfold cIdx; split
  · exact (mix_lt hj hi).trans_eq (Nat.mul_comm _ _)
  · exact mix_lt hi hj

theorem cIdx_inj {M N i j i' j' : Nat} (hi : i < M) (hj : j < N) (hi' : i' < M) (hj' : j' < N)
    (h : p.cIdx M N i j = p.cIdx M N i' j') : i = i' ∧ j = j' := by
  unfold cIdx at h; split at h
  · have h1 := congrArg (· / M) h; have h2 := congrArg (· % M) h
    simp only [mix_div hi, mix_div hi', mix_mod hi, mix_mod hi'] at h1 h2
    exact ⟨h2, h1⟩
  · have h1 := congrArg (· / N) h; have h2 := congrArg (· % N) h
    simp only [mix_div hj, mix_div hj', mix_mod hj, mix_mod hj'] at h1 h2
    exact ⟨h1, h2⟩

end Params

/-! ### Dot products over k blocks -/

theorem sum_blocks {R : Type*} [AddCommMonoid R] (k : Nat) (f : Nat → R) :
    ∀ Kt, ∑ kk ∈ Finset.range Kt, ∑ c ∈ Finset.range k, f (kk * k + c) =
      ∑ x ∈ Finset.range (k * Kt), f x
  | 0 => by simp
  | Kt + 1 => by
    rw [Finset.sum_range_succ, sum_blocks k f Kt, show k * (Kt + 1) = k * Kt + k by ring,
      Finset.sum_range_add]
    congr 1
    refine Finset.sum_congr rfl fun c _ => ?_
    rw [Nat.mul_comm]

end IronVerify.Schema.WholeArray
