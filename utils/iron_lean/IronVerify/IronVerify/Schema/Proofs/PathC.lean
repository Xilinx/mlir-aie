-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Schema.Defs
import IronVerify.Schema.Proofs.Arith
import IronVerify.Schema.Proofs.Names
import IronVerify.Schema.Proofs.NetHops
import IronVerify.Schema.Proofs.Seq

/-!
The C path for every valid `p`: drain `c`'s position `wpos row τ a b` writes element `(a, b)`
of core `(row, c)`'s `τ`-th output tile, every drain position is one of these, and so every
element of C has exactly one writer across all `cols` drains.
-/

namespace IronVerify.Schema.WholeArray.PathC

open IronVerify IronVerify.Schema.WholeArray IronVerify.Schema.WholeArray.Arith

/-- Drain position of element `(a, b)` of core row `row`'s `τ`-th output tile. -/
def wpos (p : Params) (row τ a b : Nat) : Nat :=
  τ * (p.rows * (p.m * p.n)) + row * (p.m * p.n) + p.cPos a b

theorem lt_size_of_getElem? {xs : Array (Nat × Int)} {i : Nat} {e : Nat × Int}
    (h : xs[i]? = some e) : i < xs.size := by
  by_contra hc
  rw [Array.getElem?_eq_none (by omega)] at h
  cases h

variable {p : Params} (hp : p.Valid)
include hp

omit hp in
theorem rowpos_lt {row a b : Nat} (hrow : row < p.rows) (ha : a < p.m) (hb : b < p.n) :
    row * (p.m * p.n) + p.cPos a b < p.rows * (p.m * p.n) :=
  mix_lt hrow (p.cPos_lt ha hb)

omit hp in
theorem wpos_div {row τ a b : Nat} (hrow : row < p.rows) (ha : a < p.m) (hb : b < p.n) :
    wpos p row τ a b / (p.rows * (p.m * p.n)) = τ ∧
    wpos p row τ a b % (p.rows * (p.m * p.n)) / (p.m * p.n) = row ∧
    wpos p row τ a b % (p.m * p.n) = p.cPos a b := by
  have hin := rowpos_lt hrow ha hb
  have hc := p.cPos_lt ha hb
  have e : wpos p row τ a b = τ * (p.rows * (p.m * p.n)) + (row * (p.m * p.n) + p.cPos a b) := by
    unfold wpos; ring
  refine ⟨by rw [e, mix_div hin], by rw [e, mix_mod hin, mix_div hc], ?_⟩
  rw [show wpos p row τ a b = (τ * p.rows + row) * (p.m * p.n) + p.cPos a b by unfold wpos; ring,
    mix_mod hc]

/-- Every position of a drain is a writer position. -/
theorem wpos_decode {Mt Nt Q : Nat} (hQ : Q < Mt * Nt * (p.rows * (p.m * p.n))) :
    let row := Q % (p.rows * (p.m * p.n)) / (p.m * p.n)
    let τ := Q / (p.rows * (p.m * p.n))
    let a := p.cPosRow (Q % (p.m * p.n))
    let b := p.cPosCol (Q % (p.m * p.n))
    row < p.rows ∧ τ < Mt * Nt ∧ a < p.m ∧ b < p.n ∧ Q = wpos p row τ a b := by
  intro row τ a b
  have hmn : 0 < p.m * p.n := Nat.mul_pos hp.m_pos hp.n_pos
  have hR : 0 < p.rows * (p.m * p.n) := Nat.mul_pos hp.rows_pos hmn
  have hx : Q % (p.m * p.n) < p.m * p.n := Nat.mod_lt _ hmn
  refine ⟨(Nat.div_lt_iff_lt_mul hmn).2 (Nat.mod_lt _ hR), (Nat.div_lt_iff_lt_mul hR).2 hQ,
    hp.cPosRow_lt hx, hp.cPosCol_lt hx, ?_⟩
  unfold wpos
  rw [p.cPos_cPosRow]
  have e1 := Nat.div_add_mod Q (p.rows * (p.m * p.n))
  have e2 := Nat.div_add_mod (Q % (p.rows * (p.m * p.n))) (p.m * p.n)
  rw [Nat.mod_mul_left_mod Q p.rows (p.m * p.n)] at e2
  simp only [row, τ]
  nlinarith

/-- Drain `c` writes `C[i0 + a, j0 + b]` at `wpos row τ a b`. -/
theorem hostC_wpos (Mt Kt Nt : Nat) {c row τ a b : Nat} (hc : c < p.cols) (hrow : row < p.rows)
    (hτ : τ < Mt * Nt) (ha : a < p.m) (hb : b < p.n) :
    ((schema p).host (p.dims Mt Kt Nt) (cL3 c))[wpos p row τ a b]? =
      some (2, ((p.cIdx (p.Mn Mt) (p.Nn Nt) (p.i0 Nt row τ + a) (p.j0 Nt c τ + b) : Nat) : Int)) := by
  have hNt : 0 < Nt := by
    rcases Nat.eq_zero_or_pos Nt with h | h
    · subst h; simp at hτ
    · exact h
  have hrb : τ / Nt < Mt := (Nat.div_lt_iff_lt_mul hNt).2 hτ
  have hnt : τ % Nt < Nt := Nat.mod_lt _ hNt
  have hin : τ % Nt * (p.rows * (p.m * p.n)) + row * (p.m * p.n) + p.cPos a b <
      Streams.lenC p Nt := by
    have := mix_lt hnt (rowpos_lt hrow ha hb)
    unfold Streams.lenC
    calc τ % Nt * (p.rows * (p.m * p.n)) + row * (p.m * p.n) + p.cPos a b
        = τ % Nt * (p.rows * (p.m * p.n)) + (row * (p.m * p.n) + p.cPos a b) := by ring
      _ < Nt * (p.rows * (p.m * p.n)) := this
      _ = Nt * (p.rows * p.m * p.n) := by ring
  have hw : wpos p row τ a b = τ / Nt * Streams.lenC p Nt +
      (τ % Nt * (p.rows * (p.m * p.n)) + row * (p.m * p.n) + p.cPos a b) := by
    have := Nat.div_add_mod' τ Nt
    unfold wpos Streams.lenC
    conv_lhs => rw [← this]
    ring
  rw [hw, Streams.hostC_get hp Mt Kt Nt hc hrb hin]
  have hra : row * p.m + a < p.rows * p.m := mix_lt hrow ha
  cases hcm : p.cColMaj
  · have hq : τ % Nt * (p.rows * (p.m * p.n)) + row * (p.m * p.n) + p.cPos a b =
        (τ % Nt * (p.rows * p.m) + (row * p.m + a)) * p.n + b := by
      simp only [Params.cPos, hcm, Bool.false_eq_true, ↓reduceIte]; ring
    rw [hq, (Streams.tapC_idx hp Mt Nt (τ / Nt) c hnt).1 hcm hra hb]
    simp only [Params.cIdx, hcm, Params.i0, Params.j0, Bool.false_eq_true, ↓reduceIte]
    congr 3
    ring
  · have hq : τ % Nt * (p.rows * (p.m * p.n)) + row * (p.m * p.n) + p.cPos a b =
        ((τ % Nt * p.rows + row) * p.n + b) * p.m + a := by
      simp only [Params.cPos, hcm, ↓reduceIte]; ring
    rw [hq, (Streams.tapC_idx hp Mt Nt (τ / Nt) c hnt).2 hcm hrow hb ha]
    simp only [Params.cIdx, hcm, Params.i0, Params.j0, ↓reduceIte]
    congr 3
    ring

/-- What drain `c` emits at `wpos row τ a b`: core `(row, c)`'s `τ`-th tile, slot `layoutC a b`. -/
theorem streamC_wpos (P : Int × Int × Int) {c row τ a b : Nat} (hc : c < p.cols)
    (hrow : row < p.rows) (ha : a < p.m) (hb : b < p.n) :
    (schema p).stream P (cL3Fifo p c) (wpos p row τ a b) =
      some (.core (p.worker row c) (cL1 c row) τ (p.layoutC a b)) := by
  obtain ⟨e3, e2, e1⟩ := wpos_div (τ := τ) hrow ha hb
  have hx : ((walkC p).idx (wpos p row τ a b % (p.m * p.n))).toNat < p.m * p.n := by
    rw [e1, hp.walkC_idx ha hb, Int.toNat_natCast]; exact hp.layoutC_lt ha hb
  show (wnet p).stream ((schema p).host P) (cL3Fifo p c) (wpos p row τ a b) = _
  rw [NetHops.stream_cL3 p _ hp hc _ hx (by rw [e2]; exact Names.fifo?_cL1 p hc hrow), e1, e2, e3,
    hp.walkC_idx ha hb, Int.toNat_natCast]

theorem hostC_lt {Mt Kt Nt c Q : Nat} (hc : c < p.cols) {e : Nat × Int}
    (h : ((schema p).host (p.dims Mt Kt Nt) (cL3 c))[Q]? = some e) :
    Q < Mt * Nt * (p.rows * (p.m * p.n)) := by
  have := lt_size_of_getElem? h
  rw [Streams.hostC_size hp Mt Kt Nt hc] at this
  unfold Streams.lenC at this
  calc Q < Mt * (Nt * (p.rows * p.m * p.n)) := this
    _ = Mt * Nt * (p.rows * (p.m * p.n)) := by ring

/-! ### T-C -/

variable (p)

theorem tC (Mt Kt Nt i j : Nat) (hi : i < p.Mn Mt) (hj : j < p.Nn Nt) :
    let P := p.dims Mt Kt Nt
    let e : Nat × Int := (2, ((p.cIdx (p.Mn Mt) (p.Nn Nt) i j : Nat) : Int))
    (∃! cq : Nat × Nat, cq.1 < p.cols ∧ ((schema p).host P (cL3 cq.1))[cq.2]? = some e) ∧
    (∀ c q, c < p.cols → ((schema p).host P (cL3 c))[q]? = some e →
      c = p.colC j ∧
      (schema p).stream P (cL3Fifo p c) q =
        some (.core (p.worker (p.rowC i) c) (cL1 c (p.rowC i)) (p.tauC Nt i j)
          (p.layoutC (i % p.m) (j % p.n)))) ∧
    i = p.i0 Nt (p.rowC i) (p.tauC Nt i j) + i % p.m ∧
    j = p.j0 Nt (p.colC j) (p.tauC Nt i j) + j % p.n := by
  intro P e
  have hi' : i < p.rows * p.m * Mt := hi
  have hj' : j < p.n * p.cols * Nt := hj
  have hτ := hp.tauC_lt hi' hj'
  have hrow := hp.rowC_lt i
  have hcol := hp.colC_lt j
  have ha : i % p.m < p.m := Nat.mod_lt _ hp.m_pos
  have hb : j % p.n < p.n := Nat.mod_lt _ hp.n_pos
  have di := hp.i_decomp (i := i) hj'
  have dj := hp.j_decomp (i := i) hj'
  have hNt : 0 < Nt := by
    rcases Nat.eq_zero_or_pos Nt with h | h
    · subst h; simp at hτ
    · exact h
  -- The canonical writer.
  have hw : ((schema p).host P (cL3 (p.colC j)))[wpos p (p.rowC i) (p.tauC Nt i j) (i % p.m)
      (j % p.n)]? = some e := by
    have := hostC_wpos hp Mt Kt Nt hcol hrow hτ ha hb
    rwa [di, dj] at this
  -- Any writer of `e` is the canonical one.
  have huniq : ∀ c Q, c < p.cols → ((schema p).host P (cL3 c))[Q]? = some e →
      c = p.colC j ∧ Q = wpos p (p.rowC i) (p.tauC Nt i j) (i % p.m) (j % p.n) := by
    intro c Q hc hQ
    obtain ⟨hrow', hτ', ha', hb', hQe⟩ := wpos_decode hp (hostC_lt hp hc hQ)
    set row' := Q % (p.rows * (p.m * p.n)) / (p.m * p.n)
    set τ' := Q / (p.rows * (p.m * p.n))
    set a' := p.cPosRow (Q % (p.m * p.n))
    set b' := p.cPosCol (Q % (p.m * p.n))
    have h2 := hostC_wpos hp Mt Kt Nt hc hrow' hτ' ha' hb'
    rw [← hQe, hQ] at h2
    have hidx : p.cIdx (p.Mn Mt) (p.Nn Nt) i j =
        p.cIdx (p.Mn Mt) (p.Nn Nt) (p.i0 Nt row' τ' + a') (p.j0 Nt c τ' + b') := by
      have := (Prod.mk.inj (Option.some.inj h2)).2
      exact_mod_cast this
    obtain ⟨ei, ej⟩ := p.cIdx_inj hi hj (p.i0_lt hrow' hτ' ha') (p.j0_lt hNt hc hb') hidx
    obtain ⟨oi1, oi2, -⟩ := p.own_i0 (Nt := Nt) (τ := τ') hrow' ha'
    obtain ⟨oj1, oj2, -⟩ := p.own_j0 (Nt := Nt) (τ := τ') hc hb'
    have ot := p.own_tauC (Nt := Nt) (τ := τ') hrow' hc ha' hb'
    rw [← ei] at oi1 oi2
    rw [← ej] at oj1 oj2
    rw [← ei, ← ej] at ot
    refine ⟨oj1.symm, ?_⟩
    rw [hQe, oi1, oi2, oj2, ot]
  refine ⟨⟨(p.colC j, wpos p (p.rowC i) (p.tauC Nt i j) (i % p.m) (j % p.n)), ⟨hcol, hw⟩, ?_⟩,
    ?_, di.symm, dj.symm⟩
  · rintro ⟨c, Q⟩ ⟨hc, hQ⟩
    obtain ⟨h1, h2⟩ := huniq c Q hc hQ
    simp [h1, h2]
  · intro c Q hc hQ
    obtain ⟨h1, h2⟩ := huniq c Q hc hQ
    refine ⟨h1, ?_⟩
    rw [h2, streamC_wpos hp P hc hrow ha hb]

/-- Each drain writes only elements of C in its own column group, and as many as it owns. -/
theorem tC_writes (Mt Kt Nt c : Nat) (hc : c < p.cols) :
    let h := (schema p).host (p.dims Mt Kt Nt) (cL3 c)
    h.size = Mt * Nt * (p.rows * p.m * p.n) ∧
    ∀ q < h.size, ∃ i j, i < p.Mn Mt ∧ j < p.Nn Nt ∧ p.colC j = c ∧
      h[q]? = some (2, ((p.cIdx (p.Mn Mt) (p.Nn Nt) i j : Nat) : Int)) := by
  intro h
  have hs : h.size = Mt * Nt * (p.rows * p.m * p.n) := by
    show ((schema p).host (p.dims Mt Kt Nt) (cL3 c)).size = _
    rw [Streams.hostC_size hp Mt Kt Nt hc]; unfold Streams.lenC; ring
  refine ⟨hs, fun Q hQ => ?_⟩
  have hNt : 0 < Nt := by
    rcases Nat.eq_zero_or_pos Nt with h0 | h0
    · subst h0; rw [hs] at hQ; simp at hQ
    · exact h0
  obtain ⟨hrow, hτ, ha, hb, hQe⟩ :=
    wpos_decode hp (Mt := Mt) (Nt := Nt) (by rw [hs] at hQ; calc Q < _ := hQ
      _ = Mt * Nt * (p.rows * (p.m * p.n)) := by ring)
  refine ⟨_, p.j0 Nt c (Q / (p.rows * (p.m * p.n))) + p.cPosCol (Q % (p.m * p.n)),
    p.i0_lt hrow hτ ha, p.j0_lt hNt hc hb, (p.own_j0 hc hb).1, ?_⟩
  have := hostC_wpos hp Mt Kt Nt hc hrow hτ ha hb
  rw [← hQe] at this
  exact this

end IronVerify.Schema.WholeArray.PathC
