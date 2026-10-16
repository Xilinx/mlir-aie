-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Schema.Defs
import IronVerify.Schema.Proofs.Arith
import IronVerify.Schema.Proofs.Names
import IronVerify.Schema.Proofs.NetHops
import IronVerify.Schema.Proofs.Seq

/-!
T-A for the parametric schema: every core receives the A tiles it owns, in its kernel's layout,
the right number of times. Any `rows`, `cols` (including the uneven `a_shim_rows` split), tile
and micro-tile.
-/

namespace IronVerify.Schema.WholeArray.PathA

open IronVerify IronVerify.Schema.WholeArray IronVerify.Schema.WholeArray.Names
  IronVerify.Schema.WholeArray.NetHops IronVerify.Schema.WholeArray.Streams

variable (p : Params)

/-! ### Which A shim feeds a core row -/

section Shims

variable {p}

theorem nA_pos (hp : p.Valid) : 0 < p.nA := by
  have := hp.rows_pos; have := hp.cols_pos; unfold Params.nA; omega

theorem nA_le_rows : p.nA ≤ p.rows := by unfold Params.nA; omega

theorem shimRows_pos (hp : p.Valid) (i : Nat) : 0 < p.shimRows i := by
  have h1 := nA_pos hp
  have h2 : 1 ≤ p.rows / p.nA := (Nat.one_le_div_iff h1).2 nA_le_rows
  unfold Params.shimRows; omega

/-- Core row `row` lies in the run of rows its A shim feeds. -/
theorem shimOf_spec (hp : p.Valid) {row : Nat} (hrow : row < p.rows) :
    p.shimOf row < p.nA ∧ p.shimFirst (p.shimOf row) ≤ row ∧
      row < p.shimFirst (p.shimOf row) + p.shimRows (p.shimOf row) := by
  have hn := nA_pos hp
  set q := p.rows / p.nA with hq
  set e := p.rows % p.nA with he
  have hq1 : 1 ≤ q := (Nat.one_le_div_iff hn).2 nA_le_rows
  have he_lt : e < p.nA := Nat.mod_lt _ hn
  have hrows : p.rows = p.nA * q + e := (Nat.div_add_mod _ _).symm
  unfold Params.shimOf Params.shimFirst Params.shimRows
  simp only [← hq, ← he]
  split
  · rename_i h
    have hlo : row / (q + 1) * (q + 1) ≤ row := Nat.div_mul_le_self row (q + 1)
    have hhi : row < row / (q + 1) * (q + 1) + (q + 1) := Nat.lt_div_mul_add (by omega)
    have hie : row / (q + 1) < e := (Nat.div_lt_iff_lt_mul (by omega)).2 h
    generalize row / (q + 1) = i at hlo hhi hie
    have hmin : min i e = i := by omega
    rw [hmin, ite_eq_left_iff.mpr (fun hn => absurd hie hn)]
    have hexp : i * (q + 1) = i * q + i := by ring
    refine ⟨by omega, by omega, by omega⟩
  · rename_i h
    have h' : e * (q + 1) ≤ row := by omega
    have hlo : (row - e * (q + 1)) / q * q ≤ row - e * (q + 1) := Nat.div_mul_le_self _ q
    have hhi : row - e * (q + 1) < (row - e * (q + 1)) / q * q + q := Nat.lt_div_mul_add (by omega)
    have hxlt : row - e * (q + 1) < (p.nA - e) * q := by
      have h1 : p.nA * q = (p.nA - e) * q + e * q := by
        rw [← Nat.add_mul, Nat.sub_add_cancel he_lt.le]
      have h2 : e * (q + 1) = e * q + e := by ring
      omega
    have hdlt : (row - e * (q + 1)) / q < p.nA - e := (Nat.div_lt_iff_lt_mul (by omega)).2 hxlt
    generalize (row - e * (q + 1)) / q = d at hlo hhi hdlt
    have hmin : min (e + d) e = e := by omega
    rw [hmin, ite_eq_right_iff.mpr (fun hc => absurd hc (by omega))]
    have hexp : (e + d) * q + e = e * (q + 1) + d * q := by ring
    refine ⟨by omega, by omega, by omega⟩

end Shims

variable (hp : p.Valid)
include hp

theorem tA (Mt Kt Nt row col τ kk a b : Nat) (hrow : row < p.rows) (hcol : col < p.cols)
    (hτ : τ < Mt * Nt) (hkk : kk < Kt) (ha : a < p.m) (hb : b < p.k) :
    (schema p).deliver (p.dims Mt Kt Nt) (aL2Fifo p row) (p.worker row col) (τ * Kt + kk)
        (p.layoutA a b) =
      some (.host 0 ((p.aIdx (p.Kn Kt) (p.i0 Nt row τ + a) (kk * p.k + b) : Nat) : Int)) := by
  obtain ⟨hi, hfirst, hlast⟩ := shimOf_spec hp hrow
  set i := p.shimOf row with hi_def
  set d := row - p.shimFirst i with hd_def
  set R := p.shimRows i with hR
  have hdR : d < R := by omega
  have hNt : 0 < Nt := by
    rcases Nat.eq_zero_or_pos Nt with h | h
    · subst h; simp at hτ
    · exact h
  have hm := hp.m_pos
  have hk := hp.k_pos
  have hL := hp.layoutA_lt ha hb
  set L := p.layoutA a b with hL_def
  have hmk : 0 < p.m * p.k := Nat.mul_pos hm hk
  -- core slot -> A_L2L1 stream -> A_L3L2 host stream
  show (wnet p).deliver ((schema p).host (p.dims Mt Kt Nt)) (aL2Fifo p row) (p.worker row col)
      (τ * Kt + kk) L = _
  rw [deliver_aL2 p _ hcol, stream_aL2 p _ hp hi hi_def.symm (idxOf?_aLink p hfirst hlast)
    (fifo?_aL3 p hi), Arith.mix_div hL, Arith.mix_mod hL, hp.walkA_idx ha hb, Int.toNat_natCast]
  -- the host position as row block `τ / Nt`, then the tap's mixed-radix digits
  have hτd := Nat.div_add_mod τ Nt
  have hnt : τ % Nt < Nt := Nat.mod_lt _ hNt
  have hrb : τ / Nt < Mt := (Nat.div_lt_iff_lt_mul hNt).2 hτ
  have ha' : d * p.m + a < R * p.m := by
    have := Arith.mix_lt hdR ha; exact this
  have hq : ((τ % Nt * Kt + kk) * (R * p.m) + (d * p.m + a)) * p.k + b <
      Streams.lenA p Kt Nt i := by
    unfold Streams.lenA
    exact Arith.mix_lt (Arith.mix_lt (Arith.mix_lt hnt hkk) ha') hb
  have hpos : (τ * Kt + kk) * (p.m * p.k * R) + (p.m * p.k * d + (a * p.k + b)) =
      τ / Nt * Streams.lenA p Kt Nt i +
        (((τ % Nt * Kt + kk) * (R * p.m) + (d * p.m + a)) * p.k + b) := by
    unfold Streams.lenA
    conv_lhs => rw [← hτd]
    ring
  rw [hpos, hostA_get hp Mt Kt Nt hi hrb hq, Option.map_some,
    tapA_idx hp Kt Nt _ i hnt hkk ha' hb]
  congr 3
  unfold Params.aIdx Params.i0
  have : p.shimFirst i * p.m + (d * p.m + a) = row * p.m + a := by
    have : p.shimFirst i + d = row := by omega
    rw [← this]; ring
  rw [show τ / Nt * (p.rows * p.m) + p.shimFirst i * p.m + (d * p.m + a) =
      τ / Nt * (p.rows * p.m) + (row * p.m + a) by omega]
  dsimp only
  congr 1
  ring

/-- Each core receives exactly the objects it consumes: `tiles · Kt` of each operand. -/
theorem countA (Mt Kt Nt row : Nat) (hrow : row < p.rows) :
    (wnet p).streamLen ((schema p).host (p.dims Mt Kt Nt)) (aL2Fifo p row) =
      some (Mt * Nt * Kt * (p.m * p.k)) := by
  obtain ⟨hi, -, -⟩ := shimOf_spec hp hrow
  have hR := shimRows_pos hp (p.shimOf row)
  have hmk : 0 < p.m * p.k := Nat.mul_pos hp.m_pos hp.k_pos
  rw [streamLen_aL2 p _ hp hi rfl (fifo?_aL3 p hi), hostA_size hp Mt Kt Nt hi]
  unfold Streams.lenA
  congr 1
  have : Mt * (Nt * Kt * (p.shimRows (p.shimOf row) * p.m) * p.k) =
      Mt * Nt * Kt * (p.m * p.k * p.shimRows (p.shimOf row)) := by ring
  rw [this, Nat.mul_div_cancel _ (Nat.mul_pos hmk hR)]

end IronVerify.Schema.WholeArray.PathA
