-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Schema.Defs
import IronVerify.Schema.Proofs.Names
import IronVerify.Schema.Proofs.NetHops
import IronVerify.Schema.Proofs.Seq
import IronVerify.Schema.Proofs.Arith

/-!
T-B for the schema: every core in column `col` receives B's `k × n` tiles of its column group,
`kk = 0 … Kt-1` per output tile, in its kernel's `(s × t)` layout (`(t × s)` of Bᵀ with
`bColMaj`), and exactly `tiles · Kt` of them.
-/

namespace IronVerify.Schema.WholeArray.PathB

open IronVerify IronVerify.Schema.WholeArray

variable (p : Params)

variable (hp : p.Valid)
include hp

theorem tB (Mt Kt Nt row col τ kk a b : Nat) (hrow : row < p.rows) (hcol : col < p.cols)
    (hτ : τ < Mt * Nt) (hkk : kk < Kt) (ha : a < p.k) (hb : b < p.n) :
    (schema p).deliver (p.dims Mt Kt Nt) (bL2Fifo p col) (p.worker row col) (τ * Kt + kk)
        (p.layoutB a b) =
      some (.host 1 ((p.bIdx (p.Kn Kt) (p.Nn Nt) (kk * p.k + a) (p.j0 Nt col τ + b) : Nat) : Int)) := by
  have hkn : 0 < p.k * p.n := Nat.mul_pos hp.k_pos hp.n_pos
  have hNt : 0 < Nt := Nat.pos_of_ne_zero (by rintro rfl; simp at hτ)
  have hs := hp.layoutB_lt ha hb
  set e : Nat := if p.bColMaj then b * p.k + a else a * p.n + b with he
  have he_lt : e < p.k * p.n := by
    rw [he]; split
    · calc b * p.k + a < b * p.k + p.k := by omega
        _ = (b + 1) * p.k := by ring
        _ ≤ p.n * p.k := Nat.mul_le_mul_right _ hb
        _ = p.k * p.n := by ring
    · calc a * p.n + b < a * p.n + p.n := by omega
        _ = (a + 1) * p.n := by ring
        _ ≤ p.k * p.n := Nat.mul_le_mul_right _ ha
  -- Core slot → B channel position.
  have hdeliv : (schema p).deliver (p.dims Mt Kt Nt) (bL2Fifo p col) (p.worker row col)
      (τ * Kt + kk) (p.layoutB a b) =
      ((schema p).host (p.dims Mt Kt Nt) (bL3 col))[(τ * Kt + kk) * (p.k * p.n) + e]?.map
        fun x => .host x.1 x.2 := by
    show (wnet p).deliver _ _ _ _ _ = _
    rw [NetHops.deliver_bL2 p _ hrow, NetHops.stream_bL2 p _ hp hcol (Names.fifo?_bL3 p hcol)]
    have h1 : ((τ * Kt + kk) * (p.k * p.n) + p.layoutB a b) / (p.k * p.n) = τ * Kt + kk := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hkn, Nat.div_eq_of_lt hs, Nat.zero_add]
    have h2 : ((τ * Kt + kk) * (p.k * p.n) + p.layoutB a b) % (p.k * p.n) = p.layoutB a b := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hs]
    rw [h1, h2, hp.walkB_idx ha hb, Int.toNat_natCast]
  -- Split the position into row block `τ / Nt` and the offset inside that block's transfer.
  set rb := τ / Nt
  set nt := τ % Nt
  have hrb : rb < Mt := Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hτ)
  have hnt : nt < Nt := Nat.mod_lt _ hNt
  have hτ' : τ = rb * Nt + nt := (Nat.div_add_mod' τ Nt).symm
  have hq_lt : (nt * Kt + kk) * (p.k * p.n) + e < Streams.lenB p Kt Nt := by
    unfold Streams.lenB
    calc (nt * Kt + kk) * (p.k * p.n) + e < (nt * Kt + kk) * (p.k * p.n) + p.k * p.n := by omega
      _ = (nt * Kt + kk + 1) * (p.k * p.n) := by ring
      _ ≤ Nt * Kt * (p.k * p.n) := Nat.mul_le_mul_right _ (by nlinarith)
  have hpos : (τ * Kt + kk) * (p.k * p.n) + e =
      rb * Streams.lenB p Kt Nt + ((nt * Kt + kk) * (p.k * p.n) + e) := by
    unfold Streams.lenB; rw [hτ']; ring
  rw [hdeliv, hpos, Streams.hostB_get hp Mt Kt Nt hcol hrb hq_lt]
  simp only [Option.map_some, Option.some.injEq, Src.host.injEq, true_and]
  have hj0 : p.j0 Nt col τ = nt * (p.n * p.cols) + col * p.n := rfl
  obtain ⟨hrm, hcm⟩ := Streams.tapB_idx hp Kt Nt col hnt hkk (a := a) (b := b)
  obtain ⟨hrm', hcm'⟩ := Streams.tapB_idx hp Kt Nt col hnt hkk (a := b) (b := a)
  unfold Params.bIdx
  cases hbc : p.bColMaj
  · simp only [hbc, he, Bool.false_eq_true, ite_false] at hrm ⊢
    rw [show (nt * Kt + kk) * (p.k * p.n) + (a * p.n + b) = ((nt * Kt + kk) * p.k + a) * p.n + b
      by ring, hrm trivial ha hb, hj0]
    congr 1; ring
  · simp only [hbc, he, ite_true] at hcm' ⊢
    rw [show (nt * Kt + kk) * (p.k * p.n) + (b * p.k + a) = ((nt * Kt + kk) * p.n + b) * p.k + a
      by ring, hcm' trivial hb ha, hj0]
    congr 1; ring

theorem countB (Mt Kt Nt c : Nat) (hc : c < p.cols) :
    (wnet p).streamLen ((schema p).host (p.dims Mt Kt Nt)) (bL2Fifo p c) =
      some (Mt * Nt * Kt * (p.k * p.n)) := by
  have hkn : 0 < p.k * p.n := Nat.mul_pos hp.k_pos hp.n_pos
  rw [NetHops.streamLen_bL2 p _ hp hc (Names.fifo?_bL3 p hc), Streams.hostB_size hp Mt Kt Nt hc]
  unfold Streams.lenB
  rw [show Mt * (Nt * Kt * (p.k * p.n)) = Mt * Nt * Kt * (p.k * p.n) by ring,
    Nat.mul_div_cancel _ hkn]

end IronVerify.Schema.WholeArray.PathB
