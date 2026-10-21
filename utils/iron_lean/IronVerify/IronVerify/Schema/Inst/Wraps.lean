-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Schema.Inst.Transfer
import IronVerify.Schema.Overflow
import IronVerify.Schema.Proofs.Seq

/-!
The width obligations of an export (`wraps`, emitted by `export.py --wraps`) follow from its own
guards: whole_array's seven `require`s pin the shape and keep the runtime-parameter product
inside `i32`, and its dispatch values are `i32`. With every obligation met,
`IExpr.machine_eq_exact` makes the exported exact terms the machine's.
-/

namespace IronVerify.Schema.WholeArray

open IronVerify

theorem fits_of {w : Nat} {x : Int} (h1 : -(2 : Int) ^ (w - 1) ≤ x) (h2 : x < 2 ^ (w - 1)) :
    fitsS w x = true := by
  simp [fitsS, h1, h2]

theorem bounds_of_fits {w : Nat} {x : Int} (h : fitsS w x = true) :
    -(2 : Int) ^ (w - 1) ≤ x ∧ x < 2 ^ (w - 1) := by
  simpa [fitsS] using h

theorem ok_seq {a b : Seq} (ha : a.ok = true) (hb : b.ok = true) : (Seq.seq a b).ok = true := by
  simp [Seq.ok, ha, hb]

theorem ok_check {c : Bool} (h : c = true) : (Seq.check c).ok = true := h

theorem ok_skip : Seq.skip.ok = true := rfl

theorem ok_ite {c : Bool} {t e : Seq} (ht : c = true → t.ok = true)
    (he : c = false → e.ok = true) : (Seq.ite c t e).ok = true := by
  cases c <;> simp_all [Seq.ok]

theorem ok_loop {lo hi : Int} {body : Int → Seq}
    (h : ∀ i : Nat, i < (hi - lo).toNat → (body (lo + (i : Int))).ok = true) :
    (Seq.loop lo hi body).ok = true := by
  unfold Seq.ok
  rw [List.all_eq_true]
  intro x hx
  simp only [List.mem_map, List.mem_flatMap, List.mem_range, List.mem_singleton,
    List.pure_def, List.bind_eq_flatMap] at hx
  obtain ⟨_, ⟨i, hi, rfl⟩, rfl⟩ := hx
  exact h i hi

/-- Descends a `Seq` goal exactly once per node (`<;>` recurses into each subgoal `ok_seq` /
`ok_ite` / `ok_loop` produce), instead of repeatedly rescanning every open goal: the
`iterate 600 (all_goals (try (first | ...)))` shape this replaces re-tries already-closed
siblings on every pass, which blows up memory on the deeper `Seq` trees wider column counts
unroll. Leaves one `c = true` goal per `.check` leaf, for the caller to discharge. -/
syntax "close_ok" : tactic

-- Unhygienic so the `i`, `hi` and `hc` it introduces are visible to the caller's closing steps.
set_option hygiene false in
macro_rules
  | `(tactic| close_ok) =>
    `(tactic| first
      | (with_reducible apply ok_seq) <;> close_ok
      | with_reducible exact ok_skip
      | ((with_reducible apply ok_ite) <;> (intro hc <;> close_ok))
      | ((with_reducible apply ok_loop) <;> (intro i hi <;> close_ok))
      | with_reducible apply ok_check)

theorem fdiv_max_bounds (x : Int) (hx : 0 < x) :
    0 ≤ Int.fdiv 2147483647 x ∧ Int.fdiv 2147483647 x ≤ 2147483647 := by
  rw [Int.fdiv_eq_ediv]
  simp only [hx.le, true_or, ite_true, sub_zero]
  exact ⟨Int.ediv_nonneg (by norm_num) hx.le, Int.ediv_le_self _ (by norm_num)⟩

/-- An export opens with whole_array's seven `require` guards; they pin the shape and bound the
runtime-parameter product `(M / m) · (N / n)`. -/
theorem shape_of_prefix (p : Params) (hp : p.Valid) {d : Design (Int × Int × Int)}
    (dm dk dn m n : Int)
    (hdm : dm = ((p.rows * p.m : Nat) : Int)) (hdk : dk = ((p.k : Nat) : Int))
    (hdn : dn = ((p.n * p.cols : Nat) : Int)) (hm : m = ((p.m : Nat) : Int))
    (hn : n = ((p.n : Nat) : Int))
    (hs : ∀ M K N : Int, ∃ rest : Seq, d.seq (M, K, N) =
      .seq (.check (decide (M > 0))) (.seq (.check (decide (K > 0)))
      (.seq (.check (decide (N > 0))) (.seq (.check (decide (Int.tmod M dm = 0)))
      (.seq (.check (decide (Int.tmod K dk = 0)))
      (.seq (.check (decide (Int.tmod N dn = 0)))
      (.seq (.check (decide (Int.fdiv M m ≤ Int.fdiv 2147483647 (Int.fdiv N n)))) rest))))))) :
    ∀ P : Int × Int × Int, (d.seq P).ok = true →
      ∃ Mt Kt Nt : Nat, 0 < Mt ∧ 0 < Kt ∧ 0 < Nt ∧ P = p.shapeOf Mt Kt Nt ∧
        p.rows * Mt * (p.cols * Nt) < 2 ^ 31 := by
  rintro ⟨M, K, N⟩ h
  obtain ⟨rest, hr⟩ := hs M K N
  rw [hr] at h
  simp only [Seq.ok, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨hM, hK, hN, hmd, hkd, hnd, hb, -⟩ := h
  have hm0 : 0 < dm := by subst hdm; have := hp.rows_pos; have := hp.m_pos; positivity
  have hk0 : 0 < dk := by subst hdk; have := hp.k_pos; positivity
  have hn0 : 0 < dn := by subst hdn; have := hp.n_pos; have := hp.cols_pos; positivity
  obtain ⟨Mt, hMt, rfl⟩ := multiple_of_tmod hm0 hM hmd
  obtain ⟨Kt, hKt, rfl⟩ := multiple_of_tmod hk0 hK hkd
  obtain ⟨Nt, hNt, rfl⟩ := multiple_of_tmod hn0 hN hnd
  subst hdm hdk hdn hm hn
  have hmp : (0 : Int) < (p.m : Nat) := by have := hp.m_pos; positivity
  have hnp : (0 : Int) < (p.n : Nat) := by have := hp.n_pos; positivity
  have e1 : Int.fdiv (((p.rows * p.m : Nat) : Int) * Mt) (p.m : Nat) =
      ((p.rows * p.m * Mt : Nat) : Int) / (p.m : Nat) := by
    rw [Int.fdiv_eq_ediv]; simp [hmp.le]
  have e2 : Int.fdiv (((p.n * p.cols : Nat) : Int) * Nt) (p.n : Nat) =
      ((p.n * p.cols * Nt : Nat) : Int) / (p.n : Nat) := by
    rw [Int.fdiv_eq_ediv]; simp [hnp.le]
  have e3 : ∀ x : Int, 0 ≤ x → Int.fdiv 2147483647 x = (2 ^ 31 - 1) / x := by
    intro x hx; rw [Int.fdiv_eq_ediv]; simp [hx]
  have hq : (0 : Int) ≤ ((p.n * p.cols * Nt : Nat) : Int) / (p.n : Nat) := by positivity
  rw [e1, e2, e3 _ hq] at hb
  refine ⟨Mt, Kt, Nt, hMt, hKt, hNt, ?_, (Streams.rtp_bound_iff hp Mt Nt hNt).mp hb⟩
  simp only [Params.shapeOf]; push_cast; ring_nf

end IronVerify.Schema.WholeArray
