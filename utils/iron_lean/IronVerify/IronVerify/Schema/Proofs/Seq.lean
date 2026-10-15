-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Lemmas
import IronVerify.Schema.Defs
import IronVerify.Schema.Proofs.Names

/-!
The schema's runtime sequence in closed form: each shim channel's host stream is one walk per
row block (`tapA`, `tapB`, `tapC`), every row block's walk has the same length, and the
design's guards accept exactly the shapes `dims p Mt Kt Nt`.
-/

namespace IronVerify.Schema.WholeArray.Streams

open IronVerify IronVerify.Schema.WholeArray IronVerify.Schema.WholeArray.Names

theorem stream_ofList (ch : String) :
    ∀ l : List Seq, (Seq.ofList l).stream ch = (l.map (Seq.stream ch)).flatten
  | [] => by simp [Seq.ofList, Seq.stream]
  | x :: xs => by simp [Seq.ofList, Seq.stream, stream_ofList ch xs]

theorem flatMap_natCast (l : List Nat) :
    List.flatMap (fun a : Nat => [(a : Int)]) l = l.map (Nat.cast : Nat → Int) := by
  induction l with
  | nil => rfl
  | cons x l ih => rw [List.flatMap_cons, ih]; rfl

theorem ok_ofList : ∀ l : List Seq, (Seq.ofList l).ok = l.all Seq.ok
  | [] => by simp [Seq.ofList, Seq.ok]
  | x :: xs => by simp [Seq.ofList, Seq.ok, ok_ofList xs]

/-- Concatenating over `c < n` where only `c = i` contributes. -/
theorem flatten_range_single {α : Type} (g : Nat → List α) {i : Nat} :
    ∀ {n : Nat}, i < n → (∀ c, c ≠ i → g c = []) → ((List.range n).map g).flatten = g i
  | 0, hi, _ => absurd hi (Nat.not_lt_zero _)
  | n + 1, hi, h => by
    rw [List.range_succ, List.map_append, List.flatten_append]
    by_cases hin : i < n
    · rw [flatten_range_single g hin h]
      simp [h n (by omega)]
    · have : i = n := by omega
      subst this
      have hnil : ((List.range i).map g).flatten = [] := by
        rw [List.flatten_eq_nil_iff]
        intro l hl
        obtain ⟨c, hc, rfl⟩ := List.mem_map.mp hl
        exact h c (by simp at hc; omega)
      simp [hnil]

variable (p : Params)

/-- Row block `rb`'s transfers on one channel. -/
theorem step_stream (ch : String) (M K N rb : Int) :
    (step p M K N rb).stream ch =
      ((List.range p.cols).map fun c =>
        (Seq.xfer (cL3 c) 2 (tapC p M N rb c)).stream ch ++
        ((if c < p.nA then [Seq.xfer (aL3 c) 0 (tapA p K N rb c)] else []).map
          (Seq.stream ch)).flatten ++
        (Seq.xfer (bL3 c) 1 (tapB p K N c)).stream ch).flatten := by
  simp only [step, stream_ofList, List.flatMap, List.map_flatten, List.flatten_flatten,
    List.map_map]
  congr 1
  refine List.map_congr_left fun c _ => ?_
  simp [List.flatten_append]

/-- The sequence's host stream on `ch`: the row-block loop's, the guards move nothing. -/
theorem wseq_stream (ch : String) (M K N : Int) :
    (wseq p M K N).stream ch =
      ((List.range (M / (p.rows * p.m : Nat)).toNat).map fun (rb : Nat) =>
        (step p M K N rb).stream ch).flatten := by
  simp [wseq, stream_ofList, Seq.stream, flatMap_natCast, Function.comp_def]

theorem nA_le_cols : p.nA ≤ p.cols := Nat.min_le_right _ _

theorem stream_aL3 {i : Nat} (hi : i < p.nA) (M K N rb : Int) :
    (step p M K N rb).stream (aL3 i) = (tapA p K N rb i).toList.map (0, ·) := by
  rw [step_stream]
  rw [flatten_range_single _ (Nat.lt_of_lt_of_le hi (nA_le_cols p))]
  · simp [Seq.stream, hi]
  · intro c hc
    by_cases h : c < p.nA <;> simp [Seq.stream, h, hc]

theorem stream_bL3 {c : Nat} (hc : c < p.cols) (M K N rb : Int) :
    (step p M K N rb).stream (bL3 c) = (tapB p K N c).toList.map (1, ·) := by
  rw [step_stream, flatten_range_single _ hc]
  · by_cases h : c < p.nA <;> simp [Seq.stream, h]
  · intro c' hc'
    by_cases h : c' < p.nA <;> simp [Seq.stream, h, hc']

theorem stream_cL3 {c : Nat} (hc : c < p.cols) (M K N rb : Int) :
    (step p M K N rb).stream (cL3 c) = (tapC p M N rb c).toList.map (2, ·) := by
  rw [step_stream, flatten_range_single _ hc]
  · by_cases h : c < p.nA <;> simp [Seq.stream, h]
  · intro c' hc'
    by_cases h : c' < p.nA <;> simp [Seq.stream, h, hc']

/-! ### Closed forms, for any dispatch values -/

theorem host_aL3 {i : Nat} (hi : i < p.nA) (M K N : Int) :
    (wseq p M K N).stream (aL3 i) =
      ((List.range (M / (p.rows * p.m : Nat)).toNat).map fun (rb : Nat) =>
        (tapA p K N rb i).toList.map (0, ·)).flatten := by
  rw [wseq_stream]
  simp only [stream_aL3 p hi]

theorem host_bL3 {c : Nat} (hc : c < p.cols) (M K N : Int) :
    (wseq p M K N).stream (bL3 c) =
      ((List.range (M / (p.rows * p.m : Nat)).toNat).map fun (_ : Nat) =>
        (tapB p K N c).toList.map (1, ·)).flatten := by
  rw [wseq_stream]
  simp only [stream_bL3 p hc]

theorem host_cL3 {c : Nat} (hc : c < p.cols) (M K N : Int) :
    (wseq p M K N).stream (cL3 c) =
      ((List.range (M / (p.rows * p.m : Nat)).toNat).map fun (rb : Nat) =>
        (tapC p M N rb c).toList.map (2, ·)).flatten := by
  rw [wseq_stream]
  simp only [stream_cL3 p hc]

/-! Every row block's walk has the same length. -/

theorem tapA_len (K N rb : Int) (i : Nat) : (tapA p K N rb i).len = (tapA p K N 0 i).len := rfl
theorem tapC_len (M N rb : Int) (c : Nat) : (tapC p M N rb c).len = (tapC p M N 0 c).len := by
  unfold tapC; split <;> rfl

theorem get_aL3 {i : Nat} (hi : i < p.nA) (M K N : Int) {rb q : Nat}
    (hrb : rb < (M / (p.rows * p.m : Nat)).toNat) (hq : q < (tapA p K N 0 i).len) :
    ((wseq p M K N).stream (aL3 i))[rb * (tapA p K N 0 i).len + q]? =
      some (0, (tapA p K N rb i).idx q) := by
  rw [host_aL3 p hi, getElem?_flatten_range _ _ _ (fun _ _ => by simp [Walk.length_toList,
    tapA_len]) hrb hq]
  simp [Walk.getElem?_toList, tapA_len p K N rb i, hq]

theorem get_bL3 {c : Nat} (hc : c < p.cols) (M K N : Int) {rb q : Nat}
    (hrb : rb < (M / (p.rows * p.m : Nat)).toNat) (hq : q < (tapB p K N c).len) :
    ((wseq p M K N).stream (bL3 c))[rb * (tapB p K N c).len + q]? =
      some (1, (tapB p K N c).idx q) := by
  rw [host_bL3 p hc, getElem?_flatten_range _ _ _ (fun _ _ => by simp [Walk.length_toList])
    hrb hq]
  simp [Walk.getElem?_toList, hq]

theorem get_cL3 {c : Nat} (hc : c < p.cols) (M K N : Int) {rb q : Nat}
    (hrb : rb < (M / (p.rows * p.m : Nat)).toNat) (hq : q < (tapC p M N 0 c).len) :
    ((wseq p M K N).stream (cL3 c))[rb * (tapC p M N 0 c).len + q]? =
      some (2, (tapC p M N rb c).idx q) := by
  rw [host_cL3 p hc, getElem?_flatten_range _ _ _ (fun _ _ => by simp [Walk.length_toList,
    tapC_len]) hrb hq]
  simp [Walk.getElem?_toList, tapC_len p M N rb c, hq]

theorem length_aL3 {i : Nat} (hi : i < p.nA) (M K N : Int) :
    ((wseq p M K N).stream (aL3 i)).length =
      (M / (p.rows * p.m : Nat)).toNat * (tapA p K N 0 i).len := by
  rw [host_aL3 p hi, length_flatten_range _ (tapA p K N 0 i).len _ (fun _ _ => by simp [Walk.length_toList,
    tapA_len])]

theorem length_bL3 {c : Nat} (hc : c < p.cols) (M K N : Int) :
    ((wseq p M K N).stream (bL3 c)).length =
      (M / (p.rows * p.m : Nat)).toNat * (tapB p K N c).len := by
  rw [host_bL3 p hc, length_flatten_range _ (tapB p K N c).len _ (fun _ _ => by simp [Walk.length_toList])]

theorem length_cL3 {c : Nat} (hc : c < p.cols) (M K N : Int) :
    ((wseq p M K N).stream (cL3 c)).length =
      (M / (p.rows * p.m : Nat)).toNat * (tapC p M N 0 c).len := by
  rw [host_cL3 p hc, length_flatten_range _ (tapC p M N 0 c).len _ (fun _ _ => by simp [Walk.length_toList,
    tapC_len])]

/-! ### At the guarded shapes `dims p Mt Kt Nt` -/

/-- Host elements per row block on A shim `i`, B shim `c` and C drain `c`. -/
def lenA (Kt Nt i : Nat) : Nat := Nt * Kt * (p.shimRows i * p.m) * p.k
def lenB (Kt Nt : Nat) : Nat := Nt * Kt * (p.k * p.n)
def lenC (Nt : Nat) : Nat := Nt * (p.rows * p.m * p.n)

theorem host_dims (Mt Kt Nt : Nat) (ch : String) :
    (schema p).host (p.dims Mt Kt Nt) ch =
      ((wseq p (p.Mn Mt) (p.Kn Kt) (p.Nn Nt)).stream ch).toArray := rfl

variable {p} (hp : p.Valid)
include hp

theorem rowBlocks_dims (Mt : Nat) :
    (((p.Mn Mt : Nat) : Int) / ((p.rows * p.m : Nat) : Int)).toNat = Mt := by
  have h : ((p.rows * p.m : Nat) : Int) ≠ 0 := by
    have := hp.rows_pos; have := hp.m_pos; positivity
  unfold Params.Mn
  rw [Nat.cast_mul (p.rows * p.m), Int.mul_ediv_cancel_left _ h, Int.toNat_natCast]

theorem kTiles_dims (Kt : Nat) : ((p.Kn Kt : Nat) : Int) / ((p.k : Nat) : Int) = Kt := by
  have h : ((p.k : Nat) : Int) ≠ 0 := by have := hp.k_pos; positivity
  unfold Params.Kn
  rw [Nat.cast_mul, Int.mul_ediv_cancel_left _ h]

theorem nTiles_dims (Nt : Nat) : nTiles p ((p.Nn Nt : Nat) : Int) = Nt := by
  have h : ((p.n * p.cols : Nat) : Int) ≠ 0 := by
    have := hp.n_pos; have := hp.cols_pos; positivity
  unfold nTiles Params.Nn
  rw [Nat.cast_mul (p.n * p.cols), Int.mul_ediv_cancel_left _ h]

theorem tapA_len_dims (Kt Nt i : Nat) :
    (tapA p (p.Kn Kt) (p.Nn Nt) 0 i).len = lenA p Kt Nt i := by
  simp only [tapA, Walk.len, dimsLen, nTiles_dims hp, kTiles_dims hp, lenA, Int.toNat_natCast]
  ring

theorem tapB_len_dims (Kt Nt c : Nat) :
    (tapB p (p.Kn Kt) (p.Nn Nt) c).len = lenB p Kt Nt := by
  unfold tapB
  split <;>
  · simp only [Walk.len, dimsLen, nTiles_dims hp, kTiles_dims hp, lenB, Int.toNat_natCast]
    ring

theorem tapC_len_dims (Mt Nt c : Nat) :
    (tapC p (p.Mn Mt) (p.Nn Nt) 0 c).len = lenC p Nt := by
  unfold tapC
  split <;>
  · simp only [Walk.len, dimsLen, nTiles_dims hp, lenC, Int.toNat_natCast]
    ring

/-- Element `q` of row block `rb` on A shim `i`. -/
theorem hostA_get (Mt Kt Nt : Nat) {i rb q : Nat} (hi : i < p.nA) (hrb : rb < Mt)
    (hq : q < lenA p Kt Nt i) :
    ((schema p).host (p.dims Mt Kt Nt) (aL3 i))[rb * lenA p Kt Nt i + q]? =
      some (0, (tapA p (p.Kn Kt) (p.Nn Nt) rb i).idx q) := by
  rw [host_dims, List.getElem?_toArray, ← tapA_len_dims hp]
  exact get_aL3 p hi _ _ _ (by rw [rowBlocks_dims hp]; exact hrb)
    (by rw [tapA_len_dims hp]; exact hq)

theorem hostB_get (Mt Kt Nt : Nat) {c rb q : Nat} (hc : c < p.cols) (hrb : rb < Mt)
    (hq : q < lenB p Kt Nt) :
    ((schema p).host (p.dims Mt Kt Nt) (bL3 c))[rb * lenB p Kt Nt + q]? =
      some (1, (tapB p (p.Kn Kt) (p.Nn Nt) c).idx q) := by
  rw [host_dims, List.getElem?_toArray, ← tapB_len_dims hp Kt Nt c]
  exact get_bL3 p hc _ _ _ (by rw [rowBlocks_dims hp]; exact hrb)
    (by rw [tapB_len_dims hp]; exact hq)

theorem hostC_get (Mt Kt Nt : Nat) {c rb q : Nat} (hc : c < p.cols) (hrb : rb < Mt)
    (hq : q < lenC p Nt) :
    ((schema p).host (p.dims Mt Kt Nt) (cL3 c))[rb * lenC p Nt + q]? =
      some (2, (tapC p (p.Mn Mt) (p.Nn Nt) rb c).idx q) := by
  rw [host_dims, List.getElem?_toArray, ← tapC_len_dims hp Mt Nt c]
  exact get_cL3 p hc _ _ _ (by rw [rowBlocks_dims hp]; exact hrb)
    (by rw [tapC_len_dims hp]; exact hq)

theorem hostA_size (Mt Kt Nt : Nat) {i : Nat} (hi : i < p.nA) :
    ((schema p).host (p.dims Mt Kt Nt) (aL3 i)).size = Mt * lenA p Kt Nt i := by
  rw [host_dims, List.size_toArray, length_aL3 p hi, rowBlocks_dims hp, tapA_len_dims hp]

theorem hostB_size (Mt Kt Nt : Nat) {c : Nat} (hc : c < p.cols) :
    ((schema p).host (p.dims Mt Kt Nt) (bL3 c)).size = Mt * lenB p Kt Nt := by
  rw [host_dims, List.size_toArray, length_bL3 p hc, rowBlocks_dims hp, tapB_len_dims hp]

theorem hostC_size (Mt Kt Nt : Nat) {c : Nat} (hc : c < p.cols) :
    ((schema p).host (p.dims Mt Kt Nt) (cL3 c)).size = Mt * lenC p Nt := by
  rw [host_dims, List.size_toArray, length_cL3 p hc, rowBlocks_dims hp, tapC_len_dims hp]

/-- Position `((nt·Kt + kk)·(rows of shim i)·m + a)·k + b` of row block `rb` on A shim `i`
reads `A[rb·rows·m + shimFirst i·m + a, kk·k + b]` (row-major, `K` columns). -/
theorem tapA_idx (Kt Nt : Nat) (rb i : Nat) {nt kk a b : Nat} (hnt : nt < Nt) (hkk : kk < Kt)
    (ha : a < p.shimRows i * p.m) (hb : b < p.k) :
    (tapA p (p.Kn Kt) (p.Nn Nt) rb i).idx
        (((nt * Kt + kk) * (p.shimRows i * p.m) + a) * p.k + b) =
      (((rb * (p.rows * p.m) + p.shimFirst i * p.m + a) * p.Kn Kt + kk * p.k + b : Nat) : Int) := by
  have hw : tapA p (p.Kn Kt) (p.Nn Nt) rb i =
      { offset := (rb : Int) * (p.rows * p.m : Nat) * (p.Kn Kt : Nat) +
          (p.shimFirst i * p.m : Nat) * (p.Kn Kt : Nat),
        dims := [((Nt : Nat), 0), ((Kt : Nat), (p.k : Nat)), ((p.shimRows i * p.m : Nat),
          (p.Kn Kt : Nat)), ((p.k : Nat), 1)] } := by
    simp only [tapA, nTiles_dims hp, kTiles_dims hp]
  have hq : ((nt * Kt + kk) * (p.shimRows i * p.m) + a) * p.k + b =
      encode (tapA p (p.Kn Kt) (p.Nn Nt) rb i).dims [nt, kk, a, b] := by
    rw [hw]; simp only [encode, dimsLen, Int.toNat_natCast]; ring
  have hfits : Fits (tapA p (p.Kn Kt) (p.Nn Nt) rb i).dims [nt, kk, a, b] := by
    rw [hw]; simp only [Fits, Int.toNat_natCast]; exact ⟨hnt, hkk, ha, hb, trivial⟩
  rw [hq, Walk.idx_encode _ hfits, hw]
  simp only [dot, Params.Kn]
  push_cast
  ring

/-- B shim `c`'s column tile `nt`, k-step `kk`: position `((nt·Kt + kk)·k + a)·n + b` reads
`B[kk·k + a, (nt·cols + c)·n + b]`; with `bColMaj` (B given as `N × K`) position
`((nt·Kt + kk)·n + a)·k + b` reads `Bᵀ[(nt·cols + c)·n + a, kk·k + b]`. -/
theorem tapB_idx (Kt Nt c : Nat) {nt kk a b : Nat} (hnt : nt < Nt) (hkk : kk < Kt) :
    (p.bColMaj = false → a < p.k → b < p.n →
      (tapB p (p.Kn Kt) (p.Nn Nt) c).idx (((nt * Kt + kk) * p.k + a) * p.n + b) =
        (((kk * p.k + a) * p.Nn Nt + (nt * p.cols + c) * p.n + b : Nat) : Int)) ∧
    (p.bColMaj = true → a < p.n → b < p.k →
      (tapB p (p.Kn Kt) (p.Nn Nt) c).idx (((nt * Kt + kk) * p.n + a) * p.k + b) =
        ((((nt * p.cols + c) * p.n + a) * p.Kn Kt + kk * p.k + b : Nat) : Int)) := by
  constructor
  · intro hcm ha hb
    have hw : tapB p (p.Kn Kt) (p.Nn Nt) c =
        { offset := ((c * p.n : Nat) : Int),
          dims := [((Nt : Nat), (p.n * p.cols : Nat)), ((Kt : Nat), (p.k : Nat) * (p.Nn Nt : Nat)),
            ((p.k : Nat), (p.Nn Nt : Nat)), ((p.n : Nat), 1)] } := by
      simp only [tapB, hcm, nTiles_dims hp, kTiles_dims hp]; rfl
    have hq : ((nt * Kt + kk) * p.k + a) * p.n + b =
        encode (tapB p (p.Kn Kt) (p.Nn Nt) c).dims [nt, kk, a, b] := by
      rw [hw]; simp only [encode, dimsLen, Int.toNat_natCast]; ring
    have hfits : Fits (tapB p (p.Kn Kt) (p.Nn Nt) c).dims [nt, kk, a, b] := by
      rw [hw]; simp only [Fits, Int.toNat_natCast]; exact ⟨hnt, hkk, ha, hb, trivial⟩
    rw [hq, Walk.idx_encode _ hfits, hw]
    simp only [dot, Params.Nn]
    push_cast
    ring
  · intro hcm ha hb
    have hw : tapB p (p.Kn Kt) (p.Nn Nt) c =
        { offset := ((c * p.n : Nat) : Int) * (p.Kn Kt : Nat),
          dims := [((Nt : Nat), (p.cols * p.n : Nat) * (p.Kn Kt : Nat)), ((Kt : Nat), (p.k : Nat)),
            ((p.n : Nat), (p.Kn Kt : Nat)), ((p.k : Nat), 1)] } := by
      simp only [tapB, hcm, nTiles_dims hp, kTiles_dims hp]; rfl
    have hq : ((nt * Kt + kk) * p.n + a) * p.k + b =
        encode (tapB p (p.Kn Kt) (p.Nn Nt) c).dims [nt, kk, a, b] := by
      rw [hw]; simp only [encode, dimsLen, Int.toNat_natCast]; ring
    have hfits : Fits (tapB p (p.Kn Kt) (p.Nn Nt) c).dims [nt, kk, a, b] := by
      rw [hw]; simp only [Fits, Int.toNat_natCast]; exact ⟨hnt, hkk, ha, hb, trivial⟩
    rw [hq, Walk.idx_encode _ hfits, hw]
    simp only [dot, Params.Kn]
    push_cast
    ring

/-- C drain `c`, row block `rb`, column tile `nt`: position `(nt·rows·m + a)·n + b` writes
`C[rb·rows·m + a, (nt·cols + c)·n + b]`; with `cColMaj` (C drained as `N × M`) position
`((nt·rows + ro)·n + a)·m + b` writes `Cᵀ[(nt·cols + c)·n + a, rb·rows·m + ro·m + b]`. -/
theorem tapC_idx (Mt Nt rb c : Nat) {nt : Nat} (hnt : nt < Nt) :
    (∀ {a b}, p.cColMaj = false → a < p.rows * p.m → b < p.n →
      (tapC p (p.Mn Mt) (p.Nn Nt) rb c).idx ((nt * (p.rows * p.m) + a) * p.n + b) =
        (((rb * (p.rows * p.m) + a) * p.Nn Nt + (nt * p.cols + c) * p.n + b : Nat) : Int)) ∧
    (∀ {ro a b}, p.cColMaj = true → ro < p.rows → a < p.n → b < p.m →
      (tapC p (p.Mn Mt) (p.Nn Nt) rb c).idx (((nt * p.rows + ro) * p.n + a) * p.m + b) =
        ((((nt * p.cols + c) * p.n + a) * p.Mn Mt + rb * (p.rows * p.m) + ro * p.m + b : Nat) :
          Int)) := by
  constructor
  · intro a b hcm ha hb
    have hw : tapC p (p.Mn Mt) (p.Nn Nt) rb c =
        { offset := (rb : Int) * (p.rows * p.m : Nat) * (p.Nn Nt : Nat) + ((c * p.n : Nat) : Int),
          dims := [((Nt : Nat), (p.n * p.cols : Nat)), ((p.rows * p.m : Nat), (p.Nn Nt : Nat)),
            ((p.n : Nat), 1)] } := by
      simp only [tapC, hcm, nTiles_dims hp]; rfl
    have hq : (nt * (p.rows * p.m) + a) * p.n + b =
        encode (tapC p (p.Mn Mt) (p.Nn Nt) rb c).dims [nt, a, b] := by
      rw [hw]; simp only [encode, dimsLen, Int.toNat_natCast]; ring
    have hfits : Fits (tapC p (p.Mn Mt) (p.Nn Nt) rb c).dims [nt, a, b] := by
      rw [hw]; simp only [Fits, Int.toNat_natCast]; exact ⟨hnt, ha, hb, trivial⟩
    rw [hq, Walk.idx_encode _ hfits, hw]
    simp only [dot, Params.Nn]
    push_cast
    ring
  · intro ro a b hcm hro ha hb
    have hw : tapC p (p.Mn Mt) (p.Nn Nt) rb c =
        { offset := (rb : Int) * (p.rows * p.m : Nat) + ((c * p.n : Nat) : Int) * (p.Mn Mt : Nat),
          dims := [((Nt : Nat), (p.cols * p.n : Nat) * (p.Mn Mt : Nat)), ((p.rows : Nat), (p.m : Nat)),
            ((p.n : Nat), (p.Mn Mt : Nat)), ((p.m : Nat), 1)] } := by
      simp only [tapC, hcm, nTiles_dims hp]; rfl
    have hq : ((nt * p.rows + ro) * p.n + a) * p.m + b =
        encode (tapC p (p.Mn Mt) (p.Nn Nt) rb c).dims [nt, ro, a, b] := by
      rw [hw]; simp only [encode, dimsLen, Int.toNat_natCast]; ring
    have hfits : Fits (tapC p (p.Mn Mt) (p.Nn Nt) rb c).dims [nt, ro, a, b] := by
      rw [hw]; simp only [Fits, Int.toNat_natCast]; exact ⟨hnt, hro, ha, hb, trivial⟩
    rw [hq, Walk.idx_encode _ hfits, hw]
    simp only [dot, Params.Mn]
    push_cast
    ring

omit hp in
theorem step_ok (M K N rb : Int) : (step p M K N rb).ok = true := by
  simp only [step, ok_ofList, List.all_eq_true, List.mem_flatMap, List.mem_range]
  rintro x ⟨c, -, hx⟩
  simp only [List.mem_append, List.mem_singleton] at hx
  rcases hx with (rfl | hx) | rfl
  · rfl
  · split at hx
    · simp only [List.mem_singleton] at hx; subst hx; rfl
    · simp at hx
  · rfl

/-- The design's guards accept exactly the shapes `dims p Mt Kt Nt` with every count positive
(the same statement as `Theorems.guards_iff`). -/
theorem guards_iff (M K N : Int) :
    (wseq p M K N).ok = true ↔
      ∃ Mt Kt Nt : Nat, 0 < Mt ∧ 0 < Kt ∧ 0 < Nt ∧ (M, K, N) = p.dims Mt Kt Nt := by
  have hrm : (0 : Int) < ((p.rows * p.m : Nat) : Int) := by
    have := hp.rows_pos; have := hp.m_pos; positivity
  have hk : (0 : Int) < ((p.k : Nat) : Int) := by have := hp.k_pos; positivity
  have hnc : (0 : Int) < ((p.n * p.cols : Nat) : Int) := by
    have := hp.n_pos; have := hp.cols_pos; positivity
  have hloop : ∀ l : List Int, (l.map fun i => (step p M K N (0 + i)).ok).all id = true := by
    intro l; simp [step_ok]
  simp only [wseq, ok_ofList, List.all_cons, List.all_nil, Seq.ok, Bool.and_true,
    Bool.and_eq_true, decide_eq_true_eq, hloop]
  constructor
  · rintro ⟨hM, hK, hN, hMd, hKd, hNd⟩
    obtain ⟨a, ha⟩ := Int.dvd_of_emod_eq_zero hMd
    obtain ⟨b, hb⟩ := Int.dvd_of_emod_eq_zero hKd
    obtain ⟨c, hc⟩ := Int.dvd_of_emod_eq_zero hNd
    have ha0 : 0 < a := by rw [ha] at hM; exact pos_of_mul_pos_right hM hrm.le
    have hb0 : 0 < b := by rw [hb] at hK; exact pos_of_mul_pos_right hK hk.le
    have hc0 : 0 < c := by rw [hc] at hN; exact pos_of_mul_pos_right hN hnc.le
    refine ⟨a.toNat, b.toNat, c.toNat, by omega, by omega, by omega, ?_⟩
    simp only [Params.dims, Params.Mn, Params.Kn, Params.Nn, Prod.mk.injEq]
    refine ⟨?_, ?_, ?_⟩
    · rw [ha]; push_cast [Int.toNat_of_nonneg ha0.le]; ring
    · rw [hb]; push_cast [Int.toNat_of_nonneg hb0.le]; ring
    · rw [hc]; push_cast [Int.toNat_of_nonneg hc0.le]; ring
  · rintro ⟨Mt, Kt, Nt, hMt, hKt, hNt, h⟩
    simp only [Params.dims, Params.Mn, Params.Kn, Params.Nn, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl, rfl⟩ := h
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
    · exact_mod_cast Nat.mul_pos (Nat.mul_pos hp.rows_pos hp.m_pos) hMt
    · exact_mod_cast Nat.mul_pos hp.k_pos hKt
    · exact_mod_cast Nat.mul_pos (Nat.mul_pos hp.n_pos hp.cols_pos) hNt
    · exact Int.emod_eq_zero_of_dvd ⟨Mt, by push_cast; ring⟩
    · exact Int.emod_eq_zero_of_dvd ⟨Kt, by push_cast; ring⟩
    · exact Int.emod_eq_zero_of_dvd ⟨Nt, by push_cast; ring⟩

end IronVerify.Schema.WholeArray.Streams
