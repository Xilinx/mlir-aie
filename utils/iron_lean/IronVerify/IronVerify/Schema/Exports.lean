-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Schema.Theorems
import IronVerify.Schema.Inst.Transfer

/-!
The headline theorems, about an exported design `d` rather than the schema: for any `d` that is
an `Instance` of a valid `schema p`, and every `M`, `K`, `N` whose exported `cf.assert`s pass
(`(d.seq (M, K, N)).ok = true`, the only hypothesis on the shape), the export's cores receive
the right A and B tiles in their kernel's layout, each element of C is written once by the core
that owns it, and the value written is `C = A @ B`. Each is the schema theorem moved across
`Instance` once; the corollaries at the end apply them to two real IRON exports.
-/

namespace IronVerify.Schema.WholeArray

open IronVerify

theorem Params.dims_eq_shapeOf (p : Params) : p.dims = p.shapeOf := rfl

namespace Export

/-- The tile counts of a dispatch shape: `M = rows·m·Mt`, `K = k·Kt`, `N = n·cols·Nt`. -/
def tilesM (p : Params) (M : Int) : Nat := (M / ((p.rows * p.m : Nat) : Int)).toNat
def tilesK (p : Params) (K : Int) : Nat := (K / (p.k : Int)).toNat
def tilesN (p : Params) (N : Int) : Nat := (N / ((p.n * p.cols : Nat) : Int)).toNat

theorem toNat_div_cancel {a b : Nat} (ha : 0 < a) : (((a * b : Nat) : Int) / (a : Int)).toNat = b := by
  rw [Nat.cast_mul, Int.mul_ediv_cancel_left _ (by exact_mod_cast ha.ne'), Int.toNat_natCast]

section Shape

variable {d : Design (Int × Int × Int)} {p : Params} (hi : Instance d p) (hp : p.Valid)
include hi hp

/-- What the export's guards accept is `dims` of its own tile counts, each positive. -/
theorem shape {M K N : Int} (hok : (d.seq (M, K, N)).ok = true) :
    0 < tilesM p M ∧ 0 < tilesK p K ∧ 0 < tilesN p N ∧
      (M, K, N) = p.dims (tilesM p M) (tilesK p K) (tilesN p N) := by
  obtain ⟨Mt, Kt, Nt, hM, hK, hN, hP⟩ := hi.shape hok
  simp only [Params.shapeOf, Prod.mk.injEq] at hP
  obtain ⟨rfl, rfl, rfl⟩ := hP
  have eM : tilesM p ((p.rows * p.m * Mt : Nat) : Int) = Mt :=
    toNat_div_cancel (Nat.mul_pos hp.rows_pos hp.m_pos)
  have eK : tilesK p ((p.k * Kt : Nat) : Int) = Kt := toNat_div_cancel hp.k_pos
  have eN : tilesN p ((p.n * p.cols * Nt : Nat) : Int) = Nt :=
    toNat_div_cancel (Nat.mul_pos hp.n_pos hp.cols_pos)
  rw [eM, eK, eN]
  exact ⟨hM, hK, hN, rfl⟩

/-- The export's guards fix `M`, `K`, `N` to whole tiles. -/
theorem toNats {M K N : Int} (hok : (d.seq (M, K, N)).ok = true) :
    M.toNat = p.Mn (tilesM p M) ∧ K.toNat = p.Kn (tilesK p K) ∧ N.toNat = p.Nn (tilesN p N) := by
  obtain ⟨-, -, -, hP⟩ := shape hi hp hok
  simp only [Params.dims, Prod.mk.injEq] at hP
  refine ⟨?_, ?_, ?_⟩
  · have h := congrArg Int.toNat hP.1; rwa [Int.toNat_natCast] at h
  · have h := congrArg Int.toNat hP.2.1; rwa [Int.toNat_natCast] at h
  · have h := congrArg Int.toNat hP.2.2; rwa [Int.toNat_natCast] at h

theorem host_eq {M K N : Int} (hok : (d.seq (M, K, N)).ok = true) :
    d.host (M, K, N) = (schema p).host (p.dims (tilesM p M) (tilesK p K) (tilesN p N)) := by
  obtain ⟨hM, hK, hN, hP⟩ := shape hi hp hok
  rw [hP]; exact hi.host_eq hM hK hN

end Shape

/-- A name the schema gives a fifo names one in the export too. -/
theorem fifo?_exists {d : Design (Int × Int × Int)} {p : Params} (hi : Instance d p)
    {name : String} {g : FifoD} (hg : (wnet p).fifo? name = some g) :
    ∃ f, d.net.fifo? name = some f := by
  have e := Net.fifo?_norm d.net name
  rw [hi.net, Net.fifo?_norm, hg] at e
  cases h : d.net.fifo? name with
  | none => simp [h] at e
  | some f => exact ⟨f, rfl⟩

/-! ### The values a drain of the export writes -/

section Values

variable {R : Type*} [Zero R] (d : Design (Int × Int × Int)) (p : Params) (hA hB : Nat → R)
  (P : Int × Int × Int)

/-- Worker `w`'s `j`-th A and B objects in the export, read off its fifos by name. -/
def opA (w j : Nat) : Nat → R := fun s =>
  if s < p.m * p.k then
    match d.net.fifo? (aL2 (w / p.cols)) with
    | some f => hostVal hA hB (d.deliver P f w j s)
    | none => 0
  else 0

def opB (w j : Nat) : Nat → R := fun s =>
  if s < p.k * p.n then
    match d.net.fifo? (bL2 (w % p.cols)) with
    | some f => hostVal hA hB (d.deliver P f w j s)
    | none => 0
  else 0

/-- The value drain `cL3 c` of the export writes at stream position `q`, given what one kernel
call computes and the core contract (`coreTile`). -/
def drainVal (call : KernelCall R) (zero : Nat → Nat → R) (c q : Nat) : R :=
  match d.net.fifo? (cL3 c) with
  | some f =>
    match d.stream P f q with
    | some (.core w _ τ s) =>
      coreTile call zero (opA d p hA hB P w) (opB d p hA hB P w) (tilesK p P.2.1) τ
        (p.unlayoutC s).1 (p.unlayoutC s).2
    | _ => 0
  | none => 0

end Values

section Theorems

variable {d : Design (Int × Int × Int)} {p : Params} (hi : Instance d p) (hp : p.Valid)
include hi hp

/-- **T-A** for the export: core `(row, col)`'s `(τ·Kt + kk)`-th A object holds row `i0 + a`,
column `kk·k + b` of A at the kernel's layout slot. -/
theorem tA {M K N : Int} (hok : (d.seq (M, K, N)).ok = true) {row col τ kk a b : Nat}
    (hrow : row < p.rows) (hcol : col < p.cols) (hτ : τ < tilesM p M * tilesN p N)
    (hkk : kk < tilesK p K) (ha : a < p.m) (hb : b < p.k) {f : FifoD}
    (hf : d.net.fifo? (aL2 row) = some f) :
    d.deliver (M, K, N) f (p.worker row col) (τ * tilesK p K + kk) (p.layoutA a b) =
      some (.host 0 ((p.aIdx K.toNat (p.i0 (tilesN p N) row τ + a) (kk * p.k + b) : Nat) :
        Int)) := by
  obtain ⟨hM, hK, hN, hP⟩ := shape hi hp hok
  obtain ⟨-, eK, -⟩ := toNats hi hp hok
  rw [hP, Params.dims_eq_shapeOf, hi.deliver_eq hM hK hN hf (fifo?_aL2 p hrow),
    ← Params.dims_eq_shapeOf, eK]
  exact WholeArray.tA p hp _ _ _ row col τ kk a b hrow hcol hτ hkk ha hb

/-- **T-B** for the export: core `(row, col)`'s `(τ·Kt + kk)`-th B object holds row `kk·k + a`,
column `j0 + b` of B at the kernel's layout slot (either B layout). -/
theorem tB {M K N : Int} (hok : (d.seq (M, K, N)).ok = true) {row col τ kk a b : Nat}
    (hrow : row < p.rows) (hcol : col < p.cols) (hτ : τ < tilesM p M * tilesN p N)
    (hkk : kk < tilesK p K) (ha : a < p.k) (hb : b < p.n) {f : FifoD}
    (hf : d.net.fifo? (bL2 col) = some f) :
    d.deliver (M, K, N) f (p.worker row col) (τ * tilesK p K + kk) (p.layoutB a b) =
      some (.host 1 ((p.bIdx K.toNat N.toNat (kk * p.k + a) (p.j0 (tilesN p N) col τ + b) :
        Nat) : Int)) := by
  obtain ⟨hM, hK, hN, hP⟩ := shape hi hp hok
  obtain ⟨-, eK, eN⟩ := toNats hi hp hok
  rw [hP, Params.dims_eq_shapeOf, hi.deliver_eq hM hK hN hf (fifo?_bL2 p hcol),
    ← Params.dims_eq_shapeOf, eK, eN]
  exact WholeArray.tB p hp _ _ _ row col τ kk a b hrow hcol hτ hkk ha hb

/-- **T-C** for the export: across all `cols` drains exactly one write puts `C[i, j]` in the C
buffer; it comes from the core that owns `(i, j)`, from its output tile `tauC`, at the C layout
slot of `(i mod m, j mod n)`, with the same `i0`/`j0` ownership as T-A and T-B. -/
theorem tC {M K N : Int} (hok : (d.seq (M, K, N)).ok = true) {i j : Nat} (hiM : i < M.toNat)
    (hjN : j < N.toNat) :
    (∃! cq : Nat × Nat, cq.1 < p.cols ∧ (d.host (M, K, N) (cL3 cq.1))[cq.2]? =
      some (2, ((p.cIdx M.toNat N.toNat i j : Nat) : Int))) ∧
    (∀ c q f, c < p.cols → d.net.fifo? (cL3 c) = some f →
      (d.host (M, K, N) (cL3 c))[q]? = some (2, ((p.cIdx M.toNat N.toNat i j : Nat) : Int)) →
      c = p.colC j ∧
      d.stream (M, K, N) f q =
        some (.core (p.worker (p.rowC i) c) (cL1 c (p.rowC i)) (p.tauC (tilesN p N) i j)
          (p.layoutC (i % p.m) (j % p.n)))) ∧
    i = p.i0 (tilesN p N) (p.rowC i) (p.tauC (tilesN p N) i j) + i % p.m ∧
    j = p.j0 (tilesN p N) (p.colC j) (p.tauC (tilesN p N) i j) + j % p.n := by
  obtain ⟨hM, hK, hN, hP⟩ := shape hi hp hok
  obtain ⟨eM, -, eN⟩ := toNats hi hp hok
  rw [eM] at hiM ⊢
  rw [eN] at hjN ⊢
  have hh := host_eq hi hp hok
  obtain ⟨h1, h2, h3, h4⟩ := WholeArray.tC p hp (tilesM p M) (tilesK p K) (tilesN p N) i j hiM hjN
  refine ⟨by rw [hh]; exact h1, fun c q f hc hf hq => ?_, h3, h4⟩
  rw [hh] at hq
  obtain ⟨hcj, hs⟩ := h2 c q hc hq
  refine ⟨hcj, ?_⟩
  rw [hP, Params.dims_eq_shapeOf, hi.stream_eq hM hK hN hf (fifo?_cL3 p hc),
    ← Params.dims_eq_shapeOf]
  exact hs

/-- The export's drain writes the same value as the schema's wherever it writes `C[i, j]`. -/
theorem drainVal_eq {R : Type*} [Zero R] (hA hB : Nat → R) (call : KernelCall R)
    (zero : Nat → Nat → R) {M K N : Int} (hok : (d.seq (M, K, N)).ok = true) {i j c q : Nat}
    (hiM : i < M.toNat) (hjN : j < N.toNat) (hc : c < p.cols)
    (hq : (d.host (M, K, N) (cL3 c))[q]? = some (2, ((p.cIdx M.toNat N.toNat i j : Nat) : Int))) :
    drainVal d p hA hB (M, K, N) call zero c q =
      WholeArray.drainVal p hA hB (tilesM p M) (tilesK p K) (tilesN p N) call zero c q := by
  obtain ⟨hM, hK, hN, hP⟩ := shape hi hp hok
  obtain ⟨f, hf⟩ := fifo?_exists hi (fifo?_cL3 p hc)
  obtain ⟨-, h2, -, -⟩ := tC hi hp hok hiM hjN
  obtain ⟨-, hs⟩ := h2 c q f hc hf hq
  have hs' := hs
  rw [hP, Params.dims_eq_shapeOf, hi.stream_eq hM hK hN hf (fifo?_cL3 p hc),
    ← Params.dims_eq_shapeOf] at hs'
  have hrow := hp.rowC_lt i
  have hcols := hp.cols_pos
  have hdiv : p.worker (p.rowC i) c / p.cols = p.rowC i := by
    unfold Params.worker
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hcols, Nat.div_eq_of_lt hc, Nat.zero_add]
  have hmod : p.worker (p.rowC i) c % p.cols = c := by
    unfold Params.worker
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hc]
  have eKt : tilesK p (M, K, N).2.1 = tilesK p K := rfl
  have hopA : opA d p hA hB (M, K, N) (p.worker (p.rowC i) c) =
      WholeArray.opA p hA hB (tilesM p M) (tilesK p K) (tilesN p N) (p.worker (p.rowC i) c) := by
    funext jj s
    unfold opA WholeArray.opA
    split_ifs
    · have hr : p.worker (p.rowC i) c / p.cols < p.rows := by rw [hdiv]; exact hrow
      obtain ⟨fa, hfa⟩ := fifo?_exists hi (fifo?_aL2 p hr)
      simp only [hfa]
      rw [hP, Params.dims_eq_shapeOf, hi.deliver_eq hM hK hN hfa (fifo?_aL2 p hr),
        ← Params.dims_eq_shapeOf]
    · rfl
  have hopB : opB d p hA hB (M, K, N) (p.worker (p.rowC i) c) =
      WholeArray.opB p hA hB (tilesM p M) (tilesK p K) (tilesN p N) (p.worker (p.rowC i) c) := by
    funext jj s
    unfold opB WholeArray.opB
    split_ifs
    · have hr : p.worker (p.rowC i) c % p.cols < p.cols := by rw [hmod]; exact hc
      obtain ⟨fb, hfb⟩ := fifo?_exists hi (fifo?_bL2 p hr)
      simp only [hfb]
      rw [hP, Params.dims_eq_shapeOf, hi.deliver_eq hM hK hN hfb (fifo?_bL2 p hr),
        ← Params.dims_eq_shapeOf]
    · rfl
  unfold drainVal WholeArray.drainVal
  simp only [hf]
  rw [hs, hs']
  dsimp only
  rw [eKt, hopA, hopB]

/-- **C = A @ B** for the export, over any commutative semiring: wherever a drain writes
`C[i, j]`, the value is `Σ_{x < K} A[i, x] · B[x, j]`. -/
theorem tMM {R : Type*} [CommSemiring R] (hA hB : Nat → R) {M K N : Int}
    (hok : (d.seq (M, K, N)).ok = true) {i j c q : Nat} (hiM : i < M.toNat) (hjN : j < N.toNat)
    (hc : c < p.cols)
    (hq : (d.host (M, K, N) (cL3 c))[q]? = some (2, ((p.cIdx M.toNat N.toNat i j : Nat) : Int))) :
    drainVal d p hA hB (M, K, N) p.matmul (fun _ _ => 0) c q =
      ∑ x ∈ Finset.range K.toNat, hA (p.aIdx K.toNat i x) * hB (p.bIdx K.toNat N.toNat x j) := by
  rw [drainVal_eq hi hp hA hB _ _ hok hiM hjN hc hq]
  obtain ⟨eM, eK, eN⟩ := toNats hi hp hok
  rw [eM] at hiM
  rw [eM, eN] at hq
  rw [eN] at hjN
  rw [host_eq hi hp hok] at hq
  rw [eK, eN]
  exact WholeArray.tMM p hp hA hB _ _ _ i j c q hiM hjN hc hq

/-- `w`-bit integers with saturation off (measured on the NPU for i16): the stored C is the exact
integer product reduced mod `2 ^ w`. -/
theorem tMM_wrap (w : Nat) (hA hB : Nat → ℤ) {M K N : Int} (hok : (d.seq (M, K, N)).ok = true)
    {i j c q : Nat} (hiM : i < M.toNat) (hjN : j < N.toNat) (hc : c < p.cols)
    (hq : (d.host (M, K, N) (cL3 c))[q]? = some (2, ((p.cIdx M.toNat N.toNat i j : Nat) : Int))) :
    drainVal d p (fun x => (hA x : Wrap w)) (fun x => (hB x : Wrap w)) (M, K, N) p.matmul
        (fun _ _ => 0) c q =
      ((∑ x ∈ Finset.range K.toNat,
        hA (p.aIdx K.toNat i x) * hB (p.bIdx K.toNat N.toNat x j) : ℤ) : Wrap w) := by
  rw [tMM hi hp _ _ hok hiM hjN hc hq]
  push_cast
  rfl

/-- Integers with saturating stores: if no partial sum over the first `kk` k-blocks leaves
`[lo, up]`, the clamps never fire and the stored C is the exact integer product. -/
theorem tMM_sat (lo up : ℤ) (hlo : lo ≤ 0) (hup : 0 ≤ up) (hA hB : Nat → ℤ) {M K N : Int}
    (hok : (d.seq (M, K, N)).ok = true) {i j c q : Nat} (hiM : i < M.toNat) (hjN : j < N.toNat)
    (hc : c < p.cols)
    (hq : (d.host (M, K, N) (cL3 c))[q]? = some (2, ((p.cIdx M.toNat N.toNat i j : Nat) : Int)))
    (hpart : ∀ kk ≤ tilesK p K, lo ≤ ∑ x ∈ Finset.range (kk * p.k),
        hA (p.aIdx K.toNat i x) * hB (p.bIdx K.toNat N.toNat x j) ∧
      ∑ x ∈ Finset.range (kk * p.k),
        hA (p.aIdx K.toNat i x) * hB (p.bIdx K.toNat N.toNat x j) ≤ up) :
    drainVal d p hA hB (M, K, N) (p.matmulSat lo up) (fun _ _ => 0) c q =
      ∑ x ∈ Finset.range K.toNat, hA (p.aIdx K.toNat i x) * hB (p.bIdx K.toNat N.toNat x j) := by
  rw [drainVal_eq hi hp hA hB _ _ hok hiM hjN hc hq]
  obtain ⟨eM, eK, eN⟩ := toNats hi hp hok
  rw [eM] at hiM
  rw [eM, eN] at hq
  rw [eN] at hjN
  rw [host_eq hi hp hok] at hq
  rw [eK, eN] at hpart ⊢
  exact WholeArray.tMM_sat p hp lo up hlo hup hA hB _ _ _ i j c q hiM hjN hc hq hpart

/-- Ordered accumulation for any kernel (any dtype, any rounding): the export's drain writes the
kernel folded over `kk = 0 … Kt - 1` on exactly the A row block and B column block of `(i, j)`. -/
theorem orderedAcc {R : Type*} [Zero R] (hA hB : Nat → R) (call : KernelCall R)
    (zero : Nat → Nat → R) {M K N : Int} (hok : (d.seq (M, K, N)).ok = true) {i j c q : Nat}
    (hiM : i < M.toNat) (hjN : j < N.toNat) (hc : c < p.cols)
    (hq : (d.host (M, K, N) (cL3 c))[q]? = some (2, ((p.cIdx M.toNat N.toNat i j : Nat) : Int))) :
    drainVal d p hA hB (M, K, N) call zero c q =
      (List.range (tilesK p K)).foldl
        (fun C kk => call (specA p hA (tilesK p K) (i - i % p.m) kk)
          (specB p hB (tilesK p K) (tilesN p N) (j - j % p.n) kk) C)
        zero (i % p.m) (j % p.n) := by
  rw [drainVal_eq hi hp hA hB _ _ hok hiM hjN hc hq]
  obtain ⟨eM, -, eN⟩ := toNats hi hp hok
  rw [eM] at hiM
  rw [eM, eN] at hq
  rw [eN] at hjN
  rw [host_eq hi hp hok] at hq
  exact WholeArray.orderedAcc p hp hA hB call zero _ _ _ i j c q hiM hjN hc hq

end Theorems

end Export

end IronVerify.Schema.WholeArray
