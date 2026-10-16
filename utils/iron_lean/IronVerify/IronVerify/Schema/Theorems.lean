-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Schema.Defs
import IronVerify.Schema.Proofs.Seq
import IronVerify.Schema.Proofs.Names
import IronVerify.Schema.Proofs.PathA
import IronVerify.Schema.Proofs.PathB
import IronVerify.Schema.Proofs.PathC
import IronVerify.Schema.Proofs.Matmul
import IronVerify.Schema.Proofs.Overflow

/-!
The parametric theorems, stated once for every valid `p` (any `rows`, `cols`, tile, micro-tile
and column-major switch) and every shape the guards accept. Each is proved in `Schema/Proofs/`;
`Schema/Instances` transfers them to each export.

`dims p Mt Kt Nt` is the shape `M = rows·m·Mt`, `K = k·Kt`, `N = n·cols·Nt`; `guards_iff` says
these, with the runtime-parameter product `(rows·Mt)·(cols·Nt)` inside `i32`, are exactly the
shapes whose `require(...)` checks pass.
-/

namespace IronVerify.Schema.WholeArray

open IronVerify

variable (p : Params)

variable (hp : p.Valid)
include hp

/-! ### Guards -/

theorem guards_iff (M K N : Int) :
    (wseq p M K N).ok = true ↔
      ∃ Mt Kt Nt : Nat, 0 < Mt ∧ 0 < Kt ∧ 0 < Nt ∧ (M, K, N) = p.dims Mt Kt Nt ∧
        p.rows * Mt * (p.cols * Nt) < 2 ^ 31 := by exact Streams.guards_iff (hp := hp) M K N

/-! ### Fifo lookups (the only place fifo names are compared) -/

omit hp in
theorem fifo?_aL2 {row : Nat} (h : row < p.rows) :
    (wnet p).fifo? (aL2 row) = some (aL2Fifo p row) := by exact Names.fifo?_aL2 (p := p) h

omit hp in
theorem fifo?_bL2 {c : Nat} (h : c < p.cols) :
    (wnet p).fifo? (bL2 c) = some (bL2Fifo p c) := by exact Names.fifo?_bL2 (p := p) h

omit hp in
theorem fifo?_cL3 {c : Nat} (h : c < p.cols) :
    (wnet p).fifo? (cL3 c) = some (cL3Fifo p c) := by exact Names.fifo?_cL3 (p := p) h

/-! ### T-A, T-B: every core gets the right operands, in its kernel's layout -/

theorem tA (Mt Kt Nt row col τ kk a b : Nat) (hrow : row < p.rows) (hcol : col < p.cols)
    (hτ : τ < Mt * Nt) (hkk : kk < Kt) (ha : a < p.m) (hb : b < p.k) :
    (schema p).deliver (p.dims Mt Kt Nt) (aL2Fifo p row) (p.worker row col) (τ * Kt + kk)
        (p.layoutA a b) =
      some (.host 0 ((p.aIdx (p.Kn Kt) (p.i0 Nt row τ + a) (kk * p.k + b) : Nat) : Int)) := by apply PathA.tA (hp := hp) <;> assumption

theorem tB (Mt Kt Nt row col τ kk a b : Nat) (hrow : row < p.rows) (hcol : col < p.cols)
    (hτ : τ < Mt * Nt) (hkk : kk < Kt) (ha : a < p.k) (hb : b < p.n) :
    (schema p).deliver (p.dims Mt Kt Nt) (bL2Fifo p col) (p.worker row col) (τ * Kt + kk)
        (p.layoutB a b) =
      some (.host 1 ((p.bIdx (p.Kn Kt) (p.Nn Nt) (kk * p.k + a) (p.j0 Nt col τ + b) : Nat) : Int)) := by apply PathB.tB (hp := hp) <;> assumption

/-- Each core receives exactly the objects it consumes: `tiles · Kt` of each operand. -/
theorem countA (Mt Kt Nt row : Nat) (hrow : row < p.rows) :
    (wnet p).streamLen ((schema p).host (p.dims Mt Kt Nt)) (aL2Fifo p row) =
      some (Mt * Nt * Kt * (p.m * p.k)) := by apply PathA.countA (hp := hp) <;> assumption

theorem countB (Mt Kt Nt c : Nat) (hc : c < p.cols) :
    (wnet p).streamLen ((schema p).host (p.dims Mt Kt Nt)) (bL2Fifo p c) =
      some (Mt * Nt * Kt * (p.k * p.n)) := by apply PathB.countB (hp := hp) <;> assumption

/-! ### T-C: every element of C is written once, by the core that owns it -/

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
    j = p.j0 Nt (p.colC j) (p.tauC Nt i j) + j % p.n := by apply PathC.tC (hp := hp) <;> assumption

/-- Each drain writes only elements of C in its own column group, and as many as it owns. -/
theorem tC_writes (Mt Kt Nt c : Nat) (hc : c < p.cols) :
    let h := (schema p).host (p.dims Mt Kt Nt) (cL3 c)
    h.size = Mt * Nt * (p.rows * p.m * p.n) ∧
    ∀ q < h.size, ∃ i j, i < p.Mn Mt ∧ j < p.Nn Nt ∧ p.colC j = c ∧
      h[q]? = some (2, ((p.cIdx (p.Mn Mt) (p.Nn Nt) i j : Nat) : Int)) := by apply PathC.tC_writes (hp := hp) <;> assumption


/-- Ordered accumulation, for any kernel (any dtype, any rounding): the value written for
`C[i, j]` is the kernel folded over `kk = 0 … Kt - 1` on exactly the A row block and B column
block the spec names, in order. -/
theorem orderedAcc {R : Type*} [Zero R] (hA hB : Nat → R) (call : KernelCall R)
    (zero : Nat → Nat → R) (Mt Kt Nt i j c q : Nat) (hi : i < p.Mn Mt) (hj : j < p.Nn Nt)
    (hc : c < p.cols)
    (hq : ((schema p).host (p.dims Mt Kt Nt) (cL3 c))[q]? =
      some (2, ((p.cIdx (p.Mn Mt) (p.Nn Nt) i j : Nat) : Int))) :
    drainVal p hA hB Mt Kt Nt call zero c q =
      (List.range Kt).foldl
        (fun C kk => call (specA p hA Kt (i - i % p.m) kk) (specB p hB Kt Nt (j - j % p.n) kk) C)
        zero (i % p.m) (j % p.n) := by apply Matmul.orderedAcc (hp := hp) <;> assumption

/-- Exact `C = A @ B` over any commutative semiring (ℤ, or `Wrap w` for `w`-bit integers with
saturation off). -/
theorem tMM {R : Type*} [CommSemiring R] (hA hB : Nat → R) (Mt Kt Nt i j c q : Nat)
    (hi : i < p.Mn Mt) (hj : j < p.Nn Nt) (hc : c < p.cols)
    (hq : ((schema p).host (p.dims Mt Kt Nt) (cL3 c))[q]? =
      some (2, ((p.cIdx (p.Mn Mt) (p.Nn Nt) i j : Nat) : Int))) :
    drainVal p hA hB Mt Kt Nt p.matmul (fun _ _ => 0) c q =
      ∑ x ∈ Finset.range (p.Kn Kt),
        hA (p.aIdx (p.Kn Kt) i x) * hB (p.bIdx (p.Kn Kt) (p.Nn Nt) x j) := by apply Matmul.tMM (hp := hp) <;> assumption

/-- `w`-bit integers with saturation off (the contract assumption): the stored C is the exact
integer product reduced mod `2 ^ w`. -/
theorem tMM_wrap (w : Nat) (hA hB : Nat → ℤ) (Mt Kt Nt i j c q : Nat)
    (hi : i < p.Mn Mt) (hj : j < p.Nn Nt) (hc : c < p.cols)
    (hq : ((schema p).host (p.dims Mt Kt Nt) (cL3 c))[q]? =
      some (2, ((p.cIdx (p.Mn Mt) (p.Nn Nt) i j : Nat) : Int))) :
    drainVal p (fun x => (hA x : Wrap w)) (fun x => (hB x : Wrap w)) Mt Kt Nt p.matmul
        (fun _ _ => 0) c q =
      ((∑ x ∈ Finset.range (p.Kn Kt),
        hA (p.aIdx (p.Kn Kt) i x) * hB (p.bIdx (p.Kn Kt) (p.Nn Nt) x j) : ℤ) : Wrap w) := by apply Matmul.tMM_wrap (hp := hp) <;> assumption

/-- Saturating stores: exact over ℤ as long as no partial sum of `C[i, j]` (after any number of
k steps) leaves `[lo, hi]`. -/
theorem tMM_sat (lo up : ℤ) (hlo : lo ≤ 0) (hup : 0 ≤ up) (hA hB : Nat → ℤ)
    (Mt Kt Nt i j c q : Nat) (hi : i < p.Mn Mt) (hj : j < p.Nn Nt) (hc : c < p.cols)
    (hq : ((schema p).host (p.dims Mt Kt Nt) (cL3 c))[q]? =
      some (2, ((p.cIdx (p.Mn Mt) (p.Nn Nt) i j : Nat) : Int)))
    (hpart : ∀ kk ≤ Kt, lo ≤ ∑ x ∈ Finset.range (kk * p.k),
        hA (p.aIdx (p.Kn Kt) i x) * hB (p.bIdx (p.Kn Kt) (p.Nn Nt) x j) ∧
      ∑ x ∈ Finset.range (kk * p.k),
        hA (p.aIdx (p.Kn Kt) i x) * hB (p.bIdx (p.Kn Kt) (p.Nn Nt) x j) ≤ up) :
    drainVal p hA hB Mt Kt Nt (p.matmulSat lo up) (fun _ _ => 0) c q =
      ∑ x ∈ Finset.range (p.Kn Kt),
        hA (p.aIdx (p.Kn Kt) i x) * hB (p.bIdx (p.Kn Kt) (p.Nn Nt) x j) := by apply Matmul.tMM_sat (hp := hp) <;> assumption

/-! ### Overflow: the host-buffer guard, exactly -/

/-- The compiler's runtime host-buffer guard accepts the schema's transfers exactly when A, B
and C fit their host buffers (`cap 0/1/2` elements). -/
theorem hostFits_iff (Mt Kt Nt : Nat) (hM : 0 < Mt) (hK : 0 < Kt) (hN : 0 < Nt)
    (cap : Nat → Nat) :
    (wseq p (p.Mn Mt) (p.Kn Kt) (p.Nn Nt)).HostFits cap ↔
      p.Mn Mt * p.Kn Kt ≤ cap 0 ∧ p.Kn Kt * p.Nn Nt ≤ cap 1 ∧ p.Mn Mt * p.Nn Nt ≤ cap 2 := by apply Ovf.hostFits_iff_aux (hp := hp) <;> assumption

end IronVerify.Schema.WholeArray

namespace IronVerify.Schema.WholeArray

variable (p : Params) (hp : p.Valid)
include hp

/-- No element offset, size, stride or furthest element of any transfer leaves `i64`, for every
`i32` dispatch value the guards accept (tile parameters below `2^16`). With `hostFits_iff` and
whole_array's `< 2^31`-element buffers, every byte offset is below `2^33`, which the 64-bit
`arg_plus` holds. -/
theorem noWrap64 (M K N : Int) (hM : fitsS 32 M) (hK : fitsS 32 K) (hN : fitsS 32 N)
    (hok : (wseq p M K N).ok = true)
    (hsmall : p.rows * p.m < 2 ^ 16 ∧ p.k < 2 ^ 16 ∧ p.n * p.cols < 2 ^ 16) :
    (wseq p M K N).Every fun _ _ w =>
      fitsS 64 w.offset ∧ (∀ d ∈ w.dims, fitsS 64 d.1 ∧ fitsS 64 d.2) ∧ fitsS 64 w.maxIdx := by apply Ovf.noWrap64_aux (hp := hp) <;> assumption

end IronVerify.Schema.WholeArray
