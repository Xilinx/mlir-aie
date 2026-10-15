-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Designs.WholeArrayC4
import IronVerify.Proofs.WholeArrayCommon

/-!
whole_array on a 4 × 4 array of cores (m = 64, k = 64, n = 32), exported with M, K and N symbolic.
Shim column `c` feeds A row `c`, B column `c` and drains C column `c`. The theorems hold for every
M = 256·Mt, K = 64·Kt, N = 128·Nq, the shapes its `require(...)` guards accept.
-/

set_option maxRecDepth 20000

namespace IronVerify.Proofs.WholeArrayC4

open IronVerify Designs.WholeArrayC4 Spec.WholeArray Proofs.WholeArray

def aL3 (c : Nat) : String := ["A_L3L2_0", "A_L3L2_1", "A_L3L2_2", "A_L3L2_3"][c]!
def bL3 (c : Nat) : String := ["B_L3L2_0", "B_L3L2_1", "B_L3L2_2", "B_L3L2_3"][c]!
def cL3 (c : Nat) : String := ["C_L2L3_0", "C_L2L3_1", "C_L2L3_2", "C_L2L3_3"][c]!
def aL2 (row : Nat) : String := ["A_L2L1_0", "A_L2L1_1", "A_L2L1_2", "A_L2L1_3"][row]!
def bL2 (col : Nat) : String := ["B_L2L1_0", "B_L2L1_1", "B_L2L1_2", "B_L2L1_3"][col]!
def cL1 (col row : Nat) : String :=
  ["C_L1L2_0_0", "C_L1L2_0_1", "C_L1L2_0_2", "C_L1L2_0_3",
   "C_L1L2_1_0", "C_L1L2_1_1", "C_L1L2_1_2", "C_L1L2_1_3",
   "C_L1L2_2_0", "C_L1L2_2_1", "C_L1L2_2_2", "C_L1L2_2_3",
   "C_L1L2_3_0", "C_L1L2_3_1", "C_L1L2_3_2", "C_L1L2_3_3"][col * 4 + row]!

/-- Shim `c`'s A transfer for row block `rb`: tile row `4·rb + c`, `Nq` passes over its `Kt` tiles. -/
def tapA (K : Int) (Kt Nq rb c : Nat) : Walk :=
  { offset := (rb * 4 + c) * 64 * K, dims := [(Nq, 0), (Kt, 64), (64, K), (64, 1)] }

/-- Shim `c`'s B transfer (the same for every row block): every fourth column tile from `c`. -/
def tapB (N : Int) (Kt Nq c : Nat) : Walk :=
  { offset := c * 32, dims := [(Nq, 128), (Kt, N * 64), (64, N), (32, 1)] }

/-- Shim `c`'s C drain for row block `rb`: every fourth 32-column tile from `c`. -/
def tapC (N : Int) (Nq rb c : Nat) : Walk :=
  { offset := rb * 256 * N + c * 32, dims := [(Nq, 128), (256, N), (32, 1)] }

theorem streamA (Mt Kt Nq c : Nat) (hc : c < 4) :
    (seq (256 * Mt) (64 * Kt) (128 * Nq)).stream (aL3 c) =
      ((List.range Mt).map fun rb => (tapA (64 * Kt) Kt Nq rb c).toList.map (0, ·)).flatten := by
  have hMt : (256 * (Mt : Int)) / 64 / 4 = Mt := by omega
  have h3 : ((Mt : Int) - 3).toNat = Mt - 3 := by omega
  have hN : 128 * (Nq : Int) / 32 / 4 = Nq := by omega
  interval_cases c <;>
  · simp only [aL3, seq, Seq.stream]
    simp [Int.fdiv_eq_ediv, hMt, h3, hN]
    rw [← peel3]
    simp only [tapA, flatMap_cast, List.map_map, Function.comp_def]
    push_cast
    ring_nf

theorem streamB (Mt Kt Nq c : Nat) (hc : c < 4) :
    (seq (256 * Mt) (64 * Kt) (128 * Nq)).stream (bL3 c) =
      (List.replicate Mt ((tapB (128 * Nq) Kt Nq c).toList.map (1, ·))).flatten := by
  have hMt : (256 * (Mt : Int)) / 64 / 4 = Mt := by omega
  have h3 : ((Mt : Int) - 3).toNat = Mt - 3 := by omega
  have e0 : (128 * (Nq : Int) / 32 - 1) / 4 + 1 = Nq := by omega
  have e1 : (128 * (Nq : Int) / 32 - 1 - 1) / 4 + 1 = Nq := by omega
  have e2 : (128 * (Nq : Int) / 32 - 2 - 1) / 4 + 1 = Nq := by omega
  have e3 : (128 * (Nq : Int) / 32 - 3 - 1) / 4 + 1 = Nq := by omega
  have hK : 64 * (Kt : Int) / 64 = Kt := by omega
  interval_cases c <;>
  · simp only [bL3, seq, Seq.stream]
    simp [Int.fdiv_eq_ediv, hMt, h3, e0, e1, e2, e3, hK]
    simp only [tapB]
    ring_nf
    rcases Mt with _ | _ | _ | n <;> simp [List.replicate_succ]

theorem streamC (Mt Kt Nq c : Nat) (hc : c < 4) :
    (seq (256 * Mt) (64 * Kt) (128 * Nq)).stream (cL3 c) =
      ((List.range Mt).map fun rb => (tapC (128 * Nq) Nq rb c).toList.map (2, ·)).flatten := by
  have hMt : (256 * (Mt : Int)) / 64 / 4 = Mt := by omega
  have h3 : ((Mt : Int) - 3).toNat = Mt - 3 := by omega
  have e0 : (128 * (Nq : Int) / 32 - 1) / 4 + 1 = Nq := by omega
  have e1 : (128 * (Nq : Int) / 32 - 1 - 1) / 4 + 1 = Nq := by omega
  have e2 : (128 * (Nq : Int) / 32 - 2 - 1) / 4 + 1 = Nq := by omega
  have e3 : (128 * (Nq : Int) / 32 - 3 - 1) / 4 + 1 = Nq := by omega
  interval_cases c <;>
  · simp only [cL3, seq, Seq.stream]
    simp [Int.fdiv_eq_ediv, hMt, h3, e0, e1, e2, e3]
    rw [← peel3]
    simp only [tapC, flatMap_cast, List.map_map, Function.comp_def]
    push_cast
    ring_nf

/-- Core (row, col) reads A on row `row`'s fifo: object `j`, slot `s` is host element
`j·4096 + wA(s)` of shim `row`'s A channel. -/
theorem deliverA (host : String → Array (Nat × Int)) (row col : Nat) (hrow : row < 4)
    (hcol : col < 4) (f : FifoD) (hf : net.fifo? (aL2 row) = some f) (j s : Nat) (hs : s < 4096) :
    net.deliver host f (row * 4 + col) j s =
      (host (aL3 row))[j * 4096 + (wA.idx s).toNat]?.map (fun e => .host e.1 e.2) := by
  have hlen : wA.len = 4096 := rfl
  have h1 : (j * 4096 + s) / 4096 = j := by omega
  have h2 : (j * 4096 + s) % 4096 = s := by omega
  interval_cases row <;>
  · simp [aL2, Net.fifo?, net] at hf
    subst hf
    interval_cases col <;>
    · rw [Net.deliver_linear net host (by rfl), Net.stream,
        show net.fifos.length + 1 = 36 + 1 from rfl,
        Net.streamAt_link1 net host (w := wA) rfl rfl rfl rfl rfl (by decide) rfl rfl rfl,
        show (36 : Nat) = 35 + 1 from rfl, Net.streamAt_shim net host rfl]
      simp [hlen, h1, h2, aL3]

/-- Core (row, col) reads B on column `col`'s fifo: object `j`, slot `s` is host element
`j·2048 + wB(s)` of shim `col`'s B channel. -/
theorem deliverB (host : String → Array (Nat × Int)) (row col : Nat) (hrow : row < 4)
    (hcol : col < 4) (f : FifoD) (hf : net.fifo? (bL2 col) = some f) (j s : Nat) (hs : s < 2048) :
    net.deliver host f (row * 4 + col) j s =
      (host (bL3 col))[j * 2048 + (wB.idx s).toNat]?.map (fun e => .host e.1 e.2) := by
  have hlen : wB.len = 2048 := rfl
  have h1 : (j * 2048 + s) / 2048 = j := by omega
  have h2 : (j * 2048 + s) % 2048 = s := by omega
  interval_cases col <;>
  · simp [bL2, Net.fifo?, net] at hf
    subst hf
    interval_cases row <;>
    · rw [Net.deliver_linear net host (by rfl), Net.stream,
        show net.fifos.length + 1 = 36 + 1 from rfl,
        Net.streamAt_link1 net host (w := wB) rfl rfl rfl rfl rfl (by decide) rfl rfl rfl,
        show (36 : Nat) = 35 + 1 from rfl, Net.streamAt_shim net host rfl]
      simp [hlen, h1, h2, bL3]

/-- T-A: for every M = 256·Mt, K = 64·Kt, N = 128·Nq, core (row, col)'s A object `τ·Kt + kk` holds
the 64 × 64 tile of A at rows `i0 ..` and columns `kk·64 ..`, in the kernel's blocked layout. -/
theorem tA (Mt Kt Nq : Nat) (row col : Nat) (hrow : row < 4) (hcol : col < 4) (f : FifoD)
    (hf : net.fifo? (aL2 row) = some f) (τ kk a b : Nat) (hτ : τ < Mt * Nq) (hkk : kk < Kt)
    (ha : a < 64) (hb : b < 64) :
    design.deliver ((256 * Mt : Nat), (64 * Kt : Nat), (128 * Nq : Nat)) f (row * 4 + col)
        (τ * Kt + kk) (layoutA a b) =
      some (.host 0 ((i0 Nq row τ + a : Nat) * (64 * Kt : Nat) + (kk * 64 + b : Nat))) := by
  obtain ⟨hl, hidx⟩ := wA_layoutA a b ha hb
  have hNq : 0 < Nq := by
    rcases Nat.eq_zero_or_pos Nq with h | h
    · subst h; simp at hτ
    · exact h
  simp only [Design.deliver, design]
  rw [deliverA _ row col hrow hcol f hf _ _ hl, hidx]
  simp only [Design.host]
  push_cast
  rw [streamA Mt Kt Nq row hrow, List.getElem?_toArray]
  set rb := τ / Nq
  set n := τ % Nq
  have hτ' : τ = rb * Nq + n := by rw [Nat.div_add_mod' τ Nq]
  have hrb : rb < Mt := by
    rw [Nat.div_lt_iff_lt_mul hNq]; exact hτ
  have hn : n < Nq := Nat.mod_lt _ hNq
  have hfits : Fits (tapA (64 * Kt) Kt Nq rb row).dims [n, kk, a, b] := by
    simp [Fits, tapA]; omega
  have hlen : ∀ i, i < Mt → ((tapA (64 * Kt) Kt Nq i row).toList.map (0, ·)).length
      = (tapA (64 * Kt) Kt Nq 0 row).len := by
    intro i _; simp [Walk.length_toList, Walk.len, tapA]
  have hab : ((a : Int) * 64 + b).toNat = a * 64 + b := by omega
  have hL : (tapA (64 * Kt) Kt Nq 0 row).len = Nq * (Kt * 4096) := by
    simp [Walk.len, tapA, dimsLen]
  have henc : encode (tapA (64 * Kt) Kt Nq rb row).dims [n, kk, a, b]
      = n * (Kt * 4096) + (kk * 4096 + (a * 64 + b)) := by
    simp [encode, tapA, dimsLen]
  have hpos : (τ * Kt + kk) * 4096 + (a * 64 + b)
      = rb * (Nq * (Kt * 4096)) + (n * (Kt * 4096) + (kk * 4096 + (a * 64 + b))) := by
    rw [hτ']; ring
  rw [hab, hpos, ← henc, ← hL, getElem?_flatten_range _ _ _ hlen hrb
    ((tapA (64 * Kt) Kt Nq rb row).encode_lt_len hfits |>.trans_eq (by simp [Walk.len, tapA]))]
  simp only [List.getElem?_map, Walk.getElem?_toList,
    (tapA (64 * Kt) Kt Nq rb row).encode_lt_len hfits, ite_true, Option.map_some,
    Walk.idx_encode _ hfits]
  simp [dot, tapA, i0, rb]
  ring

/-- T-B: for every M = 256·Mt, K = 64·Kt, N = 128·Nq, core (row, col)'s B object `τ·Kt + kk`
holds the 64 × 32 tile of B at rows `kk·64 ..` and columns `j0 ..`, in the kernel's blocked
layout. -/
theorem tB (Mt Kt Nq : Nat) (row col : Nat) (hrow : row < 4) (hcol : col < 4) (f : FifoD)
    (hf : net.fifo? (bL2 col) = some f) (τ kk a b : Nat) (hτ : τ < Mt * Nq) (hkk : kk < Kt)
    (ha : a < 64) (hb : b < 32) :
    design.deliver ((256 * Mt : Nat), (64 * Kt : Nat), (128 * Nq : Nat)) f (row * 4 + col)
        (τ * Kt + kk) (layoutB a b) =
      some (.host 1 ((kk * 64 + a : Nat) * (128 * Nq : Nat) + (j0 4 Nq col τ + b : Nat))) := by
  obtain ⟨hl, hidx⟩ := wB_layoutB a b ha hb
  have hNq : 0 < Nq := by
    rcases Nat.eq_zero_or_pos Nq with h | h
    · subst h; simp at hτ
    · exact h
  simp only [Design.deliver, design]
  rw [deliverB _ row col hrow hcol f hf _ _ hl, hidx]
  simp only [Design.host]
  push_cast
  rw [streamB Mt Kt Nq col hcol, List.getElem?_toArray]
  set rb := τ / Nq
  set n := τ % Nq
  have hτ' : τ = rb * Nq + n := by rw [Nat.div_add_mod' τ Nq]
  have hrb : rb < Mt := by
    rw [Nat.div_lt_iff_lt_mul hNq]; exact hτ
  have hn : n < Nq := Nat.mod_lt _ hNq
  have hfits : Fits (tapB (128 * Nq) Kt Nq col).dims [n, kk, a, b] := by
    simp [Fits, tapB]; omega
  have hab : ((a : Int) * 32 + b).toNat = a * 32 + b := by omega
  have hL : ((tapB (128 * Nq) Kt Nq col).toList.map (1, ·)).length = Nq * (Kt * 2048) := by
    simp [Walk.length_toList, Walk.len, tapB, dimsLen]
  have henc : encode (tapB (128 * Nq) Kt Nq col).dims [n, kk, a, b]
      = n * (Kt * 2048) + (kk * 2048 + (a * 32 + b)) := by
    simp [encode, tapB, dimsLen]
  have hpos : (τ * Kt + kk) * 2048 + (a * 32 + b)
      = rb * (Nq * (Kt * 2048)) + (n * (Kt * 2048) + (kk * 2048 + (a * 32 + b))) := by
    rw [hτ']; ring
  have hlt := (tapB (128 * Nq) Kt Nq col).encode_lt_len hfits
  rw [hab, hpos, ← henc, ← hL,
    getElem?_flatten_replicate _ _ hrb (by simpa [Walk.length_toList] using hlt)]
  simp only [List.getElem?_map, Walk.getElem?_toList, hlt, ite_true, Option.map_some,
    Walk.idx_encode _ hfits]
  simp [dot, tapB, j0, n]
  ring

/-- The C drain stream of shim `col`: position `J·8192 + r·2048 + (a·32 + b)` carries row `a`,
column `b` of core (r, col)'s `J`-th output object. -/
theorem streamC_src (host : String → Array (Nat × Int)) (col : Nat) (hcol : col < 4) (f : FifoD)
    (hf : net.fifo? (cL3 col) = some f) (J r a b : Nat) (hr : r < 4) (ha : a < 64) (hb : b < 32) :
    net.stream host f (J * 8192 + (r * 2048 + (a * 32 + b))) =
      some (.core (r * 4 + col) (cL1 col r) J (layoutC a b)) := by
  have hlt := (wC_layoutC a b ha hb).2
  obtain ⟨h1, h2, h3, h4, hm, hm'⟩ :
      (J * 8192 + (r * 2048 + (a * 32 + b))) / 8192 = J ∧
      (J * 8192 + (r * 2048 + (a * 32 + b))) % 8192 = r * 2048 + (a * 32 + b) ∧
      (r * 2048 + (a * 32 + b)) / 2048 = r ∧ (r * 2048 + (a * 32 + b)) % 2048 = a * 32 + b ∧
      (J * 2048 + layoutC a b) / 2048 = J ∧ (J * 2048 + layoutC a b) % 2048 = layoutC a b :=
    ⟨by omega, by omega, by omega, by omega, by omega, by omega⟩
  have hwl : wC.len = 2048 := rfl
  have hidx := (wC_layoutC a b ha hb).1
  have hseg : segmentOf [0, 2048, 4096, 6144] ([0, 2048, 4096, 6144][r]! + layoutC a b) = some r := by
    rw [show ([0, 2048, 4096, 6144] : List Nat)[r]! = r * 2048 by interval_cases r <;> rfl]
    exact segmentOf_four r (layoutC a b) hlt hr
  have hoff : ((([(0, 2048), (2048, 2048), (4096, 2048), (6144, 2048)] : List (Nat × Nat))[r]?).map
      (·.1)).getD 0 = ([0, 2048, 4096, 6144] : List Nat)[r]! := by
    interval_cases r <;> rfl
  obtain ⟨L, hprod, hL, hto, hrep, hdst, hobj, hlen, hoffs, hgs⟩ :
      ∃ L : LinkD, f.prod = .link (8 + col) ∧ net.links[8 + col]? = some L ∧
        f.toStream = some wC ∧ f.repeatCount = 1 ∧ L.dsts = [f.name] ∧ f.objSize = 8192 ∧
        1 < L.srcs.length ∧ L.srcOffsets = [0, 2048, 4096, 6144] ∧
        ∀ r, r < 4 → ∃ g : FifoD, (L.srcs[r]?).bind net.fifo? = some g ∧
          g.consFrom? (.link (8 + col)) = some none ∧ g.prod = .worker (r * 4 + col) ∧
          g.toStream = none ∧ g.repeatCount = 1 ∧ g.name = cL1 col r ∧ g.objSize = 2048 ∧
          g.consSize = 2048 := by
    interval_cases col <;>
    · simp [cL3, Net.fifo?, net] at hf
      subst hf
      refine ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, by decide, rfl, ?_⟩
      intro r hr
      interval_cases r <;> exact ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
  have hsegs : L.segments 8192 = [(0, 2048), (2048, 2048), (4096, 2048), (6144, 2048)] := by
    simp [LinkD.segments, hoffs, hlen]
  rw [Net.stream, show net.fifos.length + 1 = 36 + 1 from rfl,
    Net.streamAt_join net host hprod hL hlen hto hrep hdst, hobj, hsegs,
    show segLen [(0, 2048), (2048, 2048), (4096, 2048), (6144, 2048)] (some wC) = 8192 from rfl,
    h1, h2]
  simp only [segGather, hwl, h3, h4, hidx, Int.toNat_natCast]
  rw [hoff]
  obtain ⟨g, hg, hfrom, hprod', hto', hrep', hname, hobj', hcons⟩ := hgs r hr
  rw [← hoffs] at hseg ⊢
  rw [Net.linkObj_join net hL hlen hseg hg hfrom, hcons,
    show (36 : Nat) = 35 + 1 from rfl, Net.streamAt_worker net host hprod' hto' hrep', hobj', hname,
    hm, hm']

/-- The output tile `C[i, j]` belongs to: row block `i / 256`, column group `j / 128`. -/
def tauC (Nq i j : Nat) : Nat := (i / 256) * Nq + j / 128

/-- The shim column whose drain writes `C[i, j]`. -/
def colC (j : Nat) : Nat := (j % 128) / 32

/-- The position on shim `colC j`'s drain that writes `C[i, j]`. -/
def posC (Nq i j : Nat) : Nat := tauC Nq i j * 8192 + ((i % 256) / 64 * 2048 + (i % 64 * 32 + j % 32))

theorem hostC_len (Mt Kt Nq c : Nat) (hc : c < 4) :
    ((seq (256 * Mt) (64 * Kt) (128 * Nq)).stream (cL3 c)).length = Mt * (Nq * 8192) := by
  rw [streamC Mt Kt Nq c hc, length_flatten_range _ (Nq * 8192)]
  intro i _; simp [Walk.length_toList, Walk.len, tapC, dimsLen]

/-- `posC` on shim `colC j` hits `C[i, j]`. -/
theorem hostC_posC (Mt Kt Nq i j : Nat) (hi : i < 256 * Mt) (hj : j < 128 * Nq) :
    ((seq (256 * Mt) (64 * Kt) (128 * Nq)).stream (cL3 (colC j)))[posC Nq i j]? =
      some (2, ((i * (128 * Nq) + j : Nat) : Int)) := by
  have hc : colC j < 4 := by simp only [colC]; omega
  set rb := i / 256
  have hrb : rb < Mt := by omega
  have hfits : Fits (tapC (128 * Nq) Nq rb (colC j)).dims [j / 128, i % 256, j % 32] := by
    simp [Fits, tapC]; omega
  have hlen : ∀ k, k < Mt → ((tapC (128 * Nq) Nq k (colC j)).toList.map (2, ·)).length
      = Nq * 8192 := by
    intro k _; simp [Walk.length_toList, Walk.len, tapC, dimsLen]
  have henc : encode (tapC (128 * Nq) Nq rb (colC j)).dims [j / 128, i % 256, j % 32]
      = j / 128 * 8192 + (i % 256 * 32 + j % 32) := by
    simp [encode, tapC, dimsLen]
  have e : (i % 256) / 64 * 2048 + i % 64 * 32 = i % 256 * 32 := by omega
  have hpos : posC Nq i j
      = rb * (Nq * 8192) + encode (tapC (128 * Nq) Nq rb (colC j)).dims [j / 128, i % 256, j % 32] := by
    rw [henc, posC, tauC, ← add_assoc ((i % 256) / 64 * 2048), e]; ring
  have hlt := (tapC (128 * Nq) Nq rb (colC j)).encode_lt_len hfits
  rw [streamC Mt Kt Nq (colC j) hc, hpos, getElem?_flatten_range _ _ _ hlen hrb
    (hlt.trans_eq (by simp [Walk.len, tapC, dimsLen]))]
  simp only [List.getElem?_map, Walk.getElem?_toList, hlt, ite_true, Option.map_some,
    Walk.idx_encode _ hfits]
  simp [dot, tapC, rb, colC]
  have hj' : j = j / 128 * 128 + (j % 128) / 32 * 32 + j % 32 := by omega
  conv_rhs => rw [← Nat.div_add_mod i 256, hj']
  push_cast
  ring

/-- Every position on shim `c`'s drain is `posC` of some in-range `(i, j)` in column group `c`. -/
theorem posC_onto (Mt Nq c p : Nat) (hc : c < 4) (hp : p < Mt * (Nq * 8192)) :
    ∃ i j, i < 256 * Mt ∧ j < 128 * Nq ∧ colC j = c ∧ p = posC Nq i j := by
  have hNq : 0 < Nq := by
    rcases Nat.eq_zero_or_pos Nq with h | h
    · subst h; simp at hp
    · exact h
  have hL : 0 < Nq * 8192 := by positivity
  set rb := p / (Nq * 8192)
  set rest := p % (Nq * 8192)
  have hrb : rb < Mt := by rw [Nat.div_lt_iff_lt_mul hL]; exact hp
  have hrest : rest < Nq * 8192 := Nat.mod_lt _ hL
  have hn : rest / 8192 < Nq := by rw [Nat.div_lt_iff_lt_mul (by norm_num)]; exact hrest
  set n := rest / 8192
  set q := rest % 8192
  have hq : q < 8192 := Nat.mod_lt _ (by norm_num)
  refine ⟨rb * 256 + q / 32, n * 128 + c * 32 + q % 32, by omega, by omega,
    by simp only [colC]; omega, ?_⟩
  obtain ⟨h1, h2, h3, h4, h5, e⟩ :
      (rb * 256 + q / 32) / 256 = rb ∧ (rb * 256 + q / 32) % 256 = q / 32 ∧
      (rb * 256 + q / 32) % 64 = q / 32 % 64 ∧ (n * 128 + c * 32 + q % 32) / 128 = n ∧
      (n * 128 + c * 32 + q % 32) % 32 = q % 32 ∧
      q / 32 / 64 * 2048 + (q / 32 % 64 * 32 + q % 32) = q :=
    ⟨by omega, by omega, by omega, by omega, by omega, by omega⟩
  rw [posC, tauC, h1, h2, h3, h4, h5, e]
  have hp' : p = rb * (Nq * 8192) + (n * 8192 + q) := by
    have e1 : Nq * 8192 * rb + rest = p := Nat.div_add_mod p _
    have e2 : 8192 * n + q = rest := Nat.div_add_mod rest _
    linarith
  rw [hp']; ring

/-- T-C: for every M = 256·Mt, K = 64·Kt, N = 128·Nq, exactly one position on one of the four
drains writes `C[i, j]`, from row `i % 64`, column `j % 32` of core (r, colC j)'s output tile `τ`,
which is the tile T-A and T-B feed that core at `τ`. -/
theorem tC (Mt Kt Nq : Nat) (i j : Nat) (hi : i < 256 * Mt) (hj : j < 128 * Nq) (f : FifoD)
    (hf : net.fifo? (cL3 (colC j)) = some f) :
    (∃! cp : Nat × Nat, cp.1 < 4 ∧
      (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (128 * Nq : Nat)) (cL3 cp.1))[cp.2]? =
        some (2, ((i * (128 * Nq) + j : Nat) : Int))) ∧
    design.stream ((256 * Mt : Nat), (64 * Kt : Nat), (128 * Nq : Nat)) f (posC Nq i j) =
      some (.core ((i % 256) / 64 * 4 + colC j) (cL1 (colC j) ((i % 256) / 64)) (tauC Nq i j)
        (layoutC (i % 64) (j % 32))) ∧
    i = i0 Nq ((i % 256) / 64) (tauC Nq i j) + i % 64 ∧
    j = j0 4 Nq (colC j) (tauC Nq i j) + j % 32 := by
  have hNq : 0 < Nq := by omega
  have hj128 : j / 128 < Nq := by omega
  have hc : colC j < 4 := by simp only [colC]; omega
  refine ⟨⟨(colC j, posC Nq i j), ⟨hc, ?_⟩, ?_⟩, ?_, ?_, ?_⟩
  · simp only [Design.host, design]; push_cast
    rw [List.getElem?_toArray]; exact_mod_cast hostC_posC Mt Kt Nq i j hi hj
  · rintro ⟨c, p⟩ ⟨hc', hp⟩
    simp only [Design.host, design] at hp; push_cast at hp
    rw [List.getElem?_toArray] at hp
    have hlt : p < Mt * (Nq * 8192) := by
      rw [← hostC_len Mt Kt Nq c hc']
      exact (List.getElem?_eq_some_iff.mp (by exact_mod_cast hp)).1
    obtain ⟨i', j', hi', hj', rfl, rfl⟩ := posC_onto Mt Nq c p hc' hlt
    rw [hostC_posC Mt Kt Nq i' j' hi' hj'] at hp
    simp only [Option.some.injEq, Prod.mk.injEq, true_and] at hp
    have hnat : i' * (128 * Nq) + j' = i * (128 * Nq) + j := by exact_mod_cast hp
    have hN : 0 < 128 * Nq := by omega
    have hii : i' = i := by
      have := congrArg (· / (128 * Nq)) hnat
      rwa [Nat.add_comm, Nat.add_mul_div_right _ _ hN, Nat.div_eq_of_lt hj', Nat.zero_add,
        Nat.add_comm, Nat.add_mul_div_right _ _ hN, Nat.div_eq_of_lt hj, Nat.zero_add] at this
    subst hii
    have : j' = j := by omega
    rw [this]
  · have hr : (i % 256) / 64 < 4 := by omega
    exact streamC_src (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (128 * Nq : Nat)))
      (colC j) hc f hf (tauC Nq i j) ((i % 256) / 64) (i % 64) (j % 32) hr (by omega) (by omega)
  · have : tauC Nq i j / Nq = i / 256 := by
      rw [tauC, Nat.add_comm, Nat.add_mul_div_right _ _ hNq, Nat.div_eq_of_lt hj128, Nat.zero_add]
    simp only [i0, this]; omega
  · have : tauC Nq i j % Nq = j / 128 := by
      rw [tauC, Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj128]
    simp only [j0, this, colC]; omega

/-- Each drain writes `Mt·Nq·8192` elements, M·N over the four, all inside C and in its own
column group. -/
theorem tC_writes (Mt Kt Nq c : Nat) (hc : c < 4) :
    (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (128 * Nq : Nat)) (cL3 c)).size =
        Mt * (Nq * 8192) ∧
    ∀ (p : Nat) e,
      (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (128 * Nq : Nat)) (cL3 c))[p]? = some e →
      ∃ i j, i < 256 * Mt ∧ j < 128 * Nq ∧ colC j = c ∧ e = (2, ((i * (128 * Nq) + j : Nat) : Int)) := by
  simp only [Design.host, design]
  push_cast
  refine ⟨by rw [List.size_toArray, hostC_len Mt Kt Nq c hc], fun p e hp => ?_⟩
  rw [List.getElem?_toArray] at hp
  have hlt : p < Mt * (Nq * 8192) := by
    rw [← hostC_len Mt Kt Nq c hc]; exact (List.getElem?_eq_some_iff.mp hp).1
  obtain ⟨i, j, hi, hj, hcj, rfl⟩ := posC_onto Mt Nq c p hc hlt
  rw [← hcj, hostC_posC Mt Kt Nq i j hi hj] at hp
  exact ⟨i, j, hi, hj, hcj, (Option.some.inj hp).symm⟩

/-- Core (row, ·) receives exactly `Mt·Nq` output tiles' worth of A objects, `Kt` per tile. -/
theorem countA (Mt Kt Nq : Nat) (row : Nat) (hrow : row < 4) (f : FifoD)
    (hf : net.fifo? (aL2 row) = some f) :
    net.streamLen (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (128 * Nq : Nat))) f =
      some (Mt * Nq * Kt * 4096) := by
  have hsize : (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (128 * Nq : Nat)) (aL3 row)).size
      = Mt * (Nq * (Kt * 4096)) := by
    simp only [Design.host, design]; push_cast
    rw [List.size_toArray, streamA Mt Kt Nq row hrow, length_flatten_range _ (Nq * (Kt * 4096))]
    intro i _; simp [Walk.length_toList, Walk.len, tapA, dimsLen]
  have hdiv : Mt * (Nq * (Kt * 4096)) / 4096 = Mt * Nq * Kt := by
    rw [show Mt * (Nq * (Kt * 4096)) = Mt * Nq * Kt * 4096 by ring, Nat.mul_div_cancel _ (by norm_num)]
  interval_cases row <;>
  · simp [aL2, Net.fifo?, net] at hf
    subst hf
    rw [Net.streamLen, show net.fifos.length + 1 = 35 + 2 from rfl,
      Net.streamLenAt_link1 net _ rfl rfl rfl rfl rfl]
    simp [aL3] at hsize
    simp [Net.prodSegs, segLen, Walk.len, dimsLen, net, hsize, hdiv]

/-- Core (·, col) receives exactly `Mt·Nq` output tiles' worth of B objects, `Kt` per tile. -/
theorem countB (Mt Kt Nq : Nat) (col : Nat) (hcol : col < 4) (f : FifoD)
    (hf : net.fifo? (bL2 col) = some f) :
    net.streamLen (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (128 * Nq : Nat))) f =
      some (Mt * Nq * Kt * 2048) := by
  have hsize : (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (128 * Nq : Nat)) (bL3 col)).size
      = Mt * (Nq * (Kt * 2048)) := by
    simp only [Design.host, design]; push_cast
    rw [List.size_toArray, streamB Mt Kt Nq col hcol, List.length_flatten]
    simp [Walk.length_toList, Walk.len, tapB, dimsLen, List.sum_replicate]
  have hdiv : Mt * (Nq * (Kt * 2048)) / 2048 = Mt * Nq * Kt := by
    rw [show Mt * (Nq * (Kt * 2048)) = Mt * Nq * Kt * 2048 by ring, Nat.mul_div_cancel _ (by norm_num)]
  interval_cases col <;>
  · simp [bL2, Net.fifo?, net] at hf
    subst hf
    rw [Net.streamLen, show net.fifos.length + 1 = 35 + 2 from rfl,
      Net.streamLenAt_link1 net _ rfl rfl rfl rfl rfl]
    simp [bL3] at hsize
    simp [Net.prodSegs, segLen, Walk.len, dimsLen, net, hsize, hdiv]

end IronVerify.Proofs.WholeArrayC4
