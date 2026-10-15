-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Designs.WholeArrayC1
import IronVerify.Proofs.WholeArrayCommon

/-!
whole_array with one column of four cores (m = 64, k = 64, n = 32), exported with M, K and N
symbolic. The theorems hold for every M = 256·Mt, K = 64·Kt, N = 32·Nt, the shapes its
`require(...)` guards accept.
-/

namespace IronVerify.Proofs.WholeArrayC1

open IronVerify Designs.WholeArrayC1 Spec.WholeArray Proofs.WholeArray

/-- Row block `rb`'s A transfer: `Nt` passes over its `Kt` (256 × 64) tiles. -/
def tapA (K : Int) (Kt Nt rb : Nat) : Walk :=
  { offset := rb * 256 * K, dims := [(Nt, 0), (Kt, 64), (256, K), (64, 1)] }

theorem streamA (Mt Kt Nt : Nat) :
    (seq (256 * Mt) (64 * Kt) (32 * Nt)).stream "A_L3L2_0" =
      ((List.range Mt).map fun rb => (tapA (64 * Kt) Kt Nt rb).toList.map (0, ·)).flatten := by
  have hMt : (256 * (Mt : Int)) / 64 / 4 = Mt := by omega
  have h3 : ((Mt : Int) - 3).toNat = Mt - 3 := by omega
  simp only [seq, Seq.stream]
  simp [Int.fdiv_eq_ediv, hMt, h3]
  rw [← peel3]
  simp only [tapA, flatMap_cast, List.map_map, Function.comp_def]
  push_cast
  ring_nf

/-- Row block `rb`'s B transfer (the same for every block): `Nt` column tiles, each `Kt` (64 × 32)
tiles down. -/
def tapB (N : Int) (Kt Nt : Nat) : Walk :=
  { offset := 0, dims := [(Nt, 32), (Kt, N * 64), (64, N), (32, 1)] }

theorem streamB (Mt Kt Nt : Nat) :
    (seq (256 * Mt) (64 * Kt) (32 * Nt)).stream "B_L3L2_0" =
      (List.replicate Mt ((tapB (32 * Nt) Kt Nt).toList.map (1, ·))).flatten := by
  have hMt : (256 * (Mt : Int)) / 64 / 4 = Mt := by omega
  have h3 : ((Mt : Int) - 3).toNat = Mt - 3 := by omega
  simp only [seq, Seq.stream]
  simp [Int.fdiv_eq_ediv, hMt, h3]
  simp only [tapB]
  ring_nf
  rcases Mt with _ | _ | _ | n <;> simp [List.replicate_succ]

/-- Row block `rb`'s C drain: the 256 × N row block of C, one 32-column tile at a time. -/
def tapC (N : Int) (Nt rb : Nat) : Walk :=
  { offset := rb * 256 * N, dims := [(Nt, 32), (256, N), (32, 1)] }

theorem streamC (Mt Kt Nt : Nat) :
    (seq (256 * Mt) (64 * Kt) (32 * Nt)).stream "C_L2L3_0" =
      ((List.range Mt).map fun rb => (tapC (32 * Nt) Nt rb).toList.map (2, ·)).flatten := by
  have hMt : (256 * (Mt : Int)) / 64 / 4 = Mt := by omega
  have h3 : ((Mt : Int) - 3).toNat = Mt - 3 := by omega
  simp only [seq, Seq.stream]
  simp [Int.fdiv_eq_ediv, hMt, h3]
  rw [← peel3]
  simp only [tapC, flatMap_cast, List.map_map, Function.comp_def]
  push_cast
  ring_nf

def aName (row : Nat) : String := ["A_L2L1_0", "A_L2L1_1", "A_L2L1_2", "A_L2L1_3"][row]!

/-- Core `row` reads A on its fifo: object `j`, slot `s` is host element `j·16384 + row·4096 + wA(s)`
of the A channel. -/
theorem deliverA (host : String → Array (Nat × Int)) (row : Nat) (hrow : row < 4) (f : FifoD)
    (hf : net.fifo? (aName row) = some f) (j s : Nat) (hs : s < 4096) :
    net.deliver host f row j s =
      (host "A_L3L2_0")[j * 16384 + (row * 4096 + (wA.idx s).toNat)]?.map
        (fun e => .host e.1 e.2) := by
  have hlen : wA.len = 4096 := rfl
  have h1 : (j * 4096 + s) / 4096 = j := by omega
  have h2 : (j * 4096 + s) % 4096 = s := by omega
  interval_cases row <;>
  · simp [aName, Net.fifo?, net] at hf
    subst hf
    rw [Net.deliver_linear net host (by rfl), Net.stream, show net.fifos.length + 1 = 12 + 1 from rfl,
      Net.streamAt_link1 net host (l := 0) (src := "A_L3L2_0") (w := wA) (L := net.links[0])
        (g := net.fifos[0]) rfl rfl rfl rfl rfl (by decide) rfl rfl rfl,
      show (12 : Nat) = 11 + 1 from rfl, Net.streamAt_shim net host rfl]
    simp [hlen, h1, h2, net]

/-- T-A: for every M = 256·Mt, K = 64·Kt, N = 32·Nt, core `row`'s A object `τ·Kt + kk` holds
the 64 × 64 tile of A at rows `i0 ..` and columns `kk·64 ..`, in the kernel's blocked layout. -/
theorem tA (Mt Kt Nt : Nat) (row : Nat) (hrow : row < 4) (f : FifoD)
    (hf : net.fifo? (aName row) = some f) (τ kk a b : Nat) (hτ : τ < Mt * Nt) (hkk : kk < Kt)
    (ha : a < 64) (hb : b < 64) :
    design.deliver ((256 * Mt : Nat), (64 * Kt : Nat), (32 * Nt : Nat)) f row (τ * Kt + kk)
        (layoutA a b) =
      some (.host 0 ((i0 Nt row τ + a : Nat) * (64 * Kt : Nat) + (kk * 64 + b : Nat))) := by
  obtain ⟨hl, hidx⟩ := wA_layoutA a b ha hb
  have hNt : 0 < Nt := by
    rcases Nat.eq_zero_or_pos Nt with h | h
    · subst h; simp at hτ
    · exact h
  simp only [Design.deliver, design]
  rw [deliverA _ row hrow f hf _ _ hl, hidx]
  simp only [Design.host]
  push_cast
  rw [streamA, List.getElem?_toArray]
  set rb := τ / Nt
  set n := τ % Nt
  have hτ' : τ = rb * Nt + n := by rw [Nat.div_add_mod' τ Nt]
  have hrb : rb < Mt := by
    rw [Nat.div_lt_iff_lt_mul hNt]; exact hτ
  have hn : n < Nt := Nat.mod_lt _ hNt
  have hfits : Fits (tapA (64 * Kt) Kt Nt rb).dims [n, kk, row * 64 + a, b] := by
    simp [Fits, tapA]; omega
  have hlen : ∀ i, i < Mt → ((tapA (64 * Kt) Kt Nt i).toList.map (0, ·)).length
      = (tapA (64 * Kt) Kt Nt 0).len := by
    intro i _; simp [Walk.length_toList, Walk.len, tapA]
  have hab : ((a : Int) * 64 + b).toNat = a * 64 + b := by omega
  have hL : (tapA (64 * Kt) Kt Nt 0).len = Nt * (Kt * 16384) := by
    simp [Walk.len, tapA, dimsLen]
  have henc : encode (tapA (64 * Kt) Kt Nt rb).dims [n, kk, row * 64 + a, b]
      = n * (Kt * 16384) + (kk * 16384 + ((row * 64 + a) * 64 + b)) := by
    simp [encode, tapA, dimsLen]
  have hpos : (τ * Kt + kk) * 16384 + (row * 4096 + (a * 64 + b))
      = rb * (Nt * (Kt * 16384)) + (n * (Kt * 16384) + (kk * 16384 + ((row * 64 + a) * 64 + b))) := by
    rw [hτ']; ring
  rw [hab, hpos, ← henc, ← hL, getElem?_flatten_range _ _ _ hlen hrb
    ((tapA (64 * Kt) Kt Nt rb).encode_lt_len hfits |>.trans_eq (by simp [Walk.len, tapA]))]
  simp only [List.getElem?_map, Walk.getElem?_toList,
    (tapA (64 * Kt) Kt Nt rb).encode_lt_len hfits, ite_true, Option.map_some,
    Walk.idx_encode _ hfits]
  simp [dot, tapA, i0, rb]
  ring

/-- Core `w` reads B on the broadcast fifo: object `j`, slot `s` is host element `j·2048 + wB(s)`
of the B channel. -/
theorem deliverB (host : String → Array (Nat × Int)) (w : Nat) (hw : w < 4) (f : FifoD)
    (hf : net.fifo? "B_L2L1_0" = some f) (j s : Nat) (hs : s < 2048) :
    net.deliver host f w j s =
      (host "B_L3L2_0")[j * 2048 + (wB.idx s).toNat]?.map (fun e => .host e.1 e.2) := by
  have hlen : wB.len = 2048 := rfl
  have h1 : (j * 2048 + s) / 2048 = j := by omega
  have h2 : (j * 2048 + s) % 2048 = s := by omega
  simp [Net.fifo?, net] at hf
  subst hf
  interval_cases w <;>
  · rw [Net.deliver_linear net host (by rfl), Net.stream, show net.fifos.length + 1 = 12 + 1 from rfl,
      Net.streamAt_link1 net host (l := 1) (src := "B_L3L2_0") (w := wB) (L := net.links[1])
        (g := net.fifos[1]) rfl rfl rfl rfl rfl (by decide) rfl rfl rfl,
      show (12 : Nat) = 11 + 1 from rfl, Net.streamAt_shim net host rfl]
    simp [hlen, h1, h2, net]

/-- T-B: for every M = 256·Mt, K = 64·Kt, N = 32·Nt, each core's B object `τ·Kt + kk` holds the
64 × 32 tile of B at rows `kk·64 ..` and columns `j0 ..`, in the kernel's blocked layout. -/
theorem tB (Mt Kt Nt : Nat) (w : Nat) (hw : w < 4) (f : FifoD)
    (hf : net.fifo? "B_L2L1_0" = some f) (τ kk a b : Nat) (hτ : τ < Mt * Nt) (hkk : kk < Kt)
    (ha : a < 64) (hb : b < 32) :
    design.deliver ((256 * Mt : Nat), (64 * Kt : Nat), (32 * Nt : Nat)) f w (τ * Kt + kk)
        (layoutB a b) =
      some (.host 1 ((kk * 64 + a : Nat) * (32 * Nt : Nat) + (j0 1 Nt 0 τ + b : Nat))) := by
  obtain ⟨hl, hidx⟩ := wB_layoutB a b ha hb
  have hNt : 0 < Nt := by
    rcases Nat.eq_zero_or_pos Nt with h | h
    · subst h; simp at hτ
    · exact h
  simp only [Design.deliver, design]
  rw [deliverB _ w hw f hf _ _ hl, hidx]
  simp only [Design.host]
  push_cast
  rw [streamB, List.getElem?_toArray]
  set rb := τ / Nt
  set n := τ % Nt
  have hτ' : τ = rb * Nt + n := by rw [Nat.div_add_mod' τ Nt]
  have hrb : rb < Mt := by
    rw [Nat.div_lt_iff_lt_mul hNt]; exact hτ
  have hn : n < Nt := Nat.mod_lt _ hNt
  have hfits : Fits (tapB (32 * Nt) Kt Nt).dims [n, kk, a, b] := by
    simp [Fits, tapB]; omega
  have hab : ((a : Int) * 32 + b).toNat = a * 32 + b := by omega
  have hL : ((tapB (32 * Nt) Kt Nt).toList.map (1, ·)).length = Nt * (Kt * 2048) := by
    simp [Walk.length_toList, Walk.len, tapB, dimsLen]
  have henc : encode (tapB (32 * Nt) Kt Nt).dims [n, kk, a, b]
      = n * (Kt * 2048) + (kk * 2048 + (a * 32 + b)) := by
    simp [encode, tapB, dimsLen]
  have hpos : (τ * Kt + kk) * 2048 + (a * 32 + b)
      = rb * (Nt * (Kt * 2048)) + (n * (Kt * 2048) + (kk * 2048 + (a * 32 + b))) := by
    rw [hτ']; ring
  have hlt := (tapB (32 * Nt) Kt Nt).encode_lt_len hfits
  rw [hab, hpos, ← henc, ← hL, getElem?_flatten_replicate _ _ hrb (by simpa [Walk.length_toList] using hlt)]
  simp only [List.getElem?_map, Walk.getElem?_toList, hlt, ite_true, Option.map_some,
    Walk.idx_encode _ hfits]
  simp [dot, tapB, j0, n]
  ring

def cName (row : Nat) : String :=
  ["C_L1L2_0_0", "C_L1L2_0_1", "C_L1L2_0_2", "C_L1L2_0_3"][row]!

/-- The C drain stream: position `J·8192 + r·2048 + (a·32 + b)` carries row `a`, column `b` of
core `r`'s `J`-th output object. -/
theorem streamC_src (host : String → Array (Nat × Int)) (f : FifoD)
    (hf : net.fifo? "C_L2L3_0" = some f) (J r a b : Nat) (hr : r < 4) (ha : a < 64) (hb : b < 32) :
    net.stream host f (J * 8192 + (r * 2048 + (a * 32 + b))) =
      some (.core r (cName r) J (layoutC a b)) := by
  have hlt := (wC_layoutC a b ha hb).2
  obtain ⟨h1, h2, h3, h4, hm, hm'⟩ :
      (J * 8192 + (r * 2048 + (a * 32 + b))) / 8192 = J ∧
      (J * 8192 + (r * 2048 + (a * 32 + b))) % 8192 = r * 2048 + (a * 32 + b) ∧
      (r * 2048 + (a * 32 + b)) / 2048 = r ∧ (r * 2048 + (a * 32 + b)) % 2048 = a * 32 + b ∧
      (J * 2048 + layoutC a b) / 2048 = J ∧ (J * 2048 + layoutC a b) % 2048 = layoutC a b :=
    ⟨by omega, by omega, by omega, by omega, by omega, by omega⟩
  have hwl : wC.len = 2048 := rfl
  have hidx := (wC_layoutC a b ha hb).1
  simp [Net.fifo?, net] at hf
  subst hf
  have hseglen : segLen ((net.links[2]).segments 8192) (some wC) = 8192 := rfl
  have hx : a * 32 + b < 2048 := by omega
  rw [Net.stream, show net.fifos.length + 1 = 12 + 1 from rfl,
    Net.streamAt_join net host (l := 2) (L := net.links[2]) (w := wC) rfl rfl (by decide) rfl rfl rfl]
  have hsegs : net.links[2].segments 8192 = [(0, 2048), (2048, 2048), (4096, 2048), (6144, 2048)] :=
    rfl
  rw [hseglen, h1, h2]
  simp only [segGather, hwl, h3, h4, hidx, Int.toNat_natCast, hsegs]
  have hoff : ((([(0, 2048), (2048, 2048), (4096, 2048), (6144, 2048)] : List (Nat × Nat))[r]?).map
      (·.1)).getD 0 = net.links[2].srcOffsets[r]! := by
    interval_cases r <;> rfl
  have hseg : segmentOf net.links[2].srcOffsets (net.links[2].srcOffsets[r]! + layoutC a b) = some r := by
    rw [show net.links[2].srcOffsets[r]! = r * 2048 by interval_cases r <;> rfl]
    exact segmentOf_four r (layoutC a b) hlt hr
  obtain ⟨g, hg, hfrom, hprod, hto, hrep, hname, hobj, hcons⟩ :
      ∃ g : FifoD, (net.links[2].srcs[r]?).bind net.fifo? = some g ∧
        g.consFrom? (.link 2) = some none ∧ g.prod = .worker r ∧ g.toStream = none ∧
        g.repeatCount = 1 ∧ g.name = cName r ∧ g.objSize = 2048 ∧ g.consSize = 2048 := by
    interval_cases r <;> exact ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
  rw [hoff, Net.linkObj_join net rfl (by decide) hseg hg hfrom, hcons,
    show (12 : Nat) = 11 + 1 from rfl, Net.streamAt_worker net host hprod hto hrep, hobj, hname, hm, hm']

/-- The output tile `C[i, j]` belongs to: row block `i / 256`, column tile `j / 32`. -/
def tauC (Nt i j : Nat) : Nat := (i / 256) * Nt + j / 32

/-- The drain position that writes `C[i, j]`. -/
def posC (Nt i j : Nat) : Nat := tauC Nt i j * 8192 + ((i % 256) / 64 * 2048 + (i % 64 * 32 + j % 32))

theorem hostC_len (Mt Kt Nt : Nat) :
    ((seq (256 * Mt) (64 * Kt) (32 * Nt)).stream "C_L2L3_0").length = Mt * (Nt * 8192) := by
  rw [streamC, length_flatten_range _ (Nt * 8192)]
  intro i _; simp [Walk.length_toList, Walk.len, tapC, dimsLen]

/-- `posC` hits `C[i, j]`. -/
theorem hostC_posC (Mt Kt Nt i j : Nat) (hi : i < 256 * Mt) (hj : j < 32 * Nt) :
    ((seq (256 * Mt) (64 * Kt) (32 * Nt)).stream "C_L2L3_0")[posC Nt i j]? =
      some (2, ((i * (32 * Nt) + j : Nat) : Int)) := by
  have hNt : 0 < Nt := by omega
  set rb := i / 256
  have hrb : rb < Mt := by omega
  have hn : j / 32 < Nt := by omega
  have hfits : Fits (tapC (32 * Nt) Nt rb).dims [j / 32, i % 256, j % 32] := by
    simp [Fits, tapC]; omega
  have hlen : ∀ k, k < Mt → ((tapC (32 * Nt) Nt k).toList.map (2, ·)).length = Nt * 8192 := by
    intro k _; simp [Walk.length_toList, Walk.len, tapC, dimsLen]
  have henc : encode (tapC (32 * Nt) Nt rb).dims [j / 32, i % 256, j % 32]
      = j / 32 * 8192 + (i % 256 * 32 + j % 32) := by
    simp [encode, tapC, dimsLen]
  have e : (i % 256) / 64 * 2048 + i % 64 * 32 = i % 256 * 32 := by omega
  have hpos : posC Nt i j = rb * (Nt * 8192) + encode (tapC (32 * Nt) Nt rb).dims [j / 32, i % 256, j % 32] := by
    rw [henc, posC, tauC, ← add_assoc ((i % 256) / 64 * 2048), e]; ring
  have hlt := (tapC (32 * Nt) Nt rb).encode_lt_len hfits
  rw [streamC, hpos, getElem?_flatten_range _ _ _ hlen hrb
    (hlt.trans_eq (by simp [Walk.len, tapC, dimsLen]))]
  simp only [List.getElem?_map, Walk.getElem?_toList, hlt, ite_true, Option.map_some,
    Walk.idx_encode _ hfits]
  simp [dot, tapC, rb]
  conv_rhs => rw [← Nat.div_add_mod i 256, ← Nat.div_add_mod j 32]
  push_cast
  ring

/-- Every drain position is `posC` of some in-range `(i, j)`. -/
theorem posC_onto (Mt Nt p : Nat) (hp : p < Mt * (Nt * 8192)) :
    ∃ i j, i < 256 * Mt ∧ j < 32 * Nt ∧ p = posC Nt i j := by
  have hNt : 0 < Nt := by
    rcases Nat.eq_zero_or_pos Nt with h | h
    · subst h; simp at hp
    · exact h
  have hL : 0 < Nt * 8192 := by positivity
  set rb := p / (Nt * 8192)
  set rest := p % (Nt * 8192)
  have hrb : rb < Mt := by rw [Nat.div_lt_iff_lt_mul hL]; exact hp
  have hrest : rest < Nt * 8192 := Nat.mod_lt _ hL
  have hn : rest / 8192 < Nt := by rw [Nat.div_lt_iff_lt_mul (by norm_num)]; exact hrest
  set n := rest / 8192
  set q := rest % 8192
  have hq : q < 8192 := Nat.mod_lt _ (by norm_num)
  refine ⟨rb * 256 + q / 32, n * 32 + q % 32, by omega, by omega, ?_⟩
  have h1 : (rb * 256 + q / 32) / 256 = rb := by omega
  have h2 : (rb * 256 + q / 32) % 256 = q / 32 := by omega
  have h3 : (rb * 256 + q / 32) % 64 = q / 32 % 64 := by omega
  have h4 : (n * 32 + q % 32) / 32 = n := by omega
  have h5 : (n * 32 + q % 32) % 32 = q % 32 := by omega
  have e : q / 32 / 64 * 2048 + (q / 32 % 64 * 32 + q % 32) = q := by omega
  rw [posC, tauC, h1, h2, h3, h4, h5, e]
  have hp' : p = rb * (Nt * 8192) + (n * 8192 + q) := by
    have e1 : Nt * 8192 * rb + rest = p := Nat.div_add_mod p _
    have e2 : 8192 * n + q = rest := Nat.div_add_mod rest _
    linarith
  rw [hp']; ring

/-- T-C: for every M = 256·Mt, K = 64·Kt, N = 32·Nt, the drain writes `C[i, j]` at exactly one
position, from row `i % 64`, column `j % 32` of core `r`'s output tile `τ`, which is the tile T-A and
T-B feed core `r` at `τ`. -/
theorem tC (Mt Kt Nt : Nat) (f : FifoD) (hf : net.fifo? "C_L2L3_0" = some f) (i j : Nat)
    (hi : i < 256 * Mt) (hj : j < 32 * Nt) :
    (∃! p : Nat, (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (32 * Nt : Nat)) "C_L2L3_0")[p]? =
        some (2, ((i * (32 * Nt) + j : Nat) : Int))) ∧
    design.stream ((256 * Mt : Nat), (64 * Kt : Nat), (32 * Nt : Nat)) f (posC Nt i j) =
      some (.core ((i % 256) / 64) (cName ((i % 256) / 64)) (tauC Nt i j) (layoutC (i % 64) (j % 32))) ∧
    i = i0 Nt ((i % 256) / 64) (tauC Nt i j) + i % 64 ∧ j = j0 1 Nt 0 (tauC Nt i j) + j % 32 := by
  have hNt : 0 < Nt := by omega
  have hj32 : j / 32 < Nt := by omega
  refine ⟨⟨posC Nt i j, ?_, ?_⟩, ?_, ?_, ?_⟩
  · simp only [Design.host, design]; push_cast
    rw [List.getElem?_toArray]; exact_mod_cast hostC_posC Mt Kt Nt i j hi hj
  · intro p hp
    simp only [Design.host, design] at hp; push_cast at hp
    rw [List.getElem?_toArray] at hp
    have hlt : p < Mt * (Nt * 8192) := by
      rw [← hostC_len Mt Kt Nt]
      exact (List.getElem?_eq_some_iff.mp (by exact_mod_cast hp)).1
    obtain ⟨i', j', hi', hj', rfl⟩ := posC_onto Mt Nt p hlt
    rw [hostC_posC Mt Kt Nt i' j' hi' hj'] at hp
    simp only [Option.some.injEq, Prod.mk.injEq, true_and] at hp
    have hnat : i' * (32 * Nt) + j' = i * (32 * Nt) + j := by exact_mod_cast hp
    have hN : 0 < 32 * Nt := by omega
    have hii : i' = i := by
      have := congrArg (· / (32 * Nt)) hnat
      rwa [Nat.add_comm, Nat.add_mul_div_right _ _ hN, Nat.div_eq_of_lt hj', Nat.zero_add,
        Nat.add_comm, Nat.add_mul_div_right _ _ hN, Nat.div_eq_of_lt hj, Nat.zero_add] at this
    subst hii
    have : j' = j := by omega
    rw [this]
  · have hr : (i % 256) / 64 < 4 := by omega
    have := streamC_src (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (32 * Nt : Nat))) f hf
      (tauC Nt i j) ((i % 256) / 64) (i % 64) (j % 32) hr (by omega) (by omega)
    rw [posC]; exact this
  · have : tauC Nt i j / Nt = i / 256 := by
      rw [tauC, Nat.add_comm, Nat.add_mul_div_right _ _ hNt, Nat.div_eq_of_lt hj32, Nat.zero_add]
    simp only [i0, this]; omega
  · have : tauC Nt i j % Nt = j / 32 := by
      rw [tauC, Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj32]
    simp only [j0, this]; omega

/-- Every drain write lands inside C, and there are exactly M·N of them. -/
theorem tC_writes (Mt Kt Nt : Nat) :
    (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (32 * Nt : Nat)) "C_L2L3_0").size =
        (256 * Mt) * (32 * Nt) ∧
    ∀ (p : Nat) e, (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (32 * Nt : Nat)) "C_L2L3_0")[p]? = some e →
      ∃ i j, i < 256 * Mt ∧ j < 32 * Nt ∧ e = (2, ((i * (32 * Nt) + j : Nat) : Int)) := by
  simp only [Design.host, design]
  push_cast
  refine ⟨by rw [List.size_toArray, hostC_len]; ring, fun p e hp => ?_⟩
  rw [List.getElem?_toArray] at hp
  have hlt : p < Mt * (Nt * 8192) := by
    rw [← hostC_len Mt Kt Nt]; exact (List.getElem?_eq_some_iff.mp hp).1
  obtain ⟨i, j, hi, hj, rfl⟩ := posC_onto Mt Nt p hlt
  rw [hostC_posC Mt Kt Nt i j hi hj] at hp
  exact ⟨i, j, hi, hj, (Option.some.inj hp).symm⟩

/-- Core `row` receives exactly `Mt·Nt` output tiles' worth of A objects, `Kt` per tile. -/
theorem countA (Mt Kt Nt : Nat) (row : Nat) (hrow : row < 4) (f : FifoD)
    (hf : net.fifo? (aName row) = some f) :
    net.streamLen (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (32 * Nt : Nat))) f =
      some (Mt * Nt * Kt * 4096) := by
  have hsize : (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (32 * Nt : Nat)) "A_L3L2_0").size
      = Mt * (Nt * (Kt * 16384)) := by
    simp only [Design.host, design]; push_cast
    rw [List.size_toArray, streamA, length_flatten_range _ (Nt * (Kt * 16384))]
    intro i _; simp [Walk.length_toList, Walk.len, tapA, dimsLen]
  have hdiv : Mt * (Nt * (Kt * 16384)) / 16384 = Mt * Nt * Kt := by
    rw [show Mt * (Nt * (Kt * 16384)) = Mt * Nt * Kt * 16384 by ring, Nat.mul_div_cancel _ (by norm_num)]
  interval_cases row <;>
  · simp [aName, Net.fifo?, net] at hf
    subst hf
    rw [Net.streamLen, show net.fifos.length + 1 = 11 + 2 from rfl,
      Net.streamLenAt_link1 net _ rfl rfl rfl rfl rfl]
    simp only [hsize, hdiv]
    simp [Net.prodSegs, segLen, Walk.len, dimsLen, net]

/-- Every core receives exactly `Mt·Nt` output tiles' worth of B objects, `Kt` per tile. -/
theorem countB (Mt Kt Nt : Nat) (f : FifoD) (hf : net.fifo? "B_L2L1_0" = some f) :
    net.streamLen (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (32 * Nt : Nat))) f =
      some (Mt * Nt * Kt * 2048) := by
  have hsize : (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (32 * Nt : Nat)) "B_L3L2_0").size
      = Mt * (Nt * (Kt * 2048)) := by
    simp only [Design.host, design]; push_cast
    rw [List.size_toArray, streamB, List.length_flatten]
    simp [Walk.length_toList, Walk.len, tapB, dimsLen, List.sum_replicate]
  have hdiv : Mt * (Nt * (Kt * 2048)) / 2048 = Mt * Nt * Kt := by
    rw [show Mt * (Nt * (Kt * 2048)) = Mt * Nt * Kt * 2048 by ring, Nat.mul_div_cancel _ (by norm_num)]
  simp [Net.fifo?, net] at hf
  subst hf
  rw [Net.streamLen, show net.fifos.length + 1 = 11 + 2 from rfl,
    Net.streamLenAt_link1 net _ rfl rfl rfl rfl rfl]
  simp only [hsize, hdiv]
  simp [Net.prodSegs, segLen, Walk.len, dimsLen, net]

end IronVerify.Proofs.WholeArrayC1
