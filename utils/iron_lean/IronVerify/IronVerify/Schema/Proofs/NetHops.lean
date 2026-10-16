-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Lemmas
import IronVerify.Schema.WholeArray

/-!
One hop of the schema's fifo graph at a time, for every valid `p`: the memtile split that hands
core row `row` its part of A shim `shimOf row`'s object, the forward that broadcasts B column
`c`, the join that collects column `c`'s C tiles, and the shim and core ends. Fifo lookups by
name are hypotheses here (`Names` proves them), so these lemmas compare no strings.
-/

namespace IronVerify.Schema.WholeArray.NetHops

open IronVerify

variable (p : Params) (host : String → Array (Nat × Int))

/-! ### Structure of `wnet p` -/

theorem links_aLink {i : Nat} (hi : i < p.nA) : (wnet p).links[i]? = some (aLinkD p i) := by
  simp [wnet, List.getElem?_append_left, hi]

theorem links_bLink {c : Nat} (hc : c < p.cols) :
    (wnet p).links[p.bLink c]? = some (bLinkD c) := by
  simp [wnet, Params.bLink, List.getElem?_append_left, List.getElem?_append_right, hc]

theorem links_cLink {c : Nat} (hc : c < p.cols) :
    (wnet p).links[p.cLink c]? = some (cLinkD p c) := by
  simp only [wnet, Params.cLink]
  rw [List.getElem?_append_right (by simp)]
  simp [hc]

theorem two_le_fifos_length (hp : p.Valid) : 2 ≤ (wnet p).fifos.length := by
  have := hp.rows_pos
  have := hp.cols_pos
  simp [wnet]
  omega

/-- `stream` has fuel for at least two hops. -/
theorem stream_eq (f : FifoD) (q : Nat) (hp : p.Valid) :
    (wnet p).stream host f q =
      (wnet p).streamAt host ((wnet p).fifos.length - 2 + 1 + 1 + 1) f q := by
  have := two_le_fifos_length p hp
  unfold Net.stream
  congr 1
  omega

/-! ### Walk lengths -/

theorem walkA_len (hp : p.Valid) : (walkA p).len = p.m * p.k := by
  obtain ⟨a, ha⟩ := hp.r_dvd
  obtain ⟨b, hb⟩ := hp.s_dvd
  have hr := hp.r_pos
  have hs := hp.s_pos
  simp only [Walk.len, walkA, dimsLen, Int.toNat_natCast, mul_one]
  rw [ha, hb, Nat.mul_div_cancel_left _ hr, Nat.mul_div_cancel_left _ hs]
  ring

theorem walkB_len (hp : p.Valid) : (walkB p).len = p.k * p.n := by
  obtain ⟨a, ha⟩ := hp.s_dvd
  obtain ⟨b, hb⟩ := hp.t_dvd
  have hs := hp.s_pos
  have ht := hp.t_pos
  unfold walkB
  split <;>
  · simp only [Walk.len, dimsLen, Int.toNat_natCast, mul_one]
    rw [ha, hb, Nat.mul_div_cancel_left _ hs, Nat.mul_div_cancel_left _ ht]
    ring

theorem walkC_len (hp : p.Valid) : (walkC p).len = p.m * p.n := by
  obtain ⟨a, ha⟩ := hp.r_dvd
  obtain ⟨b, hb⟩ := hp.t_dvd
  have hr := hp.r_pos
  have ht := hp.t_pos
  unfold walkC
  split <;>
  · simp only [Walk.len, dimsLen, Int.toNat_natCast, mul_one]
    rw [ha, hb, Nat.mul_div_cancel_left _ hr, Nat.mul_div_cancel_left _ ht]
    ring

/-! ### Consumers: every DMA that fills an object writes it linearly -/

/-- A consumer list whose entries all scatter linearly. -/
theorem consFrom?_linear {f : FifoD} {e : End} (hall : ∀ x ∈ f.cons, x.2 = none)
    (hmem : (e, none) ∈ f.cons) : f.consFrom? e = some none := by
  unfold FifoD.consFrom?
  obtain ⟨x, hx⟩ : ∃ x, f.cons.find? (·.1 = e) = some x :=
    Option.isSome_iff_exists.mp (List.find?_isSome.mpr ⟨_, hmem, by simp⟩)
  simp [hx, hall x (List.mem_of_find?_eq_some hx)]

theorem aL2_consFrom {row c : Nat} (hc : c < p.cols) :
    (aL2Fifo p row).consFrom? (.worker (p.worker row c)) = some none :=
  consFrom?_linear (by simp [aL2Fifo]) (by simp [aL2Fifo]; exact ⟨c, hc, rfl⟩)

theorem bL2_consFrom {row c : Nat} (hrow : row < p.rows) :
    (bL2Fifo p c).consFrom? (.worker (p.worker row c)) = some none :=
  consFrom?_linear (by simp [bL2Fifo]) (by simp [bL2Fifo]; exact ⟨row, hrow, rfl⟩)

theorem aL3_consFrom (i : Nat) : (aL3Fifo p i).consFrom? (.link i) = some none := by
  simp [FifoD.consFrom?, aL3Fifo]

theorem bL3_consFrom (c : Nat) : (bL3Fifo p c).consFrom? (.link (p.bLink c)) = some none := by
  simp [FifoD.consFrom?, bL3Fifo]

theorem cL1_consFrom (c row : Nat) :
    (cL1Fifo p c row).consFrom? (.link (p.cLink c)) = some none := by
  simp [FifoD.consFrom?, cL1Fifo]

theorem cL3_consFrom (c : Nat) : (cL3Fifo p c).consFrom? .shim = some none := by
  simp [FifoD.consFrom?, cL3Fifo]

/-- Core `(row, c)` reads its A objects straight off row `row`'s stream. -/
theorem deliver_aL2 {row c : Nat} (hc : c < p.cols) (j s : Nat) :
    (wnet p).deliver host (aL2Fifo p row) (p.worker row c) j s =
      (wnet p).stream host (aL2Fifo p row) (j * (p.m * p.k) + s) :=
  Net.deliver_linear _ _ (aL2_consFrom p hc)

/-- Core `(row, c)` reads its B objects straight off column `c`'s stream. -/
theorem deliver_bL2 {row c : Nat} (hrow : row < p.rows) (j s : Nat) :
    (wnet p).deliver host (bL2Fifo p c) (p.worker row c) j s =
      (wnet p).stream host (bL2Fifo p c) (j * (p.k * p.n) + s) :=
  Net.deliver_linear _ _ (bL2_consFrom p hrow)

/-! ### Segments -/

/-- `segmentOf` on offsets `a·0, a·1, …, a·(R-1)`: position `a·d + x` (`x < a`) is in segment `d`. -/
theorem segmentOf_mul {a R d x : Nat} (hd : d < R) (hx : x < a) :
    segmentOf ((List.range R).map (a * ·)) (a * d + x) = some d := by
  unfold segmentOf
  simp only [List.length_map, List.length_range]
  have key : ∀ R', R' ≤ R → d < R' →
      (List.range R').reverse.find? (fun r => ((List.range R).map (a * ·))[r]! ≤ a * d + x) =
        some d := by
    intro R' hR' hd'
    induction R' with
    | zero => omega
    | succ R' ih =>
      rw [List.range_succ, List.reverse_append, List.reverse_singleton, List.singleton_append,
        List.find?_cons]
      have hget : ((List.range R).map (a * ·))[R']! = a * R' := by
        rw [List.getElem!_eq_getElem?_getD]; simp [show R' < R by omega]
      rw [hget]
      by_cases hRd : R' = d
      · subst hRd; simp
      · have hlt : d < R' := by omega
        have : ¬ a * R' ≤ a * d + x := by
          have : a * (d + 1) ≤ a * R' := Nat.mul_le_mul_left _ hlt
          nlinarith
        simp only [this, decide_false]
        exact ih (by omega) hlt
  exact key R le_rfl hd

/-- The join's shared object: `rows` segments of `m·n`, segment `d` at `m·n·d`. -/
theorem cLink_segments {c : Nat} (hrow : 1 < p.rows) :
    (cLinkD p c).segments (p.m * p.n * p.rows) =
      (((List.range p.rows).map (p.m * p.n * ·)).zip
        (((List.range p.rows).map (p.m * p.n * ·)).drop 1 ++ [p.m * p.n * p.rows])).map
          fun (a, b) => (a, b - a) := by
  unfold LinkD.segments
  have h1 : (cLinkD p c).srcs.length > 1 := by simp [cLinkD]; omega
  have h2 : ¬ ((List.range p.rows).map (p.m * p.n * ·)).length ≤ 1 := by simp; omega
  simp only [h1, ite_true]
  simp only [cLinkD] at h2 ⊢
  simp only [h2, ite_false]

theorem cLink_segments_length {c : Nat} (hrow : 1 < p.rows) :
    ((cLinkD p c).segments (p.m * p.n * p.rows)).length = p.rows := by
  rw [cLink_segments p hrow]
  simp
  omega

theorem cLink_segments_fst {c d : Nat} (hrow : 1 < p.rows) (hd : d < p.rows) :
    ((cLinkD p c).segments (p.m * p.n * p.rows))[d]?.map (·.1) = some (p.m * p.n * d) := by
  have hl : d < ((cLinkD p c).segments (p.m * p.n * p.rows)).length := by
    rw [cLink_segments_length p hrow]; exact hd
  rw [List.getElem?_eq_getElem hl]
  simp only [Option.map_some, Option.some.injEq]
  simp [cLink_segments p hrow]

/-! ### Hops -/

/-- The shim end: a shim-fed stream is the host walk of its channel. -/
theorem streamAt_aL3 {fuel : Nat} (i q : Nat) :
    (wnet p).streamAt host (fuel + 1) (aL3Fifo p i) q =
      (host (aL3 i))[q]?.map fun e => .host e.1 e.2 :=
  Net.streamAt_shim _ _ rfl

theorem streamAt_bL3 {fuel : Nat} (c q : Nat) :
    (wnet p).streamAt host (fuel + 1) (bL3Fifo p c) q =
      (host (bL3 c))[q]?.map fun e => .host e.1 e.2 :=
  Net.streamAt_shim _ _ rfl

/-- The core end: core `(row, c)`'s C stream is its objects, slot by slot. -/
theorem streamAt_cL1 {fuel : Nat} (c row x : Nat) :
    (wnet p).streamAt host (fuel + 1) (cL1Fifo p c row) x =
      some (.core (p.worker row c) (cL1 c row) (x / (p.m * p.n)) (x % (p.m * p.n))) :=
  Net.streamAt_worker _ _ rfl rfl rfl

/-- The A split: element `q` of core row `row`'s stream is slot `m·k·d + walkA(q mod m·k)` of
object `q / (m·k)` of its shim's stream, where `d` is the row's place among the shim's rows. -/
theorem streamAt_aL2 (hp : p.Valid) {fuel row i d : Nat} (hi : i < p.nA)
    (hrow : p.shimOf row = i) (hd : (aLinkD p i).dsts.idxOf? (aL2 row) = some d)
    (hg : (wnet p).fifo? (aL3 i) = some (aL3Fifo p i)) (q : Nat) :
    (wnet p).streamAt host (fuel + 1) (aL2Fifo p row) q =
      (wnet p).streamAt host fuel (aL3Fifo p i)
        (q / (p.m * p.k) * (p.m * p.k * p.shimRows i) +
          (p.m * p.k * d + ((walkA p).idx (q % (p.m * p.k))).toNat)) := by
  have hlen := walkA_len p hp
  have hmk : 0 < p.m * p.k := Nat.mul_pos hp.m_pos hp.k_pos
  have hdlt : d < p.shimRows i := by
    have := List.idxOf?_eq_some_iff.mp hd |>.1
    simpa [aLinkD] using this
  rw [Net.streamAt_link1 (wnet p) host (l := i) (L := aLinkD p i) (src := aL3 i)
    (by simp [aL2Fifo, hrow]) (links_aLink p hi) rfl rfl rfl (by omega) hd hg
    (aL3_consFrom p i)]
  congr 1
  rw [hlen]
  simp only [aL3Fifo, aLinkD, List.length_map, List.length_range]
  split
  · have : d = 0 := by omega
    subst this; simp
  · rw [List.getElem!_eq_getElem?_getD]; simp [hdlt]

/-- The B forward: element `q` of column `c`'s core stream is slot `walkB(q mod k·n)` of object
`q / (k·n)` of its shim's stream. -/
theorem streamAt_bL2 (hp : p.Valid) {fuel c : Nat} (hc : c < p.cols)
    (hg : (wnet p).fifo? (bL3 c) = some (bL3Fifo p c)) (q : Nat) :
    (wnet p).streamAt host (fuel + 1) (bL2Fifo p c) q =
      (wnet p).streamAt host fuel (bL3Fifo p c)
        (q / (p.k * p.n) * (p.k * p.n) + ((walkB p).idx (q % (p.k * p.n))).toNat) := by
  have hlen := walkB_len p hp
  have hkn : 0 < p.k * p.n := Nat.mul_pos hp.k_pos hp.n_pos
  rw [Net.streamAt_link1 (wnet p) host (l := p.bLink c) (L := bLinkD c) (src := bL3 c) (d := 0)
    rfl (links_bLink p hc) rfl rfl rfl (by omega) (by simp [bLinkD, bL2Fifo]) hg (bL3_consFrom p c)]
  simp [hlen, bL3Fifo, bLinkD]

/-- The C join: element `q` of column `c`'s drain stream is slot `walkC(q mod m·n)` of object
`q / (rows·m·n)` of core row `d = (q mod rows·m·n) / (m·n)`'s C stream. -/
theorem streamAt_cL3 (hp : p.Valid) {fuel c : Nat} (hc : c < p.cols) (q : Nat)
    (hx : ((walkC p).idx (q % (p.m * p.n))).toNat < p.m * p.n)
    (hg : (wnet p).fifo? (cL1 c (q % (p.rows * (p.m * p.n)) / (p.m * p.n))) =
      some (cL1Fifo p c (q % (p.rows * (p.m * p.n)) / (p.m * p.n)))) :
    (wnet p).streamAt host (fuel + 1) (cL3Fifo p c) q =
      (wnet p).streamAt host fuel (cL1Fifo p c (q % (p.rows * (p.m * p.n)) / (p.m * p.n)))
        (q / (p.rows * (p.m * p.n)) * (p.m * p.n) + ((walkC p).idx (q % (p.m * p.n))).toNat) := by
  have hlen := walkC_len p hp
  have hmn : 0 < p.m * p.n := Nat.mul_pos hp.m_pos hp.n_pos
  have hrows := hp.rows_pos
  set d := q % (p.rows * (p.m * p.n)) / (p.m * p.n) with hd_def
  have hdlt : d < p.rows := by
    rw [hd_def, Nat.div_lt_iff_lt_mul hmn]
    exact Nat.mod_lt _ (Nat.mul_pos hrows hmn)
  have hmod : q % (p.rows * (p.m * p.n)) % (p.m * p.n) = q % (p.m * p.n) :=
    Nat.mod_mod_of_dvd _ (Dvd.intro_left _ rfl)
  rcases Nat.lt_or_ge 1 p.rows with h1 | h1
  · -- several core rows: a real join
    have hsrcs : 1 < (cLinkD p c).srcs.length := by simp [cLinkD]; omega
    rw [Net.streamAt_join (wnet p) host (l := p.cLink c) (L := cLinkD p c) (w := walkC p)
      rfl (links_cLink p hc) hsrcs rfl rfl rfl]
    have hseglen : segLen ((cLinkD p c).segments (cL3Fifo p c).objSize) (some (walkC p)) =
        p.rows * (p.m * p.n) := by
      simp only [segLen, cL3Fifo, cLink_segments_length p h1, hlen]
    rw [hseglen]
    have hgather : segGather ((cLinkD p c).segments (cL3Fifo p c).objSize) (some (walkC p))
        (q % (p.rows * (p.m * p.n))) =
        p.m * p.n * d + ((walkC p).idx (q % (p.m * p.n))).toNat := by
      simp only [segGather, hlen, cL3Fifo]
      rw [show p.m * p.n * p.rows = p.m * p.n * p.rows from rfl, cLink_segments_fst p h1 hdlt,
        hmod]
      rfl
    rw [hgather]
    have hoff : (cLinkD p c).srcOffsets[d]! = p.m * p.n * d := by
      rw [List.getElem!_eq_getElem?_getD]; simp [cLinkD, hdlt]
    rw [← hoff]
    refine Net.linkObj_join (wnet p) (links_cLink p hc) hsrcs ?_ ?_ (cL1_consFrom p c d)
    · rw [hoff]; simpa [cLinkD] using segmentOf_mul (a := p.m * p.n) hdlt hx
    · simpa [cLinkD, hdlt] using hg
  · -- one core row: the join is a forward
    have hr1 : p.rows = 1 := by omega
    have hd0 : d = 0 := Nat.lt_one_iff.mp (by rw [← hr1]; exact hdlt)
    have hsrcs : (cLinkD p c).srcs = [cL1 c 0] := by simp [cLinkD, hr1]
    rw [Net.streamAt_link1 (wnet p) host (l := p.cLink c) (L := cLinkD p c) (src := cL1 c 0)
      (d := 0) rfl (links_cLink p hc) hsrcs rfl rfl (by omega) (by simp [cLinkD, cL3Fifo])
      (by rw [← hd0]; exact hg) (cL1_consFrom p c 0)]
    simp [hd0, hlen, cL1Fifo, cLinkD, hr1]

/-! ### Whole paths, from a core's object back to the host -/

/-- Core row `row`'s A stream, from its shim's host walk. -/
theorem stream_aL2 (hp : p.Valid) {row i d : Nat} (hi : i < p.nA) (hrow : p.shimOf row = i)
    (hd : (aLinkD p i).dsts.idxOf? (aL2 row) = some d)
    (hg : (wnet p).fifo? (aL3 i) = some (aL3Fifo p i)) (q : Nat) :
    (wnet p).stream host (aL2Fifo p row) q =
      (host (aL3 i))[q / (p.m * p.k) * (p.m * p.k * p.shimRows i) +
          (p.m * p.k * d + ((walkA p).idx (q % (p.m * p.k))).toNat)]?.map
        fun e => .host e.1 e.2 := by
  rw [stream_eq p host _ _ hp, streamAt_aL2 p host hp hi hrow hd hg, streamAt_aL3]

/-- Column `c`'s B stream, from its shim's host walk. -/
theorem stream_bL2 (hp : p.Valid) {c : Nat} (hc : c < p.cols)
    (hg : (wnet p).fifo? (bL3 c) = some (bL3Fifo p c)) (q : Nat) :
    (wnet p).stream host (bL2Fifo p c) q =
      (host (bL3 c))[q / (p.k * p.n) * (p.k * p.n) + ((walkB p).idx (q % (p.k * p.n))).toNat]?.map
        fun e => .host e.1 e.2 := by
  rw [stream_eq p host _ _ hp, streamAt_bL2 p host hp hc hg, streamAt_bL3]

/-- Column `c`'s drain stream: element `q` comes from core `(d, c)`'s object `q / (rows·m·n)`,
at the slot the C walk reads. -/
theorem stream_cL3 (hp : p.Valid) {c : Nat} (hc : c < p.cols) (q : Nat)
    (hx : ((walkC p).idx (q % (p.m * p.n))).toNat < p.m * p.n)
    (hg : (wnet p).fifo? (cL1 c (q % (p.rows * (p.m * p.n)) / (p.m * p.n))) =
      some (cL1Fifo p c (q % (p.rows * (p.m * p.n)) / (p.m * p.n)))) :
    (wnet p).stream host (cL3Fifo p c) q =
      some (.core (p.worker (q % (p.rows * (p.m * p.n)) / (p.m * p.n)) c)
        (cL1 c (q % (p.rows * (p.m * p.n)) / (p.m * p.n))) (q / (p.rows * (p.m * p.n)))
        ((walkC p).idx (q % (p.m * p.n))).toNat) := by
  have hmn : 0 < p.m * p.n := Nat.mul_pos hp.m_pos hp.n_pos
  have e1 : (q / (p.rows * (p.m * p.n)) * (p.m * p.n) + ((walkC p).idx (q % (p.m * p.n))).toNat) /
      (p.m * p.n) = q / (p.rows * (p.m * p.n)) := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hmn, Nat.div_eq_of_lt hx, Nat.zero_add]
  have e2 : (q / (p.rows * (p.m * p.n)) * (p.m * p.n) + ((walkC p).idx (q % (p.m * p.n))).toNat) %
      (p.m * p.n) = ((walkC p).idx (q % (p.m * p.n))).toNat := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hx]
  rw [stream_eq p host _ _ hp, streamAt_cL3 p host hp hc q hx hg, streamAt_cL1, e1, e2]

/-! ### Stream lengths -/

/-- Core row `row` receives one A object per row tile its shim's host walk moves. -/
theorem streamLen_aL2 (hp : p.Valid) {row i : Nat} (hi : i < p.nA) (hrow : p.shimOf row = i)
    (hg : (wnet p).fifo? (aL3 i) = some (aL3Fifo p i)) :
    (wnet p).streamLen host (aL2Fifo p row) =
      some ((host (aL3 i)).size / (p.m * p.k * p.shimRows i) * (p.m * p.k)) := by
  have h2 := two_le_fifos_length p hp
  unfold Net.streamLen
  rw [show (wnet p).fifos.length + 1 = (wnet p).fifos.length - 1 + 2 by omega,
    Net.streamLenAt_link1 (wnet p) host (l := i) (L := aLinkD p i) (src := aL3 i)
      (by simp [aL2Fifo, hrow]) (links_aLink p hi) rfl hg rfl]
  simp [Net.prodSegs, aL2Fifo, hrow, links_aLink p hi, aLinkD, segLen, walkA_len p hp, aL3Fifo]

/-- Column `c` receives one B object per object its shim's host walk moves. -/
theorem streamLen_bL2 (hp : p.Valid) {c : Nat} (hc : c < p.cols)
    (hg : (wnet p).fifo? (bL3 c) = some (bL3Fifo p c)) :
    (wnet p).streamLen host (bL2Fifo p c) =
      some ((host (bL3 c)).size / (p.k * p.n) * (p.k * p.n)) := by
  have h2 := two_le_fifos_length p hp
  unfold Net.streamLen
  rw [show (wnet p).fifos.length + 1 = (wnet p).fifos.length - 1 + 2 by omega,
    Net.streamLenAt_link1 (wnet p) host (l := p.bLink c) (L := bLinkD c) (src := bL3 c)
      rfl (links_bLink p hc) rfl hg rfl]
  simp [Net.prodSegs, bL2Fifo, links_bLink p hc, bLinkD, segLen, walkB_len p hp, bL3Fifo]

end IronVerify.Schema.WholeArray.NetHops
