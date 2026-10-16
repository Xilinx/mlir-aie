-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Mathlib
import IronVerify.Net

namespace IronVerify

theorem Walk.length_toList (w : Walk) : w.toList.length = w.len := by
  simp [Walk.toList]

theorem Walk.getElem?_toList (w : Walk) (p : Nat) :
    w.toList[p]? = if p < w.len then some (w.idx p) else none := by
  unfold Walk.toList
  split <;> simp_all

theorem length_flatten_range {α : Type} (f : Nat → List α) (L : Nat) :
    ∀ R, (∀ i, i < R → (f i).length = L) → ((List.range R).map f).flatten.length = R * L
  | 0, _ => by simp
  | R + 1, hL => by
    rw [List.range_succ, List.map_append, List.flatten_append, List.length_append,
      length_flatten_range f L R (fun i hi => hL i (by omega))]
    simp [hL R (by omega), Nat.succ_mul]

/-- Indexing the concatenation of `R` lists that each have length `L`. -/
theorem getElem?_flatten_range {α : Type} (f : Nat → List α) (L R : Nat)
    (hL : ∀ i, i < R → (f i).length = L) {i q : Nat} (hi : i < R) (hq : q < L) :
    ((List.range R).map f).flatten[i * L + q]? = (f i)[q]? := by
  induction R with
  | zero => omega
  | succ R ih =>
    have hlen := length_flatten_range f L R (fun j hj => hL j (by omega))
    rw [List.range_succ, List.map_append, List.flatten_append]
    by_cases h : i < R
    · rw [List.getElem?_append_left]
      · exact ih (fun j hj => hL j (by omega)) h
      · rw [hlen]; nlinarith
    · have hiR : i = R := by omega
      subst hiR
      rw [List.getElem?_append_right]
      · simp [hlen]
      · rw [hlen]; exact Nat.le_add_right _ _

theorem getElem?_flatten_replicate {α : Type} (l : List α) (R : Nat) {i q : Nat} (hi : i < R)
    (hq : q < l.length) : (List.replicate R l).flatten[i * l.length + q]? = l[q]? := by
  have := getElem?_flatten_range (fun _ => l) l.length R (fun _ _ => rfl) hi hq
  simpa [List.map_const'] using this

/-- Mixed-radix digits, outermost first, against a walk's dims. -/
def Fits : List (Int × Int) → List Nat → Prop
  | [], [] => True
  | (s, _) :: ds, d :: dd => d < s.toNat ∧ Fits ds dd
  | _, _ => False

/-- The stream position of digits `dd`. -/
def encode : List (Int × Int) → List Nat → Nat
  | (_, _) :: ds, d :: dd => d * dimsLen ds + encode ds dd
  | _, _ => 0

/-- The element offset digits `dd` reach. -/
def dot : List (Int × Int) → List Nat → Int
  | (_, t) :: ds, d :: dd => d * t + dot ds dd
  | _, _ => 0

theorem encode_lt : ∀ {ds : List (Int × Int)} {dd : List Nat}, Fits ds dd →
    encode ds dd < dimsLen ds
  | [], [], _ => by simp [encode, dimsLen]
  | (s, t) :: ds, d :: dd, ⟨hd, h⟩ => by
    have := encode_lt h
    simp only [encode, dimsLen]
    have : (d + 1) * dimsLen ds ≤ s.toNat * dimsLen ds := Nat.mul_le_mul_right _ hd
    nlinarith

theorem walkAt_encode : ∀ {ds : List (Int × Int)} {dd : List Nat}, Fits ds dd →
    walkAt ds (encode ds dd) = dot ds dd
  | [], [], _ => by simp [walkAt, dot]
  | (s, t) :: ds, d :: dd, ⟨_, h⟩ => by
    have hlt := encode_lt h
    have hpos : 0 < dimsLen ds := by omega
    simp only [walkAt, encode, dot]
    rw [show (d * dimsLen ds + encode ds dd) / dimsLen ds = d by
          rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hlt, Nat.zero_add],
        show (d * dimsLen ds + encode ds dd) % dimsLen ds = encode ds dd by
          rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hlt],
        walkAt_encode h]

theorem Walk.idx_encode (w : Walk) {dd : List Nat} (h : Fits w.dims dd) :
    w.idx (encode w.dims dd) = w.offset + dot w.dims dd := by
  simp [Walk.idx, walkAt_encode h]

theorem Walk.encode_lt_len (w : Walk) {dd : List Nat} (h : Fits w.dims dd) :
    encode w.dims dd < w.len := encode_lt h

namespace Net

variable (net : Net) (host : String → Array (Nat × Int))

theorem deliver_linear {f : FifoD} {w j s : Nat} (h : f.consFrom? (.worker w) = some none) :
    net.deliver host f w j s = net.stream host f (j * f.consSize + s) := by
  simp [deliver, consObj, h, segScatter]

theorem streamAt_shim {fuel : Nat} {f : FifoD} {p : Nat} (h : f.prod = .shim) :
    net.streamAt host (fuel + 1) f p = (host f.name)[p]?.map fun e => .host e.1 e.2 := by
  rw [streamAt]; simp [h]

theorem streamAt_worker {fuel i : Nat} {g : FifoD} {p : Nat} (h : g.prod = .worker i)
    (hto : g.toStream = none) (hrep : g.repeatCount = 1) :
    net.streamAt host (fuel + 1) g p = some (.core i g.name (p / g.objSize) (p % g.objSize)) := by
  rw [streamAt]; simp [h, hto, hrep, prodSegs, segLen, segGather, prodObj]

/-- A link with one source (a forward or a distribute), seen from one of its destinations. -/
theorem streamAt_link1 {fuel l d : Nat} {f g : FifoD} {L : LinkD} {src : String} {w : Walk}
    {p : Nat} (hprod : f.prod = .link l) (hL : net.links[l]? = some L) (hsrcs : L.srcs = [src])
    (hto : f.toStream = some w) (hrep : f.repeatCount = 1) (hw : 0 < w.len)
    (hd : L.dsts.idxOf? f.name = some d) (hg : net.fifo? src = some g)
    (hfrom : g.consFrom? (.link l) = some none) :
    net.streamAt host (fuel + 1) f p =
      net.streamAt host fuel g ((p / w.len) * g.consSize +
        (if L.dsts.length = 1 then (w.idx (p % w.len)).toNat
         else L.dstOffsets[d]! + (w.idx (p % w.len)).toNat)) := by
  have hq : p % w.len / w.len = 0 := Nat.div_eq_of_lt (Nat.mod_lt _ hw)
  rw [streamAt]
  simp only [hprod, prodSegs, hL, hsrcs, List.length_singleton, gt_iff_lt, lt_self_iff_false,
    ite_false, segLen, hto, one_mul, hrep, max_self, mul_one, segGather, hq, prodObj]
  simp only [Option.bind_eq_bind, Option.bind_some, hd, Nat.mod_mod]
  split <;> simp [linkObj, hL, hsrcs, hg, consObj, hfrom, segScatter]

/-- A join, seen from its destination: object `p / per` of the shared buffer, at the slot
the destination's walk of the source segments reads. -/
theorem streamAt_join {fuel l : Nat} {f : FifoD} {L : LinkD} {w : Walk} {p : Nat}
    (hprod : f.prod = .link l) (hL : net.links[l]? = some L) (hlen : 1 < L.srcs.length)
    (hto : f.toStream = some w) (hrep : f.repeatCount = 1) (hdst : L.dsts = [f.name]) :
    net.streamAt host (fuel + 1) f p =
      net.linkObj (net.streamAt host fuel) l (p / segLen (L.segments f.objSize) (some w))
        (segGather (L.segments f.objSize) (some w) (p % segLen (L.segments f.objSize) (some w))) := by
  rw [streamAt]
  simp [hprod, prodSegs, hL, hlen, hto, hrep, prodObj, hdst]

/-- Source `r` of a join fills its own segment of the shared buffer. -/
theorem linkObj_join {l r : Nat} {L : LinkD} {g : FifoD} {rec : FifoD → Nat → Option Src}
    {j x : Nat} (hL : net.links[l]? = some L) (hlen : 1 < L.srcs.length)
    (hseg : segmentOf L.srcOffsets (L.srcOffsets[r]! + x) = some r)
    (hg : (L.srcs[r]?).bind net.fifo? = some g) (hfrom : g.consFrom? (.link l) = some none) :
    net.linkObj rec l j (L.srcOffsets[r]! + x) = rec g (j * g.consSize + x) := by
  unfold linkObj
  simp only [hL, Option.bind_eq_bind, Option.bind_some]
  rcases hs : L.srcs with _ | ⟨a, _ | ⟨b, t⟩⟩
  · simp [hs] at hlen
  · simp [hs] at hlen
  · rw [hs] at hg
    rcases hr : (a :: b :: t)[r]? with _ | src
    · simp [hr] at hg
    · simp only [hr, Option.bind_some] at hg
      simp only [List.getElem!_eq_getElem?_getD, Nat.default_eq_zero] at hseg
      simp [hseg, hr, hg, consObj, hfrom, segScatter]

/-- A link with one shim-fed source: its destinations carry one walk per object the source delivers. -/
theorem streamLenAt_link1 {fuel l : Nat} {f g : FifoD} {L : LinkD} {src : String}
    (hprod : f.prod = .link l) (hL : net.links[l]? = some L) (hsrcs : L.srcs = [src])
    (hg : net.fifo? src = some g) (hgp : g.prod = .shim) :
    net.streamLenAt host (fuel + 2) f =
      some ((host g.name).size / g.consSize * segLen (net.prodSegs f) f.toStream *
        max f.repeatCount 1) := by
  rw [streamLenAt]
  simp [hprod, hL, hsrcs, hg, streamLenAt, hgp]

end Net

end IronVerify
