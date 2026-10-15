-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Kernel
import IronVerify.Schema.Normalize
import IronVerify.Schema.WholeArray
import IronVerify.Schema.Pipelined

/-!
Translation validation: an exported design is an `Instance` of `schema p` when its fifo graph
normalizes to the schema's, its runtime sequence moves the same host elements on every channel
for every shape the guards accept, and its guards accept only those shapes. Theorems about
`schema p` then transfer to the export (`Instance.deliver_eq`, `stream_eq`, `streamLen_eq`,
`host_eq`, `shape`). Shapes are `Params.shapeOf`.
-/

namespace IronVerify.Schema.WholeArray

open IronVerify

/-- The dispatch values of a shape given in tiles; `Params.dims` in `Schema/Theorems.lean` is the
same function (`rfl`), kept separate so instances build without the theorem statements. -/
def Params.shapeOf (p : Params) (Mt Kt Nt : Nat) : Int × Int × Int :=
  (((p.rows * p.m * Mt : Nat) : Int), ((p.k * Kt : Nat) : Int), ((p.n * p.cols * Nt : Nat) : Int))

/-! ### Channels -/

/-- Every transfer `S` issues (taken branches, every iteration) is on a channel satisfying `P`. -/
def _root_.IronVerify.Seq.AllChans (P : String → Prop) : Seq → Prop
  | .xfer c _ _ => P c
  | .skip => True
  | .seq a b => a.AllChans P ∧ b.AllChans P
  | .ite c t e => if c then t.AllChans P else e.AllChans P
  | .loop lo hi body => ∀ i < (hi - lo).toNat, (body (lo + i)).AllChans P
  | .check _ => True

theorem _root_.IronVerify.Seq.stream_eq_nil_of_every {P : String → Prop} {ch : String}
    (hch : ¬ P ch) : ∀ {S : Seq}, S.AllChans P → S.stream ch = []
  | .xfer c _ _, h => by
    have : c ≠ ch := fun e => hch (e ▸ h)
    simp [Seq.stream, this]
  | .skip, _ => rfl
  | .seq a b, h => by
    simp [Seq.stream, stream_eq_nil_of_every hch h.1, stream_eq_nil_of_every hch h.2]
  | .ite c t e, h => by
    cases c
    · simp only [Seq.AllChans, Bool.false_eq_true, ↓reduceIte] at h
      simp [Seq.stream, stream_eq_nil_of_every hch h]
    · simp only [Seq.AllChans, ↓reduceIte] at h
      simp [Seq.stream, stream_eq_nil_of_every hch h]
  | .loop lo hi body, h => by
    simp only [Seq.stream]
    rw [List.flatten_eq_nil_iff]
    intro l hl
    rw [List.mem_map] at hl
    obtain ⟨i, hi, rfl⟩ := hl
    simp at hi
    obtain ⟨a, ha, rfl⟩ := hi
    exact stream_eq_nil_of_every hch (h a (by omega))
  | .check _, _ => rfl

theorem chanEvery_ofList {P : String → Prop} :
    ∀ {xs : List Seq}, (∀ x ∈ xs, x.AllChans P) →
      (Seq.ofList xs).AllChans P
  | [], _ => trivial
  | x :: xs, h => ⟨h x (by simp), chanEvery_ofList fun y hy => h y (List.mem_cons_of_mem _ hy)⟩

/-- The shim channels of `schema p`. -/
def chans (p : Params) : List String :=
  (List.range p.nA).map aL3 ++ (List.range p.cols).map bL3 ++ (List.range p.cols).map cL3

theorem wseq_every (p : Params) (M K N : Int) :
    (wseq p M K N).AllChans (· ∈ chans p) := by
  apply chanEvery_ofList
  intro x hx
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
  rcases hx with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> try trivial
  intro i _
  apply chanEvery_ofList
  intro x hx
  simp only [List.mem_flatMap, List.mem_range] at hx
  obtain ⟨c, hc, hx⟩ := hx
  simp only [List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hx
  simp only [chans, List.mem_append, List.mem_map, List.mem_range]
  rcases hx with (rfl | hx) | rfl
  · exact Or.inr ⟨c, hc, rfl⟩
  · split at hx
    · rename_i hA
      simp only [List.mem_singleton] at hx
      subst hx
      exact Or.inl (Or.inl ⟨c, hA, rfl⟩)
    · simp at hx
  · exact Or.inl (Or.inr ⟨c, hc, rfl⟩)

/-! ### Instances -/

/-- `d` is `schema p` up to how the exporter wrote it. -/
structure Instance (d : Design (Int × Int × Int)) (p : Params) : Prop where
  net : d.net.normWalks = (wnet p).normWalks
  pos : d.net.Pos
  wpos : (wnet p).Pos
  streams : ∀ Mt Kt Nt : Nat, 0 < Mt → 0 < Kt → 0 < Nt → ∀ ch : String,
    (d.seq (p.shapeOf Mt Kt Nt)).stream ch = ((schema p).seq (p.shapeOf Mt Kt Nt)).stream ch
  guard : ∀ P : Int × Int × Int, (d.seq P).ok = true →
    ∃ Mt Kt Nt : Nat, 0 < Mt ∧ 0 < Kt ∧ 0 < Nt ∧ P = p.shapeOf Mt Kt Nt

namespace Instance

variable {d : Design (Int × Int × Int)} {p : Params} (hi : Instance d p)
include hi

theorem host_eq {Mt Kt Nt : Nat} (hM : 0 < Mt) (hK : 0 < Kt) (hN : 0 < Nt) :
    d.host (p.shapeOf Mt Kt Nt) = (schema p).host (p.shapeOf Mt Kt Nt) := by
  funext ch
  simp only [Design.host, hi.streams Mt Kt Nt hM hK hN ch]

theorem deliver_eq {Mt Kt Nt : Nat} (hM : 0 < Mt) (hK : 0 < Kt) (hN : 0 < Nt) {name : String}
    {f₁ f₂ : FifoD} (hf₁ : d.net.fifo? name = some f₁) (hf₂ : (wnet p).fifo? name = some f₂)
    (w j s : Nat) :
    d.deliver (p.shapeOf Mt Kt Nt) f₁ w j s = (schema p).deliver (p.shapeOf Mt Kt Nt) f₂ w j s := by
  unfold Design.deliver
  rw [hi.host_eq hM hK hN]
  exact Net.deliver_congr _ hi.net hi.pos hi.wpos hf₁ hf₂ w j s

theorem stream_eq {Mt Kt Nt : Nat} (hM : 0 < Mt) (hK : 0 < Kt) (hN : 0 < Nt) {name : String}
    {f₁ f₂ : FifoD} (hf₁ : d.net.fifo? name = some f₁) (hf₂ : (wnet p).fifo? name = some f₂)
    (q : Nat) :
    d.stream (p.shapeOf Mt Kt Nt) f₁ q = (schema p).stream (p.shapeOf Mt Kt Nt) f₂ q := by
  unfold Design.stream
  rw [hi.host_eq hM hK hN]
  exact Net.stream_congr _ hi.net hi.pos hi.wpos hf₁ hf₂ q

theorem streamLen_eq {Mt Kt Nt : Nat} (hM : 0 < Mt) (hK : 0 < Kt) (hN : 0 < Nt) {name : String}
    {f₁ f₂ : FifoD} (hf₁ : d.net.fifo? name = some f₁) (hf₂ : (wnet p).fifo? name = some f₂) :
    d.net.streamLen (d.host (p.shapeOf Mt Kt Nt)) f₁ =
      (wnet p).streamLen ((schema p).host (p.shapeOf Mt Kt Nt)) f₂ := by
  rw [hi.host_eq hM hK hN]
  exact Net.streamLen_congr _ hi.net hi.pos hi.wpos hf₁ hf₂

/-- Whatever passes the export's guards is a shape the schema theorems cover. -/
theorem shape {P : Int × Int × Int} (h : (d.seq P).ok = true) :
    ∃ Mt Kt Nt : Nat, 0 < Mt ∧ 0 < Kt ∧ 0 < Nt ∧ P = p.shapeOf Mt Kt Nt :=
  hi.guard P h

end Instance

/-! ### Building instances -/

/-- Streams agree on every channel once they agree on the schema's channels. -/
theorem streams_of_chans {d : Design (Int × Int × Int)} {p : Params}
    (hd : ∀ P, (d.seq P).AllChans (· ∈ chans p))
    (hc : ∀ Mt Kt Nt : Nat, 0 < Mt → 0 < Kt → 0 < Nt → ∀ ch ∈ chans p,
      (d.seq (p.shapeOf Mt Kt Nt)).stream ch = ((schema p).seq (p.shapeOf Mt Kt Nt)).stream ch) :
    ∀ Mt Kt Nt : Nat, 0 < Mt → 0 < Kt → 0 < Nt → ∀ ch : String,
      (d.seq (p.shapeOf Mt Kt Nt)).stream ch = ((schema p).seq (p.shapeOf Mt Kt Nt)).stream ch := by
  intro Mt Kt Nt hM hK hN ch
  by_cases h : ch ∈ chans p
  · exact hc Mt Kt Nt hM hK hN ch h
  · rw [Seq.stream_eq_nil_of_every h (hd _)]
    exact (Seq.stream_eq_nil_of_every h (wseq_every p _ _ _)).symm

theorem multiple_of_tmod {M d : Int} (hd : 0 < d) (hM : M > 0) (h : Int.tmod M d = 0) :
    ∃ t : Nat, 0 < t ∧ M = d * t := by
  rw [Int.tmod_eq_emod] at h
  simp [le_of_lt hM] at h
  obtain ⟨q, hq⟩ := h
  have hq0 : 0 < q := by
    rcases lt_or_ge 0 q with h | h
    · exact h
    · nlinarith
  exact ⟨q.toNat, by omega, by rw [Int.toNat_of_nonneg hq0.le]; exact hq⟩

/-- An export opens with whole_array's six `require` guards; they pin the shape. -/
theorem guard_of_prefix (p : Params) {d : Design (Int × Int × Int)} (dm dk dn : Int)
    (hdm : dm = ((p.rows * p.m : Nat) : Int)) (hdk : dk = ((p.k : Nat) : Int))
    (hdn : dn = ((p.n * p.cols : Nat) : Int)) (hm0 : 0 < dm) (hk0 : 0 < dk) (hn0 : 0 < dn)
    (hs : ∀ M K N : Int, ∃ rest : Seq, d.seq (M, K, N) =
      .seq (.check (decide (M > 0))) (.seq (.check (decide (K > 0)))
      (.seq (.check (decide (N > 0))) (.seq (.check (decide (Int.tmod M dm = 0)))
      (.seq (.check (decide (Int.tmod K dk = 0)))
      (.seq (.check (decide (Int.tmod N dn = 0))) rest)))))) :
    ∀ P : Int × Int × Int, (d.seq P).ok = true →
      ∃ Mt Kt Nt : Nat, 0 < Mt ∧ 0 < Kt ∧ 0 < Nt ∧ P = p.shapeOf Mt Kt Nt := by
  rintro ⟨M, K, N⟩ h
  obtain ⟨rest, hr⟩ := hs M K N
  rw [hr] at h
  simp only [Seq.ok, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨hM, hK, hN, hm, hk, hn, -⟩ := h
  obtain ⟨Mt, hMt, rfl⟩ := multiple_of_tmod hm0 hM hm
  obtain ⟨Kt, hKt, rfl⟩ := multiple_of_tmod hk0 hK hk
  obtain ⟨Nt, hNt, rfl⟩ := multiple_of_tmod hn0 hN hn
  refine ⟨Mt, Kt, Nt, hMt, hKt, hNt, ?_⟩
  subst hdm hdk hdn
  simp only [Params.shapeOf]
  push_cast
  ring_nf

theorem schemaStream (p : Params) (M K N : Int) (ch : String) :
    (wseq p M K N).stream ch =
      ((List.range (M / (p.rows * p.m : Nat)).toNat).map fun rb =>
        (step p M K N (rb : Int)).stream ch).flatten := by
  simp [wseq, Seq.ofList, Seq.stream]

/-! ### Kernel contracts -/

def Layout.normalize (L : Layout) : Layout := { L with stream := L.stream.map Walk.normalize }

/-- The matmul operand layouts `schema p` assumes: `mm_stream_dims` A, B and C. -/
def mmLayouts (p : Params) : List (Option Layout) :=
  [some { rows := p.m, cols := p.k, transposed := false, stream := some (walkA p), inverse := false },
   some { rows := p.k, cols := p.n, transposed := p.bColMaj, stream := some (walkB p),
          inverse := false },
   some { rows := p.m, cols := p.n, transposed := p.cColMaj, stream := some (walkC p),
          inverse := true }]

/-- Every worker runs a three-operand kernel whose layouts are the schema's, up to normalization. -/
def kernelsMatch (p : Params) (ks : List (Nat × List KernelD)) : Bool :=
  ks.length == p.rows * p.cols &&
    ks.all fun wk => wk.2.any fun k =>
      k.args.map (fun a => a.layout.map Layout.normalize) ==
        (mmLayouts p).map (Option.map Layout.normalize)

end IronVerify.Schema.WholeArray
