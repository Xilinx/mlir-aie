-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Net

/-!
taplib drops and merges dims when it builds a walk (leading unit dims get stride 0, `inverse()`
coalesces), so an exported walk and the schema's clean mixed-radix walk can differ in shape
while reading the same elements. `Walk.normalize` drops unit dims and merges contiguous ones;
instances compare nets after normalizing every walk, and `Net.deliver_congr`,
`Net.stream_congr` and `Net.streamLen_congr` say the semantics cannot tell the difference.
-/

namespace IronVerify

/-- Merge `(s₁, t₁), (s₂, t₂)` into `(s₁ · s₂, t₂)` whenever `t₁ = s₂ · t₂`, innermost first. -/
def mergeDims : List (Int × Int) → List (Int × Int)
  | [] => []
  | d :: ds =>
    match mergeDims ds with
    | (s₂, t₂) :: rest => if d.2 = s₂ * t₂ then (d.1 * s₂, t₂) :: rest else d :: (s₂, t₂) :: rest
    | [] => [d]

/-- Whether a dim is kept: unit dims read one element, so they can go. -/
def keepDim (d : Int × Int) : Bool := d.1 != 1

/-- The same walk with unit dims dropped and contiguous dims merged. -/
def Walk.normalize (w : Walk) : Walk :=
  { offset := w.offset, dims := mergeDims (w.dims.filter keepDim) }

def FifoD.normWalks (f : FifoD) : FifoD :=
  { f with toStream := f.toStream.map Walk.normalize,
           cons := f.cons.map fun c => (c.1, c.2.map Walk.normalize) }

def Net.normWalks (n : Net) : Net := { n with fifos := n.fifos.map FifoD.normWalks }

/-! ### Walks -/

/-- Every dim of every walk has a positive size; checked per instance by `decide`. -/
def Walk.posB (w : Walk) : Bool := w.dims.all fun d => decide (0 < d.1)

def FifoD.posB (f : FifoD) : Bool :=
  (f.toStream.map Walk.posB).getD true && f.cons.all fun c => (c.2.map Walk.posB).getD true

def Net.posB (n : Net) : Bool := n.fifos.all FifoD.posB

theorem dimsLen_pos : ∀ {ds : List (Int × Int)}, (∀ d ∈ ds, 0 < d.1) → 0 < dimsLen ds
  | [], _ => by simp [dimsLen]
  | (s, t) :: ds, h => by
    have h1 : 0 < s := h (s, t) (by simp)
    have h2 := dimsLen_pos (fun d hd => h d (List.mem_cons_of_mem _ hd))
    simp only [dimsLen]
    exact Nat.mul_pos (by omega) h2

theorem filter_unit_sound : ∀ (ds : List (Int × Int)), (∀ d ∈ ds, 0 < d.1) →
    dimsLen (ds.filter keepDim) = dimsLen ds ∧
      ∀ q < dimsLen ds, walkAt (ds.filter keepDim) q = walkAt ds q
  | [], _ => by simp
  | (s, t) :: ds, h => by
    have hds := fun d hd => h d (List.mem_cons_of_mem (s, t) hd)
    obtain ⟨hl, hw⟩ := filter_unit_sound ds hds
    have hL := dimsLen_pos hds
    by_cases hs : s = 1
    · subst hs
      have hk : keepDim (1, t) = false := by simp [keepDim]
      refine ⟨?_, fun q hq => ?_⟩
      · rw [List.filter_cons, hk]; simp [dimsLen, hl]
      · simp only [dimsLen, Int.toNat_one, Nat.one_mul] at hq
        rw [List.filter_cons, hk]
        simp only [Bool.false_eq_true, ↓reduceIte, walkAt, Nat.div_eq_of_lt hq,
          Nat.mod_eq_of_lt hq, hw q hq]
        simp
    · have hk : keepDim (s, t) = true := by simp [keepDim, hs]
      refine ⟨?_, fun q hq => ?_⟩
      · rw [List.filter_cons, hk]; simp [dimsLen, hl]
      · rw [List.filter_cons, hk]
        simp only [↓reduceIte, walkAt, hl]
        rw [hw _ (Nat.mod_lt _ hL)]

theorem mergeDims_pos : ∀ (ds : List (Int × Int)), (∀ d ∈ ds, 0 < d.1) →
    ∀ d ∈ mergeDims ds, 0 < d.1
  | [], _ => by simp [mergeDims]
  | d :: ds, h => by
    have hd : 0 < d.1 := h d (by simp)
    have ih := mergeDims_pos ds (fun e he => h e (List.mem_cons_of_mem d he))
    intro e he
    simp only [mergeDims] at he
    split at he
    · rename_i s₂ t₂ rest hm
      have h2 : 0 < s₂ := ih (s₂, t₂) (by rw [hm]; simp)
      split at he
      · simp only [List.mem_cons] at he
        rcases he with rfl | he
        · exact Int.mul_pos hd h2
        · exact ih e (by rw [hm]; exact List.mem_cons_of_mem _ he)
      · simp only [List.mem_cons] at he
        rcases he with rfl | he
        · exact hd
        · exact ih e (by rw [hm]; simpa using he)
    · simp only [List.mem_singleton] at he
      subst he
      exact hd

theorem mergeDims_sound : ∀ (ds : List (Int × Int)), (∀ d ∈ ds, 0 < d.1) →
    dimsLen (mergeDims ds) = dimsLen ds ∧
      ∀ q < dimsLen ds, walkAt (mergeDims ds) q = walkAt ds q
  | [], _ => by simp [mergeDims]
  | (d1, d2) :: ds, h => by
    have hds := fun e he => h e (List.mem_cons_of_mem (d1, d2) he)
    have hd : 0 < d1 := h (d1, d2) (by simp)
    obtain ⟨hl, hw⟩ := mergeDims_sound ds hds
    have hL := dimsLen_pos hds
    have hpos := mergeDims_pos ds hds
    rcases hm : mergeDims ds with _ | ⟨⟨s₂, t₂⟩, rest⟩
    · rw [hm] at hl hw
      simp only [dimsLen] at hl
      have hw0 := hw 0 (by omega)
      simp only [walkAt] at hw0
      refine ⟨by simp [mergeDims, hm, dimsLen, ← hl], fun q _ => ?_⟩
      simp only [mergeDims, hm, walkAt, dimsLen, ← hl, Nat.div_one, Nat.mod_one, ← hw0]
    · rw [hm] at hl hw hpos
      have h2 : 0 < s₂ := hpos (s₂, t₂) (by simp)
      have hR := dimsLen_pos (fun e he => hpos e (List.mem_cons_of_mem _ he))
      simp only [dimsLen] at hl
      generalize hRdef : dimsLen rest = R at hl hR hw
      have hS : ((s₂.toNat : Nat) : Int) = s₂ := Int.toNat_of_nonneg h2.le
      generalize hSdef : s₂.toNat = S at hl hS
      by_cases hc : d2 = s₂ * t₂
      · refine ⟨?_, fun q hq => ?_⟩
        · simp only [mergeDims, hm, hc, ↓reduceIte, dimsLen]
          rw [Int.toNat_mul hd.le h2.le, Nat.mul_assoc, hSdef, hRdef, hl]
        · simp only [mergeDims, hm, hc, ↓reduceIte, walkAt, hRdef]
          rw [← hw _ (Nat.mod_lt _ hL)]
          simp only [walkAt, hRdef]
          rw [← hl, Nat.mod_mul_left_mod]
          have key : q / R = S * (q / (S * R)) + q % (S * R) / R := by
            have e : q = R * (S * (q / (S * R))) + q % (S * R) := by
              have := Nat.div_add_mod q (S * R)
              calc q = S * R * (q / (S * R)) + q % (S * R) := this.symm
                _ = R * (S * (q / (S * R))) + q % (S * R) := by ring
            conv_lhs => rw [e]
            rw [Nat.mul_add_div hR]
          rw [key]
          push_cast
          rw [← hS]
          ring
      · have hlen : dimsLen ((s₂, t₂) :: rest) = dimsLen ds := by
          simp only [dimsLen, hSdef, hRdef]; exact hl
        refine ⟨?_, fun q hq => ?_⟩
        · simp only [mergeDims, hm, hc, ↓reduceIte]
          simp only [dimsLen] at hlen ⊢
          rw [hlen]
        · have hwq := hw (q % dimsLen ds) (Nat.mod_lt _ hL)
          simp only [walkAt] at hwq
          simp only [mergeDims, hm, hc, ↓reduceIte]
          simp only [walkAt, hlen]
          rw [hwq]

/-- A walk whose dims all have positive sizes. -/
def Walk.Pos (w : Walk) : Prop := ∀ d ∈ w.dims, 0 < d.1

theorem Walk.pos_of_posB {w : Walk} (h : w.posB = true) : w.Pos := by
  simpa [Walk.posB, Walk.Pos] using h

theorem Walk.len_pos {w : Walk} (h : w.Pos) : 0 < w.len := dimsLen_pos h

/-- Normalizing keeps a walk's length and every element it reads. -/
theorem Walk.normalize_sound (w : Walk) (hpos : ∀ d ∈ w.dims, 0 < d.1) :
    w.normalize.len = w.len ∧ ∀ q < w.len, w.normalize.idx q = w.idx q := by
  have hf := filter_unit_sound w.dims hpos
  have hfp : ∀ d ∈ w.dims.filter keepDim, 0 < d.1 := fun d hd => hpos d (List.mem_filter.1 hd).1
  have hm := mergeDims_sound _ hfp
  refine ⟨by simp [Walk.normalize, Walk.len, hm.1, hf.1], fun q hq => ?_⟩
  simp only [Walk.normalize, Walk.idx]
  rw [hm.2 q (by rw [hf.1]; exact hq), hf.2 q hq]

theorem Walk.normalize_len {w : Walk} (h : w.Pos) : w.normalize.len = w.len :=
  (w.normalize_sound h).1

theorem Walk.normalize_idx {w : Walk} (h : w.Pos) {q : Nat} (hq : q < w.len) :
    w.normalize.idx q = w.idx q :=
  (w.normalize_sound h).2 q hq

/-! ### The semantics only reads a walk's length and its elements -/

theorem segLen_norm (segs : List (Nat × Nat)) {w : Option Walk} (h : ∀ v, w = some v → v.Pos) :
    segLen segs (w.map Walk.normalize) = segLen segs w := by
  cases w with
  | none => rfl
  | some v => simp [segLen, Walk.normalize_len (h v rfl)]

theorem segGather_norm (segs : List (Nat × Nat)) {w : Option Walk}
    (h : ∀ v, w = some v → v.Pos) (q : Nat) :
    segGather segs (w.map Walk.normalize) q = segGather segs w q := by
  cases w with
  | none => rfl
  | some v =>
    have hv := h v rfl
    have hl := Walk.normalize_len hv
    simp only [Option.map_some, segGather, hl]
    rw [Walk.normalize_idx hv (Nat.mod_lt _ (Walk.len_pos hv))]

theorem find?_congr_mem {α : Type} {p q : α → Bool} :
    ∀ {l : List α}, (∀ x ∈ l, p x = q x) → l.find? p = l.find? q
  | [], _ => rfl
  | x :: xs, h => by
    simp only [List.find?_cons, h x (by simp)]
    split <;> [rfl; exact find?_congr_mem fun y hy => h y (List.mem_cons_of_mem _ hy)]

theorem scatterPos_norm {w : Walk} (hw : w.Pos) (slot : Nat) :
    scatterPos w.normalize slot = scatterPos w slot := by
  unfold scatterPos
  rw [Walk.normalize_len hw]
  apply find?_congr_mem
  intro q hq
  simp only [List.mem_reverse, List.mem_range] at hq
  rw [Walk.normalize_idx hw hq]

theorem segScatter_norm (segs : List (Nat × Nat)) {w : Option Walk}
    (h : ∀ v, w = some v → v.Pos) (slot : Nat) :
    segScatter segs (w.map Walk.normalize) slot = segScatter segs w slot := by
  cases w with
  | none => rfl
  | some v =>
    have hv := h v rfl
    simp only [Option.map_some, segScatter, scatterPos_norm hv, Walk.normalize_len hv]

/-! ### Nets -/

/-- Every walk of `f` has positive dims. -/
def FifoD.Pos (f : FifoD) : Prop :=
  (∀ v, f.toStream = some v → v.Pos) ∧ ∀ c ∈ f.cons, ∀ v, c.2 = some v → v.Pos

def Net.Pos (n : Net) : Prop := ∀ f ∈ n.fifos, f.Pos

theorem FifoD.pos_of_posB {f : FifoD} (h : f.posB = true) : f.Pos := by
  simp only [FifoD.posB, Bool.and_eq_true, List.all_eq_true] at h
  refine ⟨fun v hv => ?_, fun c hc v hv => ?_⟩
  · simpa [hv] using Walk.pos_of_posB (by simpa [hv] using h.1)
  · have := h.2 c hc
    rw [hv] at this
    exact Walk.pos_of_posB (by simpa using this)

theorem Net.pos_of_posB {n : Net} (h : n.posB = true) : n.Pos := by
  intro f hf
  simp only [Net.posB, List.all_eq_true] at h
  exact FifoD.pos_of_posB (h f hf)

@[simp] theorem FifoD.normWalks_name (f : FifoD) : f.normWalks.name = f.name := rfl
@[simp] theorem FifoD.normWalks_objSize (f : FifoD) : f.normWalks.objSize = f.objSize := rfl
@[simp] theorem FifoD.normWalks_consSize (f : FifoD) : f.normWalks.consSize = f.consSize := rfl
@[simp] theorem FifoD.normWalks_repeatCount (f : FifoD) :
    f.normWalks.repeatCount = f.repeatCount := rfl
@[simp] theorem FifoD.normWalks_prod (f : FifoD) : f.normWalks.prod = f.prod := rfl
@[simp] theorem FifoD.normWalks_toStream (f : FifoD) :
    f.normWalks.toStream = f.toStream.map Walk.normalize := rfl
@[simp] theorem Net.normWalks_links (n : Net) : n.normWalks.links = n.links := rfl
@[simp] theorem Net.normWalks_fifos (n : Net) : n.normWalks.fifos = n.fifos.map FifoD.normWalks :=
  rfl

theorem FifoD.consFrom?_norm (f : FifoD) (e : End) :
    f.normWalks.consFrom? e = (f.consFrom? e).map (Option.map Walk.normalize) := by
  simp only [FifoD.consFrom?, FifoD.normWalks]
  induction f.cons with
  | nil => rfl
  | cons c cs ih =>
    by_cases h : c.1 = e
    · simp [h]
    · simpa [List.find?_cons, h] using ih

theorem Net.fifo?_norm (n : Net) (name : String) :
    n.normWalks.fifo? name = (n.fifo? name).map FifoD.normWalks := by
  simp only [Net.fifo?, Net.normWalks_fifos, List.find?_map]
  rfl

theorem Net.prodSegs_norm (n : Net) (f : FifoD) :
    n.normWalks.prodSegs f.normWalks = n.prodSegs f := by
  unfold Net.prodSegs; simp

theorem Net.consSegs_norm (n : Net) (g : FifoD) (e : End) :
    n.normWalks.consSegs g.normWalks e = n.consSegs g e := by
  unfold Net.consSegs; simp

theorem Net.mem_of_fifo? {n : Net} {name : String} {g : FifoD} (h : n.fifo? name = some g) :
    g ∈ n.fifos := List.mem_of_find?_eq_some h

section Congr

variable (n : Net) (host : String → Array (Nat × Int))

theorem mapM_congr_mem {α β : Type} {f g : α → Option β} :
    ∀ {l : List α}, (∀ x ∈ l, f x = g x) → l.mapM f = l.mapM g
  | [], _ => rfl
  | x :: xs, h => by
    simp only [List.mapM_cons, h x (by simp),
      mapM_congr_mem (fun y hy => h y (List.mem_cons_of_mem _ hy))]

theorem Net.consObj_norm (hn : n.Pos) (rec rec' : FifoD → Nat → Option Src)
    (hrec : ∀ g ∈ n.fifos, ∀ p, rec' g.normWalks p = rec g p) {g : FifoD} (hg : g ∈ n.fifos)
    (e : End) (j s : Nat) :
    n.normWalks.consObj rec' g.normWalks e j s = n.consObj rec g e j s := by
  unfold Net.consObj
  rw [FifoD.consFrom?_norm, Net.consSegs_norm]
  rcases hc : g.consFrom? e with _ | from_
  · rfl
  · have hfrom : ∀ v, from_ = some v → v.Pos := by
      intro v hv
      simp only [FifoD.consFrom?, Option.map_eq_some_iff] at hc
      obtain ⟨c, hc, rfl⟩ := hc
      exact (hn g hg).2 c (List.mem_of_find?_eq_some hc) v hv
    simp only [Option.map_some, Option.bind_eq_bind, Option.bind_some]
    rw [segScatter_norm _ hfrom]
    rcases segScatter (n.consSegs g e) from_ s with _ | q
    · rfl
    · simp [hrec g hg]

theorem Net.linkObj_norm (hn : n.Pos) (rec rec' : FifoD → Nat → Option Src)
    (hrec : ∀ g ∈ n.fifos, ∀ p, rec' g.normWalks p = rec g p) (l j s : Nat) :
    n.normWalks.linkObj rec' l j s = n.linkObj rec l j s := by
  unfold Net.linkObj
  simp only [Net.normWalks_links]
  rcases n.links[l]? with _ | L
  · rfl
  · simp only [Option.bind_eq_bind, Option.bind_some]
    split
    · rw [Net.fifo?_norm]
      rcases hg : n.fifo? _ with _ | g
      · rfl
      · simp only [Option.map_some, Option.bind_some]
        exact n.consObj_norm hn rec rec' hrec (Net.mem_of_fifo? hg) _ _ _
    · rcases segmentOf L.srcOffsets s with _ | r
      · rfl
      · simp only [Option.bind_some]
        rcases L.srcs[r]? with _ | name
        · rfl
        · simp only [Option.bind_some]
          rw [Net.fifo?_norm]
          rcases hg : n.fifo? name with _ | g
          · rfl
          · simp only [Option.map_some, Option.bind_some]
            exact n.consObj_norm hn rec rec' hrec (Net.mem_of_fifo? hg) _ _ _

theorem Net.prodObj_norm (hn : n.Pos) (rec rec' : FifoD → Nat → Option Src)
    (hrec : ∀ g ∈ n.fifos, ∀ p, rec' g.normWalks p = rec g p) (f : FifoD) (j s : Nat) :
    n.normWalks.prodObj rec' f.normWalks j s = n.prodObj rec f j s := by
  unfold Net.prodObj
  simp only [FifoD.normWalks_prod, FifoD.normWalks_name, Net.normWalks_links]
  split
  · rfl
  · rfl
  · rcases n.links[_]? with _ | L
    · rfl
    · simp only [Option.bind_eq_bind, Option.bind_some]
      split
      · split <;> exact n.linkObj_norm hn rec rec' hrec _ _ _
      · rfl

theorem Net.streamAt_norm (hn : n.Pos) :
    ∀ fuel, ∀ f ∈ n.fifos, ∀ p,
      n.normWalks.streamAt host fuel f.normWalks p = n.streamAt host fuel f p
  | 0, _, _, _ => rfl
  | fuel + 1, f, hf, p => by
    have ih := Net.streamAt_norm hn fuel
    have hto : ∀ v, f.toStream = some v → v.Pos := (hn f hf).1
    unfold Net.streamAt
    simp only [FifoD.normWalks_prod, FifoD.normWalks_name, FifoD.normWalks_toStream,
      FifoD.normWalks_repeatCount, Net.prodSegs_norm]
    split
    · rfl
    · rw [segLen_norm _ hto, segGather_norm _ hto]
      exact n.prodObj_norm hn _ _ ih f _ _

theorem Net.streamLenAt_norm (hn : n.Pos) :
    ∀ fuel, ∀ f ∈ n.fifos, n.normWalks.streamLenAt host fuel f.normWalks = n.streamLenAt host fuel f
  | 0, _, _ => rfl
  | fuel + 1, f, hf => by
    have ih := Net.streamLenAt_norm hn fuel
    have hto : ∀ v, f.toStream = some v → v.Pos := (hn f hf).1
    unfold Net.streamLenAt
    simp only [FifoD.normWalks_prod, FifoD.normWalks_name, FifoD.normWalks_toStream,
      FifoD.normWalks_repeatCount, Net.prodSegs_norm, Net.normWalks_links]
    split
    · rfl
    · rfl
    · rcases n.links[_]? with _ | L
      · rfl
      · simp only [Option.bind_eq_bind, Option.bind_some, segLen_norm _ hto]
        congr 1
        apply mapM_congr_mem
        intro s _
        rw [Net.fifo?_norm]
        rcases hg : n.fifo? s with _ | g
        · rfl
        · simp [ih g (Net.mem_of_fifo? hg)]

theorem Net.stream_norm (hn : n.Pos) {f : FifoD} (hf : f ∈ n.fifos) (p : Nat) :
    n.normWalks.stream host f.normWalks p = n.stream host f p := by
  unfold Net.stream
  simp only [Net.normWalks_fifos, List.length_map]
  exact n.streamAt_norm host hn _ f hf p

theorem Net.streamLen_norm (hn : n.Pos) {f : FifoD} (hf : f ∈ n.fifos) :
    n.normWalks.streamLen host f.normWalks = n.streamLen host f := by
  unfold Net.streamLen
  simp only [Net.normWalks_fifos, List.length_map]
  exact n.streamLenAt_norm host hn _ f hf

theorem Net.deliver_norm (hn : n.Pos) {f : FifoD} (hf : f ∈ n.fifos) (w j s : Nat) :
    n.normWalks.deliver host f.normWalks w j s = n.deliver host f w j s := by
  unfold Net.deliver
  exact n.consObj_norm hn _ _ (fun g hg p => n.stream_norm host hn hg p) hf _ _ _

end Congr

/-! ### Two nets that normalize equal have the same semantics -/

section TwoNets

variable {n₁ n₂ : Net} (host : String → Array (Nat × Int))

/-- The fifos two nets that normalize equal give a name. -/
theorem Net.normWalks_fifo? (h : n₁.normWalks = n₂.normWalks) {name : String} {f₁ f₂ : FifoD}
    (hf₁ : n₁.fifo? name = some f₁) (hf₂ : n₂.fifo? name = some f₂) :
    f₁.normWalks = f₂.normWalks := by
  have e₁ := Net.fifo?_norm n₁ name
  have e₂ := Net.fifo?_norm n₂ name
  rw [hf₁, h, e₂, hf₂] at e₁
  exact (Option.some.inj e₁).symm

theorem Net.stream_congr (h : n₁.normWalks = n₂.normWalks) (h₁ : n₁.Pos) (h₂ : n₂.Pos)
    {name : String} {f₁ f₂ : FifoD} (hf₁ : n₁.fifo? name = some f₁)
    (hf₂ : n₂.fifo? name = some f₂) (p : Nat) :
    n₁.stream host f₁ p = n₂.stream host f₂ p := by
  rw [← n₁.stream_norm host h₁ (Net.mem_of_fifo? hf₁), ← n₂.stream_norm host h₂
    (Net.mem_of_fifo? hf₂), h, Net.normWalks_fifo? h hf₁ hf₂]

theorem Net.streamLen_congr (h : n₁.normWalks = n₂.normWalks) (h₁ : n₁.Pos) (h₂ : n₂.Pos)
    {name : String} {f₁ f₂ : FifoD} (hf₁ : n₁.fifo? name = some f₁)
    (hf₂ : n₂.fifo? name = some f₂) :
    n₁.streamLen host f₁ = n₂.streamLen host f₂ := by
  rw [← n₁.streamLen_norm host h₁ (Net.mem_of_fifo? hf₁), ← n₂.streamLen_norm host h₂
    (Net.mem_of_fifo? hf₂), h, Net.normWalks_fifo? h hf₁ hf₂]

theorem Net.deliver_congr (h : n₁.normWalks = n₂.normWalks) (h₁ : n₁.Pos) (h₂ : n₂.Pos)
    {name : String} {f₁ f₂ : FifoD} (hf₁ : n₁.fifo? name = some f₁)
    (hf₂ : n₂.fifo? name = some f₂) (w j s : Nat) :
    n₁.deliver host f₁ w j s = n₂.deliver host f₂ w j s := by
  rw [← n₁.deliver_norm host h₁ (Net.mem_of_fifo? hf₁), ← n₂.deliver_norm host h₂
    (Net.mem_of_fifo? hf₂), h, Net.normWalks_fifo? h hf₁ hf₂]

end TwoNets

end IronVerify
