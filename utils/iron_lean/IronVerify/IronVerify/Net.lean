-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Seq

namespace IronVerify

/-- Where an element on a stream came from. -/
inductive Src where
  | host (buf : Nat) (idx : Int)
  | core (worker : Nat) (fifo : String) (obj slot : Nat)
  deriving DecidableEq, Repr

inductive End where
  | shim
  | worker (i : Nat)
  | link (i : Nat)
  deriving DecidableEq, Repr

structure FifoD where
  name : String
  objSize : Nat
  consSize : Nat
  toStream : Option Walk
  repeatCount : Nat
  prod : End
  cons : List (End × Option Walk)
  deriving Repr

structure LinkD where
  srcs : List String
  dsts : List String
  srcOffsets : List Nat
  dstOffsets : List Nat
  deriving Repr

structure Net where
  fifos : List FifoD
  links : List LinkD
  deriving Repr

namespace FifoD

def consFrom? (f : FifoD) (e : End) : Option (Option Walk) :=
  (f.cons.find? (·.1 = e)).map (·.2)

end FifoD

/-- The `(offset, size)` segments a link's offsets cut a `shared`-element object into. -/
def LinkD.segments (L : LinkD) (shared : Nat) : List (Nat × Nat) :=
  let offs := if L.srcs.length > 1 then L.srcOffsets else L.dstOffsets
  if offs.length ≤ 1 then [(0, shared)]
  else (offs.zip (offs.drop 1 ++ [shared])).map fun (a, b) => (a, b - a)

/-- Elements one object's worth of a DMA walking `segs`, each with `w`, moves. -/
def segLen (segs : List (Nat × Nat)) : Option Walk → Nat
  | some w => segs.length * w.len
  | none => (segs.map (·.2)).sum

/-- The object slot a gathering DMA reads at position `q` of one object's worth. -/
def segGather (segs : List (Nat × Nat)) (to_ : Option Walk) (q : Nat) : Nat :=
  match to_ with
  | some w => (segs[q / w.len]?.map (·.1)).getD 0 + (w.idx (q % w.len)).toNat
  | none => q

/-- The last position of a chunk that a DMA walking `w` writes into `slot`. -/
def scatterPos (w : Walk) (slot : Nat) : Option Nat :=
  (List.range w.len).reverse.find? (fun q => w.idx q = slot)

/-- The largest index `r` with `offsets[r] ≤ s`. -/
def segmentOf (offsets : List Nat) (s : Nat) : Option Nat :=
  ((List.range offsets.length).reverse.find? fun r => offsets[r]! ≤ s)

/-- The position within one object's chunk that a scattering DMA writes into `slot`. -/
def segScatter (segs : List (Nat × Nat)) (from_ : Option Walk) (slot : Nat) : Option Nat :=
  match from_ with
  | none => some slot
  | some w => do
    let d ← segmentOf (segs.map (·.1)) slot
    let (off, _) ← segs[d]?
    let q ← scatterPos w (slot - off)
    pure (d * w.len + q)

namespace Net

variable (net : Net) (host : String → Array (Nat × Int))

def fifo? (name : String) : Option FifoD := net.fifos.find? (·.name = name)

/-- The segments `f`'s producer DMA walks in the object it reads. -/
def prodSegs (f : FifoD) : List (Nat × Nat) :=
  match f.prod with
  | .link l => match net.links[l]? with
    | some L => if L.srcs.length > 1 then L.segments f.objSize else [(0, f.objSize)]
    | none => []
  | _ => [(0, f.objSize)]

/-- The segments consumer `e` of `g` scatters its stream into. -/
def consSegs (g : FifoD) (e : End) : List (Nat × Nat) :=
  match e with
  | .link l => match net.links[l]? with
    | some L => if L.dsts.length > 1 then L.segments g.consSize else [(0, g.consSize)]
    | none => []
  | _ => [(0, g.consSize)]

/-- Slot `s` of object `j` at consumer `e` of `g`. -/
def consObj (rec : FifoD → Nat → Option Src) (g : FifoD) (e : End) (j s : Nat) : Option Src := do
  let from_ ← g.consFrom? e
  let q ← segScatter (net.consSegs g e) from_ s
  rec g (j * g.consSize + q)

/-- Slot `s` of object `j` of link `l`'s shared buffer. -/
def linkObj (rec : FifoD → Nat → Option Src) (l : Nat) (j s : Nat) : Option Src := do
  let L ← net.links[l]?
  match L.srcs with
  | [src] => consObj net rec (← net.fifo? src) (.link l) j s
  | srcs =>
    let r ← segmentOf L.srcOffsets s
    consObj net rec (← net.fifo? (← srcs[r]?)) (.link l) j (s - L.srcOffsets[r]!)

/-- Slot `s` of object `j` at `f`'s producer. -/
def prodObj (rec : FifoD → Nat → Option Src) (f : FifoD) (j s : Nat) : Option Src :=
  match f.prod with
  | .worker i => some (.core i f.name j s)
  | .shim => none
  | .link l => do
    let L ← net.links[l]?
    match L.dsts.idxOf? f.name with
    | some d =>
      if L.dsts.length = 1 then linkObj net rec l j s
      else linkObj net rec l j (L.dstOffsets[d]! + s)
    | none => none

/-- Element `p` of the stream `f`'s producer DMA emits. `fuel` bounds the walk up the graph. -/
def streamAt : Nat → FifoD → Nat → Option Src
  | 0, _, _ => none
  | fuel + 1, f, p =>
    match f.prod with
    | .shim => (host f.name)[p]?.map fun e => .host e.1 e.2
    | _ =>
      let segs := net.prodSegs f
      let per := segLen segs f.toStream
      prodObj net (streamAt fuel) f (p / (per * max f.repeatCount 1))
        (segGather segs f.toStream (p % per))

def stream (f : FifoD) (p : Nat) : Option Src := streamAt net host (net.fifos.length + 1) f p

/-- Elements on the stream `f`'s producer emits; `none` when a core produces it. -/
def streamLenAt : Nat → FifoD → Option Nat
  | 0, _ => none
  | fuel + 1, f =>
    match f.prod with
    | .shim => some (host f.name).size
    | .worker _ => none
    | .link l => do
      let L ← net.links[l]?
      let counts ← L.srcs.mapM fun s => do
        let g ← net.fifo? s
        let n ← streamLenAt fuel g
        pure (n / g.consSize)
      let objs ← counts.min?
      pure (objs * segLen (net.prodSegs f) f.toStream * max f.repeatCount 1)

def streamLen (f : FifoD) : Option Nat := streamLenAt net host (net.fifos.length + 1) f

/-- Slot `s` of the `j`-th object worker `w` receives on `f`. -/
def deliver (f : FifoD) (w j s : Nat) : Option Src :=
  consObj net (stream net host) f (.worker w) j s

end Net

/-- A design: its fifo graph and its runtime sequence, given the dispatch values. -/
structure Design (P : Type) where
  net : Net
  seq : P → Seq

namespace Design

variable {P : Type} (d : Design P) (params : P)

def host (ch : String) : Array (Nat × Int) := ((d.seq params).stream ch).toArray

def stream (f : FifoD) (p : Nat) : Option Src := d.net.stream (d.host params) f p

def deliver (f : FifoD) (w j s : Nat) : Option Src := d.net.deliver (d.host params) f w j s

end Design

end IronVerify
