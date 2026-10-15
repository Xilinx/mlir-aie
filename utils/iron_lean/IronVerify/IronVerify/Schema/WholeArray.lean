-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Net

/-!
whole_array as one Lean `Design`, a function of its parameters: the core grid (`rows × cols`),
the core tile `m × k × n`, the kernel's micro-tile `r × s × t` (how the dtype enters) and the
column-major switches. It mirrors whole_array.py: every export of the real design is checked
against `schema p` for its parameters (fifos and links up to walk normalization, host streams
under the design's guards), so theorems proved here hold for each validated export.

Fifos and links are listed in the exporter's canonical order (`export.py --order canonical`:
by name, numbers compared numerically).
-/

namespace IronVerify.Schema.WholeArray

open IronVerify

structure Params where
  rows : Nat
  cols : Nat
  m : Nat
  k : Nat
  n : Nat
  r : Nat
  s : Nat
  t : Nat
  bColMaj : Bool
  cColMaj : Bool
  deriving Repr, DecidableEq

/-- What whole_array.py asserts before it builds anything. -/
structure Params.Valid (p : Params) : Prop where
  rows_pos : 0 < p.rows
  cols_pos : 0 < p.cols
  r_pos : 0 < p.r
  s_pos : 0 < p.s
  t_pos : 0 < p.t
  m_pos : 0 < p.m
  k_pos : 0 < p.k
  n_pos : 0 < p.n
  r_dvd : p.r ∣ p.m
  s_dvd : p.s ∣ p.k
  t_dvd : p.t ∣ p.n

namespace Params

variable (p : Params)

/-- A shims: one per column, at most one per core row. -/
def nA : Nat := min p.rows p.cols

/-- Core rows A shim `i` feeds (`a_shim_rows`): the first `rows % nA` shims take one more. -/
def shimRows (i : Nat) : Nat := p.rows / p.nA + if i < p.rows % p.nA then 1 else 0

/-- The first core row A shim `i` feeds. -/
def shimFirst (i : Nat) : Nat := i * (p.rows / p.nA) + min i (p.rows % p.nA)

/-- The A shim that feeds core row `row`. -/
def shimOf (row : Nat) : Nat :=
  let q := p.rows / p.nA
  let e := p.rows % p.nA
  if row < e * (q + 1) then row / (q + 1) else e + (row - e * (q + 1)) / q

/-- Worker index of the core at `(row, col)`: `Worker.grid` is row-major. -/
def worker (row col : Nat) : Nat := row * p.cols + col

def bLink (c : Nat) : Nat := p.nA + c
def cLink (c : Nat) : Nat := p.nA + p.cols + c

end Params

def aL3 (i : Nat) : String := s!"A_L3L2_{i}"
def aL2 (row : Nat) : String := s!"A_L2L1_{row}"
def bL3 (c : Nat) : String := s!"B_L3L2_{c}"
def bL2 (c : Nat) : String := s!"B_L2L1_{c}"
def cL1 (c row : Nat) : String := s!"C_L1L2_{c}_{row}"
def cL3 (c : Nat) : String := s!"C_L2L3_{c}"

section Walks

variable (p : Params)

/-- `mm_stream_dims` A: the `m × k` tile, row-major, streamed as `(r × s)` blocks. -/
def walkA : Walk :=
  { offset := 0,
    dims := [((p.m / p.r : Nat), (p.r * p.k : Nat)), ((p.k / p.s : Nat), (p.s : Nat)),
             ((p.r : Nat), (p.k : Nat)), ((p.s : Nat), 1)] }

/-- `mm_stream_dims` B: `(s × t)` blocks of the `k × n` tile, or `(t × s)` blocks of its
`n × k` transpose with `bColMaj`. -/
def walkB : Walk :=
  if p.bColMaj then
    { offset := 0,
      dims := [((p.n / p.t : Nat), (p.t * p.k : Nat)), ((p.k / p.s : Nat), (p.s : Nat)),
               ((p.t : Nat), (p.k : Nat)), ((p.s : Nat), 1)] }
  else
    { offset := 0,
      dims := [((p.k / p.s : Nat), (p.s * p.n : Nat)), ((p.n / p.t : Nat), (p.t : Nat)),
               ((p.s : Nat), (p.n : Nat)), ((p.t : Nat), 1)] }

/-- `mm_stream_dims` C: reads the core's `(r × t)`-blocked buffer and emits the tile row-major
(`(t × r)` blocks and the `n × m` transpose with `cColMaj`). -/
def walkC : Walk :=
  if p.cColMaj then
    { offset := 0,
      dims := [((p.n / p.t : Nat), (p.t * p.m : Nat)), ((p.t : Nat), (p.r : Nat)),
               ((p.m / p.r : Nat), (p.t * p.r : Nat)), ((p.r : Nat), 1)] }
  else
    { offset := 0,
      dims := [((p.m / p.r : Nat), (p.r * p.n : Nat)), ((p.r : Nat), (p.t : Nat)),
               ((p.n / p.t : Nat), (p.r * p.t : Nat)), ((p.t : Nat), 1)] }

end Walks

section Net

variable (p : Params)

def aL2Fifo (row : Nat) : FifoD :=
  { name := aL2 row, objSize := p.m * p.k, consSize := p.m * p.k, toStream := some (walkA p),
    repeatCount := 1, prod := .link (p.shimOf row),
    cons := (List.range p.cols).map fun c => (.worker (p.worker row c), none) }

def aL3Fifo (i : Nat) : FifoD :=
  { name := aL3 i, objSize := p.m * p.k * p.shimRows i, consSize := p.m * p.k * p.shimRows i,
    toStream := none, repeatCount := 1, prod := .shim, cons := [(.link i, none)] }

def bL2Fifo (c : Nat) : FifoD :=
  { name := bL2 c, objSize := p.k * p.n, consSize := p.k * p.n, toStream := some (walkB p),
    repeatCount := 1, prod := .link (p.bLink c),
    cons := (List.range p.rows).map fun row => (.worker (p.worker row c), none) }

def bL3Fifo (c : Nat) : FifoD :=
  { name := bL3 c, objSize := p.k * p.n, consSize := p.k * p.n, toStream := none,
    repeatCount := 1, prod := .shim, cons := [(.link (p.bLink c), none)] }

def cL1Fifo (c row : Nat) : FifoD :=
  { name := cL1 c row, objSize := p.m * p.n, consSize := p.m * p.n, toStream := none,
    repeatCount := 1, prod := .worker (p.worker row c), cons := [(.link (p.cLink c), none)] }

def cL3Fifo (c : Nat) : FifoD :=
  { name := cL3 c, objSize := p.m * p.n * p.rows, consSize := p.m * p.n * p.rows,
    toStream := some (walkC p), repeatCount := 1, prod := .link (p.cLink c),
    cons := [(.shim, none)] }

/-- A shim `i`'s memtile splits its rows' tiles to their cores. -/
def aLinkD (i : Nat) : LinkD :=
  { srcs := [aL3 i], dsts := (List.range (p.shimRows i)).map fun j => aL2 (p.shimFirst i + j),
    srcOffsets := [], dstOffsets := (List.range (p.shimRows i)).map (p.m * p.k * ·) }

/-- B column `c` is forwarded to its column of cores. -/
def bLinkD (c : Nat) : LinkD :=
  { srcs := [bL3 c], dsts := [bL2 c], srcOffsets := [], dstOffsets := [0] }

/-- Column `c`'s cores' C tiles are joined, one segment per core row. -/
def cLinkD (c : Nat) : LinkD :=
  { srcs := (List.range p.rows).map (cL1 c), dsts := [cL3 c],
    srcOffsets := (List.range p.rows).map (p.m * p.n * ·), dstOffsets := [] }

def wnet : Net where
  fifos :=
    (List.range p.rows).map (aL2Fifo p) ++ (List.range p.nA).map (aL3Fifo p) ++
    (List.range p.cols).map (bL2Fifo p) ++ (List.range p.cols).map (bL3Fifo p) ++
    ((List.range p.cols).flatMap fun c => (List.range p.rows).map (cL1Fifo p c)) ++
    (List.range p.cols).map (cL3Fifo p)
  links :=
    (List.range p.nA).map (aLinkD p) ++ (List.range p.cols).map bLinkD ++
    (List.range p.cols).map (cLinkD p)

end Net

section Seq

variable (p : Params) (M K N : Int)

/-- Column tiles of C per column of cores: `N / (n · cols)`. -/
def nTiles : Int := N / (p.n * p.cols : Nat)

/-- Row block `rb`'s A for shim `i`: its rows' `m × k` tiles, every k step, `nTiles` times. -/
def tapA (rb : Int) (i : Nat) : Walk :=
  { offset := rb * (p.rows * p.m : Nat) * K + (p.shimFirst i * p.m : Nat) * K,
    dims := [(nTiles p N, 0), (K / (p.k : Nat), (p.k : Nat)), ((p.shimRows i * p.m : Nat), K),
             ((p.k : Nat), 1)] }

/-- Column `c`'s B: its column tiles, each all the way down K (`B` given as `N × K` with
`bColMaj`). The same for every row block. -/
def tapB (c : Nat) : Walk :=
  if p.bColMaj then
    { offset := (c * p.n : Nat) * K,
      dims := [(nTiles p N, (p.cols * p.n : Nat) * K), (K / (p.k : Nat), (p.k : Nat)),
               ((p.n : Nat), K), ((p.k : Nat), 1)] }
  else
    { offset := (c * p.n : Nat),
      dims := [(nTiles p N, (p.n * p.cols : Nat)), (K / (p.k : Nat), (p.k : Nat) * N),
               ((p.k : Nat), N), ((p.n : Nat), 1)] }

/-- Row block `rb`'s C for column `c` (`C` drained as `N × M` with `cColMaj`). -/
def tapC (rb : Int) (c : Nat) : Walk :=
  if p.cColMaj then
    { offset := rb * (p.rows * p.m : Nat) + (c * p.n : Nat) * M,
      dims := [(nTiles p N, (p.cols * p.n : Nat) * M), ((p.rows : Nat), (p.m : Nat)),
               ((p.n : Nat), M), ((p.m : Nat), 1)] }
  else
    { offset := rb * (p.rows * p.m : Nat) * N + (c * p.n : Nat),
      dims := [(nTiles p N, (p.n * p.cols : Nat)), ((p.rows * p.m : Nat), N),
               ((p.n : Nat), 1)] }

def _root_.IronVerify.Seq.ofList : List Seq → Seq
  | [] => .skip
  | x :: xs => .seq x (Seq.ofList xs)

/-- One row block's transfers, in `step_transfers` order. -/
def step (rb : Int) : Seq :=
  Seq.ofList <| (List.range p.cols).flatMap fun c =>
    [.xfer (cL3 c) 2 (tapC p M N rb c)] ++
    (if c < p.nA then [.xfer (aL3 c) 0 (tapA p K N rb c)] else []) ++
    [.xfer (bL3 c) 1 (tapB p K N c)]

/-- The design's `require(...)` guards, then every row block. `TaskGroup.pipelined` peels the
first steps off this loop; the exports' streams equal it (see `Schema/Instances`). -/
def wseq : Seq :=
  Seq.ofList [
    .check (decide (M > 0)), .check (decide (K > 0)), .check (decide (N > 0)),
    .check (decide (M % (p.rows * p.m : Nat) = 0)), .check (decide (K % (p.k : Nat) = 0)),
    .check (decide (N % (p.n * p.cols : Nat) = 0)),
    .loop 0 (M / (p.rows * p.m : Nat)) (step p M K N)]

end Seq

def schema (p : Params) : Design (Int × Int × Int) where
  net := wnet p
  seq := fun (M, K, N) => wseq p M K N

end IronVerify.Schema.WholeArray
