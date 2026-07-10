<!-- Copyright (C) 2024-2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Lower-level IRON primitives

Advanced primitives for hand-routed data movement and explicit DMA
programming. Reach for these when the managed [`ObjectFifo`][iron.ObjectFifo]
abstraction on the [High-level IRON](iron_highlevel.md) page is not enough and
you need direct control over routing, DMA descriptors, and locks.

All symbols documented here are importable directly from the `iron`
namespace (e.g. `from iron import Flow, TileDma, Lock`).

---

## Explicit routing

### Flow / PacketFlow

Circuit-switched ([`Flow`][iron.Flow]) and packet-switched
([`PacketFlow`][iron.PacketFlow]) stream connections, plus the
[`PacketDest`][iron.PacketDest] endpoint descriptor.

::: iron.dataflow.flow
    options:
      show_root_heading: false

### CascadeFlow

Directed cascade-stream connection between two adjacent Workers.

::: iron.dataflow.cascadeflow
    options:
      show_root_heading: false

---

## DMA programming

### TileDma / DmaChannel / Bd

Explicit tile DMA programs: [`TileDma`][iron.TileDma],
[`DmaChannel`][iron.DmaChannel], buffer descriptors ([`Bd`][iron.Bd]), and the
[`Acquire`][iron.Acquire] / [`Release`][iron.Release] lock actions.

::: iron.dataflow.tile_dma
    options:
      show_root_heading: false

### Lock

::: iron.lock
    options:
      show_root_heading: false

---

## Runtime tasks

Lower-level runtime task types scheduled by the
[`Runtime`][iron.runtime.runtime.Runtime]. For the high-level `Runtime` entry
point itself, see the [High-level IRON](iron_highlevel.md#runtime) page.

::: iron.runtime.task
    options:
      show_root_heading: false

::: iron.runtime.taskgroup
    options:
      show_root_heading: false

::: iron.runtime.dmatask
    options:
      show_root_heading: false
