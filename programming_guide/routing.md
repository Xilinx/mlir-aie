<!---//===- routing.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2021 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Routing

Every tile has a stream switch, and data moves between tiles over the links
between neighbouring switches. A design names only the two ends of each
stream: an ObjectFifo, an IRON `Flow` or `PacketFlow`, or an `aie.flow` or
`aie.packet_flow` in MLIR. The `aie-create-pathfinder-flows` pass, which
`aiecc` runs for you, picks the path each stream takes and writes the switch
settings that carry it.

Most designs never need to think about routing. This page is for when you
read the routed MLIR, write flows by hand, or hit a routing error. It covers
what the router produces, the hang that packet-switched streams can cause and
how the router avoids it, and how to act on its errors.

## Stream switches

A stream switch connects *slave* ports, where streams come in, to *master*
ports, where they go out. Ports come in bundles: `DMA` and `Core` for the
tile's own channels; `North`, `South`, `East` and `West` for the links to the
neighbouring switches; and a few more, such as `TileControl` and `Trace`.

A stream is one of two kinds:

- **Circuit-switched** (`aie.flow`): it owns every master port on its path,
  and no other stream can use them.
- **Packet-switched** (`aie.packet_flow`): each packet starts with a header
  that carries a 5-bit ID. Switches forward the packet by that ID, so streams
  with different IDs can share ports.

Packet switching shares ports through *arbiters*. Each switch has 6 arbiters,
and each arbiter has 4 master selects (*msels*) that pick the master ports a
packet goes out on. An arbiter passes one packet at a time from the slave
ports routed through it.

## Circuit-switched flows

`aie.flow` connects one port to another:

```mlir
aie.device(npu2) {
  %t00 = aie.tile(0, 0)
  %t02 = aie.tile(0, 2)
  aie.flow(%t00, DMA : 0, %t02, DMA : 0)
  aie.flow(%t02, DMA : 0, %t00, DMA : 0)
}
```

The router replaces the flows with an `aie.switchbox` for each tile they
cross. Each switchbox holds an `aie.connect<source, destination>` for each
hop:

```mlir
%switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
  aie.connect<South : 3, North : 1>
  aie.connect<North : 0, South : 2>
}
%shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
  aie.connect<DMA : 0, North : 3>
  aie.connect<North : 2, DMA : 0>
}
%switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
  aie.connect<South : 1, North : 1>
  aie.connect<North : 0, South : 0>
}
%switchbox_0_2 = aie.switchbox(%tile_0_2) {
  aie.connect<South : 1, DMA : 0>
  aie.connect<DMA : 0, South : 0>
}
```

`North : 1` leaving tile (0,0) arrives at tile (0,1) as `South : 1`. A master
port takes one connection. A slave port can feed several, which is how a flow
broadcasts.

A shim tile's DMA reaches its switch through the `aie.shim_mux`, which maps
DMA channels onto fixed stream channels. On the way into the array, DMA 0 and
DMA 1 enter as `North : 3` and `North : 7`. On the way out, `North : 2` and
`North : 3` drain into DMA 0 and DMA 1. The router writes the shim mux for
you.

You can also write `aie.switchbox` and `aie.connect` yourself. The router
keeps what is already there and routes the flows around the ports it uses.

## Packet-switched flows

Here two packet flows from different tiles end at the same DMA channel:

```mlir
aie.device(npu2) {
  %t02 = aie.tile(0, 2)
  %t03 = aie.tile(0, 3)
  %t04 = aie.tile(0, 4)
  aie.packet_flow(1) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t04, DMA : 0> }
  aie.packet_flow(2) { aie.packet_source<%t03, DMA : 0> aie.packet_dest<%t04, DMA : 0> }
}
```

For each switch on the way, the router writes three kinds of ops:

- `aie.amsel<arbiter> (msel)` names an arbiter and one of its msels.
- `aie.masterset(port, amsels...)` names the master port each amsel drives.
- `aie.packet_rules(slave port) { aie.rule(mask, value, amsel) }` says which
  amsel a packet arriving at that slave port takes. A rule matches when
  `id & mask == value`.

```mlir
%switchbox_0_3 = aie.switchbox(%tile_0_3) {
  %0 = aie.amsel<0> (0)
  %1 = aie.amsel<1> (0)
  %2 = aie.masterset(North : 3, %0)
  %3 = aie.masterset(North : 4, %1)
  aie.packet_rules(DMA : 0) {
    aie.rule(31, 2, %0)
  }
  aie.packet_rules(South : 3) {
    aie.rule(31, 1, %1)
  }
}
%switchbox_0_4 = aie.switchbox(%tile_0_4) {
  %0 = aie.amsel<0> (0)
  %1 = aie.masterset(DMA : 0, %0)
  aie.packet_rules(South : 3) {
    aie.rule(31, 2, %0)
  }
  aie.packet_rules(South : 4) {
    aie.rule(31, 1, %0)
  }
}
```

At (0,3) the two flows leave on separate links through separate arbiters. At
(0,4) they go out of the same master port, so they share its arbiter, and
their packets take turns.

A packet flow takes three optional attributes:

- `mask` claims a range of IDs: `aie.packet_flow(0x10, mask = 0x1c)` claims
  0x10 through 0x13.
- `keep_pkt_header` says whether the destination keeps the header. By
  default, a DMA destination, or a shim's `South` port, drops it, and any
  other destination keeps it.
- `priority_route` routes the flow before the others; see
  [Priority routes and the control overlay](#priority-routes-and-the-control-overlay).

In IRON, `ObjectFifo(..., packet=True)` routes a fifo as a packet flow, and
`PacketFlow` declares one directly (see
[Section 2g](./section-2/section-2g/README.md)).

## Arbiters and deadlock

An arbiter holds its grant until the packet it is passing ends. Two slave
ports whose packets go through one arbiter therefore take turns, even when
they head to different master ports. Suppose the packet being passed can't
finish until the other stream moves, for example because its receiver waits
on a lock that the other stream's receiver releases. Then the array hangs.

The router avoids this. Before routing, it works out which packet flows can
hold each other up. It reads the design's locks, DMA programs, buffer
descriptor (BD) lengths and the runtime sequence's waits. Then it:

1. keeps flows that can hold each other up off a shared arbiter;
2. where a switch has more packet master ports in use than free arbiters,
   turns a hop into a circuit connection when its slave port is the only one
   feeding its master ports toward the neighbouring switches. The connection
   needs no arbiter and passes the header on to the next switch.
3. routes the flows elsewhere where no assignment of arbiters works;
4. breaks *hold cycles*, where packets into receivers they share each hold an
   arbiter that another needs, across several switches.

Where the router can't see something, it assumes the worst. A DMA channel the
design doesn't program (in `aie.mem`, `aie.memtile_dma` or `aie.shim_dma`) is
assumed to wait on anything on its tile. A flow whose length is unknown is
assumed to fill its receiver.

### When routing fails

Sometimes the flows that must stay apart need more arbiters at a switch than
it has, whatever the routing. Then the pass fails, names the flows, and says
why it thinks they can hold each other up. For example,
[`arbiter_cut_tile_clique.mlir`](../test/create-packet-flows/arbiter_cut_tile_clique.mlir)
with `circuit-switch-hops=false` fails with:

```
error: Unable to find a legal routing: at tile (0, 3), no two of packet flow (0, 2) DMA:0 -> (0, 4) DMA:0 (id 0), packet flow (0, 2) DMA:1 -> (0, 4) DMA:1 (id 1), packet flow (0, 2) Core:0 -> (0, 4) Core:0 (id 2) can share an arbiter, and each takes one there whatever the routing, but the switchbox has 2 free. For example, packet flow (0, 2) DMA:0 -> (0, 4) DMA:0 (id 0) can fill its receiver, and draining that waits on (0, 4) S2MM 1, which receives packet flow (0, 2) DMA:1 -> (0, 4) DMA:1 (id 1). The volume packet flow (0, 2) DMA:0 -> (0, 4) DMA:0 (id 0) carries is unknown, so it is assumed to overrun its receiver. Nothing in the design programs (0, 4) S2MM 0, so it is assumed to wait on anything on its tile.
```

The last two sentences are assumptions. If the design states what was
assumed, such as the receiving DMA programs and their BD lengths, the
conflict often goes away. Otherwise, spread the flows over more tiles or
columns, or make some of them circuit flows. In an IRON design, the error is
reported at the ObjectFifo's name.

Two errors point at the design, not the routing:

- *Flows can deadlock however they are routed*: one DMA channel sends or
  receives two flows in an order that lets one hold up the other, so no
  routing can keep them apart.
- *Packet flows into receivers they share can deadlock holding arbiters across
  switchboxes, and no routing found avoids it.*

If you know the hang can't happen, for example because the runtime never
issues the transfers in that order, set `allow-deadlock-prone`. The router
then routes the flows anyway and warns instead.

## Priority routes and the control overlay

A packet flow with `priority_route = true` routes first, on its own, and the
other flows route around it. If the design can't route that way, the flow
routes with the others. The control overlay that `aiecc
--generate-ctrl-pkt-overlay` adds, which carries control packets to and from
tiles, uses priority routes.

A control-packet reload (`aiecc --load-pdi-to-ctrl-pkt`) goes further. It
loads the `@ctrl_pkt_overlay` device first and configures the rest of the
design through it, so the overlay's switch settings can't change. Such a
design carries `has_ctrl_pkt_overlay` on its `aie.device`. The router keeps
the overlay's flows (those to or from a `TileControl` port) on the routes
they take alone, and fails if the design's flows don't fit around them.

## Options

| Pass option | Default | Effect |
|---|---|---|
| `route-circuit` | `true` | Route `aie.flow`s. |
| `route-packet` | `true` | Route `aie.packet_flow`s. |
| `circuit-switch-hops` | `true` | Turn packet hops into circuit connections where a switch runs short of arbiters (step 2 above). |
| `allow-deadlock-prone` | `false` | Warn instead of failing on the two design errors above. From `aiecc`, pass `--allow-deadlock-prone-routing`; from IRON, use `@iron.jit(aiecc_flags=["--allow-deadlock-prone-routing"])`. |

For example:
`aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" design.mlir`.

In a build with assertions enabled,
`-debug-only=aie-pathfinder,aie-stream-dependency` prints each routing
iteration and the dependencies found between streams. See the
[pass reference](../docs/AIEPasses.md) and the
[dialect reference](../docs/AIEDialect.md) for every op and attribute.

## Visualizing routing

[`visualize.py`](../tools/aie-routing-command-line/visualize.py) draws each
routed flow as text. It reads JSON from `aie-translate --aie-flows-to-json`,
which traces the flows through the switchboxes. Keep those switchboxes with
`--aie-find-flows=remove-lifted=false`:

```
aie-opt --aie-create-pathfinder-flows --aie-find-flows=remove-lifted=false design.mlir \
    | aie-translate --aie-flows-to-json > design.json
python3 tools/aie-routing-command-line/visualize.py -j design.json
```

The script writes one text file per flow into `./design/`. Here is a flow
from shim (0,0) to tiles (0,3) and (1,3):

```
    ┌─────┐ ₁ ┌─────┐ ₁ ┌─────┐ ₁ ┌─────┐
 →→→│ 0,0 ├→→→┤ 0,1 ├→→→┤ 0,2 ├→→→┤ 0,3 │
╶───┤S #  ├───┤  #  ├───┤  #  │   │  # D│
  ¹ └─────┘ ¹ └─────┘ ¹ └───┬─┘   └─┬───┘
                            │¹     ¹↓
                        ┌───┴─┐   ┌─┴───┐
                        │ 1,2 │   │ 1,3 │
                        │  *  │   │  # D│
                        └─────┘   └─────┘
```

Each box is a tile, labeled `col,row`. Arrows mark the current flow, and the
numbers on the links count the streams that use them. `S` and `D` mark the
flow's source and destinations, `#` the tiles it crosses, and `*` the tiles
other flows use. Run `python3 visualize.py --help` for the other options.

## Benchmarking routing

[`router_performance.py`](../utils/router_performance.py) runs the
`aie-opt` command in every test in a directory, with `--debug`, which needs
a build with assertions enabled. It writes the wall-clock time and the path
length of each routing to `routing_performance_results.csv` in that
directory:

```
python3 utils/router_performance.py test/create-flows/
python3 utils/router_performance.py test/create-packet-flows/
```
