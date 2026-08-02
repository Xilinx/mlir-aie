# Readings: observing a simulation run

A **record** is one deterministic run, observed, as JSON. It is the unit that attaches to a PR, diffs
against last week, and gets read by a person or an agent from the same file. Schema:
[`readings-schema.json`](readings-schema.json). Emitter: `runtime_lib/aie-sim/lib/Readings.cpp`.

Recording cannot perturb the result -- there is no real-time constraint and nothing in the recorder
touches the active set -- so it is exhaustive rather than sampled. On-device trace consumes bandwidth
and changes what it measures; this does not.

## Shapes are a closed set

Each shape has exactly one viewer, so new instrumentation emits into an existing shape and gets its
picture for free. The alternative -- a bespoke schema per panel -- makes the frontend N special cases
and nobody adds the N+1th.

| shape | for a quantity that is | viewer |
|---|---|---|
| `scalar` | one value per run | table + run-to-run diff |
| `containment` | conserved AND strictly nested | treemap |
| `flow` | conserved AND directional | Sankey |
| `interval` | concurrent, not nested | timeline / flame (export Chrome Trace Event JSON) |
| `series` | a value over time | line, or heatmap when entities are many |
| `graph` | topology rather than magnitude | node-link |
| `coverage` | a set with a seen/unseen mark | heatmap or list |

Pick by the quantity's algebra, not by taste. Memory nests and is conserved, so it is a treemap.
Occupancy is concurrent and does not nest; forcing it into a treemap is the way this goes wrong.

## Three things the record does that a pile of numbers does not

**Verdicts are computed here, not left to the reader.** The characteristic failure of a consumer
reading raw metrics is deciding for itself that a number is acceptable. A verdict ships the judgment
with the fact behind it, so the reader reports rather than adjudicates. It is also where a doctrine
invariant becomes a mechanical check.

`unknown` is not `pass`. When the instrument was off, the record says so -- otherwise a measurement
nobody took reads as a measurement of zero, which is the same silent-zero mistake `RegisterFile`
exists to prevent.

**Units and derivations travel with values.** A bare `6.0` in a record is the same defect as a bare
constant in source. `{"value": 6.0, "unit": "GB/s", "derivedFrom": ...}` cannot be misreported, and
`inputs` lets a consumer recheck the arithmetic instead of trusting it.

**Records are byte-stable.** Emission order is fixed and no wall-clock value appears outside the
paths listed in `diffIgnore`, so identical inputs produce identical bytes. Without that, every diff
is key-order churn and the record cannot be a regression gate.

## Reading one

Start at `summary`. It carries the headline scalars, verdict counts, and an `index` naming every
observation with its element count -- so a consumer chooses what to fetch before paying for it. A
full interval record for a 48-tile array does not fit in a context window; the summary is a few KB.

## Using it

```cpp
#include "aiesim/Readings.h"

readings::enableMemoryTracking(array);   // BEFORE the run, or touched figures are 0
// ... run ...
readings::CaptureConfig cfg;
cfg.design = "my_design";
cfg.runId  = "...";        // content-derived, never a timestamp
cfg.device = "npu2_1col";  // required: DeviceModel keeps geometry, not the name
std::string json = readings::capture(array, cfg).toJson();
```

## What is emitted today

`containment/tile-memory` (touched bytes per tile memory, against capacity),
`coverage/unclaimed-registers` (every register the design wrote that nothing models), four scalars,
and two verdicts. The schema covers all seven shapes; the emitter fills three.

Touched bytes are counted in 32-byte granules and round **up**, so the figure is an upper bound on
live data. The honest reading is "how much of this memory did the design touch", not byte accounting.

## Next, in the order the value falls out

1. **Memory regions** -- split the containment leaves into `.text` / `.data` / `.bss` / stack /
   objectFIFO buffers from the linker script aiecc already emits. `capacity` minus the children's sum
   is then the free gap, and a stack abutting a buffer with zero clearance becomes visible instead of
   being a silent overwrite. This is the reading that retires the `stack_size` bug class.
2. **Stall attribution** (`interval`) -- for every stalled cycle, why: lock, backpressure, BD
   dependency, memory conflict. Nothing available today answers this.
3. **L2 residency** (`flow` + a verdict) -- turn "inside a block the stream never leaves L2" from a
   rule people remember into a per-block check.
4. **Opcode coverage** (`coverage`) -- which instructions have semantics and which this run hit,
   which makes the gap a per-kernel number rather than an estimate.
