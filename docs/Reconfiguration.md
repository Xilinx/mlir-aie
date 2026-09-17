<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Control-packet reconfiguration

Reconfiguration lets one xclbin host several device designs and switch between
them **on device**, without reloading a PDI or returning to the host to swap
xclbins. A single **resident control overlay** is stood up once, and each
subsequent configuration is delivered as a small payload that reprograms the
array's buffer descriptors, locks, routing, and core programs in place.

Several designs are folded into **one full ELF** with multiple dispatch
entrypoints; the host then dispatches those entrypoints (typically batched in a
single `pyxrt.runlist`) so intermediate reconfigurations never leave the NPU.

## Delivery methods

`--reconfig-method` selects how a configuration payload reaches the array. All
three fold the same set of designs into one ELF; they differ only in transport:

| Method | Transport | Notes |
|---|---|---|
| `loadpdi` | Per-config PDI reload | The un-expanded baseline: each config is its own PDI, reloaded in place. |
| `write32` | Out-of-band direct register writes | Expands each config to direct writes; always synthesizes a shared `main:init` reset entry (see below). |
| `ctrlpkt` | In-band control packets streamed through the shim DMA | The resident-overlay method: reconfigures via control packets over a persistent overlay, no new PDI. |

The minimal correct and performant invocation is:

```
aiecc.py --get-full-elf --reconfig-method=ctrlpkt <inputs...>
```

`--get-full-elf` produces the folded full ELF; `--reconfig-method=ctrlpkt`
selects control-packet delivery. Every correctness default (control-fabric
pinning, self-clear teardown, shim-ingress auto-packetize) and the performance
default (column-parallel delivery) is on already, so no other flag is required.

## Command-line options

All options below are no-ops outside the reconfiguration flow, so a plain
(non-reconfiguration) compile is unaffected.

### `--reconfig-method={loadpdi | write32 | ctrlpkt}`

Selects the delivery method (above). Default empty (no reconfiguration fold).

Reset for `write32` is a **runtime dispatch decision, not a compile flag**:
`--reconfig-method=write32` always synthesizes a shared `main:init` entry
carrying the `@empty` whole-column reset. The host dispatches `main:init` for an
explicit reset, or skips it (the firmware resets the partition on context
teardown). There is no `--reconfig-with-reset`.

### `--ctrlpkt-pinned-overlay={adapt | blind | off}` (default `adapt`)

Control-fabric pinning mode for the `ctrlpkt` overlay. Pins one canonical
control routing across all config devices so a config's data route can no longer
repoint a resident control master mid-delivery.

- `adapt` (default) — pin control *around* the ports config data uses (eager
  avoidance; least disruptive).
- `blind` — pin all control routing (blind Layer-0 capture).
- `off` — do not pin. Reintroduces the multi-column co-tenancy wedge; an
  ablation / escape hatch only.

The pass self-gates to a no-op without a control overlay, so plain builds stay
byte-identical.

### `--ctrlpkt-parallel-columns` (default on)

Collapses each column's control packets into one shim DMA and runs the columns
in parallel. On by default; pass `--ctrlpkt-parallel-columns=false` for serial
delivery. Only engages under `--reconfig-method=ctrlpkt`.

### `--ctrlpkt-auto-packetize` (default on)

Auto-packetizes the minimal set of shim-ingress objectFifos so control ingress
always has a shim MM2S channel to share, instead of hitting the shim-MM2S
exhaustion wall when every input design is circuit-switched. On by default in
the overlay flow; `--ctrlpkt-auto-packetize=false` disables it (the flow then
falls back to the exhaustion wall).

### `--dma-fence-shared-mem` (opt-in, default off)

Carries every cross-tile core-to-core lock-only shared-memory objectFifo on DMA.
The lock-only shared path has no write-completion barrier and can read stale data
under the resident control-packet overlay; DMA completion supplies the barrier.
Opt-in because slack-satisfying pre-existing designs keep fast shared memory and
are never forced onto DMA.

## Self-clear teardown (automatic)

The resident-overlay methods (`ctrlpkt` in-band, `write32` out-of-band) apply a
per-config **self-clear teardown** unconditionally — there is no flag to gate
it, because running a resident-overlay method without teardown is incorrect
behavior, not a supported mode. Each configuration appends:

- a switch-disable epilogue (config ports minus overlay ports),
- a circuit-switch-connect teardown (for objectFifo / circuit-switched routes),
- a DMA-channel reset (assert/deassert `Ctrl.Reset` on each active non-shim
  channel),

so the persistent overlay does not accrue stream-switch / circuit bindings
across reconfigurations, and a busy DMA channel is drained before the next
config reprograms it. Each teardown is demand-scoped: a no-op for a config that
uses none of that resource class.

## IRON Python API

`iron.Reconfiguration` folds several `@iron.jit` designs into one reconfigurable
full ELF:

```python
import aie.iron as iron

r = iron.Reconfiguration("my_fold", method="ctrlpkt")
r.add(design_a, input_tensor, output_a)   # one add() per design
r.add(design_b, input_tensor, output_b)
elf = r.compile()                         # -> FullElf
```

**Constructor** — `Reconfiguration(name, method=None, output_dir=None, extra_aiecc_args=None)`:

- `name` — bare identifier naming the output `<name>.elf` and aiecc's working
  directory `<name>.prj` (no path separators).
- `method` — `"loadpdi"`, `"write32"`, `"ctrlpkt"`, or `None` (aiecc's default).
  Only `"ctrlpkt"` prepends a `main:init` entrypoint and reserves a
  control-packet slot.
- `output_dir` — directory for staged inputs, kernel objects, and the ELF
  (defaults to the current directory). Use a fresh directory per fold, or clear
  stale `.o` files: kernel compilation skips rebuilding when the target object
  already exists.
- `extra_aiecc_args` — extra aiecc flags appended to the fold build (e.g. an
  ablation flag such as `--ctrlpkt-pinned-overlay=off`). Empty by default, so the
  ordinary fold is byte-identical.

**`add(design, *args, **kwargs)`** stages one design's MLIR and external kernels.
`design` is a `CallableDesign` (an `@iron.jit`-decorated function); the positional
args mirror `as_mlir()`'s signature. Each design must carry a distinct
`@iron.jit(name=)` — the fold hard-fails on duplicate entrypoints.

**`compile()`** returns a frozen `FullElf` descriptor: the output path, the
entrypoints in dispatch order, the `main:init` standup entry (or `None`), and
whether the fold reserves a control-packet slot.

## Dispatching the fold

Each entrypoint of the folded `FullElf` is dispatched by name. For an on-device
reconfiguration chain, batch every entrypoint into a single `pyxrt.runlist` so
the intermediate switches stay on the NPU and never round-trip through the host.
For the `ctrlpkt` method, dispatch the `main:init` standup entry first, then each
configuration entrypoint in order.
