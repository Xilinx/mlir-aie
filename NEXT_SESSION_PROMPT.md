Continue the program-memory overlay work on branch `program-memory-fun` in
mlir-aie. Read `test/npu-xrt/program_memory_overlay/README.md` and
`python/iron/overlay/slot.py`'s module/class docstrings first; they're the
source of truth for what exists and why.

## State

The IRON API (commit d447b942f4c) has all three transports — host,
ping-pong, tile-sourced — implemented and hardware-verified on Strix.
Since then, six more commits, all unpushed:

- `b93b31bf8b8` [AIE] Assign BD ids per tile, not per DMA region
- `458e567e811` [test] Move the no-hardware overlay tests out of a gitignored directory
- `52b66f310ce` [IRON] Catch an overlay that overflows the core's stack at build time
- `3cf52f4541b` [IRON] Tile-sourced overlays: lift the size ceiling, allow many phases
- `46246a947a8` [test] Guard the BD-id fix with a three-phase tile-sourced run
- `5b532c068ce` [AIEX] Keep the runtime BD pool off statically-assigned ids

Of the original 5 items, 2/3/4 are done: the stack-budget guard, the
tile-sourced size ceiling (now a whole write granule, 0x1FF0, via one
self-looping `next="self"` BD instead of one BD per chunk), and multi-phase
tile-sourced scheduling (`load()` once per phase, paced by a reverse ack
channel).

**The multi-phase failure turned out not to be an overlay bug at all.** BD
ids are a per-*tile* table, but `--aie-assign-bd-ids` numbered them per
`aie.mem` region, so the ack rig's TileDma and the forward transport's
TileDma — both on the source tile — were handed the same ids and silently
overwrote each other's slots. Fixed in that pass, plus two more holders of
BD ids with the same gap (the runtime-sequence allocator never seeded from
static BDs; the dynamic free-list pool seeded itself with every id on the
tile). That fix is broader than this branch and would probably land faster
as its own PR against main — it's first in the stack and touches only the
two passes plus its own tests.

## What's left

### 1. A `programming_example` (never started)

Item 5 of the original ask. Nothing exists yet.

### 2. Overlay `.rodata` support (investigated, not built)

Currently *refused* at build time — `_link.py`'s `verify()` requires exactly
one allocatable `.text`, and `nohw/reject_overlay.lit` locks that in. The
refusal is correct as far as it goes: `.rodata` is addressed in data memory,
which nothing swaps, so a kernel with a lookup table silently reads whatever
the last overlay left there.

Design (see the `pm_overlay_rodata` memory note for detail): a per-slot
rodata region — an `aie.buffer` of declared size on the slot's tile. The
awkward part is already solved elsewhere: `_ctrl_done_buf` is exactly this
pattern, a Buffer whose address is unknown until the resident links and is
recovered in pass 1 via `defined_symbols()` in `design.py`. Then the overlay
linker script gains a `.rodata` rule, `verify()` accepts and size-checks it,
and `load()` writes a second block. `.data`/`.bss` should stay refused. Note
`.rodata` costs chunks on *every* phase of a tile-sourced transfer.

### 3. Before pushing

Local clang-format is 17.0.1; CI pins 20.1.0 (check `lintAndFormat.yml`).
Re-check the C++ files in the commits above against the pinned version.

## Housekeeping

- `third_party/aie-rt` still shows as locally modified (submodule pointer
  drift) — investigated earlier and benign (CMake-applied vendored patches
  from already-merged upstream PR #3530), intentionally left uncommitted.
  Don't "fix" it without re-confirming that's still true.
- Branch is unpushed (user pushes / opens PRs themselves — don't push or
  open a PR without being asked).
- When testing on hardware: source `/opt/xilinx/xrt/setup.sh`, activate
  `ironenv/bin/activate`, then `source utils/env_setup.sh install` — all in
  ONE bash invocation (shell state doesn't persist across separate tool
  calls). Use `lit` (from `ironenv/bin/lit`) scoped to a single
  `build/test/npu-xrt/.../*.lit` file, never the whole suite. After editing
  any `python/...` file, copy it to both `install/python/aie/...` and
  `build/python/aie/...` before re-running (three parallel trees, no
  symlinks).

## Two debugging lessons that cost real time here

- **A core cannot read tile MMIO registers on AIE2P** — every such load
  returns 0, while writes work fine. A long stretch of the multi-phase
  investigation was spent interpreting all-zero register reads as real
  state. Read registers from the runtime sequence with `aiex.npu.maskpoll`
  instead (it's a predicate, so run complementary value pairs).
- **Sample size.** At a ~94% failure rate, 3-4 hardware runs look
  deterministic in either direction, and several "clear" conclusions
  reversed under 16-run trials. Run 16+ before believing anything about a
  flaky hardware symptom.

And the thing that actually cracked it, worth reaching for earlier next
time: when a hardware symptom resists explanation, diff the *emitted
configuration* against golden. `aie-opt --aie-place-tiles
--aie-objectFifo-stateful-transform --aie-assign-lock-ids --aie-assign-bd-ids
--aie-assign-buffer-addresses | aie-translate --aie-generate-cdo
--cdo-debug=true`, then look for one register address written twice. That
found in minutes what behavioural probing had missed for hours.
