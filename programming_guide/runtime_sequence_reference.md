<!---//===- runtime_sequence_reference.md ------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Runtime Sequence Op Reference

A runtime sequence is the host-side program that drives an NPU design: it fills
tensors into the AIE-array, launches DMAs, waits on completion tokens, and
reconfigures state between dispatches. In IRON you write it as the body of
`Runtime.sequence()`; in the AIE dialect it is the region of an
`aie.runtime_sequence` operation.

[Section 2d](./section-2/section-2d/) teaches the runtime sequence by example
and [Section 3](./section-3/) shows a full end-to-end program. This page is the
companion **reference**: a catalog of the `aiex` control ops that appear in a
runtime sequence, grouped by the resource they act on and the verb they perform.
Most you write directly; a few are emitted by the compiler on the dynamic
buffer-descriptor path and are marked *(compiler-emitted)* below. This is a
curated companion to the auto-generated, per-op
[AIEX dialect reference](https://xilinx.github.io/mlir-aie/AIEXDialect.html);
that page and [`AIEX.td`](../include/aie/Dialect/AIEX/IR/AIEX.td) (whose
`let description` fields these entries condense) are authoritative.

## How to read this page

* Op names use their `aiex.` dialect prefix (for example `aiex.npu.write32`).
  Many have IRON Python wrappers; where an op is most often reached through a
  wrapper, [Section 2d](./section-2/section-2d/) shows the Python form.
* An op is legal inside a sequence for one of three reasons:
  * its verifier carries `HasParent<AIE::RuntimeSequenceOp>` or
    `HasAncestor<AIE::RuntimeSequenceOp>` (the DMA-task ops use `HasAncestor`,
    which keeps them legal inside a nested `scf.for`);
  * it is `HasParent<ConfigureOp>`, legal transitively inside an `aiex.configure`
    that is itself in the sequence (this is `aiex.run`); or
  * it is one of the `npu.*` instruction family (plus `aiex.control_packet`), the
    low-level NPU instructions a sequence is built from and lowered to.
* **Tile classes.** `shim` tiles sit at the array edge (row 0) and move data
  to and from host memory; `mem` tiles hold shared L2 scratchpad; `core` tiles
  run compute. An op's "Applies to" column names the tile classes it can target.
* **DMA direction.** `MM2S` (memory-to-stream) moves data from a memory into the
  stream network, that is host or tile memory into the array. `S2MM`
  (stream-to-memory) moves the other way.
* **Address resolution.** Several ops take an `address` plus optional `buffer` /
  `column` / `row`. If `buffer` is set, `address` is an offset into that
  `aie.buffer`. If `column` and `row` are set, `address` is an offset into that
  tile's memory space. Otherwise `address` is a full absolute address in the
  array.

## Which interface should I use?

For moving data between host and array, reach for one of the two high-level
interfaces that [Section 2d](./section-2/section-2d/) teaches, rather than the
raw register and buffer-descriptor ops:

* `aiex.npu.dma_memcpy_nd` with `aiex.npu.dma_wait`: program a BD and wait on its
  completion token by symbol; or
* the DMA-task family (`aiex.dma_configure_task`, `dma_start_task`,
  `dma_await_task`, `dma_free_task`, or `dma_start_bd_chain`), which lets the
  compiler allocate BD ids for you and can chain BDs.

The raw ops in sections 1 and 2 (`write32`, `maskwrite32`, `blockwrite`,
`writebd`, `push_queue`, `address_patch`) and the raw `npu.sync` are the
primitives these interfaces lower to; reach for them directly only for low-level
register or buffer-descriptor control. Ops tagged *(compiler-emitted)* are
produced by the compiler on the dynamic buffer-descriptor path and are not
hand-written.

## Contents

1. [Registers and tile memory](#1-registers-and-tile-memory-write--masked-write--block-write)
2. [Buffer descriptors and DMA queues](#2-buffer-descriptors-and-dma-queues-program--push--patch)
3. [DMA tasks](#3-dma-tasks-configure--start--await--free)
4. [Synchronization](#4-synchronization-wait-on-a-task-complete-token)
5. [Locks](#5-locks-set)
6. [Runtime parameters and scratchpad](#6-runtime-parameters-and-scratchpad-write--declare--read--sync)
7. [Device image and execution](#7-device-image-and-execution-load--configure--run--preempt)
8. [Control packets](#8-control-packets-emit)

-----

## 1. Registers and tile memory (write / masked-write / block-write)

Direct writes to addresses in the array. Address resolution follows the
`buffer` / `column,row` / absolute rule described above.

| Op | What it does | Applies to | Example |
|----|--------------|------------|---------|
| `aiex.npu.write32` | Write a 32-bit `value` to a resolved `address` in the array. | any address (shim, mem, core, array registers) | `aiex.npu.write32(%addr, %val) : i32, i32` |
| `aiex.npu.maskwrite32` | Read-modify-write of a 32-bit word: only the bits set in `mask` take `value`, the rest are preserved. Use when a register packs several fields in one word. | any address | `aiex.npu.maskwrite32(%addr, %val, %mask) : i32, i32, i32` |
| `aiex.npu.blockwrite` | Write a whole `memref` of words to the array starting at the resolved address, in one instruction. | any address | `aiex.npu.blockwrite(%data) {address = 119300 : ui32} : memref<8xi32>` |

`address` and `value` are SSA operands: use `arith.constant` for a
compile-time-known value, or pass a runtime sequence argument for a
runtime-parameterized write.

## 2. Buffer descriptors and DMA queues (program / push / patch)

A DMA transfer is described by a buffer descriptor (BD). These ops program BDs
and launch them onto a tile's DMA task queue.

| Op | What it does | Applies to | Example |
|----|--------------|------------|---------|
| `aiex.npu.dma_memcpy_nd` | Program a BD and issue an n-dimensional DMA transfer (a "half-DMA": it configures one endpoint of the transfer). The target tile, channel and direction come from the `metadata` symbol (an `aie.shim_dma_allocation`, or an `aie.objectfifo` symbol directly). `offsets`/`sizes`/`strides` describe the access pattern; `issue_token` requests a completion token. | shim (host <-> array); `MM2S` / `S2MM` from the allocation | `aiex.npu.dma_memcpy_nd(%arg0[0,0,0,0][1,1,32,32][0,0,64,1]) {id = 0 : i64, metadata = @out0} : memref<32x64xi32>` |
| `aiex.npu.writebd` | Write a complete buffer descriptor (all fields explicit: buffer length/offset, the `d0..d2` sizes/strides, iteration, lock acquire/release, packet header, next-BD) to a BD slot on a tile. The low-level form behind `dma_memcpy_nd`. | any tile with a DMA engine (shim, mem, core) | `aiex.npu.writebd {column = 0 : i32, row = 0 : i32, bd_id = 0 : i32, buffer_length = 1024 : i32, /* ... */}` |
| `aiex.npu.push_queue` | Push a BD id onto a tile channel's task queue to launch it, with an outer-dimension `repeat_count` and optional completion token. | any tile channel; `MM2S` / `S2MM` | `aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %bd repeat %rep {issue_token = false} : i32, i32` |
| `aiex.npu.address_patch` | Patch a runtime buffer address into the instruction stream: takes runtime argument `arg_idx` plus `arg_plus` offset and writes it into a BD address field. Used to bind host buffers whose address is only known at launch. | shim BD addresses | `aiex.npu.address_patch(%off : i32) {addr = 0 : ui32, arg_idx = 0 : i32}` |
| `aiex.npu.assert_bd_field` *(compiler-emitted)* | Guard a runtime BD-field value against its hardware field width (`value <= max`); emitted by the dynamic DMA lowering when a runtime scalar lands in a narrow BD field (e.g. a 10-bit wrap). On the host-C++ path it aborts stream generation for an out-of-range value; on the binary path `value` is a verified constant and it is a no-op. | dynamic-BD lowering path | `aiex.npu.assert_bd_field(%v) {max = 1023 : i32} : i32` |
| `aiex.npu.assert_bd_divisible` *(compiler-emitted)* | Guard a runtime BD size/stride against the address-gen granule (`value % divisor == 0`, or `value == 1` when `allow_unit`); emitted by the dynamic DMA lowering. Same host-C++ / binary behavior as `assert_bd_field`. | dynamic-BD lowering path | `aiex.npu.assert_bd_divisible(%v) {divisor = 4 : i32} : i32` |

See [Section 2d](./section-2/section-2d/DMATasks.md) for a full walkthrough of
`dma_memcpy_nd` access patterns and buffer-descriptor reuse.

## 3. DMA tasks (configure / start / await / free)

The DMA-task interface is a higher-level way to drive DMAs than raw BDs: you
configure a task (a chain of BDs) once, then start, await and free it. The
compiler allocates the BD ids for you.

| Op | What it does | Applies to | Example |
|----|--------------|------------|---------|
| `aiex.dma_configure_task` | Instantiate a BD chain as a task on a given `tile`, `direction` and `channel`; the BDs are written in the op's region. Returns a task handle. | any tile; `MM2S` / `S2MM` | `%t = aiex.dma_configure_task(%tile, S2MM, 1) { /* aie.dma_bd ... */ }` |
| `aiex.dma_configure_task_for` | As `dma_configure_task`, but the tile/direction/channel are taken from a referenced `aie.shim_dma_allocation` symbol. | shim (via allocation) | `%t = aiex.dma_configure_task_for @alloc { /* ... */ }` |
| `aiex.dma_start_task` | Submit a previously configured task to its channel's queue for execution. | the task's tile/channel | `aiex.dma_start_task(%t)` |
| `aiex.dma_await_task` | Block the sequence until a submitted task completes. The task must have been configured with `issue_token = true`. | the task's tile/channel | `aiex.dma_await_task(%t)` |
| `aiex.dma_free_task` | Tell the BD allocator the task's BD ids may be reused. Only safe once the task is known complete. | the task's tile/channel | `aiex.dma_free_task(%t)` |
| `aiex.dma_start_bd_chain` | Materialize an abstract `aie.bd_chain` with concrete arguments as a task on `tile`/`direction`/`channel` and start it immediately. | any tile; `MM2S` / `S2MM` | `%t = aiex.dma_start_bd_chain @chain(%a) : (memref<32xi32>) on (%tile, MM2S, 0)` |
| `aiex.dma_start_bd_chain_for` | As `dma_start_bd_chain`, but the endpoint is a referenced `aie.shim_dma_allocation` symbol. | shim (via allocation) | `%t = aiex.dma_start_bd_chain_for @chain(%a) : (memref<32xi32>) for @alloc` |
| `aiex.dma_bd_pool_pop` *(compiler-emitted)* | Draw a free BD id from the per-tile runtime free-list pool and return it as an SSA value: the dynamic counterpart to the static BD-id allocator, inserted by the `aie-lower-dynamic-bd-pool` pass inside a runtime-bound `scf.for`. | any tile (dynamic-BD path) | `%bd = aiex.dma_bd_pool_pop(0, 0) : i32` |
| `aiex.dma_bd_pool_push` *(compiler-emitted)* | Return a BD id obtained from `dma_bd_pool_pop` to the per-tile pool for reuse: the dynamic counterpart to `dma_free_task`. | any tile (dynamic-BD path) | `aiex.dma_bd_pool_push(0, 0) bd_id %bd : i32` |

## 4. Synchronization (wait on a task-complete token)

DMA channels signal completion with a task-complete token (TCT). These ops block
the runtime sequence until the expected token arrives. A BD only emits a TCT if
it was configured to issue one (`S2MM` channels do by default; `MM2S` needs
`issue_token = true`).

| Op | What it does | Applies to | Example |
|----|--------------|------------|---------|
| `aiex.npu.sync` | Block until a TCT is received on `column`, `row`, `direction` (`0` = `S2MM`, `1` = `MM2S`), `channel`, optionally over a `column_num` x `row_num` range of tiles. All six are SSA operands. | shim (host-side TCT wait) | `aiex.npu.sync(%c0, %c0, %c0, %c1, %c1, %c1) : i32, i32, i32, i32, i32, i32` |
| `aiex.npu.dma_wait` | Convenience form: wait on the DMA identified by a `symbol` (an `aie.shim_dma_allocation` or ObjectFifo). Lowers to the matching `aiex.npu.sync`. | shim (via allocation) | `aiex.npu.dma_wait {symbol = @out0}` |

## 5. Locks (set)

| Op | What it does | Applies to | Example |
|----|--------------|------------|---------|
| `aiex.set_lock` | Set the value of an `aie.lock` from the sequence. Non-blocking and offers no synchronization guarantee on its own; pair it with a blocking op. | any tile that owns the referenced lock (core, mem, shim) | `aiex.set_lock(%lock, 5)` |

## 6. Runtime parameters and scratchpad (write / declare / read / sync)

Runtime parameters let a host pass values into a design at launch without
recompiling. `rtp_write` writes into a buffer directly; the scratchpad mechanism
carries values from host memory through the command processor into the array.

| Op | What it does | Applies to | Example |
|----|--------------|------------|---------|
| `aiex.npu.rtp_write` | Write `value` at `index` into an `aie.buffer` used to hold runtime parameters. | a buffer used for runtime parameters | `aiex.npu.rtp_write(@rtp_buf, 0, %val) : i32` |
| `aiex.npu.create_scratchpad` | Allocate a control-code scratchpad of `size` bytes on the host and copy it into the command processor's memory. Values are read from it by `update_from_scratchpad`. | NPU command processor | `aiex.npu.create_scratchpad {size = 128 : ui32}` |
| `aiex.npu.update_from_scratchpad` | Compute a value from a scratchpad `StateTable` entry (`mul` / `incr` / `decr` by `func_arg`) and add it to an 8-byte location at a resolved address. Always additive; writes both registers of the pair. | resolved address (shim BD address pairs) | `aiex.npu.update_from_scratchpad<incr> {state_table_idx = 0 : ui8, address = 0 : ui32}` |
| `aiex.scratchpad_parameter` | Declare a named scratchpad runtime parameter. Declared at **module scope** (outside `aie.device`), not inside the sequence; shared by all PDIs in the module. | module scope (declaration) | `aiex.scratchpad_parameter @foo : i32` |
| `aiex.read_scratchpad_parameter` | Read a declared scratchpad parameter on a core. Used inside an `aie.core`, not inside the sequence. | core scope (read site) | `%v = aiex.read_scratchpad_parameter @foo : i32` |
| `aiex.sync_scratchpad_parameters_from_host` | Marker in the sequence for where the host-to-core scratchpad parameter sync should be materialized (expands to create_scratchpad + per-parameter writes + lock releases). | in-sequence marker | `aiex.sync_scratchpad_parameters_from_host` |

`aiex.scratchpad_parameter` and `aiex.read_scratchpad_parameter` are the
companion declaration and core-read ops of the scratchpad mechanism; only
`create_scratchpad`, `update_from_scratchpad` and
`sync_scratchpad_parameters_from_host` are placed in the sequence itself.

## 7. Device image and execution (load / configure / run / preempt)

Ops that (re)configure the partition or hand off control.

| Op | What it does | Applies to | Example |
|----|--------------|------------|---------|
| `aiex.npu.load_pdi` | Load a PDI (Programmable Device Image) to configure the NPU, identified by `id`. Note: firmware skips a reload of the same PDI, so interpose a different PDI to force a device reset. | whole partition (device image) | `aiex.npu.load_pdi {id = 0 : i32}` |
| `aiex.configure` | Set up a configuration (program memories, stream switches) for a referenced device symbol. Contains `aiex.run` ops in its region. | in-sequence; targets a device | `aiex.configure @dev { aiex.run @seq(%a) : (memref<32xi32>) }` |
| `aiex.run` | Execute a named `aie.runtime_sequence`, inlining its instructions at the call site. Legal inside an `aiex.configure`. | inside `aiex.configure` | `aiex.run @seq(%a) : (memref<32xi32>)` |
| `aiex.npu.preempt` | Mark a point where the instruction stream may be interrupted to yield to higher-priority tasks. `level`: `0` no-op, `1` mem tile, `2` AIE tile, `3` AIE registers. | whole instruction stream | `aiex.npu.preempt {level = 2 : ui8}` |

## 8. Control packets (emit)

| Op | What it does | Applies to | Example |
|----|--------------|------------|---------|
| `aiex.control_packet` | Emit a low-level AIE control packet with a header (`address`, `opcode`, `stream_id`, optional `length`) and optional `data` payload. The target row and column are derived from `address`. | any addressable target | `aiex.control_packet {address = 0 : ui32, opcode = 0 : i32, stream_id = 0 : i32}` |

-----

For the authoritative operand lists, attributes and verifiers, see
[`AIEX.td`](../include/aie/Dialect/AIEX/IR/AIEX.td). For the IRON Python
interface and worked examples, see
[Section 2d - Runtime Data Movement](./section-2/section-2d/) and
[Section 3 - My First Program](./section-3/).
