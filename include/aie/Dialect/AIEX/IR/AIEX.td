//===- AIE.td ----------------------------------------------*- tablegen -*-===//
//
// Copyright (C) 2019-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIEX_OPS
#define AIEX_OPS

include "aie/Dialect/AIE/IR/AIEAttrs.td"
include "aie/Dialect/AIE/IR/AIEInterfaces.td"


include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"

def AIEX_Dialect : Dialect {
  let name = "aiex";
  let cppNamespace = "::xilinx::AIEX";
  let description = [{
This is a dialect for experimental work related to AIEngine processors. The
expectation is that new ideas can be developed here before migration to the more
mature AIE dialect.

## Runtime sequence op reference

A runtime sequence is the host-side program that drives an NPU design: it fills
tensors into the AIE-array, launches DMAs, waits on completion tokens, and
reconfigures state between dispatches. In IRON you write it as the body of
`Runtime.sequence()`; in the AIE dialect it is the region of an
`aie.runtime_sequence` operation.

[Section 2d](programming_guide/section-2/section-2d/README.md) teaches the runtime
sequence by example and [Section 3](programming_guide/section-3/README.md) shows a
full end-to-end program. The tables below are a curated catalog of the `aiex`
control ops that appear in a runtime sequence, grouped by the resource they act on
and the verb they perform. Most you write directly; a few are emitted by the
compiler on the dynamic buffer-descriptor path and are marked *(compiler-emitted)*.
The authoritative operand lists, attributes and verifiers for each op follow in the
per-op sections further down this page.

### How to read this catalog

* Op names use their `aiex.` dialect prefix (for example `aiex.npu.write32`).
  Many have IRON Python wrappers; where an op is most often reached through a
  wrapper, [Section 2d](programming_guide/section-2/section-2d/README.md) shows the
  Python form.
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

### Which interface should I use?

For moving data between host and array, reach for one of the two high-level
interfaces that [Section 2d](programming_guide/section-2/section-2d/README.md)
teaches, rather than the raw register and buffer-descriptor ops:

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

### Contents

1. [Registers and tile memory](#1-registers-and-tile-memory-write--masked-write--block-write)
2. [Buffer descriptors and DMA queues](#2-buffer-descriptors-and-dma-queues-program--push--patch)
3. [DMA tasks](#3-dma-tasks-configure--start--await--free)
4. [Synchronization](#4-synchronization-wait-on-a-task-complete-token)
5. [Locks](#5-locks-set)
6. [Runtime parameters and scratchpad](#6-runtime-parameters-and-scratchpad-write--declare--read--sync)
7. [Device image and execution](#7-device-image-and-execution-load--configure--run--preempt)
8. [Control packets](#8-control-packets-emit)

### 1. Registers and tile memory (write / masked-write / block-write)

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

### 2. Buffer descriptors and DMA queues (program / push / patch)

A DMA transfer is described by a buffer descriptor (BD). These ops program BDs
and launch them onto a tile's DMA task queue.

| Op | What it does | Applies to | Example |
|----|--------------|------------|---------|
| `aiex.npu.dma_memcpy_nd` | Program a BD and issue an n-dimensional DMA transfer (a "half-DMA": it configures one endpoint of the transfer). The target tile, channel and direction come from the `metadata` symbol (an `aie.shim_dma_allocation`, or an `aie.objectfifo` symbol directly). `offsets`/`sizes`/`strides` describe the access pattern; `issue_token` requests a completion token on an `MM2S` transfer, while `S2MM` channels always issue one. | shim (host <-> array); `MM2S` / `S2MM` from the allocation | `aiex.npu.dma_memcpy_nd(%arg0[0,0,0,0][1,1,32,32][0,0,64,1]) {id = 0 : i64, metadata = @out0} : memref<32x64xi32>` |
| `aiex.npu.writebd` | Write a complete buffer descriptor (all fields explicit: buffer length/offset, the `d0..d2` sizes/strides, iteration, lock acquire/release, packet header, next-BD) to a BD slot on a tile. The low-level form behind `dma_memcpy_nd`. | any tile with a DMA engine (shim, mem, core) | `aiex.npu.writebd {column = 0 : i32, row = 0 : i32, bd_id = 0 : i32, buffer_length = 1024 : i32, /* ... */}` |
| `aiex.npu.push_queue` | Push a BD id onto a tile channel's task queue to launch it, with an outer-dimension `repeat_count` and optional completion token. | any tile channel; `MM2S` / `S2MM` | `aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %bd repeat %rep {issue_token = false} : i32, i32` |
| `aiex.npu.address_patch` | Patch a runtime buffer address into the instruction stream: takes runtime argument `arg_idx` plus `arg_plus` offset and writes it into a BD address field. Used to bind host buffers whose address is only known at launch. | shim BD addresses | `aiex.npu.address_patch(%off : i32) {addr = 0 : ui32, arg_idx = 0 : i32}` |
| `aiex.npu.assert_bd_field` *(compiler-emitted)* | Guard a runtime BD-field value against its hardware field width (`value <= max`); emitted by the dynamic DMA lowering when a runtime scalar lands in a narrow BD field (e.g. a 10-bit wrap). On the host-C++ path it aborts stream generation for an out-of-range value; on the binary path `value` is a verified constant and it is a no-op. | dynamic-BD lowering path | `aiex.npu.assert_bd_field(%v) {max = 1023 : i32} : i32` |
| `aiex.npu.assert_bd_divisible` *(compiler-emitted)* | Guard a runtime BD size/stride against the address-gen granule (`value % divisor == 0`, or `value == 1` when `allow_unit`); emitted by the dynamic DMA lowering. Same host-C++ / binary behavior as `assert_bd_field`. | dynamic-BD lowering path | `aiex.npu.assert_bd_divisible(%v) {divisor = 4 : i32} : i32` |

See [Section 2d](programming_guide/section-2/section-2d/DMATasks.md) for a full
walkthrough of `dma_memcpy_nd` access patterns and buffer-descriptor reuse.

### 3. DMA tasks (configure / start / await / free)

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

### 4. Synchronization (wait on a task-complete token)

DMA channels signal completion with a task-complete token (TCT). These ops block
the runtime sequence until the expected token arrives. A BD only emits a TCT if
it was configured to issue one (`S2MM` channels do by default; `MM2S` needs
`issue_token = true`).

| Op | What it does | Applies to | Example |
|----|--------------|------------|---------|
| `aiex.npu.sync` | Block until a TCT is received on `column`, `row`, `direction` (`0` = `S2MM`, `1` = `MM2S`), `channel`, optionally over a `column_num` x `row_num` range of tiles. All six are SSA operands. | shim (host-side TCT wait) | `aiex.npu.sync(%c0, %c0, %c0, %c1, %c1, %c1) : i32, i32, i32, i32, i32, i32` |
| `aiex.npu.dma_wait` | Convenience form: wait on the DMA identified by a `symbol` (an `aie.shim_dma_allocation` or ObjectFifo). Lowers to the matching `aiex.npu.sync`. | shim (via allocation) | `aiex.npu.dma_wait {symbol = @out0}` |

### 5. Locks (set)

| Op | What it does | Applies to | Example |
|----|--------------|------------|---------|
| `aiex.set_lock` | Set the value of an `aie.lock` from the sequence. Non-blocking and offers no synchronization guarantee on its own; pair it with a blocking op. | any tile that owns the referenced lock (core, mem, shim) | `aiex.set_lock(%lock, 5)` |

### 6. Runtime parameters and scratchpad (write / declare / read / sync)

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

### 7. Device image and execution (load / configure / run / preempt)

Ops that (re)configure the partition or hand off control.

| Op | What it does | Applies to | Example |
|----|--------------|------------|---------|
| `aiex.npu.load_pdi` | Load a PDI (Programmable Device Image) to configure the NPU, identified by `id`. Note: firmware skips a reload of the same PDI, so interpose a different PDI to force a device reset. | whole partition (device image) | `aiex.npu.load_pdi {id = 0 : i32}` |
| `aiex.configure` | Set up a configuration (program memories, stream switches) for a referenced device symbol. Contains `aiex.run` ops in its region. | in-sequence; targets a device | `aiex.configure @dev { aiex.run @seq(%a) : (memref<32xi32>) }` |
| `aiex.run` | Execute a named `aie.runtime_sequence`, inlining its instructions at the call site. Legal inside an `aiex.configure`. | inside `aiex.configure` | `aiex.run @seq(%a) : (memref<32xi32>)` |
| `aiex.npu.preempt` | Mark a point where the instruction stream may be interrupted to yield to higher-priority tasks. `level`: `0` no-op, `1` mem tile, `2` AIE tile, `3` AIE registers. | whole instruction stream | `aiex.npu.preempt {level = 2 : ui8}` |

### 8. Control packets (emit)

| Op | What it does | Applies to | Example |
|----|--------------|------------|---------|
| `aiex.control_packet` | Emit a low-level AIE control packet with a header (`address`, `opcode`, `stream_id`, optional `length`) and optional `data` payload. The target row and column are derived from `address`. | any addressable target | `aiex.control_packet {address = 0 : ui32, opcode = 0 : i32, stream_id = 0 : i32}` |

For the IRON Python interface and worked examples, see
[Section 2d - Runtime Data Movement](programming_guide/section-2/section-2d/README.md)
and [Section 3 - My First Program](programming_guide/section-3/README.md).
  }];

  let extraClassDeclaration = [{
  }];

  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// AIEX Attributes
//===----------------------------------------------------------------------===//

include "aie/Dialect/AIEX/IR/AIEXAttrs.td"


//===----------------------------------------------------------------------===//
// AIEX Types
//===----------------------------------------------------------------------===//
def AIEX_BlockFloatingPointType : TypeDef<AIEX_Dialect, "BlockFloat", [
    MemRefElementTypeInterface,
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface>
  ]> {
  let summary = "AIEX type representing a block floating point type.";
  let mnemonic = "bfp";
  let description = [{
    This is a type representing a block floating point.
    It is meant to eventually be lowered into a standard type further down the pipeline.
    It the meantime, it can be used for blocked fp related dataflow adaptations.
    Available types are v8bfp16ebs8 and v16bfp16ebs16.
  }];

  let parameters = (
    ins StringRefParameter<"">:$block_type
  );

  let assemblyFormat = "`<` $block_type `>`";

  let extraClassDeclaration = [{
    int getBlockSize() const;
    int getMantissaBits() const;
    int getExponentBits() const;
    int getSubtileShiftBits() const;

    using BlockFormat = struct BlockFormat {
      int blockSize;
      int mantissaBits;
      int exponentBits;
      int subtileShiftBits;
    };

    std::optional<BlockFormat> getBlockFormat() const;
    static std::optional<BlockFormat> getBlockFormat(llvm::StringRef blockType);

    uint64_t getTotalSizeInBits() const;
  }];

  let extraClassDefinition = [{
    int $cppClass::getBlockSize() const { return getBlockFormat().value().blockSize; }
    int $cppClass::getMantissaBits() const { return getBlockFormat().value().mantissaBits; }
    int $cppClass::getExponentBits() const { return getBlockFormat().value().exponentBits; }
    int $cppClass::getSubtileShiftBits() const { return getBlockFormat().value().subtileShiftBits; }

    std::optional<$cppClass::BlockFormat> $cppClass::getBlockFormat() const {
      return getBlockFormat(getBlockType());
    }
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// AIEX Operations
//===----------------------------------------------------------------------===//

class AIEX_Op<string mnemonic, list<Trait> traits = []> :
    Op<AIEX_Dialect, mnemonic, traits>;




def AIE_MulticastOp: AIEX_Op<"multicast", [SingleBlockImplicitTerminator<"AIE::EndOp">]> {
  let arguments = (
    ins Index:$tile,
        WireBundle:$bundle,
        AIEI32Attr:$channel
  );
  let regions = (region AnyRegion:$ports);
  let summary = "An abstraction of multicast";
  let description = [{
    An abstraction of broadcast. During place and
    route, it will be replaced by multiple flows.

    Example:
    ```
      %70 = AIE.tile(7, 0)
      %73 = AIE.tile(7, 3)
      %74 = AIE.tile(7, 4)
      %63 = AIE.tile(6, 3)
      %64 = AIE.tile(6, 4)
      aiex.multicast(%70, "DMA" : 0){
        aiex.multi_dest<%73, "DMA" : 0>
        aiex.multi_dest<%74, "DMA" : 0>
        aiex.multi_dest<%63, "DMA" : 0>
        aiex.multi_dest<%64, "DMA" : 0>
      }
    ```
  }];
  let assemblyFormat = [{ `(` $tile `,` $bundle `:` $channel `)` regions attr-dict }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    int channelIndex() { return getChannel(); }
    AIE::Port port() { return {getBundle(), channelIndex()}; }
  }];
}

def AIE_MultiDestOp: AIEX_Op<"multi_dest", [HasParent<"MulticastOp">]> {
  let arguments = (
    ins Index:$tile,
        WireBundle:$bundle,
        AIEI32Attr:$channel
  );
  let summary = "A destination port of multicast flow";
  let description = [{
    An object representing the destination of a multicast flow. This must exist
    within an [aiex.multicast] operation. There can be multiple destinations within an
    aiex.multicast Op.

    See [aiex.multicast]for an example.
  }];
  let assemblyFormat = [{
    `<` $tile `,` $bundle `:` $channel `>` attr-dict
  }];
  let extraClassDeclaration = [{
    int channelIndex() { return getChannel(); }
    AIE::Port port() { return {getBundle(), channelIndex()}; }
  }];
}

def AIE_BroadcastPacketOp: AIEX_Op<"broadcast_packet", [SingleBlockImplicitTerminator<"AIE::EndOp">]> {
  let arguments = (
    ins Index:$tile,
        WireBundle:$bundle,
        AIEI32Attr:$channel
  );
  let regions = (region AnyRegion:$ports);
  let summary = "Combination of broadcast and packet-switch";
  let description = [{
    An abstraction of broadcast and packet-switched flow. During place and
    route, it will be replaced by packet-switched flow and further replaced
    by MasterSets and PacketRules inside switchboxes.

    Example:
    ```
      %70 = AIE.tile(7, 0)
      %73 = AIE.tile(7, 3)
      %74 = AIE.tile(7, 4)
      %63 = AIE.tile(6, 3)
      %64 = AIE.tile(6, 4)
      AIE.broadcast_packet(%70, "DMA" : 0){
        AIE.bp_id(0x0){
          AIE.bp_dest<%73, "DMA" : 0>
          AIE.bp_dest<%63, "DMA" : 0>
        }
        AIE.bp_id(0x1){
          AIE.bp_dest<%74, "DMA" : 0>
          AIE.bp_dest<%64, "DMA" : 0>
        }
      }
    ```
  }];
  let assemblyFormat = [{ `(` $tile `,` $bundle `:` $channel `)` regions attr-dict }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    int channelIndex() { return getChannel(); }
    AIE::Port port() { return {getBundle(), channelIndex()}; }
  }];
}

def AIE_BPIDOp: AIEX_Op<"bp_id", [SingleBlockImplicitTerminator<"AIE::EndOp">]> {
  let arguments = (ins AIEI8Attr:$ID);
  let regions = (region AnyRegion:$ports);
  let summary = "A set of packets that share the same ID";
  let description = [{
    A set of destination packets that share the same source and ID. This must exist
    within an [AIE.broadcast_packet] operation.
    See [AIE.broadcast_packet]for an example.
  }];
  let assemblyFormat = [{ `(` $ID `)` regions attr-dict }];
  let extraClassDeclaration = [{
    int IDInt() { return getID(); }
  }];
}

def AIE_BPDestOp: AIEX_Op<"bp_dest", [HasParent<"BPIDOp">]> {
  let arguments = (
    ins Index:$tile,
        WireBundle:$bundle,
        AIEI32Attr:$channel
  );
  let summary = "A destination port";
  let description = [{
    An object representing the destination of a  Broad Packet. This must exist
    within an [AIE.bp_id] operation.
    See [AIE.broadcast_packet] for an example.
  }];
  let assemblyFormat = [{
    `<` $tile `,` $bundle `:` $channel `>` attr-dict
  }];
  let extraClassDeclaration = [{
    int channelIndex() { return getChannel(); }
    AIE::Port port() { return {getBundle(), channelIndex()}; }
  }];
}

def AIE_TokenOp: AIEX_Op<"token", [Symbol]> {
  let summary = "Declare a token (a logical lock)";
  let description = [{
    This operation creates a logical lock. We use Symbol so that it can be referenced globally.
    Unlike phsical locks, logical locks are unlimited, and we can specify any integer value
    associated with a lock. The logical lock is used to manually specify the dependence of tasks, or
    core executions.

    The operation can also be generated automatically if the Dependence Analysis can be leveraged.

    Example:
      AIE.token(0) {sym_name = "token0"} // Declare token0 with initial value of 0

      ...

      AIE.useToken @token0("Acquire", 0) // acquire token0 if its value is 0

      ...

      AIE.useToken @token0("Release", 5) // release token0 and set its value to 5

  }];
  let arguments = (ins AIEI32Attr:$value);
  let assemblyFormat = [{ `(` $value `)` attr-dict }];
  let extraClassDeclaration = [{
    int getTokenValue() { return getValue(); }
  }];
}

def AIE_UseTokenOp: AIEX_Op<"useToken", []> {
  let summary = "acquire/release a logical lock";
  let description = [{
    This operation uses token (logical lock). A logical lock can be acquired or released with a value.
    Similar to UseLockOp, this operation can be understood as "blocking" op.
  }];
  let arguments = (
    ins FlatSymbolRefAttr:$tokenName,
        AIEI32Attr:$value,
        LockAction:$action
  );
  let assemblyFormat = [{ $tokenName `(` $action `,` $value `)` attr-dict }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    bool acquire() { return (getAction() == AIE::LockAction::Acquire); }
    bool release() { return (getAction() == AIE::LockAction::Release); }
    int getTokenValue() { return getValue(); }
  }];
}

def AIE_MemcpyOp: AIEX_Op<"memcpy", []> {
  let summary = "A memcpy op";
  let description = [{
    This operation defines a logical data transfer of a buffer from a source tile to another buffer
    from a destination tile.

    This operation should be lowered to Mem ops with DMA setup and Flow ops for routing data from
    the source tile to the dest. tile.
  }];
  let arguments = (
    ins FlatSymbolRefAttr:$tokenName,
        AIEI32Attr:$acqValue,
        AIEI32Attr:$relValue,
        Index:$srcTile,
        AnyMemRef:$srcBuf,
        AIEI32Attr:$srcOffset,
        AIEI32Attr:$srcLen,
        Index:$dstTile,
        AnyMemRef:$dstBuf,
        AIEI32Attr:$dstOffset,
        AIEI32Attr:$dstLen
  );
  let assemblyFormat = [{
    $tokenName `(` $acqValue `,` $relValue `)` `(`
      $srcTile `:` `<` $srcBuf `,` $srcOffset `,` $srcLen `>` `,`
      $dstTile `:` `<` $dstBuf `,` $dstOffset `,` $dstLen `>` `)`
        attr-dict `:` `(` type($srcBuf) `,` type($dstBuf) `)`
  }];
  let extraClassDeclaration = [{
    int getAcquireTokenValue() { return getAcqValue(); }
    int getReleaseTokenValue() { return getRelValue(); }
    int getSrcOffsetValue() { return getSrcOffset(); }
    int getDstOffsetValue() { return getDstOffset(); }
    int getSrcLenValue() { return getSrcLen(); }
    int getDstLenValue() { return getDstLen(); }
  }];
}





// NOTE: runtime_sequence operation has been moved to the AIE dialect (AIEOps.td)
// Use aie.runtime_sequence instead of aiex.runtime_sequence

def AIE_ConfigureOp: AIEX_Op<"configure", [
  HasParent<"AIE::RuntimeSequenceOp">,
  NoTerminator
]>
 {
  let summary = "Set up a configuration (program memories, stream switches, etc.) on the NPU device.";
  let arguments = (ins FlatSymbolRefAttr:$symbol,
      OptionalAttr<AIEX_ExpandMode>:$expand_mode);

  let assemblyFormat = [{
     $symbol regions attr-dict
  }];

  let extraClassDeclaration = [{
    AIE::DeviceOp getReferencedDeviceOp();
  }];

  let regions = (region
    AnyRegion:$body
  );

  let hasVerifier = 1;
}

def AIE_RunOp: AIEX_Op<"run", [HasParent<"ConfigureOp">]> {
  let arguments = (
    ins FlatSymbolRefAttr:$runtime_sequence_symbol,
        Variadic<AnyType>:$args
  );

  let assemblyFormat = [{
     $runtime_sequence_symbol `(` $args `)` `:` `(` type($args) `)` attr-dict
  }];

  let extraClassDeclaration = [{
    AIE::DeviceOp getCalleeDeviceOp();
    AIE::RuntimeSequenceOp getCalleeRuntimeSequenceOp();
  }];

  let summary = "Execute a runtime sequence";
  let description = [{
    Executes an `aiex.runtime_sequence` with the given name and arguments by inlining its instructions at the call site.
  }];
}

def AIE_NpuDmaMemcpyNdOp: AIEX_Op<"npu.dma_memcpy_nd", [
    AttrSizedOperandSegments,
    MyOffsetSizeAndStrideOpInterface
  ]> {
  let summary = "half DMA operator";

  let description = [{
    An n-dimensional half DMA operator.

    Programs a DMA to access a memory `memref` with an access pattern specified by `offsets`,
    `sizes` and `strides` or `static_offsets`, `static_sizes` and `static_strides`. The operator
    references the target DMA coordinates (`x`, `y`) and channel through the `metadata`
    symbol and specifies a descriptor `id` to be used, which will become the `bd_id` to be used
    when lowered further. The `issue_token` attribute specifies whether the execution of this
    operation should issue a token which can be received and read for synchronization purposes.
    This `issue_token` attribute is set to `false` by default for `MM2S` for backward compatibility
    and **is always set to true for** `S2MM` channels.
    The burst length attribute specifies the burst length in bytes for the DMA operation. A value
    of 0 indicates that the burst length is not specified and the maximal burst length is used.

    The `axcache` attribute specifies the raw 4-bit `AxCACHE` value used for the shim DMA's
    AXI-MM transfers to/from host memory. If unset, `AIETargetModel::getDefaultAxCache()` supplies
    the value.

    #### `metadata` -- Specifying Tile, Channel, Direction and Linking a `dma_memcpy_nd` to its Other Half

    The `metadata` attribute must point to a symbol referencing a
    [`aie.shim_dma_allocation` operation](AIEDialect.html#aiedma_bd-xilinxaiedmabdop).
    The tile coordinates of the DMA to configure, the channel number and the direction (`MM2S` or `S2MM`) are taken from this operation.

    To connect the DMA to its other half (i.e. a `MM2S` DMA to its receiving end and a `S2MM` to the sending end),
    the user must configure a flow (`aie.flow`) between the tile and channel referenced in the `aie.shim_dma_allocation` and the corresponding other end.

    When using ObjectFIFOs, the `aie.shim_dma_allocation` operations and the `aie.flows` are generated automatically.
    The symbol of the `aie.objectfifo` (create) operation can be used directly in `metadata` in this case.

    #### Notes on Synchronization and Reusing Buffer Descriptor IDs

    When the `dma_memcpy_nd` operation executes, it immediately reprograms the buffer descriptor with ID `bd_id` on tile (`x`, `y`), even if that buffer descriptor is currently executing.
    Without proper synchronization, this inevitably leads to nondeterministic results.

    Programming a buffer descriptor that is not currently executing is harmless.
    Thus, the first `dma_memcpy_nd` call for each `bd_id` requires no synchronization.

    However, if you wish to later re-use a `bd_id` on the same tile, you must wait for the previous buffer descriptor to complete.
    The `sync` or `dma_wait` operations can be used for this.

    `sync` blocks until it receives a _task completion token_ (TCT).
    To properly synchronize, you must thus configure your BD to issue a TCT using the `issue_token` attribute, then wait on that token before reusing the BD.

    `dma_wait` is a convenience operation that lowers to the corresponding `sync` operation for the refrenced symbol.

    Note that if you have multiple concurrently running BDs and you can reason one BD will always complete after all others, it is not strictly necessary to issue and wait on the TC token for every BD.
    For example, if you have input and output BDs on the shim, and you know the cores will only push output onto the output BD after the input BDs have completed, it may be sufficient to synchronize only on the output BD before reusing input BDs.

    #### Data Layout Transformations

    The `sizes` and `strides` attributes describe a data layout transformation to be performed by the DMA.
    These transformations are described in more depth in the documentation for the
    [`aie.dma_bd` operation](AIEDialect.html#aiedma_bd-xilinxaiedmabdop).
    Note that the syntax here differs from that of the `dma_bd` operation:
    offsets and strides are given as separate arrays instead of tuples.

    The `offsets` array is used to calculate a static offset into the memref.
    Each offset in the array is understood in relation to the shape of the memref;
    the lowest-dimension `offset` is a direct offset in units of memref element type, and the higher dimensions are multiplied by the size of the memref in those dimensions.
    Note that this is for convenience of the user only.
    The hardware only supports a single static offset, and this offset is calculated at compile time.
    Thus, all offsets can be equivalently expressed with the lowest dimension only.

    #### Automatic Linearization of Contiguous Accesses

    A canonicalization pattern automatically folds a contiguous row-major access pattern into
    the canonical linear form `[s3, 1, 1, N][st3, 0, 0, 1]`, where N is the product of the
    inner three sizes. An access is contiguous when `strides[0] == 1` and each outer stride
    equals the product of the inner sizes (i.e. a standard row-major scan).

    This means users can express naturally multidimensional accesses such as a 2D image
    `[1, 1, height, width][0, 0, width, 1]` or a 3D activation tensor
    `[1, H, W, C][0, W*C, C, 1]` without worrying about hardware dimension size limits.
    The compiler will fold them to the linear form, which uses a wider hardware register
    and avoids the 10-bit d0 wrap-size constraint that applies to ND transfers.

    #### Packet Header Attribute
    The optional `packet` attribute defines the packet header and packet type that gets issued per DMA BD.
    If the attribute is set, then every time the DMA BD gets issued, a packet header is generated prior to the transmission of data.
    The packet header is used to guide arbitration throughout a packet-routed data flow, where each switch box arbitrates the data packet to stream to a successor based on the packet header.

  }];

  let arguments = (ins AnyRankedOrUnrankedMemRef:$memref,
      // NOTE: these are in reverse order: offset3, offset2, ...
      Variadic<I64>:$offsets, Variadic<I64>:$sizes, Variadic<I64>:$strides,
      ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<4>]>:$static_offsets,
      ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<4>]>:$static_sizes,
      ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<4>]>:$static_strides,
      OptionalAttr<PacketInfoAttr>:$packet, SymbolRefAttr:$metadata,
      I64Attr:$id, DefaultValuedOptionalAttr<BoolAttr, "false">:$issue_token,
      DefaultValuedOptionalAttr<I64Attr, "0">:$d0_zero_before,
      DefaultValuedOptionalAttr<I64Attr, "0">:$d1_zero_before,
      DefaultValuedOptionalAttr<I64Attr, "0">:$d2_zero_before,
      DefaultValuedOptionalAttr<I64Attr, "0">:$d0_zero_after,
      DefaultValuedOptionalAttr<I64Attr, "0">:$d1_zero_after,
      DefaultValuedOptionalAttr<I64Attr, "0">:$d2_zero_after,
      DefaultValuedOptionalAttr<I64Attr, "0">:$burst_length,
      OptionalAttr<ConfinedAttr<I64Attr, [IntMinValue<0>,
                                          IntMaxValue<15>]>>:$axcache,
      // if set, the aiex.scratchpad_parameter value added to the BD's address
      // as an offset
      OptionalAttr<FlatSymbolRefAttr>:$offset_parameter,
      // Set by --aie-lower-scratchpad-parameters from $offset_parameter: the
      // scratchpad state-table index whose value is used as the BD address
      // offset. After lowering, $offset_parameter is removed in favor of this
      // attribute.
      OptionalAttr<UI8Attr>:$offset_state_table_idx);

  let assemblyFormat = [{
    `(` $memref ``
    custom<DynamicIndexList>($offsets, $static_offsets) ``
    custom<DynamicIndexList>($sizes, $static_sizes) ``
    custom<DynamicIndexList>($strides, $static_strides) ``
    (`,` `packet` `=` $packet^)? `)`
    attr-dict `:` type($memref)
  }];

  let extraClassDeclaration = [{
    static unsigned getOffsetSizeAndStrideStartOperandIndex();
    std::array<unsigned, 3> getArrayAttrMaxRanks();

    /* Returns the data transfer offset in bytes, i.e. the first N bytes of the
       target buffer will be skipped. In the IR, offsets are expressed in units
       of memref element data type size. */
    int64_t getOffsetInBytes();

    bool isLinearTransferWithoutTransformation();

    /* Returns the bitwidth of the type inside the memref through a
    call to DataLayout */
    uint64_t getElementTypeBitwidth();

    /* Verifies the supported scope and bounds for a transfer carrying runtime
       (SSA) sizes/strides. Called from verify() on the dynamic path. */
    llvm::LogicalResult verifyDynamicSizesStrides(
        const xilinx::AIE::AIETargetModel &targetModel,
        mlir::BaseMemRefType buffer);

    /* $axcache, or the target's default when unset. */
    uint32_t getAxcacheOrDefault();
  }];

  let extraClassDefinition = [{
    unsigned $cppClass::getOffsetSizeAndStrideStartOperandIndex() { return 1; }
    std::array<unsigned, 3> $cppClass::getArrayAttrMaxRanks() { return {4, 4, 4}; }
    uint64_t $cppClass::getElementTypeBitwidth() {
      DataLayout dataLayout = DataLayout::closest(*this);
      return dataLayout.getTypeSizeInBits(getMemref().getType().getElementType());
    }
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def AIE_NpuDmaWaitOp: AIEX_Op<"npu.dma_wait", []> {
  let summary = "Blocking operation to wait for a DMA to complete execution.";
  let description = [{
    The NpuDmaWaitOp blocks until the DMA referenced through `symbol` completes execution
    and issues a task-complete-token (TCT).

    `symbol` is a reference to a `aie.shim_dma_allocation`, which contains information about the column, channel and channel direction on which to wait for a TCT.
    The `aie.shim_dma_allocation` may be generated from an ObjectFIFO, in which case you can directly pass the ObjectFIFO symbol refrence.
    `npu.dma_wait` will be lowered to the corresponding `npu.sync` operation using the information from `symbol`.

    Example:
    ```mlir
      ...
      aie.objectfifo @out0(%tile_0_1, {% raw %}{%tile_0_0}{% endraw %}, 4 : i32) : !aie.objectfifo<memref<32x32xi32>>
      ...
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[1, 1, 0, 0][1, 1, 32, 32][1, 1, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @out0} : memref<32x64xi32>
      ...
      aiex.npu.dma_wait { symbol = @out0 }
    ```
    Here, we have an objectfifo with symbol name `out0`, which is then referenced in the
    `npu.dma_memcpy_nd` operation as the target for the respective DMA operation. Afterwards,
    an `npu.dma_wait` operation references the same symbol to block until the respective DMA
    has executed all of its tasks.
  }];
  let arguments = (
    ins FlatSymbolRefAttr:$symbol
  );
  let assemblyFormat = [{
    attr-dict
  }];
  let hasVerifier = 1;
}

// Write RTP
def AIE_NpuWriteRTPOp: AIEX_Op<"npu.rtp_write", []> {
  let summary = "rtp write operator";
  let arguments = (
    ins FlatSymbolRefAttr:$buffer,
        UI32Attr:$index,
        I32:$value
  );
  let results = (outs );
  let assemblyFormat = [{
    `(` $buffer `,` $index `,` $value `)` attr-dict `:` type($value)
  }];
  let description = [{
    rtp write operator. `value` is an SSA operand; use arith.constant for a
    compile-time-known value or pass a runtime sequence value to supply the RTP
    value at runtime.
  }];
}

// Push BD to Queue
def AIE_NpuPushQueueOp: AIEX_Op<"npu.push_queue", []> {
  let summary = "bd queue push operator";
  let arguments = (
    ins I32Attr:$column,
        I32Attr:$row,
        DMAChannelDir:$direction,
        I32Attr:$channel,
        BoolAttr:$issue_token,
        I32:$repeat_count,
        I32:$bd_id
  );
  let results = (outs );
  let assemblyFormat = [{
    `(` $column `,` $row `,` $direction `:` $channel `)`
    `bd_id` $bd_id `repeat` $repeat_count attr-dict `:` type($bd_id) `,` type($repeat_count)
  }];
  let hasVerifier = 1;
  let description = [{
    bd queue push operator.

    'bd_id' (which physical buffer descriptor slot to launch) and 'repeat_count'
    (outer-dimension repeat for the launch) are SSA operands; use arith.constant
    for compile-time-known values or pass runtime sequence values to select the
    BD / repeat count at runtime.
  }];
}

// WRITE32
def AIE_NpuWrite32Op: AIEX_Op<"npu.write32", []> {
  let summary = "write32 operator";
  let arguments = (
    ins I32:$address,
        I32:$value,
        OptionalAttr<FlatSymbolRefAttr>:$buffer,
        OptionalAttr<I32Attr>:$column,
        OptionalAttr<I32Attr>:$row
  );
  let results = (outs );
  let assemblyFormat = [{
    `(` $address `,` $value `)` attr-dict `:` type($address) `,` type($value)
  }];
  let description = [{
    NPU write32 operator writes a 32bit value to the AIE array.

    'address' and 'value' are SSA operands; use arith.constant for
    compile-time-known values or pass runtime sequence values for
    runtime-parameterized sequences.

    If 'buffer' is present then 'address' is interpreted as an offset into the
    aie.buffer with symbol name 'buffer'.
    If 'column' and 'row' are present then 'address' is interpreted as an offset
    into the memory space of aie.tile(column, row).
    If 'buffer' is not present and 'column' and 'row' are not present then
    'address' is interpreted as a full 32-bit address in the AIE array.
  }];
  let extraClassDeclaration = [{
    // Compile-time absolute address, or nullopt if 'address' is not constant.
    std::optional<uint32_t> getAbsoluteAddress();
  }];
}

// ASSERT_BD_FIELD
def AIE_NpuAssertBdFieldOp: AIEX_Op<"npu.assert_bd_field", []> {
  let summary = "Guard a runtime BD-field value against its hardware field width";
  let arguments = (
    ins I32:$value,
        I32Attr:$max
  );
  let results = (outs );
  let assemblyFormat = [{
    `(` $value `)` attr-dict `:` type($value)
  }];
  let description = [{
    Asserts, at TXN-build time, that a runtime value assigned to a buffer-
    descriptor field fits its hardware field: `value <= max`. Emitted by the
    dynamic DMA lowering for a runtime scalar landing in a narrow field (e.g. the
    10-bit d0/d1 wrap or 6-bit iteration wrap), where masking would silently
    truncate an out-of-range value.

    This op only ever appears on the dynamic (runtime-value) DMA lowering path;
    a fully static sequence never produces it. When `value` is a constant, the
    verifier hard-errors if it exceeds `max`, so it is caught at compile time.

    The TXN instruction stream has no on-device trap, so a runtime guard runs
    host-side in the generated C++ TXN builder: --aie-npu-to-cpp lowers this op
    to an early `if (value > max) return std::nullopt;` so the builder yields no
    stream for an illegal value instead of a corrupt one. On the binary
    (--aie-npu-to-binary) path `value` is always a verified constant, so the op
    carries no runtime check and is emitted as a no-op (skipped, contributing no
    TXN words).
  }];
  let hasVerifier = 1;
}

// ASSERT_BD_DIVISIBLE
def AIE_NpuAssertBdDivisibleOp: AIEX_Op<"npu.assert_bd_divisible", []> {
  let summary = "Guard a runtime BD size/stride against the address-gen granule";
  let arguments = (
    ins I32:$value,
        I32Attr:$divisor,
        UnitAttr:$allow_unit
  );
  let results = (outs );
  let assemblyFormat = [{
    `(` $value `)` attr-dict `:` type($value)
  }];
  let description = [{
    Asserts, at TXN-build time, that a runtime size/stride is realizable in
    hardware: its byte extent is a whole number of address-gen granules, i.e.
    `value % divisor == 0`. The hardware transfers whole granules only, so a
    size or stride whose `value * elemWidth` is not a granule multiple (e.g. an
    int8 innermost size of 2 = 16 bits against a 32-bit granule) cannot be
    encoded. `divisor` is the compile-time granule/element ratio.

    `allow_unit` exempts a value of 1 from the divisibility rule: the innermost
    stride is allowed to be sub-granule when it is exactly 1 (the contiguous
    scan the encoder collapses to hardware stride 0), but a non-unit sub-granule
    stride (e.g. int8 stride 2) is still unrealizable. So with `allow_unit` the
    predicate is `value == 1 || value % divisor == 0`.

    Like `assert_bd_field`, this op only appears on the dynamic DMA lowering
    path, and a constant `value` is hard-errored by the verifier at compile
    time. The TXN stream has no on-device trap, so a runtime guard runs
    host-side: --aie-npu-to-cpp lowers it to an early
    `if (value % divisor != 0) return std::nullopt;` (or, with `allow_unit`,
    `if (value != 1 && value % divisor != 0) ...`). On the binary
    (--aie-npu-to-binary) path `value` is always a verified constant, so the op
    carries no runtime check and is emitted as a no-op (skipped, contributing no
    TXN words).
  }];
  let hasVerifier = 1;
}

// MASKWRITE
def AIE_NpuMaskWrite32Op: AIEX_Op<"npu.maskwrite32", []> {
  let summary = "Write a masked 32-bit value to the AIE array";
  let arguments = (
    ins I32:$address,
        I32:$value,
        I32:$mask,
        OptionalAttr<FlatSymbolRefAttr>:$buffer,
        OptionalAttr<I32Attr>:$column,
        OptionalAttr<I32Attr>:$row
  );
  let results = (outs );
  let assemblyFormat = [{
    `(` $address `,` $value `,` $mask `)` attr-dict `:` type($address) `,` type($value) `,` type($mask)
  }];
  let description = [{
    NPU mask write32 operator writes a masked 32bit value to the AIE array.

    'address', 'value', and 'mask' are SSA operands; use arith.constant for
    compile-time-known values or pass runtime sequence values for
    runtime-parameterized sequences.

    If 'buffer' is present then 'address' is interpreted as an offset into the
    aie.buffer with symbol name 'buffer'.
    If 'column' and 'row' are present then 'address' is interpreted as an offset
    into the memory space of aie.tile(column, row).
    If 'buffer' is not present and 'column' and 'row' are not present then
    'address' is interpreted as a full 32-bit address in the AIE array.
  }];
  let extraClassDeclaration = [{
    // Compile-time absolute address, or nullopt if 'address' is not constant.
    std::optional<uint32_t> getAbsoluteAddress();
  }];
}

// BLOCKWRITE
def AIE_NpuBlockWriteOp: AIEX_Op<"npu.blockwrite", []> {
  let summary = "blockwrite operator";
  let arguments = (
    ins UI32Attr:$address,
        AnyMemRef:$data,
        OptionalAttr<FlatSymbolRefAttr>:$buffer,
        OptionalAttr<I32Attr>:$column,
        OptionalAttr<I32Attr>:$row
  );
  let results = (outs );
  let assemblyFormat = [{
    `(` $data `)` attr-dict `:` type($data)
  }];
  let description = [{
    blockwrite operator writes the data from the memref 'data' to the AIE array.
    If 'buffer' is present then 'address' is interpreted as an offset into the
    aie.buffer with symbol name 'buffer'.
    If 'column' and 'row' are present then 'address' is interpreted as an offset
    into the memory space of aie.tile(column, row).
    If 'buffer' is not present and 'column' and 'row' are not present then
    'address' is interpreted as a full 32-bit address in the AIE array.
  }];
  let extraClassDeclaration = [{
    std::optional<uint32_t> getAbsoluteAddress();
    mlir::DenseIntElementsAttr getDataWords();
  }];
}

// BLOCKWRITE_VALUES
def AIE_NpuBlockWriteValuesOp : AIEX_Op<"npu.blockwrite_values", []> {
  let summary = "blockwrite operator with a runtime-computed payload";
  let arguments = (ins I32:$address, Variadic<I32>:$values);
  let results = (outs);
  let assemblyFormat = [{
    `(` $address `:` type($address) `)` `values` $values `:` type($values)
    attr-dict
  }];
  let description = [{
    Writes `values`, in order, as a single contiguous block-write of words
    starting at `address`. Unlike `npu.blockwrite`, whose address is a
    compile-time attribute and whose payload is a compile-time-constant memref,
    both are SSA operands here, so the payload is itself computed at TXN-build
    time, by reference to calculated SSA values.

    This is used for BD register configuration. aiebu expects one blockwrite
    per BD configuration; it applies `npu.address_patch` instructions to the
    last preceding BD-configuration-blockwrite, erroring if there's none.

    This op appears only on the dynamic DMA lowering path (a runtime
    size/stride/length or a runtime BD id) and is supported only by the EmitC
    (C++ TXN) target; the static binary target has no representation for a
    runtime-valued block-write and rejects it.
  }];
}

// OP_SYNC
def AIE_NpuSyncOp: AIEX_Op<"npu.sync", []> {
  let summary = "sync operator";
  let arguments = (
    ins I32:$column,
        I32:$row,
        I32:$direction,
        I32:$channel,
        I32:$column_num,
        I32:$row_num
  );
  let results = (outs );
  let assemblyFormat = [{
    `(` $column `,` $row `,` $direction `,` $channel `,` $column_num `,` $row_num `)` attr-dict `:` type(operands)
  }];
  let description = [{
    The sync operation blocks execution of the instruction stream until a task-complete token (TCT) is received on `column`, `row`, channel `channel`, direction `direction` (where `0` is `S2MM` and `1` is `MM2S`).

    All parameters are SSA values. Use `arith.constant` for compile-time-known
    values; pass runtime sequence arguments (or values derived from them)
    directly for runtime-parameterized sequences.

    #### Troubleshooting

    If this operation appears to deadlock, ensure that at least one buffer descriptor is configured to issue a TCT on the channel you expect.
    By default, `dma_memcpy_nd` operations only issue tokens for `S2MM` channels, and `issue_token` must be set to `true` to issue tokens for `MM2S` channels.
  }];
}

// XAIE_IO_CUSTOM_OP_BEGIN + 1 (address patch)
def AIE_NpuAddressPatchOp: AIEX_Op<"npu.address_patch", []> {
  let summary = "address patch operator";
  let arguments = (
    ins UI32Attr:$addr,
        Optional<I32>:$addr_val,
        I32Attr:$arg_idx,
        I32:$arg_plus
  );
  let results = (outs );
  let assemblyFormat = [{
    `(` $arg_plus `:` type($arg_plus) `)` (`addr` $addr_val^ `:` type($addr_val))?
    attr-dict
  }];
  let description = [{
    address patch operator. `arg_plus` is an SSA value; use arith.constant for
    a compile-time-known offset or pass a runtime sequence value for a
    runtime-parameterized buffer offset.

    `addr` is the register to patch (the BD's buffer-address word). It is a
    compile-time attribute on the static path; when the BD id is chosen at
    runtime (dynamic free-list pool) the register address is itself runtime and
    is supplied via the optional `addr_val` operand instead, which then takes
    precedence. Only the EmitC (C++ TXN) target supports a runtime `addr_val`;
    the static binary target requires the constant `addr`.
  }];
  let hasVerifier = 1;
}

def AIE_NpuPreemptOp: AIEX_Op<"npu.preempt", []> {
  let summary = "Preempt transaction operation";
  let arguments = (
    ins UI8Attr:$level
  );
  let results = (outs );
  let assemblyFormat = [{
    attr-dict
  }];
  let description = [{
    Yield to higher priority task(s). Indicates to the transaction processor that the instruction stream can be interrupted at this point.
    Levels:
    0: Noop.
    1: Mem tile.
    2: AIE tile.
    3: AIE registers.
  }];
}

// XAIE_IO_CREATE_SCRATCHPAD (opcode 10)
def AIE_NpuCreateScratchpadOp: AIEX_Op<"npu.create_scratchpad", []> {
  let summary = "Create a control code scratchpad memory region";
  let arguments = (
    ins UI32Attr:$size,
        DefaultValuedOptionalAttr<UI8Attr, "0">:$usage_type
  );
  let results = (outs );
  let assemblyFormat = [{
    attr-dict
  }];
  let description = [{
    Create a scratchpad memory for data exchange between the host's main memory and the NPU command processor and copy its contents to the command processor's memory.

    When the runtime (XRT) observes that this instruction is present in the runtime sequence, it will allocate a scratchpad memory of the specified size on the host.
    When the command processor firmware executes this instruction, it copies the data in the runtime-allocated scratchpad region from the host's main memory to the NPU command processor's memory.
    From there, you can write values from the copy of the scratchpad memory in the command processor to arbitrary locations in the NPU (with restrictions) via the `npu.update_from_scratchpad` op.

    The flow of data therefore looks like this:
    ```
                    create_scratchpad                                update_from_scratchpad
    [Host memory]  ------------------->  [Command processor memory]  ------------------------>  [NPU partition memory/registers]
        ^                 |
        |                 |
         -----------------
          XRT sees inst.
          and allocates
    ```

    To get a handle on the allocated scratchpad memory from XRT, use the `run.get_ctrl_scratchpad_bo()` method on the `xrt::run` object in your host application (`test.cpp`).
    An example can be found in `test/npu-xrt/scratchpad_regwrite`.

    The `usage_type` attribute specifies the scratchpad layout; currently only a value of `0` is supported.

    The `size` attribute specifies the size of the scratchpad in bytes.

    The host's main memory address is patched into the instruction at runtime by XRT based on the `.ctrl.scratchpad` section in the ELF. The assembler (aiebu) generates patching information for this address when it encounters this opcode.

    The scratchpad memory contains a `StateTable`, indexed by 32-bit words by the `npu.update_from_scratchpad` op. `StateTable` constraints: max 32 entries, max total scratchpad size 128 bytes.
  }];
  let hasVerifier = 1;
}

// XAIE_IO_UPDATE_REG (opcode 12)
def AIE_NpuUpdateFromScratchpadOp: AIEX_Op<"npu.update_from_scratchpad", []> {
  let summary = "Add a computed value to an 8-byte section of NPU memory from scratchpad state";
  let arguments = (
    ins UI8Attr:$state_table_idx,
        DefaultValuedAttr<StateTableFunc, "StateTableFunc::Incr">:$func,
        DefaultValuedOptionalAttr<UI32Attr, "0">:$func_arg,
        UI32Attr:$address,
        OptionalAttr<FlatSymbolRefAttr>:$buffer,
        OptionalAttr<I32Attr>:$column,
        OptionalAttr<I32Attr>:$row
  );
  let results = (outs );
  let assemblyFormat = [{
    (`<` $func^ `>`)? attr-dict
  }];
  let description = [{
    Add a computed value based on scratchpad contents to an 8-byte section at the target address.

    This instruction reads the contents of scratchpad memory created using the `npu.create_scratchpad` op (the `StateTable`), calculates a value, then adds the result to the memory location or register denoted by the given address as follows:

    1. Reads the existing 64-bit value from the register pair:

       ```
       existing = (*(addr+4) & 0xFFFF) << 32 | (*addr & 0xFFFFFFFC)
       ```

    2. Computes a delta based on the contents of the scratchpad memory created using `npu.create_scratchpad` (the `StateTable`) based on the selected function:

       - `mul`:  `delta = StateTable[state_table_idx] * func_arg`
       - `incr`: `delta = StateTable[state_table_idx] + func_arg`
       - `decr`: `delta = StateTable[state_table_idx] - func_arg`

    3. Adds the delta to the existing value:

      ```
      result = existing + delta
      ```

    4. Writes back the lower 48 bits of the result:

       ```
       *addr     = result & 0xFFFFFFFC    (bits [1:0] forced to 0)
       *(addr+4) = (*(addr+4) & 0xFFFF0000) | ((result >> 32) & 0xFFFF)
       ```

    ### Constraints

    - This is always additive. It adds the computed delta to whatever value is already in the register pair. It cannot set an absolute value.
    - The lower 2 bits of the first register are always cleared (4-byte aligned).
    - The upper 16 bits of the second register are preserved unchanged.
    - Always writes 8 contiguous bytes (both registers in the pair).

    ### Rationale

    The firmware instruction underpinning this operation was originally intended to patch shim buffer descriptor addresses only.
    Because of this, this always writes 48 bits (size of BD addresses) and the lower bits are zeroed (assuring the value is a multiple of the addressable word size).

    ### Attributes

    - `state_table_idx` is a 32-bit-word index offset into the scratchpad memory. The source value will be read from this offset into the scratchpad.
    - `func` is the function applied to the state table value in firmware, which can be one of `mul`, `incr` or `decr`.
    - `func_arg` is the argument to the function.
    - `address`, `buffer`, `column` and `row` together resolve to the destination address that the value will be written to.
      Address resolution is the same as for `npu.write32`:
      - If `buffer` is specified, `address` is a word offset relative to that
        buffer's start address.
      - If `column` and `row` are present, `address` is a local offset within
        that tile's address space.
      - Otherwise, `address` is an absolute address into the AIE array.

  }];
  let extraClassDeclaration = [{
    std::optional<uint32_t> getAbsoluteAddress();
  }];
  let hasVerifier = 1;
}

def AIE_NpuControlPacketOp: AIEX_Op<"control_packet", []> {
  let summary = "AIE control packet";
  let arguments = (
    ins UI32Attr:$address,
        OptionalAttr<I32Attr>:$length,
        I32Attr:$opcode,
        I32Attr:$stream_id,
        OptionalAttr<DenseI32ArrayAttr>:$data
  );
  let results = (outs );
  let assemblyFormat = [{ attr-dict }];
  let description = [{
    The control_packet operation represents a low-level AIE control packet header
    and payload.
  }];
  let extraClassDeclaration = [{
    uint32_t getRowFromAddr();
    uint32_t getColumnFromAddr();
  }];
}

// NPU Bd Write operation
def AIE_NpuWriteBdOp: AIEX_Op<"npu.writebd", []> {
  let summary = "dma operator";
  let arguments = (ins I32Attr:$column, I32Attr:$bd_id, I32Attr:$buffer_length,
      I32Attr:$buffer_offset, I32Attr:$enable_packet, I32Attr:$out_of_order_id,
      I32Attr:$packet_id, I32Attr:$packet_type, I32Attr:$d0_size,
      I32Attr:$d0_stride, I32Attr:$d1_size, I32Attr:$d1_stride,
      I32Attr:$d2_size, I32Attr:$d2_stride, I32Attr:$iteration_current,
      I32Attr:$iteration_size, I32Attr:$iteration_stride, I32Attr:$next_bd,
      I32Attr:$row, I32Attr:$use_next_bd, I32Attr:$valid_bd,
      I32Attr:$lock_rel_val, I32Attr:$lock_rel_id, I32Attr:$lock_acq_enable,
      I32Attr:$lock_acq_val, I32Attr:$lock_acq_id, I32Attr:$d0_zero_before,
      I32Attr:$d1_zero_before, I32Attr:$d2_zero_before, I32Attr:$d0_zero_after,
      I32Attr:$d1_zero_after, I32Attr:$d2_zero_after,
      DefaultValuedOptionalAttr<I32Attr, "0">:$burst_length,
      OptionalAttr<ConfinedAttr<I32Attr, [IntMinValue<0>,
                                          IntMaxValue<15>]>>:$axcache);
  let results = (outs );
  let assemblyFormat = [{ attr-dict }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    /* $axcache, or the target's default when unset. */
    uint32_t getAxcacheOrDefault();
  }];
  let description = [{
    Write a complete buffer descriptor (all fields explicit) to a BD slot.

    ###  Iteration convention: raw register values

    The `iteration_*` attributes here are RAW register values: stored off by one
    (`iteration_size = 3` means four executions) and `iteration_stride` in
    32-bit words. The iteration state on `aie.dma_bd` (`#aie.bd_iteration<...>`)
    holds TRUE / element values; the lowering applies the `-1` bias and the
    element->word scaling there. Do NOT copy an iteration number between the
    two ops. See the "## BD iteration" section of `aie.dma_bd` for more details.
  }];
}

// NPU Load PDI operation
def AIE_NpuLoadPdiOp: AIEX_Op<"npu.load_pdi", []> {
  let summary = "load pdi operator";
  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$device_ref,
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      DefaultValuedOptionalAttr<I32Attr, "0">:$size,
      DefaultValuedOptionalAttr<UI64Attr, "0">:$address,
      OptionalAttr<AIEX_ExpandMode>:$expand_mode);
  let results = (outs );
  let assemblyFormat = [{ attr-dict }];
  let description = [{
    Load a PDI (Programmable Device Image) to configure the NPU.
    The PDI is identified by `id`. `address` and `size` are typically written at
    runtime by the driver or host program.

    If a symbol reference is provided, the compiler driver (aiecc) will match it to a device symbol name and assign the PDI ID field based on it.

    ### Watch Out!

    The firmware optimizes out repeated `load_pdi` operations if they refer to the same PDI.
    To force a reload / device reset, intersperse a `load_pdi` to a different PDI between same-device reloads.
    The easiest workaround is to create an empty `aie.device(npu2) @empty {}` and load that before the main device load.
  }];
  let hasCanonicalizeMethod = 1;
}

def AIE_DMAConfigureTaskOp : AIEX_Op<"dma_configure_task", [HasAncestor<"AIE::RuntimeSequenceOp">, TileElement]>, Results<(outs Index:$result)> {
  let summary = "Concrete Instantiation of a Buffer Descriptor Chain as a Task on a Channel and Direction on a Tile";
  let description = [{
    Encapsulates the DMA configuration of one task, that is the (chain of) buffer descriptors to be executed on a given channel and direction on a tile.

    Such configurations are generated by materializing abstract aie.bd_chains using aiex.start_task, or can be created manually using this op.

    Once configured, a task can be submitted for execution using `aiex.dma_start_configured_task`, after which its execution completion can be awaited using `aiex.dma_await_task`.
  }];

  let arguments = (
    ins Index:$tile,
        DMAChannelDir:$direction,
        I32Attr:$channel,
        DefaultValuedOptionalAttr<BoolAttr, "false">:$issue_token,
        DefaultValuedOptionalAttr<I32Attr, "0">:$repeat_count,
        Optional<I32>:$repeat_count_val,
        OptionalAttr<PacketInfoAttr>:$packet,
        // Put the channel into out-of-order mode (S2MM only).
        UnitAttr:$out_of_order
  );

  let regions = (
    region AnyRegion:$body
  );

  let assemblyFormat = [{
    `(` $tile `,` $direction `,` $channel (`,` $packet^)? `)`
    (`repeat` $repeat_count_val^ `:` type($repeat_count_val))?
    regions attr-dict
  }];

  let extraClassDeclaration = [{
    AIE::TileOp getTileOp();
    // Null when the tile is still an aie.logical_tile, so passes that can run
    // before placement defer instead of aborting inside getTileOp().
    AIE::TileOp tryGetTileOp();
    // The task's head BD, or null when the body has none.
    AIE::DMABDOp getFirstBd();
    // The repeat count as an OpFoldResult: the runtime operand if present, else
    // the compile-time attribute. Mirrors dma_bd's offset/len duality.
    mlir::OpFoldResult getRepeatCountValue() {
      if (mlir::Value v = getRepeatCountVal())
        return v;
      return mlir::IntegerAttr::get(
          mlir::IntegerType::get(getContext(), 32), getRepeatCount());
    }
  }];

  let extraClassDefinition = [{
    AIE::TileOp DMAConfigureTaskOp::getTileOp() { return cast<AIE::TileElement>(this->getOperation()).getTileOp(); }
    AIE::TileOp DMAConfigureTaskOp::tryGetTileOp() { return cast<AIE::TileElement>(this->getOperation()).tryGetTileOp(); }
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;

  let builders = [
    OpBuilder<(ins "::mlir::Type":$result, "mlir::Value":$tile, "xilinx::AIE::DMAChannelDir":$dir, "int":$channel_index,
               "bool":$token, "int":$repeat), [{
      build($_builder, $_state, result, tile,
            xilinx::AIE::DMAChannelDirAttr::get(odsBuilder.getContext(), dir),
            $_builder.getI32IntegerAttr(channel_index), $_builder.getBoolAttr(token),
            $_builder.getI32IntegerAttr(repeat), /*repeat_count_val=*/nullptr,
            /*packet=*/nullptr, /*out_of_order=*/nullptr);
    }]>
  ];
}

def AIE_DMAConfigureTaskForOp : AIEX_Op<"dma_configure_task_for", [HasAncestor<"AIE::RuntimeSequenceOp">]>, Results<(outs Index:$result)> {
  let summary = "As dma_configure_task, but specify tile, direction and channel by reference to a Shim DMA allocation op";

  let arguments = (
    ins SymbolRefAttr:$alloc,
        DefaultValuedOptionalAttr<BoolAttr, "false">:$issue_token,
        DefaultValuedOptionalAttr<I32Attr, "0">:$repeat_count,
        Optional<I32>:$repeat_count_val
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    $alloc (`repeat` $repeat_count_val^ `:` type($repeat_count_val))? regions attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // The repeat count as an OpFoldResult: runtime operand if present, else the
    // compile-time attribute.
    mlir::OpFoldResult getRepeatCountValue() {
      if (mlir::Value v = getRepeatCountVal())
        return v;
      return mlir::IntegerAttr::get(
          mlir::IntegerType::get(getContext(), 32), getRepeatCount());
    }
  }];
}

def AIE_DMAFreeTaskOp : AIEX_Op<"dma_free_task", [HasAncestor<"AIE::RuntimeSequenceOp">]> {
  let summary = "Free all Buffer Descriptor IDs Associated with the Given Task";
  let description = [{
    This operation informs the static buffer descriptor allocator pass in the compiler that the buffer descriptor IDs it has allocated to the BDs inside the referenced task can be reused thereafter.

    Potential future implementations of dynamic buffer descriptor allocators may lower this to a `free` instruction.
  }];
  let arguments = (
    ins Index:$task
  );

  let assemblyFormat = [{
    `(` $task `)` attr-dict
  }];

  let extraClassDeclaration = [{
    DMAConfigureTaskOp getTaskOp();
  }];

  let extraClassDefinition = [{
    DMAConfigureTaskOp DMAFreeTaskOp::getTaskOp() { return dyn_cast_or_null<DMAConfigureTaskOp>(getTask().getDefiningOp()); }
  }];
}

def AIE_DMAStartTaskOp : AIEX_Op<"dma_start_task", [HasAncestor<"AIE::RuntimeSequenceOp">]> {
  let summary = "Submit a Preconfigured Task to the Task Queue";
  let description = [{
    Submits the referenced task for execution on the tile, channel and direction it has been configured to run on.
    Once submitted, if the task is configured to issue a token, you can await completion of the task using `aiex.await_task`.
  }];

  let arguments = (
    ins Index:$task
  );

  let assemblyFormat = [{
    `(` $task `)` attr-dict
  }];

  let extraClassDeclaration = [{
    DMAConfigureTaskOp getTaskOp();
  }];

  let extraClassDefinition = [{
    DMAConfigureTaskOp DMAStartTaskOp::getTaskOp() { return dyn_cast_or_null<DMAConfigureTaskOp>(getTask().getDefiningOp()); }
  }];
}

def AIE_DMAAwaitTaskOp : AIEX_Op<"dma_await_task", [HasAncestor<"AIE::RuntimeSequenceOp">]> {
  let summary = "Await Completion of a Previously Submitted DMA Task";
  let description = [{
    This operation blocks execution of the runtime sequence until a previously
    started DMA task completes, by waiting on its task-completion token (TCT).
    To be able to wait on a task, it must issue a token (`issue_token = true`).

    DMA tasks can be started using `aiex.start_task` using abstract BD chains
    declared using `aie.bd_chain`, or using `aiex.start_configured_task` using a
    manually configured task.

    The task is identified by its `$task` SSA value, which must resolve to the
    configure that set it up. Under the dynamic BD pool path this configure must
    dominate the await; an await after an `scf.if` on a task configured inside a
    branch is rejected (move the await inside the branch that configures it).
  }];

  let arguments = (
    ins Index:$task
  );

  let assemblyFormat = [{
    `(` $task `)` attr-dict
  }];

  let extraClassDeclaration = [{
    DMAConfigureTaskOp getTaskOp();
  }];

  let extraClassDefinition = [{
    DMAConfigureTaskOp DMAAwaitTaskOp::getTaskOp() {
      return dyn_cast_or_null<DMAConfigureTaskOp>(getTask().getDefiningOp());
    }
  }];
}

def AIE_DMABdPoolPopOp : AIEX_Op<"dma_bd_pool_pop", [HasAncestor<"AIE::RuntimeSequenceOp">]>, Results<(outs I32:$bd_id)> {
  let summary = "Pop a free Buffer Descriptor ID from the runtime free-list pool for a tile";
  let description = [{
    Draws a buffer-descriptor ID from the per-tile runtime free-list pool and
    returns it as an SSA value. This is the dynamic counterpart to the static
    BD-ID allocator: inside a runtime-bound `scf.for` the ID is chosen at
    runtime rather than baked at compile time.

    Only meaningful on the dynamic EmitC lowering path (the static binary TXN
    stream has no runtime pool). The `aie-lower-dynamic-bd-pool` pass inserts
    this op in place of the implicit allocation a `dma_configure_task` performs.
    If the pool is empty at runtime, the generated builder yields no stream
    (`std::nullopt`).
  }];

  let arguments = (
    ins I32Attr:$column,
        I32Attr:$row
  );

  let assemblyFormat = [{
    `(` $column `,` $row `)` attr-dict `:` type($bd_id)
  }];
}

def AIE_DMABdPoolPushOp : AIEX_Op<"dma_bd_pool_push", [HasAncestor<"AIE::RuntimeSequenceOp">]> {
  let summary = "Return a Buffer Descriptor ID to the runtime free-list pool for a tile";
  let description = [{
    Returns a buffer-descriptor ID previously obtained from
    `aiex.dma_bd_pool_pop` to the per-tile runtime free-list pool, making it
    available for reuse. This is the dynamic counterpart to the static
    allocator's free/await recycling.

    Only meaningful on the dynamic EmitC lowering path. The
    `aie-lower-dynamic-bd-pool` pass inserts this op in place of a
    `dma_free_task` / the recycling half of a `dma_await_task`.
  }];

  let arguments = (
    ins I32Attr:$column,
        I32Attr:$row,
        I32:$bd_id
  );

  let assemblyFormat = [{
    `(` $column `,` $row `)` `bd_id` $bd_id attr-dict `:` type($bd_id)
  }];
}

def AIE_DMAStartBdChainOp: AIEX_Op<"dma_start_bd_chain", [HasParent<"AIE::RuntimeSequenceOp">, TileElement]>,
    Results<(outs Index:$result)>
  {

  let summary = "Materialize an Abstract BD Chain as a DMA Task on the Given Tile, Channel and Direction and Immediately Start It";
  let description = [{
    This operation will configure a new DMA task on the given tile, channel and direction by concretizing an abstract BD chain, previously defined using `aie.bd_chain`, with the given input arguments.

    Completion of the DMA task, i.e. the data transfer, can be awaited using `aiex.await_task` if the attribute `issue_token` is set to `true`.
  }];

  let arguments = (
    ins FlatSymbolRefAttr:$symbol,
        Variadic<AnyType>:$args,
        Index:$tile,
        DMAChannelDir:$direction,
        I32Attr:$channel,
        DefaultValuedOptionalAttr<BoolAttr, "false">:$issue_token,
        DefaultValuedOptionalAttr<I32Attr, "0">:$repeat_count
  );

  let assemblyFormat = [{
     $symbol `(` $args `)` `:` `(` type($args) `)` ` ` `on` ` ` `(` $tile `,` $direction `,` $channel `)` attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    AIE::TileOp getTileOp();
    AIE::BDChainOp getBDChainOp();
  }];

  let extraClassDefinition = [{
    AIE::TileOp DMAStartBdChainOp::getTileOp() { return cast<AIE::TileElement>(this->getOperation()).getTileOp(); }
  }];

}

def AIE_DMAStartBdChainForOp: AIEX_Op<"dma_start_bd_chain_for", [HasParent<"AIE::RuntimeSequenceOp">]>,
    Results<(outs Index:$result)>
  {
  let summary = "As dma_start_bd_chain, but specify tile, direction and channel by reference to a Shim DMA allocation op";

  let arguments = (
    ins FlatSymbolRefAttr:$symbol,
        Variadic<AnyType>:$args,
        FlatSymbolRefAttr:$alloc,
        DefaultValuedOptionalAttr<BoolAttr, "false">:$issue_token,
        DefaultValuedOptionalAttr<I32Attr, "0">:$repeat_count
  );

  let assemblyFormat = [{
     $symbol `(` $args `)` `:` `(` type($args) `)` ` ` `for` ` ` $alloc attr-dict
  }];
}

def AIEX_SetLockOp : AIEX_Op<"set_lock", [HasAncestor<"AIE::RuntimeSequenceOp">,
                                          SkipAccessibilityCheckTrait]> {
  let summary = "Set the value of a lock";
  let description = [{
    This operation sets the value of `lock` inside of a RuntimeSequenceOp.
    The operation is non blocking and does not offer any synchronization guarantees.
    Should be used in combination with blocking operations.

    Example:
    ```
      %tile22 = aie.tile(2, 2)
      %lock22_0 = aie.lock(%tile22, 0)
      ...
      aiex.set_lock(%lock22_0, 5)
    ```
  }];
  let arguments = (
    ins Index:$lock,
        I32Attr:$value
  );
  let assemblyFormat = [{ `(` $lock `,` $value `)` attr-dict }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    AIE::LockOp getLockOp();
  }];

  let extraClassDefinition = [{
    AIE::LockOp SetLockOp::getLockOp() { return cast<AIE::LockOp>(getLock().getDefiningOp()); }
  }];
}

def AIEX_DmaChannelResetOp: AIEX_Op<"dma_channel_reset", [HasParent<"AIE::RuntimeSequenceOp">, SkipAccessibilityCheckTrait]> {
  let summary = "Reset a DMA channel from a runtime sequence";
  let description = [{
    Resets the DMA channel `channel` in direction `direction` on `tile` by
    pulsing the channel control register reset bit. Intended for re-arming a
    resident objectFIFO across dispatches: it clears the channel's residual
    queue state so it does not carry into a subsequent dispatch. This resets the
    DMA channel state only; it does not by itself establish a clean dispatch
    baseline in general. The operation is non blocking and offers no
    synchronization guarantees.

    The pulse is emitted as two `npu.maskwrite32` writes masked to the reset bit,
    so it preserves the other control fields (out-of-order, compression,
    controller id, FoT mode) instead of clobbering them.

    Only core and mem tiles have a per-channel DMA reset bit; shim NOC/PL tiles
    are rejected by the verifier because their DMA control register has no reset
    field.

    `tile` is an SSA `aie.tile` value, so the target is named the same way as
    `aiex.dma_configure_task`.

    To re-arm one resident objectFIFO, prefer `aiex.dma_channel_reset_for`:
    resetting a channel is only one third of the sequence a resident FIFO needs
    -- the reset also freezes the channel's bound lock counters, and an aie2p
    DMA channel has no enable bit, so once the reset drains its queue only a
    START_QUEUE write restarts it. That op resolves the whole trio (reset,
    `aiex.set_lock` per bound lock, `aiex.npu.push_queue` re-push) from the
    fifo's `aie.objectfifo_rearm_binding`. Note that
    `AIEVerifyRuntimeRearmPass` checks only the lock half, so satisfying it
    after a hand-written reset still leaves a channel that never restarts.

    Resetting every DMA channel on a tile has no dedicated op, because it is a
    loop a caller can already write: emit one `aiex.dma_channel_reset` per
    channel per direction, MM2S 0..N-1 then S2MM 0..N-1, where N is the tile's
    channel count in that direction
    (`AIETargetModel::getNumSourceSwitchboxConnections` /
    `getNumDestSwitchboxConnections` with `WireBundle::DMA`, the same query this
    op's verifier uses to bound `channel`). That is exactly what aie-rt's
    `XAie_DmaChannelResetAll` is -- a bare loop over `XAie_DmaChannelReset` with
    no tile-wide reset-all bit underneath it. Resetting a channel that was never
    configured is safe: aie-rt gates the reset only on the channel index being
    in range for the tile and direction, not on configuration state.

    Example:
    ```
      %tile = aie.tile(2, 3)
      aiex.dma_channel_reset(%tile, S2MM, 0)
    ```
  }];
  let arguments = (
    ins Index:$tile,
        DMAChannelDir:$direction,
        I32Attr:$channel
  );
  let assemblyFormat = [{ `(` $tile `,` $direction `,` $channel `)` attr-dict }];
  let extraClassDeclaration = [{
    AIE::TileOp getTileOp();
  }];
  let extraClassDefinition = [{
    AIE::TileOp DmaChannelResetOp::getTileOp() {
      return cast<AIE::TileOp>(getTile().getDefiningOp());
    }
  }];
  let hasVerifier = 1;
}

def AIEX_CoreResetOp: AIEX_Op<"core_reset", [HasParent<"AIE::RuntimeSequenceOp">, SkipAccessibilityCheckTrait]> {
  let summary = "Reset a compute core from a runtime sequence";
  let description = [{
    Pulses the reset bit of the CORE_CONTROL register on `tile`. The pulse is two
    `npu.maskwrite32` writes masked to the reset bit (assert, then deassert), so
    it preserves the ENABLE field packed in the same CORE_CONTROL word instead of
    clobbering it. This mirrors aie-rt's XAie_CoreReset followed by
    XAie_CoreUnreset. The operation is non blocking and offers no synchronization
    guarantees.

    Resetting clears the core's processor state (program counter and registers);
    it does not touch the tile's data memory, locks, or DMA. The intended use is
    re-arming a resident kernel across dispatches without a full PDI reload:
    discard stale core state while keeping resident data in place. This assumes
    the resident core is still enabled; re-arming a core that has cleared its
    ENABLE bit also needs a re-enable, which this op does not emit.

    Only core tiles have a CORE_CONTROL register; mem and shim tiles are rejected
    by the verifier because they have no core to reset. This matches aie-rt's
    XAie_CoreReset, which errors on non core tiles.

    `tile` is an SSA `aie.tile` value, so the target is named the same way as
    `aiex.dma_channel_reset` and `aiex.dma_configure_task`.

    Example:
    ```
      %tile = aie.tile(2, 3)
      aiex.core_reset(%tile)
    ```
  }];
  let arguments = (
    ins Index:$tile
  );
  let assemblyFormat = [{ `(` $tile `)` attr-dict }];
  let extraClassDeclaration = [{
    AIE::TileOp getTileOp();
  }];
  let extraClassDefinition = [{
    AIE::TileOp CoreResetOp::getTileOp() {
      return cast<AIE::TileOp>(getTile().getDefiningOp());
    }
  }];
  let hasVerifier = 1;
}

def AIEX_DmaChannelResetForOp: AIEX_Op<"dma_channel_reset_for", [HasParent<"AIE::RuntimeSequenceOp">, SkipAccessibilityCheckTrait]> {
  let summary = "Reset and re-arm all resident DMA channels of an objectFIFO";
  let description = [{
    Resets and re-arms every resident core/mem DMA channel of the objectFIFO
    `objfifo`, so a resident (acquire-once, held on-chip) objectFIFO can be
    reused across dispatches without carrying stale control state into the next
    dispatch. It names the fifo instead of a raw channel, so the un-paired
    (deadlocking) form -- resetting a channel without also re-establishing its
    start queue and locks -- is unrepresentable.

    For each non-shim endpoint of the fifo it lowers to the re-arm trio the
    aie2p DMA model requires: `aiex.dma_channel_reset` (drain the channel queue +
    clear the run FSM), a `START_QUEUE` re-push (`aiex.npu.push_queue`, lowered to
    the queue `write32` by `--aie-dma-to-npu`) and
    `aiex.set_lock` (re-arm the producer/consumer semaphore counts to the values
    the fifo was armed with). The re-push is required because an aie2p DMA
    channel has no enable bit, so once the reset drains its queue the only way to
    restart it is a `START_QUEUE` write; the head BD id and repeat count are read
    from the fifo's `aie.objectfifo_rearm_binding` (folded there by
    `--aie-assign-bd-ids`), so the re-push reproduces the channel's load-time
    arming exactly without re-scanning the emitted DMA chain. Shim endpoints are
    skipped: the host re-pushes those with its own BD program.

    This resets the DMA control state only; like `aiex.dma_channel_reset` it does
    not by itself establish a clean dispatch baseline in general (it does not
    re-enable cores or restream BDs). It is non blocking and offers no
    synchronization guarantees. It re-arms a single objectFIFO, so a resident
    pipeline of several fifos needs one `dma_channel_reset_for` per fifo, and any
    resident cores need a separate `aiex.core_reset`.

    `objfifo` is resolved through the objectFIFO lowering: before
    `aie.objectfifo_stateful_transform` it names the `aie.objectfifo`; the
    transform retargets it to the fifo's `aie.objectfifo_rearm_binding`.

    Example:
    ```
      aiex.dma_channel_reset_for(@of)
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$objfifo);
  let assemblyFormat = [{ `(` $objfifo `)` attr-dict }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ScratchpadParameter ops
//===----------------------------------------------------------------------===//

def AIEX_ScratchpadParameterOp: AIEX_Op<"scratchpad_parameter", [Symbol]> {
  let summary = "Declare a scratchpad runtime parameter";
  let description = [{
    Declares a named runtime parameter that uses the scratchpad mechanism.
    Using this declaration and the MLIR-AIE runtime library, the host can write
    these named runtime parameters to DDR, and AIE cores read it using
    `aiex.read_scratchpad_parameter`. The scratchpad memory mechanism
    (CREATE_SCRATCHPAD + UPDATE_FROM_SCRATCHPAD firmware opcodes) communicates
    the values.

    `aiex.scratchpad_parameter` ops are declared at **module scope** (outside any `aie.device`).
    The scratchpad is a single hardware resource shared by all PDIs loaded by a
    runtime sequence, so parameters are global to the whole module and may be
    referenced from any device.

    Parameters can alternatively offset BD addresses when used as the
    `offset_parameter` attribute in `aiex.dma_bd` and `aiex.dma_memcpy_nd`.
    The two kinds of use are exclusive: parameters used as BD address offsets
    cannot also be read from cores.
    If used as an address offset on a BD, the parameter is a multiple of the BD's element size.

    Each parameter occupies one StateTable entry (4 bytes in the scratchpad) in DDR.
    The `--aie-lower-scratchpad-parameters` pass assigns the `state_table_idx` and the `kind` attribute (derived from the parameter's usage:
    `core` if read by a core via `aiex.read_scratchpad_parameter`, `addr` if used as a DMA offset via the `offset_parameter` attribute on a DMA op).
    Indices are unique across the entire module.

    The `type` attribute specifies the data type of the parameter (`bf16` or an integer type up to `i32`).
    For `kind == addr`, the `type` must be `i32`.
    For `kind == core`, the encoding uses a 30-bit value range because the firmware masks
    the lower 2 bits of the scratchpad register (see `aiex.read_scratchpad_parameter`
    for details on how values are shifted to work around this).

    You can manually specify the parameter synchronization point inside the runtime sequence using the `aiex.sync_scratchpad_parameters_from_host` op.

    On the host, you can write parameters using the runtime library's `parameter_scratchpad.h` header library.

    See `test/npu-xrt/scratchpad_params` and `test/npu-xrt/scratchpad_params_python` for a full usage example.

    Example (at module scope):
    ```mlir
    aiex.scratchpad_parameter @foo : i32
    aiex.scratchpad_parameter @bar : bf16
    aie.device(npu2) { ... }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    // assigned by `--aie-lower-scratchpad-parameters` pass:
    OptionalAttr<UI8Attr>:$state_table_idx,
    OptionalAttr<ScratchpadParameterKind>:$kind
  );
  let results = (outs);
  let assemblyFormat = [{ $sym_name `:` $type attr-dict }];
}

def AIEX_ReadScratchpadParameterOp: AIEX_Op<"read_scratchpad_parameter", []> {
  let summary = "Read a scratchpad runtime parameter value on an AIE core";
  let description = [{
    Reads a runtime parameter previously declared with `aiex.scratchpad_parameter` in an `aie.core`.

    The `--aie-lower-scratchpad-parameters` creates an `aie.buffer` on the core for each unique parameter read, and then replaces each instance of this op with:
    1. A `memref.load` from that buffer
    2. Arithmetic to decode the value (right-shift by 2 to undo firmware masking)

    Example:
    ```mlir
    %val = aiex.read_scratchpad_parameter @foo : i32
    ```
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$parameter,
    // assigned by `--aie-lower-scratchpad-parameters` pass:
    OptionalAttr<FlatSymbolRefAttr>:$buffer
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{ $parameter `:` type($result) attr-dict }];
  let hasVerifier = 1;
}

def AIEX_SyncScratchpadParametersFromHostOp: AIEX_Op<"sync_scratchpad_parameters_from_host",
    [HasParent<"AIE::RuntimeSequenceOp">, SkipAccessibilityCheckTrait]> {
  let summary = "Marker for where host-to-core scratchpad parameter sync should be emitted";
  let description = [{
    Marker op indicating the point in a runtime sequence where the
    host-to-core scratchpad parameter synchronisation should be materialised.
    Carries no operands or attributes.  `--aie-lower-scratchpad-parameters`
    replaces each occurrence (including a compiler-inserted preamble,
    controlled by the parent `aie.runtime_sequence`'s
    `emit_parameter_sync_preamble` attribute) with the lowered sequence:

    1. `aiex.npu.create_scratchpad`
    2. For each core-kind parameter of the enclosing `aie.device`:
       a. `aiex.npu.write32` to zero the destination buffer
       b. `aiex.npu.update_from_scratchpad` to copy the value
    3. `aiex.set_lock(lock, 1)` for each parameter-sync lock in the device
       (releasing the cores that read parameters)

    Placement: users who manually emit `aiex.npu.load_pdi` should manually
    place this op *after* the load_pdi, since PDI loading resets core state
    (including lock values).

    Example:
    ```mlir
    aiex.sync_scratchpad_parameters_from_host
    ```
  }];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = [{ attr-dict }];
}

// Include CERT operations
include "aie/Dialect/AIEX/IR/CERTOps.td"

#endif // AIEX_OPS
