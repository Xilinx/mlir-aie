Splits the monolithic `--objectFifo-stateful-transform` into multiple passes, capturing intermediate state in the IR with a new set of operations.

The old stateful transform did a lot of useful things but was "all or nothing": if the whole stack didn't meet a user's needs, they had to drop all the way down to implementing everything by hand. After this PR, users can enter the stack at any point with custom IR. All passes are idempotent and respect hard-coded and preset attributes and ops.


# New IR Operations

- Pool `aie.objectfifo.pool`
  > A set of buffers and the locks that guard how to access them.

- Endpoint `aie.objectfifo.dma_endpoint`, `aie.objectfifo.core_endpoint` 
  > Something that fills or drains a pool (a user of the buffers in a pool).

- Flow `aie.objectfifo.flow` 
  > A circuit- or packet-flow connection between two DMA endpoints.

### Examples

Below is the exact syntax as implemented, but I'm leaving off the attribute dictionary where it is not relevant (indicated with `...`).

#### Typical ObjectFifo through DMAs

```mlir
aie.objectfifo.pool @producer_pool(%t12) ...
aie.objectfifo.core_endpoint @prod_core(%t12) fills  @producer_pool
aie.objectfifo.dma_endpoint @prod_dma(%t12) drains @producer_pool

aie.objectfifo.pool @consumer_pool(%t33) ...
aie.objectfifo.dma_endpoint @cons_dma(%t33) fills  @consumer_pool
aie.objectfifo.core_endpoint @cons_core(%t33) drains @consumer_pool

aie.objectfifo.flow from @prod_dma to [@cons_dma]
```

#### MemTile passthrough (the existing `objectfifo.link`)

```mlir
aie.objectfifo.pool @memtile_pool(%t21) ...
aie.objectfifo.dma_endpoint @memtile_cons(%t21) fills @memtile_pool
aie.objectfifo.dma_endpoint @memtile_prod(%t21) drains @memtile_pool
```

#### `N`-way join on a MemTile

```mlir
aie.objectfifo.pool @out_pool(%mem_tile_2_1) 
    { depth = 2 : i32,
      segments = [#aie.objectfifo_segment<offset = 0,  size = 16>,
                  #aie.objectfifo_segment<offset = 16, size = 32>]} : memref<48xi32>

aie.objectfifo.dma_endpoint @fill_0(%t21) fills @out_pool {segments = array<i32: 0>}
aie.objectfifo.dma_endpoint @fill_1(%t21) fills @out_pool {segments = array<i32: 1>}
aie.objectfifo.dma_endpoint @drain_all(%t21) drains @out_pool
```

Note we have multiple DMAs filling different _segments_ (slices) of the same pool. Each segment will get its own locks to order access to them. One DMA drains the entire buffer (all segments), effectively joining the data.

Distribute is the same picture with the arrows reversed: one endpoint fills the whole object and `N` endpoints each drain one segment.

### Example: memory-only connection

No DMAs -- an ObjectFifo used within the same core, or across neighboring cores with shared memory:

```mlir
aie.objectfifo.pool @smem(%t12) {depth = 4 : i32,
    segments = [#aie.objectfifo_segment<offset = 0, size = 16>]} : memref<16xi32>
aie.objectfifo.core_endpoint @prod(%t12) fills  @smem
aie.objectfifo.core_endpoint @cons(%t12) drains @smem
```

Note that if our single `pool` has multiple segments, we get join/distribute "for free" on cores. This feature was previously unsupported.

## New Passes

1. **`--aie-objectfifo-split`**: splits `aie.objectfifo` and `aie.objectfifo.link` into pools, endpoints and flows; decides which tiles hold objects and which ends need DMAs

   ```mlir
   // before
   aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
   %elem = aie.objectfifo.acquire @of1(Produce) : memref<16xi32>

   // after
   aie.objectfifo.pool @of1_pool(%tile_1_2) {depth = 2 : i32, fifoName = "of1",
       segments = [#aie.objectfifo_segment<offset = 0, size = 16>]} : memref<16xi32>
   aie.objectfifo.core_endpoint @of1_prod(%tile_1_2) fills @of1_pool
   aie.objectfifo.dma_endpoint  @of1_prod_dma(%tile_1_2) drains @of1_pool {fifoName = "of1"}
   aie.objectfifo.pool @of1_cons_pool(%tile_3_3) ...
   aie.objectfifo.core_endpoint @of1_cons(%tile_3_3) drains @of1_cons_pool
   aie.objectfifo.dma_endpoint  @of1_cons_dma(%tile_3_3) fills @of1_cons_pool {fifoName = "of1"}
   aie.objectfifo.flow from @of1_prod_dma to [@of1_cons_dma]

   %elem = aie.objectfifo.acquire @of1_prod : memref<16xi32>
   ```

2. **`--aie-objectfifo-verify`**: checks completeness rules -- every segment has one filler and one drainer, segments tile the object exactly, every flow reaches a destination. Provided as a debugging tool, and kept out of the default pipeline so that incomplete definitions stay legal and users can implement portions of an ObjectFifo at a lower level.

3. **`--aie-objectfifo-allocate`**: gives pools their buffers and locks, and endpoints their DMA channels, `aie.flow`/`aie.packet_flow` and shim allocations.

   ```mlir
   %of1_buff_0 = aie.buffer(%tile_1_2) {sym_name = "of1_buff_0"} : memref<16xi32>
   %of1_buff_1 = aie.buffer(%tile_1_2) {sym_name = "of1_buff_1"} : memref<16xi32>
   %of1_prod_lock_0 = aie.lock(%tile_1_2) {init = 2 : i32, sym_name = "of1_prod_lock_0"}
   %of1_cons_lock_0 = aie.lock(%tile_1_2) {init = 0 : i32, sym_name = "of1_cons_lock_0"}

   aie.objectfifo.pool @of1_pool(%tile_1_2) {buffers = [@of1_buff_0, @of1_buff_1], depth = 2 : i32,
       segments = [#aie.objectfifo_segment<offset = 0, size = 16,
                    produceLock = @of1_prod_lock_0, consumeLock = @of1_cons_lock_0>]} : memref<16xi32>
   aie.objectfifo.dma_endpoint @of1_prod_dma(%tile_1_2) drains @of1_pool {
       channel = #aie.objectfifo_channel<MM2S : 0>}
   aie.flow(%tile_1_2, DMA : 0, %tile_3_3, DMA : 0)
   ```

4. **`--aie-objectfifo-lower-dmas`**: turns `dma_endpoint`s into DMA programs (BD chains)

   ```mlir
   %mem_1_2 = aie.mem(%tile_1_2) {
     %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
   ^bb1:
     aie.use_lock(%of1_cons_lock_0, AcquireGreaterEqual, %c1_i32)
     aie.dma_bd(%of1_buff_0 : memref<16xi32> offset = 0 len = 16)
     aie.use_lock(%of1_prod_lock_0, Release, %c1_i32_0)
     aie.next_bd ^bb2
   ^bb2:  // same, with %of1_buff_1
   ```

5. **`--aie-objectfifo-lower-cores`**: turns `acquire`/`release` ops on cores into `use_lock` plus a rotating buffer selection.

   ```mlir
   %0 = arith.subi %c1_i32, %c0_i32 : i32      // acquire(N) is absolute:
   %1 = arith.maxsi %0, %c0_i32_0 : i32        // delta = max(N - held, 0)
   aie.use_lock(%of1_prod_lock_0, AcquireGreaterEqual, %1)
   %4 = scf.index_switch %3 -> memref<16xi32>
   case 0 { scf.yield %of1_buff_0 : memref<16xi32> }
   case 1 { scf.yield %of1_buff_1 : memref<16xi32> }
   ```

   The `index_switch` folds to a concrete buffer once the loops unroll.

6. **`--aie-objectfifo-erase-pools`**: drops the pool metadata, which is no longer needed once everything is lowered. Skippable, because the record of which buffers and locks belong to which ObjectFifo is often worth keeping.

# Other notes

- The separation of concerns and modularity introduced in this PR make the code easier to understand and maintain. Each individual pass is around 500 lines or less, and despite adding new operations, the lowering code is an overall lines-of-code reduction while preserving the old functionality.
- This PR also drops the `subview` op. I don't know why it existed -- `acquire` now simply returns the objects you asked for.
- After splitting, core `acquire`s no longer need a Producer/Consumer port, which previously allowed expressing illegal combinations (e.g. an ObjectFifo with several producers, DMA and core, and no consumers). A core's `acquire` must name a `core_endpoint` on its own tile, and may not carry a port -- the endpoint's `fills`/`drains` role already says which end, so the two cannot disagree.
- Because each pass erases what it consumes, running the pipeline over its own output is a no-op. That is tested directly, and it is what keeps "the IR captures the state" honest: anything a pass forgot to record would show up as a leftover op rather than as a silently lost side table.