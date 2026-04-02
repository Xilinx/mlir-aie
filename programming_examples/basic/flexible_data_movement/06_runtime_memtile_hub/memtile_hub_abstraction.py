# memtile_hub_abstraction.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
#
# PROPOSED ABSTRACTION: MemTileHub
#
# This file sketches a higher-level Python API that wraps the low-level
# buffer/lock/flow/dma_configure_task pattern demonstrated in Prototype 6
# into an ergonomic interface for runtime-programmable MemTile data distribution.
#
# STATUS: Design sketch only — not yet implemented in MLIR-AIE.
# The underlying primitives (dma_configure_task on MemTile, locks, flows)
# are proven working on hardware in Prototype 6.
#
# POOL ALLOCATION (Prototype 7):
# Instead of individual alloc() calls that each create a separate aie.buffer(),
# pool mode allocates ONE large buffer and returns offset-based regions:
#
#   hub = MemTileHub(tile(0, 1), shim=tile(0, 0), pool_size=8192)
#   region_a = hub.pool_alloc(size=256)   # returns PoolRegion(offset=0, size=256)
#   region_b = hub.pool_alloc(size=256)   # returns PoolRegion(offset=256, size=256)
#   hub.pool_free(region_a)               # marks region available for reuse
#   region_c = hub.pool_alloc(size=512)   # reuses freed space at offset 0
#
# At compile time, this emits ONE aie.buffer of pool_size. At runtime,
# dma_bd(pool, offset=region.offset, len=region.size) selects sub-regions.
# Proven on hardware in Prototype 7 (07_memtile_pool_alloc).
#
# ==========================================================================
#
# TODAY (Prototype 6 — ~100 lines of boilerplate):
#
#   data_buf_a = buffer(MemTile, chunk_ty, name="data_buf_a")
#   data_a_prod = lock(MemTile, lock_id=0, init=1, sym_name="data_a_prod")
#   data_a_cons = lock(MemTile, lock_id=1, init=0, sym_name="data_a_cons")
#   flow(ShimTile, WireBundle.DMA, 0, MemTile, WireBundle.DMA, 0)
#   flow(MemTile, WireBundle.DMA, 0, CoreA, WireBundle.DMA, 0)
#   ... (6 flows, 8 locks, 4 buffers, 8 dma_configure_task blocks)
#
# PROPOSED (MemTileHub — ~15 lines):
#
#   hub = MemTileHub(tile(0, 1), shim=tile(0, 0))
#   buf_a = hub.alloc("data_a", shape=(256,), dtype=np.uint8)
#   buf_b = hub.alloc("data_b", shape=(256,), dtype=np.uint8)
#   hub.connect(tile(0, 2))  # pre-wire route to Core A
#   hub.connect(tile(0, 3))  # pre-wire route to Core B
#   ...
#   rt = Runtime()
#   with rt.sequence(in_ty, out_ty) as (inp, out):
#       hub.load(buf_a, inp, offset=0, size=256)    # DDR → MemTile buf_a
#       hub.load(buf_b, inp, offset=256, size=256)  # DDR → MemTile buf_b
#       hub.send(buf_a, target=tile(0, 2))           # MemTile → Core A
#       hub.send(buf_b, target=tile(0, 3))           # MemTile → Core B
#       hub.recv(buf_a, source=tile(0, 2))           # Core A → MemTile
#       hub.recv(buf_b, source=tile(0, 3))           # Core B → MemTile
#       hub.drain(buf_a, out, offset=0)              # MemTile → DDR
#       hub.drain(buf_b, out, offset=256)            # MemTile → DDR
#
# ==========================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Core Abstraction Classes
# ---------------------------------------------------------------------------

@dataclass
class MemTileBuffer:
    """A named buffer region allocated in MemTile L2 memory."""
    name: str
    shape: tuple
    dtype: np.dtype
    # Internal: managed by MemTileHub
    _aie_buffer: object = None     # aie.buffer() handle
    _prod_lock: object = None      # producer lock
    _cons_lock: object = None      # consumer lock


@dataclass
class PoolRegion:
    """A sub-region within a pool buffer, identified by offset and size.

    Unlike MemTileBuffer (which maps 1:1 to an aie.buffer()), a PoolRegion
    is a slice of a single large pool buffer. At the MLIR level, all
    PoolRegions share the same aie.buffer and use dma_bd(pool, offset=N)
    to select their sub-region.

    Proven on hardware in Prototype 7 (07_memtile_pool_alloc).
    """
    offset: int        # Offset in elements within the pool buffer
    size: int          # Size in elements
    name: str = ""     # Optional label for debugging
    _freed: bool = False
    # Locks are managed per-region for synchronization
    _prod_lock: object = None
    _cons_lock: object = None


@dataclass
class CoreConnection:
    """A pre-wired route between MemTile and a compute core."""
    target_tile: object            # aie.tile() handle (col, row)
    send_channel: int = -1         # MemTile MM2S channel (auto-assigned)
    recv_channel: int = -1         # MemTile S2MM channel (auto-assigned)
    # Internal
    _send_flow: object = None
    _recv_flow: object = None


class MemTileHub:
    """
    A programmable data distribution hub backed by a MemTile.

    Manages:
    - Buffer allocation in MemTile L2 (512KB)
    - Lock allocation for DMA synchronization
    - Flow (route) setup to connected cores
    - Runtime DMA task generation for data movement

    Usage:
        # 1. Create hub (compile time)
        hub = MemTileHub(tile(0, 1), shim=tile(0, 0))

        # 2. Allocate buffers (compile time)
        weights = hub.alloc("weights", shape=(2048,), dtype=np.int8)
        acts = hub.alloc("activations", shape=(1024,), dtype=np.uint8)

        # 3. Connect cores (compile time — pre-wires flows)
        hub.connect(tile(0, 2))  # Core A
        hub.connect(tile(0, 3))  # Core B

        # 4. Program data movement (runtime — generates dma_configure_task)
        with runtime_sequence(...) as (inp, out):
            hub.load(weights, inp, offset=0, size=2048)
            hub.send(weights, target=tile(0, 2))           # unicast
            hub.broadcast(acts, targets=[tile(0,2), tile(0,3)])  # broadcast
            hub.recv(result_a, source=tile(0, 2))
            hub.drain(result_a, out, offset=0)
    """

    # Only even MemTile DMA channels (0, 2, 4) to work around BD ID bug
    EVEN_CHANNELS = [0, 2, 4]

    def __init__(self, memtile, shim, capacity_bytes: int = 512 * 1024,
                 pool_size: Optional[int] = None):
        self.memtile = memtile
        self.shim = shim
        self.capacity = capacity_bytes
        self.used_bytes = 0

        self.buffers: list[MemTileBuffer] = []
        self.connections: list[CoreConnection] = []

        self._next_lock_id = 0
        self._next_send_ch_idx = 0  # index into EVEN_CHANNELS
        self._next_recv_ch_idx = 1  # start recv from channel 2
        self._shim_send_ch = 0      # shimDMA MM2S channel for loading
        self._shim_recv_ch = 0      # shimDMA S2MM channel for draining

        # Pool allocator state
        self._pool_size = pool_size
        self._pool_buffer = None     # aie.buffer() handle (created on first pool_alloc)
        self._pool_regions: list[PoolRegion] = []
        self._pool_watermark = 0     # next free offset (bump allocator)

    # ------ Compile-time API ------

    def alloc(self, name: str, shape: tuple, dtype=np.uint8) -> MemTileBuffer:
        """Allocate a named buffer in MemTile L2 memory."""
        buf = MemTileBuffer(name=name, shape=shape, dtype=np.dtype(dtype))
        size = int(np.prod(shape)) * buf.dtype.itemsize
        if self.used_bytes + size > self.capacity:
            raise MemoryError(
                f"MemTile capacity exceeded: {self.used_bytes + size} > {self.capacity}"
            )
        self.used_bytes += size

        # --- Would emit at compile time: ---
        # buf._aie_buffer = buffer(self.memtile, ndarray_type, name=name)
        # buf._prod_lock = lock(self.memtile, lock_id=N, init=1)
        # buf._cons_lock = lock(self.memtile, lock_id=N+1, init=0)

        self.buffers.append(buf)
        return buf

    def pool_alloc(self, size: int, name: str = "") -> PoolRegion:
        """
        Allocate a sub-region from the pool buffer.

        Returns a PoolRegion with an offset into the single pool buffer.
        At the MLIR level, this offset is used with dma_bd(pool, offset=N, len=M).

        First call lazily creates the pool buffer:
            pool = buffer(self.memtile, pool_ty, name="pool")

        Allocation strategy: simple bump allocator. For reuse after pool_free(),
        a first-fit search checks freed regions before bumping the watermark.

        Args:
            size: Size in elements (bytes for uint8)
            name: Optional label for debugging

        Returns:
            PoolRegion with offset and size set

        Example:
            hub = MemTileHub(tile(0,1), shim=tile(0,0), pool_size=8192)
            data_a = hub.pool_alloc(256, name="data_a")   # offset=0
            data_b = hub.pool_alloc(256, name="data_b")   # offset=256
            result_a = hub.pool_alloc(256, name="result_a") # offset=512
        """
        if self._pool_size is None:
            raise ValueError(
                "Pool mode not enabled. Pass pool_size= to MemTileHub constructor."
            )

        # --- Would lazily create pool buffer on first call: ---
        # if self._pool_buffer is None:
        #     pool_ty = np.ndarray[(self._pool_size,), np.dtype[np.uint8]]
        #     self._pool_buffer = buffer(self.memtile, pool_ty, name="pool")

        # First-fit search in freed regions
        for region in self._pool_regions:
            if region._freed and region.size >= size:
                # Reuse freed region (exact or split)
                if region.size == size:
                    region._freed = False
                    region.name = name
                    return region
                # Split: reuse first 'size' elements, leave remainder freed
                new_region = PoolRegion(
                    offset=region.offset, size=size, name=name
                )
                region.offset += size
                region.size -= size
                self._pool_regions.append(new_region)
                return new_region

        # Bump allocation
        if self._pool_watermark + size > self._pool_size:
            raise MemoryError(
                f"Pool exhausted: need {size} at offset {self._pool_watermark}, "
                f"pool_size={self._pool_size}"
            )

        region = PoolRegion(
            offset=self._pool_watermark, size=size, name=name
        )
        self._pool_watermark += size

        # --- Would emit lock pair for this region: ---
        # region._prod_lock = lock(self.memtile, lock_id=N, init=1)
        # region._cons_lock = lock(self.memtile, lock_id=N+1, init=0)

        self._pool_regions.append(region)
        return region

    def pool_free(self, region: PoolRegion):
        """
        Mark a pool region as available for reuse.

        Does NOT zero the memory or release locks — just marks the offset
        range as available for future pool_alloc() calls.

        At the hardware level, this is a host-side bookkeeping operation.
        The MemTile memory is not affected until a new DMA BD is configured
        to write to the freed region.

        Args:
            region: A PoolRegion previously returned by pool_alloc()
        """
        if region._freed:
            raise ValueError(f"Double-free of pool region at offset {region.offset}")
        region._freed = True

    def pool_reset(self):
        """
        Reset the pool allocator, freeing all regions.

        Use between runtime_sequence invocations to start with a fresh pool layout.
        """
        self._pool_regions.clear()
        self._pool_watermark = 0

    def connect(self, core_tile) -> CoreConnection:
        """
        Pre-wire a bidirectional route between MemTile and a compute core.

        Allocates:
        - 1 MemTile MM2S channel + flow for sending data to core
        - 1 MemTile S2MM channel + flow for receiving data from core
        - Core DMA programs (static, loop forever)
        """
        send_ch = self.EVEN_CHANNELS[self._next_send_ch_idx]
        recv_ch = self.EVEN_CHANNELS[self._next_recv_ch_idx]
        self._next_send_ch_idx += 1
        self._next_recv_ch_idx += 1

        conn = CoreConnection(
            target_tile=core_tile,
            send_channel=send_ch,
            recv_channel=recv_ch,
        )

        # --- Would emit at compile time: ---
        # flow(self.memtile, WireBundle.DMA, send_ch, core_tile, WireBundle.DMA, 0)
        # flow(core_tile, WireBundle.DMA, 0, self.memtile, WireBundle.DMA, recv_ch)
        # Also emits @mem(core_tile) with static DMA loop programs

        self.connections.append(conn)
        return conn

    # ------ Runtime API (inside runtime_sequence) ------

    def load(self, buf: MemTileBuffer, ddr_tensor, offset: int = 0,
             size: Optional[int] = None):
        """
        Load data from DDR into a MemTile buffer.

        Generates:
        - dma_configure_task(ShimTile, MM2S, ch) with shim_dma_bd
        - dma_configure_task(MemTile, S2MM, 0) with locks on buf
        - dma_start_task for both
        """
        # --- Would emit in runtime_sequence: ---
        # t_shim = dma_configure_task(self.shim, MM2S, self._shim_send_ch)
        #   shim_dma_bd(ddr_tensor, offset=offset, sizes=[1,1,1,size])
        # t_recv = dma_configure_task(self.memtile, S2MM, 0)
        #   use_lock(buf._prod_lock, Acquire, 1)
        #   dma_bd(buf._aie_buffer)
        #   use_lock(buf._cons_lock, Release, 1)
        # dma_start_task(t_shim, t_recv)
        pass

    def send(self, buf: MemTileBuffer, target):
        """
        Send data from a MemTile buffer to a specific core (unicast).

        Generates:
        - dma_configure_task(MemTile, MM2S, conn.send_channel) with locks
        - dma_start_task
        """
        # conn = self._find_connection(target)
        # t = dma_configure_task(self.memtile, MM2S, conn.send_channel)
        #   use_lock(buf._cons_lock, Acquire, 1)
        #   dma_bd(buf._aie_buffer)
        #   use_lock(buf._prod_lock, Release, 1)
        # dma_start_task(t)
        pass

    def broadcast(self, buf: MemTileBuffer, targets: list):
        """
        Send same data from a MemTile buffer to multiple cores.

        Option A: Sequential sends (reuse same buffer, different MM2S channels)
        Option B: Use lock with init = len(targets) to allow multiple reads
        """
        # For each target:
        #   conn = self._find_connection(target)
        #   t = dma_configure_task(self.memtile, MM2S, conn.send_channel)
        #     dma_bd(buf._aie_buffer)  # same buffer, different channel
        #   dma_start_task(t)
        pass

    def recv(self, buf: MemTileBuffer, source):
        """
        Receive data from a core into a MemTile buffer.

        Generates:
        - dma_configure_task(MemTile, S2MM, conn.recv_channel) with locks
        """
        # conn = self._find_connection(source)
        # t = dma_configure_task(self.memtile, S2MM, conn.recv_channel)
        #   use_lock(buf._prod_lock, Acquire, 1)
        #   dma_bd(buf._aie_buffer)
        #   use_lock(buf._cons_lock, Release, 1)
        # dma_start_task(t)
        pass

    def drain(self, buf: MemTileBuffer, ddr_tensor, offset: int = 0,
              size: Optional[int] = None, wait: bool = False):
        """
        Drain data from a MemTile buffer back to DDR.

        Generates:
        - dma_configure_task(MemTile, MM2S, drain_ch) with locks
        - dma_configure_task(ShimTile, S2MM, ch) with shim_dma_bd
        - optionally dma_await_task for blocking
        """
        # t_drain = dma_configure_task(self.memtile, MM2S, drain_channel)
        #   use_lock(buf._cons_lock, Acquire, 1)
        #   dma_bd(buf._aie_buffer)
        #   use_lock(buf._prod_lock, Release, 1)
        # t_shim = dma_configure_task(self.shim, S2MM, ch, issue_token=wait)
        #   shim_dma_bd(ddr_tensor, offset=offset, sizes=[1,1,1,size])
        # dma_start_task(t_drain, t_shim)
        # if wait: dma_await_task(t_shim)
        pass

    def _find_connection(self, core_tile) -> CoreConnection:
        for conn in self.connections:
            if conn.target_tile == core_tile:
                return conn
        raise ValueError(f"No connection to tile {core_tile}")


# ---------------------------------------------------------------------------
# Example: What Prototype 6 would look like with MemTileHub
# ---------------------------------------------------------------------------

def example_with_abstraction():
    """
    This is what the Prototype 6 design WOULD look like using MemTileHub.
    Compare with aie2.py which is ~200 lines of boilerplate.
    """

    # from aie.iron import MemTileHub, Kernel, Worker, Runtime, Program
    # from aie.iron.device import NPU2

    # hub = MemTileHub(tile(0, 1), shim=tile(0, 0))
    #
    # # Allocate buffers in MemTile L2
    # buf_a = hub.alloc("data_a", shape=(256,), dtype=np.uint8)
    # buf_b = hub.alloc("data_b", shape=(256,), dtype=np.uint8)
    # res_a = hub.alloc("result_a", shape=(256,), dtype=np.uint8)
    # res_b = hub.alloc("result_b", shape=(256,), dtype=np.uint8)
    #
    # # Pre-wire routes to cores by coordinate
    # hub.connect(tile(0, 2))  # Core A
    # hub.connect(tile(0, 3))  # Core B
    #
    # # Define kernel
    # passthrough = Kernel("passThroughLine", "passThrough.cc.o",
    #                      [chunk_ty, chunk_ty, np.int32])
    #
    # # Workers on each core (static programs)
    # worker_a = Worker(core_fn, [hub.input(tile(0,2)), hub.output(tile(0,2)),
    #                             passthrough])
    # worker_b = Worker(core_fn, [hub.input(tile(0,3)), hub.output(tile(0,3)),
    #                             passthrough])
    #
    # # Runtime: program the data movement
    # rt = Runtime()
    # with rt.sequence(full_ty, full_ty) as (inp, out):
    #     # Load from DDR into MemTile buffers
    #     hub.load(buf_a, inp, offset=0, size=256)
    #     hub.load(buf_b, inp, offset=256, size=256)
    #
    #     # Send to cores by coordinate (runtime decision)
    #     hub.send(buf_a, target=tile(0, 2))
    #     hub.send(buf_b, target=tile(0, 3))
    #
    #     # Receive results
    #     hub.recv(res_a, source=tile(0, 2))
    #     hub.recv(res_b, source=tile(0, 3))
    #
    #     # Drain to DDR
    #     hub.drain(res_a, out, offset=0)
    #     hub.drain(res_b, out, offset=256, wait=True)
    #
    # return Program(NPU2(), rt).resolve_program(SequentialPlacer())
    pass


def example_with_pool_allocator():
    """
    This is what Prototype 7 WOULD look like using MemTileHub pool mode.
    Instead of 4 separate alloc() calls, we use 1 pool with offset-based regions.

    Key difference: ONE aie.buffer() at MLIR level, sub-regions via dma_bd offset.
    Proven on hardware in Prototype 7 (07_memtile_pool_alloc/aie2.py).
    """

    # from aie.iron import MemTileHub, Kernel, Worker, Runtime, Program
    # from aie.iron.device import NPU2

    # # Pool mode: ONE large buffer, sub-regions via offset
    # hub = MemTileHub(tile(0, 1), shim=tile(0, 0), pool_size=2048)
    #
    # # Allocate sub-regions from the pool (returns PoolRegion, not MemTileBuffer)
    # data_a = hub.pool_alloc(256, name="data_a")     # offset=0
    # data_b = hub.pool_alloc(256, name="data_b")     # offset=256
    # result_a = hub.pool_alloc(256, name="result_a") # offset=512
    # result_b = hub.pool_alloc(256, name="result_b") # offset=768
    # # pool[1024:2048] remains free for future use
    #
    # # Connect cores (same as non-pool mode)
    # hub.connect(tile(0, 2))
    # hub.connect(tile(0, 3))
    #
    # # Runtime: data movement using pool regions
    # # Under the hood, each operation emits dma_bd(pool, offset=region.offset)
    # rt = Runtime()
    # with rt.sequence(full_ty, full_ty) as (inp, out):
    #     hub.pool_load(data_a, inp, ddr_offset=0)    # DDR[0:256] → pool[0:256]
    #     hub.pool_load(data_b, inp, ddr_offset=256)  # DDR[256:512] → pool[256:512]
    #     hub.pool_send(data_a, target=tile(0, 2))    # pool[0:256] → Core A
    #     hub.pool_send(data_b, target=tile(0, 3))    # pool[256:512] → Core B
    #     hub.pool_recv(result_a, source=tile(0, 2))  # Core A → pool[512:768]
    #     hub.pool_recv(result_b, source=tile(0, 3))  # Core B → pool[768:1024]
    #     hub.pool_drain(result_a, out, ddr_offset=0)
    #     hub.pool_drain(result_b, out, ddr_offset=256, wait=True)
    #
    # # Between invocations, can reallocate pool regions:
    # hub.pool_free(data_a)
    # hub.pool_free(data_b)
    # big_region = hub.pool_alloc(512, name="big_input")  # reuses freed space
    # # Or reset entirely:
    # hub.pool_reset()
    #
    # return Program(NPU2(), rt).resolve_program(SequentialPlacer())
    pass


# ---------------------------------------------------------------------------
# Design Notes for Implementation
# ---------------------------------------------------------------------------
#
# 1. CHANNEL ALLOCATION STRATEGY
#    - Use only even MemTile DMA channels (0, 2, 4) until the BD ID
#      allocator bug in AIEAssignRuntimeSequenceBDIDs.cpp is fixed
#    - This limits us to 3 send + 3 recv channels = 3 connected cores
#    - After fix: 6 send + 6 recv = 6 connected cores per MemTile
#
# 2. LOCK MANAGEMENT
#    - Each buffer gets a prod/cons lock pair (semaphore, init prod=1/cons=0)
#    - load() acquires prod, releases cons
#    - send()/broadcast() acquires cons, releases prod
#    - recv() acquires prod, releases cons
#    - drain() acquires cons, releases prod
#    - For broadcast: init prod lock = N (number of readers) so multiple
#      MM2S channels can read the same buffer
#
# 3. FLOW PRE-WIRING
#    - connect() creates bidirectional flows at compile time
#    - The switch routing is static, but which DMA channels are activated
#      (and what data they move) is decided at runtime
#    - This is the "pre-wire everything, activate selectively" pattern
#
# 4. INTEGRATION WITH IRON
#    - MemTileHub.connect() should auto-generate the @mem() DMA programs
#      for connected cores (static S2MM/MM2S loops with locks)
#    - Workers should accept hub.input()/hub.output() as ObjectFifo-like
#      endpoints, backed by the hub's core-side buffers
#    - The Runtime context should track hub operations and lower them
#      to dma_configure_task blocks in the runtime_sequence
#
# 5. COMPILER BUG TO FIX
#    - File: lib/Dialect/AIEX/Transforms/AIEAssignRuntimeSequenceBDIDs.cpp
#    - Line 76: nextBdId(/*channelIndex=*/0) should use actual channel
#    - Fix: auto task_op = bd_op->getParentOfType<DMAConfigureTaskOp>();
#           int ch = task_op ? task_op.getChannel() : 0;
#           nextBdId(ch);
#    - This would enable odd channels and double the available connections
#
# 6. SCALABILITY
#    - Single MemTile: 6 MM2S + 6 S2MM = up to 6 connected cores
#    - Multi-column: one MemTileHub per column, cross-column linking
#    - 512KB L2 budget shared across all allocated buffers
#    - 64 locks shared across all buffer lock pairs (32 buffers max)
#
# 7. POOL ALLOCATION (Prototype 7)
#    - pool_alloc()/pool_free() provide a host-side memory manager
#    - At MLIR level: ONE aie.buffer, sub-regions via dma_bd offset
#    - Bump allocator with first-fit reuse of freed regions
#    - pool_reset() clears all allocations between invocations
#    - Advantages over individual alloc():
#      * Fewer aie.buffer() objects → simpler MLIR
#      * Dynamic sub-region sizing (within pool bounds)
#      * Region reuse without recompiling
#    - Limitations:
#      * Offsets are compile-time constants in dma_bd (not SSA values)
#      * Truly dynamic offsets need npu_writebd or dynamic-runtime-sequences
#      * Fragmentation possible with many alloc/free cycles
#    - Proven on hardware: 07_memtile_pool_alloc
