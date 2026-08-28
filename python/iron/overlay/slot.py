# slot.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""ProgramMemorySlot: a reserved, run-time-writable region of a core's program memory."""

import subprocess
from pathlib import Path

import numpy as np

from ... import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ...dialects import memref  # pyright: ignore[reportAttributeAccessIssue]
from ...dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    DMAChannelDir,
    WireBundle,
)
from ...dialects.aie import external_func  # pyright: ignore[reportAttributeAccessIssue]
from ...dialects.aiex import (
    npu_blockwrite,
)  # pyright: ignore[reportAttributeAccessIssue]
from ...extras.dialects.arith import (
    constant as arith_constant,
)  # pyright: ignore[reportMissingImports]
from ...dialects.arith import addi, muli  # pyright: ignore[reportAttributeAccessIssue]
from ...helpers.dialects.func import call  # pyright: ignore[reportMissingImports]
from ...helpers.dialects.scf import (
    _for as range_,
)  # pyright: ignore[reportMissingImports]
from ...utils import get_current_device
from ..buffer import Buffer
from ..dataflow.flow import Flow, PacketFlow
from ..dataflow.tile_dma import Acquire, Bd, DmaChannel, Release, TileDma
from ..device import Tile
from ..kernel import BaseKernel
from ..lock import Lock
from ..runtime._context import active_sequence  # pyright: ignore[reportMissingImports]
from ..worker import WorkerRuntimeBarrier
from ._bootstrap import Bootstrap, emit_payload_global
from ._elf import peano
from ._geometry import PROG_MEM_LINE, Geometry, Slot
from ._link import OverlayError
from ._tile_transport import (
    MAX_DATA_WORDS_PER_PACKET,
    chunk_for_control_packets,
    done_chunk,
    wire_words,
)
from .overlay import ProgramMemoryOverlay


class ProgramMemorySlotError(Exception):
    """A ProgramMemorySlot cannot be placed, or was used from the wrong context."""


def _enclosing_op(op_name: str):
    """Walk up from the current (nested) insertion point to the nearest
    enclosing op named `op_name` (e.g. ``"aie.core"``, ``"aie.device"``).

    `Worker.resolve()` wraps every `core_fn` in an `scf.for`, so the
    immediate block owner one level up from a call made *inside* `core_fn`
    is that loop, not the `aie.core` op itself -- this walks past however
    many such wrapper levels exist instead of assuming a fixed depth.
    """
    op = ir.InsertionPoint.current.block.owner
    while op.operation.name != op_name:
        parent = op.operation.parent
        if parent is None:
            raise ProgramMemorySlotError(
                f"_enclosing_op('{op_name}'): reached the top of the IR "
                f"without finding one; this must be called from inside "
                f"a Worker's core_fn."
            )
        op = parent
    return op


class ProgramMemorySlot(BaseKernel):
    """A region of program memory whose contents are decided at run time.

    Calling a `ProgramMemorySlot` instance inside a `Worker`'s `core_fn` emits
    a call to whichever overlay is currently loaded -- exactly like calling a
    [`Kernel`][iron.Kernel], because every overlay assigned to one slot shares
    one call site (the slot's, not the overlay's). `slot.wait()` blocks the
    core until a payload has been written; `slot.load(overlay)`, called from
    a `Runtime` sequence (the default) or a tile-sourced loader `Worker`'s
    `core_fn` (see `source=`), schedules that write.

    Placement (which address, how much of program memory to reserve for the
    resident) is computed automatically from `size` and the current device's
    characterised program-memory write granule -- never hand-picked. A device
    with no characterised granule (e.g. npu1) is refused at construction,
    before any build is attempted.
    """

    def __init__(
        self,
        name: str,
        arg_types: list[type[np.ndarray] | np.dtype] | None = None,
        *,
        tile: Tile,
        size: int,
        source=None,  # a Worker, once implemented -- see `source=` below
    ):
        """Construct a ProgramMemorySlot.

        Args:
            name: A label for the slot, used to derive its entry symbol and
                generated linker-script filename. Must be unique among the
                slots on one Worker.
            arg_types: Call signature every overlay assigned to this slot must
                share -- the slot has one call site, reused by whichever
                overlay is currently loaded.
            tile: The compute tile whose program memory this slot lives in.
            size: Slot size in bytes. Rounded-up internally to whole
                program-memory write granules for placement (see class
                docstring); the *slot itself* is exactly `size` bytes -- an
                overlay larger than that is rejected at link time with a
                clear error naming both sizes.
            source: Who writes into this slot. `None` (the default) means the
                host, via `slot.load(overlay)` called from a `Runtime`
                sequence -- emits `npu_blockwrite`. A `Worker` means that
                Worker's own DMA writes the slot via a control-packet send
                into this slot's tile's `TileControl` port instead;
                `slot.load(overlay)` must then be called from *that* Worker's
                `core_fn` instead, and is rejected from a Runtime sequence.
                `source`'s tile need not be adjacent to this slot's tile --
                the packet-switched route may hop through intermediate
                switchboxes -- but if no route exists at all, `aiecc` itself
                refuses at build time; this never surfaces as a runtime hang.
                `load(overlay)` may be called more than once, one per
                phase, exactly like the host-written and ping-pong
                transports -- see `load`'s own docstring for how the next
                phase is kept from overwriting the slot before this one has
                actually finished being used. Hardware-verified on Strix at
                3 phases and a full-granule slot. `size` is not bounded by
                `source`'s tile's BD table: the transfer uses one reused,
                self-looping `aie.dma_bd` (see `_load_tile_sourced`'s
                comments for why a self-loop reframes each round's packet
                correctly where `BdIteration` does not, hardware-verified on
                Strix), so a single BD table entry covers a slot of any
                size.
        """
        from ..worker import Worker  # deferred: see the Worker-side comment

        if source is not None and not isinstance(source, Worker):
            raise TypeError(
                f"ProgramMemorySlot '{name}': source= must be a Worker (the "
                f"tile whose own DMA writes this slot) or None (host-written, "
                f"the default); got {type(source).__name__}."
            )
        if size <= 0 or size % PROG_MEM_LINE:
            raise ValueError(
                f"ProgramMemorySlot '{name}': size must be a positive multiple "
                f"of {PROG_MEM_LINE} (the program-memory line width), got {size}"
            )
        entry = f"overlay_entry_{name}"
        super().__init__(entry, arg_types)
        self._slot_name = name
        self._tile = tile
        self._requested_size = size
        self._source = source
        self._overlays: list[ProgramMemoryOverlay] = []
        self._barrier = WorkerRuntimeBarrier()
        # Tile-sourced transport only, below. A plain data-memory flag on
        # this slot's own tile that the source Worker's control-packet burst
        # writes `1` into once the payload has landed -- not a hardware
        # lock, because writing a *lock's* value register remotely needs its
        # hardware-assigned address, not knowable until a much later
        # compiler pass; a Buffer's address is knowable from the linked
        # resident ELF (ProgramMemoryOverlayDesign reads it back, the same
        # way it already does for pingpong()'s bootstrap park). wait() polls
        # it via a tiny separately compiled C++ stub (see _poll_ctrl_done),
        # the same volatile-read idiom the bootstrap park's own stub already
        # proves correct on hardware -- an MLIR-emitted software loop reading
        # a plain memref has no such guarantee against being hoisted as
        # loop-invariant, since nothing in the core's own instruction stream
        # ever writes that memory.
        self._ctrl_done_buf: Buffer | None = None
        self._ctrl_done_addr: int | None = None
        self._ctrl_wait_op = None
        if source is not None:
            # Sized to MAX_DATA_WORDS_PER_PACKET, not just the one real flag
            # word: the done signal is sent through the same reused BD as
            # every other chunk (see `_load_tile_sourced`), which sends a
            # fixed-length transfer every round, so the done chunk's payload
            # is padded to that width. Padding into this Buffer's own
            # trailing words is safe -- unlike padding into a hardware
            # register -- because this module sizes and owns it.
            word_ty = np.ndarray[(MAX_DATA_WORDS_PER_PACKET,), np.dtype[np.int32]]
            self._ctrl_done_buf = Buffer(
                word_ty,
                initial_value=np.zeros(MAX_DATA_WORDS_PER_PACKET, dtype=np.int32),
                name=f"{name}_ctrl_done",
                tile=tile,
            )
        # Multi-phase tile-sourced scheduling: how many times wait() has run
        # (for phase indexing) and the lazily-built reverse ack channel (see
        # `_ensure_ack_rig`) letting the source know a phase's overlay has
        # actually finished being used before it is safe to overwrite the
        # slot for the next phase. None of this is built at all for a
        # single-phase design -- see `_ensure_ack_rig`'s docstring.
        self._wait_calls = 0
        self._ack: dict[str, Lock] | None = None
        self._tile_source_rig: dict | None = None
        self._worker = None  # bound by Worker.__init__ when passed in fn_args
        self._geometry: Geometry | None = None  # computed on first resolve()
        self._slot_ld_path: str | None = None
        # Payload words for the *current* build pass. None (pass 1) -> emit a
        # zero-filled placeholder of `size` bytes, so the resident's memref
        # layout is identical between passes. Set once ProgramMemoryOverlay
        # bytes are known (pass 2) via `_set_pass2_payload`.
        self._pass2_words: dict[str, list[int]] | None = None
        self._load_calls: list[str] = []  # overlay names, in `load()` call order
        # Set by `pingpong()` for whichever slot shares the resident's
        # granule: wait()/load() route through the bootstrap park instead of
        # the normal barrier. None for a single-slot ProgramMemorySlot.
        self._park_via: Bootstrap | None = None
        # Set by `pingpong()` on both slots: ("low"|"high", sibling, shared
        # Bootstrap-or-None cell). Computed lazily in _compute_geometry(),
        # matching every other placement decision here, because the granule
        # size (needed to do the split) is only knowable once a device is
        # current -- not guaranteed yet at pingpong() call time. The
        # Bootstrap itself is built once, by whichever of the pair resolves
        # first, and shared via this same list cell so the second sees it.
        self._pingpong: tuple[str, "ProgramMemorySlot", list] | None = None

    @property
    def name(self) -> str:
        return self._slot_name

    @property
    def tile(self) -> Tile:
        return self._tile

    @property
    def entry_symbol(self) -> str:
        return self._name

    @property
    def base(self) -> int:
        """Byte address of this slot within program memory. Valid after resolve()."""
        if self._geometry is None:
            raise ProgramMemorySlotError(
                f"ProgramMemorySlot '{self._slot_name}': not yet placed "
                f"(resolve() has not run)."
            )
        return self._geometry.slots[0].base

    @property
    def size(self) -> int:
        return self._requested_size

    @property
    def reserved_bytes(self) -> int:
        """Bytes the owning Worker's `program_memory_reserved` must cover."""
        return self._geometry.target_model.get_program_memory_size() - self.base

    def _register_overlay(self, overlay: ProgramMemoryOverlay) -> None:
        if any(o.name == overlay.name for o in self._overlays):
            raise ValueError(
                f"ProgramMemorySlot '{self._slot_name}' already has an overlay "
                f"named '{overlay.name}'."
            )
        self._overlays.append(overlay)

    @property
    def overlays(self) -> list[ProgramMemoryOverlay]:
        return list(self._overlays)

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def _resolve_device_and_granule(self):
        device = get_current_device()
        if device is None:
            raise ProgramMemorySlotError(
                f"ProgramMemorySlot '{self._slot_name}': no current device "
                f"(iron.get_current_device() returned None). Call "
                f"iron.set_current_device(...) before building, or construct "
                f"the Program with an explicit device before resolving."
            )
        tm = device.target_model
        granule = tm.get_program_memory_write_granule()
        if granule is None:
            raise ProgramMemorySlotError(
                f"ProgramMemorySlot '{self._slot_name}': this device has no "
                f"characterised program-memory write granule "
                f"(AIETargetModel::getProgramMemoryWriteGranule() returned "
                f"None). The half-granule silent-drop behavior program-memory "
                f"overlays depend on has only been measured on npu2/AIE2P "
                f"(see test/npu-xrt/pm_write_while_running); refused rather "
                f"than guessed at for this device."
            )
        return device, tm, granule

    def _ensure_bootstrap(self) -> "Bootstrap | None":
        """Create this slot's shared Bootstrap if needed, idempotently.

        Split out from `_compute_geometry()` so `Worker.flat_fn_args` can
        force the Bootstrap (and its two Buffers) to exist *before* Program's
        resolution loop takes its one snapshot of that list -- appending to
        that list later, from inside `resolve()`, would be too late for
        Program to ever see them.
        """
        if self._pingpong is None:
            return None
        role, sibling, cell = self._pingpong
        if cell[0] is not None:
            if role == "low":
                self._park_via = cell[0]
            return cell[0]
        _, tm, granule = self._resolve_device_and_granule()
        size = self._requested_size
        fixed = granule - size
        if fixed <= 0 or fixed % PROG_MEM_LINE:
            raise ProgramMemorySlotError(
                f"ProgramMemorySlot.pingpong: size={size} leaves {fixed} "
                f"bytes for the resident and the bootstrap park in each "
                f"0x{granule:x}-byte granule, which is not a positive "
                f"multiple of {PROG_MEM_LINE}. Shrink size."
            )
        pm_size = tm.get_program_memory_size()
        if pm_size != 2 * granule:
            raise ProgramMemorySlotError(
                f"ProgramMemorySlot.pingpong: program memory is "
                f"0x{pm_size:x} bytes, not exactly two 0x{granule:x}-byte "
                f"write granules. This mechanism is only verified for the "
                f"two-granule case."
            )
        bootstrap_name = f"{self.name}_{sibling.name}"
        bootstrap_base = 2 * granule - fixed  # top of the high granule
        cell[0] = Bootstrap(bootstrap_name, self._tile, bootstrap_base, fixed)
        if role == "low":
            self._park_via = cell[0]
        return cell[0]

    def _compute_geometry(self) -> Geometry:
        device, tm, granule = self._resolve_device_and_granule()
        pm_size = tm.get_program_memory_size()

        if self._pingpong is not None:
            role, _, cell = self._pingpong
            self._ensure_bootstrap()
            bootstrap = cell[0]
            size = self._requested_size
            fixed = granule - size
            if role == "low":
                # Low granule: resident at 0, this slot right after it.
                # Written while the core parks in the bootstrap (the other
                # granule) instead of its normal in-resident wait.
                base, core_in, resident_budget = fixed, bootstrap.base, fixed
                self._park_via = bootstrap
            else:
                # High granule: this slot first, bootstrap park after it.
                # Written while the core is in the resident (address 0), as
                # normal -- no park needed.
                base, core_in, resident_budget = granule, 0, fixed
        else:
            # Top-anchor, granule-aligned: the slot occupies whole granules at
            # the top of program memory, so it can never straddle a granule
            # boundary and never shares one with the resident (which always
            # executes from granule 0 in the single-slot case). Requested
            # `size` need not be a whole granule -- the slot itself stays
            # exactly `size` bytes -- but the *placement* rounds up so the
            # boundary math is trivially safe.
            granules_needed = -(-self._requested_size // granule)  # ceil div
            base = pm_size - granules_needed * granule
            core_in = 0
            resident_budget = base
            if base <= 0:
                raise ProgramMemorySlotError(
                    f"ProgramMemorySlot '{self._slot_name}': size "
                    f"{self._requested_size} needs {granules_needed} "
                    f"program-memory write granule(s) (0x{granule:x} bytes "
                    f"each), leaving no room for the resident in a "
                    f"0x{pm_size:x}-byte program memory. Shrink the slot."
                )

        slot = Slot(self._slot_name, base, self._requested_size, core_in=core_in)
        geometry = Geometry(
            dev=device._device.name.lower(),
            tile=(self._tile.col, self._tile.row),
            slots=(slot,),
            resident_budget=resident_budget,
        )
        geometry.validate()
        return geometry

    @classmethod
    def pingpong(
        cls,
        name_a: str,
        name_b: str,
        arg_types: list[type[np.ndarray] | np.dtype] | None = None,
        *,
        tile: Tile,
        size: int,
    ) -> tuple["ProgramMemorySlot", "ProgramMemorySlot"]:
        """Two slots that alternate, each written while the core executes the
        other, plus the bootstrap park this needs and never exposes.

        Program memory splits into exactly two program-memory write granules
        (this is refused, not guessed at, if a device's granule does not
        divide its program memory in half). Each granule holds a `size`-byte
        slot plus a `(granule - size)`-byte fixed region: the resident, in
        the low granule; the bootstrap park, in the high one. `size` is
        shared between the two slots so those fixed regions come out equal --
        matching the only geometry this mechanism has been hardware-verified
        against (test/npu-xrt/program_memory_overlay/hw/pingpong.lit).

        The *second* slot returned (`name_b`, in the high granule, alongside
        the bootstrap park) must be the first one `load()`ed and `wait()`ed on
        in phase 0, not `name_a`: the core boots directly into the resident,
        so its first `wait()` must be the normal in-resident one -- `name_a`'s
        `wait()` jumps straight into the bootstrap park, whose code has not
        been written yet on phase 0 (that write itself only happens the first
        time either slot is `load()`ed, which cannot precede the core's very
        first `wait()`).

        Args:
            name_a: The low-granule slot (shares the resident's granule).
            name_b: The high-granule slot (shares the bootstrap park's granule).
            arg_types: Call signature shared by both slots and every overlay
                assigned to either.
            tile: The compute tile both slots live on.
            size: Byte size of *each* slot (both are the same size).
        """
        slot_a = cls(name_a, arg_types, tile=tile, size=size)
        slot_b = cls(name_b, arg_types, tile=tile, size=size)
        # Granule size is only knowable once a device is current, which is not
        # guaranteed yet at pingpong() call time -- so the actual split is
        # computed lazily in _compute_geometry(), matching every other
        # placement decision in this module. The shared one-element list is
        # how the second slot to resolve finds the Bootstrap the first one
        # already built, without either slot needing a back-reference set
        # after the fact.
        cell = [None]
        slot_a._pingpong = ("low", slot_b, cell)
        slot_b._pingpong = ("high", slot_a, cell)
        return slot_a, slot_b

    # ------------------------------------------------------------------
    # Resolvable
    # ------------------------------------------------------------------

    def resolve(
        self,
        loc: "ir.Location | None" = None,
        ip: "ir.InsertionPoint | None" = None,
    ) -> None:
        if self._op is not None:
            return
        self._geometry = self._compute_geometry()
        self._slot_ld_path = f"{self._slot_name}_slot.ld"
        with open(self._slot_ld_path, "w") as f:
            f.write(
                "/* Generated by iron.overlay.ProgramMemorySlot. The resident "
                "calls this address; the body is written there at run time. */\n"
                f"{self._name} = 0x{self.base:x};\n"
            )
        self._op = external_func(
            self._name,
            inputs=list(self._arg_types),
            link_with=self._slot_ld_path,
        )
        if self._park_via is not None:
            self._park_via.resolve()

    # ------------------------------------------------------------------
    # Core-side verbs
    # ------------------------------------------------------------------

    def wait(self) -> None:
        """Block the core until a payload has been written into this slot.

        Call from inside the owning `Worker`'s `core_fn`, once per phase,
        before calling the slot itself. For the low-granule slot of a
        `pingpong()` pair, this parks the core in the bootstrap stub instead
        of the normal in-resident wait -- the resident's own granule is what
        is being written while this slot's payload lands. For a tile-sourced
        slot (`source=`), this polls the plain flag Buffer the source
        Worker's control-packet burst writes on completion, instead of
        blocking on a host-released hardware lock -- there is no runtime
        sequence in that transport to release one.

        On the second and later call for a tile-sourced slot, this first
        sends the previous phase's ack (see `_ensure_ack_rig`): the source
        must not overwrite the slot for phase N+1 until this core is
        actually done executing phase N's overlay, and reaching this call
        again is exactly that point -- the same reasoning
        `iron_api_one_slot.py`'s host-transport multi-phase design gets for
        free from an output-DMA wait, made explicit here because a
        tile-sourced source has no such built-in signal to piggyback on.
        """
        if self._source is not None:
            if self._wait_calls > 0:
                self._ensure_ack_rig()
                self._ack["credit"].acquire(1)
                self._ack["go"].release(1)
            self._wait_calls += 1
            self._poll_ctrl_done()
        elif self._park_via is not None:
            self._park_via.enter()
        else:
            self._barrier.wait_for_value(1)

    def _ensure_ack_rig(self) -> None:
        """Build (once, idempotently) the reverse ack channel from this
        slot's tile back to `source`'s tile.

        Only built the first time a design actually needs a second phase --
        `wait()`/`_load_tile_sourced` only call this once `self._wait_calls`
        /`self._load_calls` show a phase beyond the first, so a single-phase
        design (still the common case) never pays for or risks this at all.

        Whichever of `wait()` (this tile's core_fn) or `_load_tile_sourced`
        (`source`'s core_fn) needs it first builds it -- same pattern as
        `pingpong()`'s shared `Bootstrap`: each caller is inside its own
        Worker's still-open `core_fn` when it calls this, so
        `_enclosing_op("aie.core")` always finds the right insertion point
        (immediately before whichever of the two cores is currently being
        built) regardless of which Worker's `resolve()` runs first.

        This rig puts a second DMA program on `source`'s tile, alongside
        the forward transport's own (`_ensure_tile_sourced_rig`). Both are
        separate `TileDma`s and so lower to separate `aie.mem` regions for
        one tile, which used to make 3+ phases fail almost always: BD ids
        are a per-*tile* table, and `--aie-assign-bd-ids` restarted
        numbering per region, so the two programs' BDs collided in the same
        slot and silently overwrote each other. Fixed in that pass (its
        allocator is now keyed by tile, and a genuine collision is a
        compile error rather than silent corruption) -- see
        `test/Passes/assign-bd-ids/multiple_mem_ops_same_tile.mlir`. Nothing
        is required of callers here; the note exists because "two TileDmas
        on one tile" reads as suspicious and is in fact fine.

        Uses this tile's own MM2S channel 1 and `source`'s S2MM channel 0.
        Assumed free: a tile-sourced slot's `source` Worker has no dataflow
        of its own (see `ProgramMemorySlot(source=...)`'s docstring), and
        this tile's own channel 0 is left for whatever `ObjectFifo`s the
        Worker that owns this slot already uses.
        """
        if self._ack is not None:
            return
        src_tile = self._source.tile
        dst_tile = self._tile
        prefix = f"{self._slot_name}_ack"

        send_bufs = [
            Buffer(
                np.ndarray[(1,), np.dtype[np.int32]],
                initial_value=np.array([1], dtype=np.int32),
                name=f"{prefix}_send_{i}",
                tile=dst_tile,
            )
            for i in range(2)
        ]
        recv_bufs = [
            Buffer(
                np.ndarray[(1,), np.dtype[np.int32]],
                name=f"{prefix}_recv_{i}",
                tile=src_tile,
            )
            for i in range(2)
        ]
        # go/credit: the core (this tile) triggers a send and only triggers
        # the next one once a buffer has actually drained -- both start
        # buffers free (credit=2), matching `of_out`'s own depth-2 producer
        # lock init.
        ack_go = Lock(dst_tile, init=0, name=f"{prefix}_go")
        ack_credit = Lock(dst_tile, init=2, name=f"{prefix}_credit")
        # recv_free/recv_ready: the mirror image on the receiving end.
        recv_free = Lock(src_tile, init=2, name=f"{prefix}_recv_free")
        recv_ready_lock = Lock(src_tile, init=0, name=f"{prefix}_recv_ready")

        send_channel = DmaChannel(
            DMAChannelDir.MM2S,
            1,
            bds=[
                Bd(
                    send_bufs[i],
                    length=1,
                    acquires=[Acquire(ack_go, 1)],
                    releases=[Release(ack_credit, 1)],
                    next=1 - i,
                )
                for i in range(2)
            ],
        )
        recv_channel = DmaChannel(
            DMAChannelDir.S2MM,
            0,
            bds=[
                Bd(
                    recv_bufs[i],
                    length=1,
                    acquires=[Acquire(recv_free, 1)],
                    releases=[Release(recv_ready_lock, 1)],
                    next=1 - i,
                )
                for i in range(2)
            ],
        )
        send_dma = TileDma(dst_tile, [send_channel])
        recv_dma = TileDma(src_tile, [recv_channel])
        flow = Flow(dst_tile, src_tile, src_channel=1, dst_channel=0)

        with ir.InsertionPoint(_enclosing_op("aie.core")):
            for lk in (ack_go, ack_credit, recv_free, recv_ready_lock):
                lk.resolve()
            for buf in (*send_bufs, *recv_bufs):
                buf.resolve()
            flow.resolve()
            send_dma.resolve()
            recv_dma.resolve()

        self._ack = {
            "go": ack_go,
            "credit": ack_credit,
            "recv_free": recv_free,
            "recv_ready": recv_ready_lock,
        }

    def _poll_ctrl_done(self) -> None:
        """Call a tiny compiled stub that spins on `_ctrl_done_buf`, then
        clears it -- safe to call more than once across a design's phases,
        each call waiting for that phase's own write to land rather than
        the first phase's flag being seen (and satisfied) forever after.

        Deliberately NOT an MLIR-emitted `scf.while` reading the buffer with
        a plain `memref.load`: nothing in this core's own instruction stream
        ever writes that memory, so an optimizer has no reason not to treat
        the load as loop-invariant and hoist it out, turning "poll until a
        remote DMA writes this" into an infinite loop regardless of what
        actually arrives. A separately compiled C++ function reading through
        a `volatile` pointer is exactly how `_bootstrap.py`'s ping-pong park
        already avoids the identical trap -- reused here as a small, ordinary
        (not slot-swapped) object-linked kernel instead of a resident-linked
        one, since this code lives at a fixed compile-time location and is
        never swapped at run time.
        """
        assert self._ctrl_done_buf is not None
        if self._ctrl_wait_op is None:
            obj = str(Path(f"{self._slot_name}_ctrl_wait.o"))
            src = (
                "// Generated by iron.overlay.ProgramMemorySlot. Spins until "
                "the tile-sourced control-packet burst signals completion, "
                "then clears the flag so the next phase can wait on it "
                "again.\n"
                "#include <cstdint>\n\n"
                f'extern "C" int32_t {self._ctrl_done_buf.name}[];\n'
                f'extern "C" void {self._name}_ctrl_wait(void) {{\n'
                f"  volatile int32_t *f = {self._ctrl_done_buf.name};\n"
                "  while (*f == 0)\n"
                "    ;\n"
                "  *f = 0;\n"
                "}\n"
            )
            src_path = f"{self._slot_name}_ctrl_wait.cc"
            with open(src_path, "w") as f:
                f.write(src)
            cmd = [
                peano("clang++"),
                "--target=aie2p-none-unknown-elf",
                "-O2",
                "-c",
                src_path,
                "-o",
                obj,
            ]
            if subprocess.run(cmd).returncode:
                raise ProgramMemorySlotError(
                    f"ProgramMemorySlot '{self._slot_name}': compiling the "
                    f"tile-sourced completion-wait stub failed."
                )
            with ir.InsertionPoint(_enclosing_op("aie.core")):
                self._ctrl_wait_op = external_func(
                    f"{self._name}_ctrl_wait", inputs=[], link_with=obj
                )
        call(self._ctrl_wait_op, [])

    def __call__(self, *args, **kwargs):
        if self._op is None:
            raise ProgramMemorySlotError(
                f"ProgramMemorySlot '{self._slot_name}' must be resolved "
                f"(placed in a Worker's fn_args) before it can be called."
            )
        super().__call__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Loader-side verb
    # ------------------------------------------------------------------

    def load(self, overlay: ProgramMemoryOverlay) -> None:
        """Schedule `overlay`'s bytes to be written into this slot, then
        release the core waiting in [`wait()`][iron.overlay.ProgramMemorySlot.wait].

        Call from inside a `Runtime` sequence for the default, host-written
        transport, or from `source`'s own `core_fn` for a tile-sourced slot
        -- calling from the wrong one raises. A tile-sourced slot supports
        any number of `load()` calls, one per phase, exactly like the
        host-written and ping-pong transports -- each call's overlay is
        written after the previous phase's has actually finished being
        used (see `_ensure_ack_rig`), not merely after it has been sent.
        """
        if overlay.slot is not self:
            raise ValueError(
                f"ProgramMemoryOverlay '{overlay.name}' belongs to slot "
                f"'{overlay.slot.name}', not '{self._slot_name}'."
            )
        if self._source is not None:
            self._load_tile_sourced(overlay)
            return
        try:
            active_sequence()
        except RuntimeError as exc:
            raise ProgramMemorySlotError(
                f"ProgramMemorySlot.load() for slot '{self._slot_name}' must "
                f"be called from inside a Runtime sequence body (this slot "
                f"is host-written; pass source=<a Worker> at construction "
                f"for tile-sourced transport instead)."
            ) from exc
        if self._pingpong is not None:
            # The bootstrap's own code is written once, as the very first
            # thing, on whichever slot's load() happens to run first -- it
            # must exist before either slot's wait() can rely on it, and the
            # low granule (where slot_a and the resident live) has not been
            # touched yet at this point in any correct phase order.
            self._pingpong[2][0].install()

        if self._park_via is not None:
            # This slot shares the resident's granule: writing it is only
            # safe once the core has actually left for the bootstrap park,
            # which is the only thing a runtime sequence can wait on the core
            # itself for (everything else it can wait on is a DMA).
            self._park_via.wait_parked(self._tile.col, self._tile.row)

        words = self._payload_words_for(overlay)
        memref_ty = ir.MemRefType.get([len(words)], ir.IntegerType.get_signless(32))
        # Suffixed by call order: the same overlay can be `load()`ed into this
        # slot more than once across a design's phases, and each occurrence
        # needs its own memref.global -- a bare (slot, overlay) name would
        # collide the second time the same overlay is loaded.
        sym = f"{self._slot_name}_{overlay.name}_payload_{len(self._load_calls)}"
        emit_payload_global(sym, memref_ty, words)
        npu_blockwrite(
            self._host_offset() + self.base,
            memref.get_global(memref_ty, sym),
            column=self._tile.col,
            row=self._tile.row,
        )
        self._load_calls.append(overlay.name)
        if self._park_via is not None:
            self._park_via.release()
        else:
            self._barrier.set(1)

    def _host_offset(self) -> int:
        return self._geometry.target_model.get_program_memory_host_offset()

    def _payload_words_for(self, overlay: ProgramMemoryOverlay) -> list[int]:
        """Words to embed for `overlay` in the *current* build pass.

        Pass 1 (no extracted bytes yet): a zero-filled placeholder the exact
        size of the slot, so the resident's module-scope globals are
        byte-identical in shape between passes -- only the *content*, not the
        *layout*, may differ, and content is exactly what `check_stability`
        (in `design.py`) does not need to compare.
        """
        if self._pass2_words is not None and overlay.name in self._pass2_words:
            return self._pass2_words[overlay.name]
        return [0] * (self._requested_size // 4)

    def _set_pass2_payload(self, overlay_name: str, words: list[int]) -> None:
        if self._pass2_words is None:
            self._pass2_words = {}
        if len(words) * 4 > self._requested_size:
            raise OverlayError(
                f"'{overlay_name}' is {len(words) * 4} bytes, larger than the "
                f"{self._requested_size}-byte slot '{self._slot_name}'; bump "
                f"ProgramMemorySlot(size=...)."
            )
        padded = words + [0] * (self._requested_size // 4 - len(words))
        self._pass2_words[overlay_name] = padded

    def _set_ctrl_done_addr(self, addr: int) -> None:
        """Inject this slot's flag Buffer's resolved address for pass 2.

        Mirrors `Bootstrap.parked_addr`: only knowable after pass 1 has
        actually linked (a Buffer's address is compiler-assigned), and unused
        by pass 1 itself (whose xclbin/ELF is discarded after extraction).
        """
        self._ctrl_done_addr = addr

    # ------------------------------------------------------------------
    # Tile-sourced transport
    # ------------------------------------------------------------------

    _ctrl_pkt_id_counter = 0

    def _ensure_tile_sourced_rig(self) -> None:
        """Build (once, idempotently) the forward transport's shared state:
        the reused staging Buffer, its pacing locks, the one self-looping
        BD, and the `PacketFlow` to this slot's tile. Every `load()` call
        (one per phase) reuses this rig, each supplying its own overlay's
        `full_payload` and core-side copy loop -- see `_load_tile_sourced`.

        Everything here is emitted at device scope (a `TileDma`'s `aie.mem`
        region is a sibling of `aie.core`, not nestable inside one),
        inserted immediately *before* `source`'s own (currently still being
        built) `aie.core` op -- not merely "somewhere at device scope":
        every tile op already precedes it (tiles are all resolved up
        front), but appending after the *current* last device-scope op
        would land after `aie.core` itself, which does not dominate a
        `use_lock` inside its own body that references a lock defined
        afterward.
        """
        if self._tile_source_rig is not None:
            return
        src_tile = self._source.tile
        chunk_stride = 1 + MAX_DATA_WORDS_PER_PACKET
        name_prefix = f"{self._slot_name}_{src_tile.col}_{src_tile.row}"

        ProgramMemorySlot._ctrl_pkt_id_counter += 1
        pkt_id = ProgramMemorySlot._ctrl_pkt_id_counter

        # `staging`: the one BD actually reads this, rewritten each round by
        # whichever phase's core-side loop is currently running (each phase
        # has its own `full_payload` table to copy from -- see
        # `_load_tile_sourced` -- but they all copy into this same buffer).
        staging = Buffer(
            np.ndarray[(chunk_stride,), np.dtype[np.int32]],
            name=f"{name_prefix}_ctrl_staging",
            tile=src_tile,
        )
        # slot_free: "the core may (over)write staging and arm a send."
        # Starts available so round 0 proceeds immediately; the BD gives it
        # back once a send has actually drained staging, gating round i+1 on
        # round i having actually completed.
        slot_free_lock = Lock(src_tile, init=1, name=f"{name_prefix}_ctrl_free")
        # xfer_ready: "staging holds a real, unsent chunk; go." The BD
        # acquires this before every send -- the first (via dma_start's own
        # automatic initial queue fetch) and every one after (via the
        # channel's own next_bd re-fetch of the same BD, below).
        xfer_ready_lock = Lock(src_tile, init=0, name=f"{name_prefix}_ctrl_ready")

        # One BD, reused for every round via next="self" -- deliberately NOT
        # one static Bd per chunk (which caps capacity at the source tile's
        # BD table, AIETargetModel::getNumBDs(), 16 on Strix, shared across
        # every DMA channel on the tile) and NOT `BdIteration` (a single
        # packet-tagged Bd repeating over N executions without re-fetching):
        # hardware-verified (Strix) to silently misdirect the packet's
        # embedded address on every execution but the first -- see
        # test/npu-xrt/tile_sourced_ctrl_pkt_spike/aie.mlir's header comment
        # (lesson 3). Root-caused: AIE2P's packet framing is tied to a
        # genuine descriptor *fetch*, and `BdIteration` deliberately avoids
        # re-fetching between repeats (that is its whole point) -- so only
        # the first repeat is framed as its own packet; the rest are
        # consumed as continuation data of the still-open first one.
        #
        # `next="self"` (an ordinary `aie.next_bd` back to the same block,
        # the documented default for "the common 'keep streaming' pattern")
        # is a *different* mechanism from `BdIteration` -- it is the same
        # kind of chain traversal that already links multiple distinct BDs
        # together elsewhere in this codebase, and a chain traversal is a
        # real descriptor fetch every hop, matching BdIteration's own
        # never-refetches contrast exactly. Hardware-verified (Strix,
        # test/npu-xrt/tile_sourced_bd_poke_spike/aie.mlir's later history):
        # a self-looping single BD reframes every round correctly, using one
        # BD table entry regardless of how many rounds there are. (An
        # earlier attempt at this same goal used a core-issued raw register
        # write to re-arm the BD instead of `next="self"`; that write did
        # work, hardware-verified, but pacing the *next* round on it never
        # got a reliable completion signal. `next="self"`'s own completion
        # signal -- the BD's ordinary trailing lock release -- turned out to
        # be exactly what was missing: it does not fire reliably for a BD
        # that terminates instead of looping, which is what the register-poke
        # version did. Looping is both simpler and the one that is provably
        # reliable.)
        bd = Bd(
            staging,
            length=chunk_stride,
            acquires=[Acquire(xfer_ready_lock, 1)],
            releases=[Release(slot_free_lock, 1)],
            packet=(1, pkt_id),
            next="self",
        )
        channel = DmaChannel(DMAChannelDir.MM2S, 0, bds=[bd])
        tile_dma = TileDma(src_tile, [channel])
        flow = PacketFlow(
            pkt_id,
            src=src_tile,
            dst=self._tile,
            dst_port=WireBundle.TileControl,
            keep_pkt_header=True,
            priority_route=True,
        )

        with ir.InsertionPoint(_enclosing_op("aie.core")):
            slot_free_lock.resolve()
            xfer_ready_lock.resolve()
            staging.resolve()
            flow.resolve()
            tile_dma.resolve()

        self._tile_source_rig = {
            "staging": staging,
            "slot_free": slot_free_lock,
            "xfer_ready": xfer_ready_lock,
            "chunk_stride": chunk_stride,
        }

    def _load_tile_sourced(self, overlay: ProgramMemoryOverlay) -> None:
        """Emit `source`'s core-side program that writes `overlay` into this
        slot for the current phase, via the shared rig `_ensure_tile_sourced_rig`
        builds once.

        Call from inside `source`'s own `core_fn` (validated below), once
        per phase -- unlike the host-written and ping-pong transports, a
        tile-sourced slot's `load()` may be called more than once, each
        call scheduling the next phase's overlay. From the second call on,
        this first waits for the *previous* phase's ack (see
        `_ensure_ack_rig`): the destination must have actually finished
        executing that phase's overlay before it is safe to overwrite the
        slot again.
        """
        try:
            active_sequence()
        except RuntimeError:
            pass
        else:
            raise ProgramMemorySlotError(
                f"ProgramMemorySlot.load() for slot '{self._slot_name}' must "
                f"be called from inside source Worker's core_fn (this slot "
                f"is tile-sourced), not from a Runtime sequence."
            )

        if self._load_calls:
            self._ensure_ack_rig()
            self._ack["recv_ready"].acquire(1)
            self._ack["recv_free"].release(1)

        self._ensure_tile_sourced_rig()
        rig = self._tile_source_rig
        staging, slot_free_lock, xfer_ready_lock, chunk_stride = (
            rig["staging"],
            rig["slot_free"],
            rig["xfer_ready"],
            rig["chunk_stride"],
        )

        words = self._payload_words_for(overlay)
        host_addr = self._host_offset() + self.base
        # Every chunk here has exactly MAX_DATA_WORDS_PER_PACKET data words:
        # `words` is this slot's whole (already size-padded) content, and
        # ProgramMemorySlot's constructor requires `size` to be a multiple of
        # PROG_MEM_LINE (16 bytes = 4 words = MAX_DATA_WORDS_PER_PACKET), so
        # `len(words)` is always a whole multiple of it -- no ragged last
        # chunk. `done_chunk` pads the one genuinely 1-word completion signal
        # to the same width. That uniformity is what lets a single BD, not
        # one per chunk, carry every round.
        chunks = chunk_for_control_packets(host_addr, words)
        done = done_chunk(self._ctrl_done_addr or 0)
        all_chunks = chunks + [done]
        wire = wire_words(all_chunks)

        src_tile = self._source.tile
        # `full_payload`: this phase's words, precomputed and static -- the
        # source core never computes a header or a data word, only copies
        # already-known content into the shared `staging` buffer.
        full_payload = Buffer(
            np.ndarray[(len(wire),), np.dtype[np.int32]],
            initial_value=np.array(wire, dtype=np.uint32).view(np.int32),
            name=f"{self._slot_name}_{src_tile.col}_{src_tile.row}_ctrl_payload_{len(self._load_calls)}",
            tile=src_tile,
        )
        with ir.InsertionPoint(_enclosing_op("aie.core")):
            full_payload.resolve()

        # The real core action: one round per chunk (main content, then the
        # padded completion signal), pacing itself on slot_free_lock rather
        # than assuming the previous round's send has already landed. A real
        # scf.for, not an unrolled Python loop -- IRON's range_() collapses
        # to a compact loop (not linear growth) once round count exceeds
        # LLVM's small-trip-count unroll threshold (~16, see
        # test/npu-xrt/program_memory_overlay/README.md), which is exactly
        # the source tile's own program-memory budget this needs to respect
        # for a genuinely large (multi-KB) overlay.
        stride_const = arith_constant(chunk_stride, index=True)
        for i in range_(len(all_chunks)):
            slot_free_lock.acquire(1)
            base_idx = muli(i, stride_const)
            for k in range(chunk_stride):
                idx = (
                    base_idx
                    if k == 0
                    else addi(base_idx, arith_constant(k, index=True))
                )
                staging[k] = full_payload[idx]
            xfer_ready_lock.release(1)

        self._load_calls.append(overlay.name)
