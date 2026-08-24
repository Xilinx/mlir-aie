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
from ...dialects.aiex import npu_blockwrite  # pyright: ignore[reportAttributeAccessIssue]
from ...helpers.dialects.func import call  # pyright: ignore[reportMissingImports]
from ...utils import get_current_device
from ..buffer import Buffer
from ..dataflow.flow import PacketFlow
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
from ._tile_transport import chunk_for_control_packets, done_chunk, wire_words
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
                Only one `load()` call per design is supported today (a
                single overlay loaded once, not a multi-phase schedule
                sourced from a tile) -- a documented v1 scope, not silently
                partial: `load()` raises clearly on a second call. `size` is
                also bounded by `source`'s tile's entire BD table (16
                descriptors on Strix, shared across every DMA channel on the
                tile): one control-packet chunk (4 payload words) per
                descriptor, plus one for the completion signal, so today's
                real ceiling is a few hundred bytes -- `load()` raises a
                clear, named error naming the actual and available
                descriptor counts rather than silently overflowing the
                table. See `_load_tile_sourced`'s comments for why a single
                iterated descriptor (which would lift this ceiling) isn't
                used: it corrupts the packet address hardware-verified.
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
            word_ty = np.ndarray[(1,), np.dtype[np.int32]]
            self._ctrl_done_buf = Buffer(
                word_ty,
                initial_value=np.array([0], dtype=np.int32),
                name=f"{name}_ctrl_done",
                tile=tile,
            )
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
        """
        if self._source is not None:
            self._poll_ctrl_done()
        elif self._park_via is not None:
            self._park_via.enter()
        else:
            self._barrier.wait_for_value(1)

    def _poll_ctrl_done(self) -> None:
        """Call a tiny compiled stub that spins on `_ctrl_done_buf`.

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
                "the tile-sourced control-packet burst signals completion.\n"
                "#include <cstdint>\n\n"
                f"extern \"C\" int32_t {self._ctrl_done_buf.name}[];\n"
                f'extern "C" void {self._name}_ctrl_wait(void) {{\n'
                f"  volatile int32_t *f = {self._ctrl_done_buf.name};\n"
                "  while (*f == 0)\n"
                "    ;\n"
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
        exactly one `load()` call for the lifetime of the design (a
        documented v1 scope: one overlay written once, not a multi-phase
        schedule sourced from a tile) -- a second call raises clearly rather
        than silently reconfiguring a route that is already live.
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

    def _load_tile_sourced(self, overlay: ProgramMemoryOverlay) -> None:
        """Emit `source`'s DMA program that writes `overlay` into this slot
        via a control-packet burst into this tile's `TileControl` port.

        Call from inside `source`'s own `core_fn` (validated by the caller,
        `load()`). Everything except the single lock release below is
        emitted at device scope (a `TileDma`'s `aie.mem` region is a sibling
        of `aie.core`, not nestable inside one), inserted immediately
        *before* `source`'s own (currently still being built) `aie.core` op
        -- not merely "somewhere at device scope": every tile op already
        precedes it (tiles are all resolved up front), but appending after
        the *current* last device-scope op would land after `aie.core`
        itself, which does not dominate a `use_lock` inside its own body
        that references a lock defined afterward.
        """
        if self._load_calls:
            raise ProgramMemorySlotError(
                f"ProgramMemorySlot '{self._slot_name}': load() was already "
                f"called once (overlay '{self._load_calls[0]}'); a "
                f"tile-sourced slot supports exactly one load() call (see "
                f"ProgramMemorySlot.load's docstring)."
            )
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

        words = self._payload_words_for(overlay)
        host_addr = self._host_offset() + self.base
        chunks = chunk_for_control_packets(host_addr, words)
        # Header + data words together: the wire format interleaves them
        # ([header0, data0..., header1, data1...], see wire_words()), and
        # the receiving TileControl port parses the header as part of the
        # SAME burst -- the packet-routing tag below is a separate, additional
        # switchbox-routing concern, not a substitute for this literal word.
        chunk_stride = 1 + len(chunks[0].data) if chunks else 0
        done = done_chunk(self._ctrl_done_addr or 0)
        wire = wire_words(chunks) + [done.header, *done.data]

        ProgramMemorySlot._ctrl_pkt_id_counter += 1
        pkt_id = ProgramMemorySlot._ctrl_pkt_id_counter
        src_tile = self._source.tile
        payload_ty = np.ndarray[(len(wire),), np.dtype[np.int32]]
        payload = Buffer(
            payload_ty,
            initial_value=np.array(wire, dtype=np.uint32).view(np.int32),
            name=f"{self._slot_name}_{self._source.tile.col}_{self._source.tile.row}_ctrl_payload",
            tile=src_tile,
        )
        cons_lock = Lock(
            src_tile, init=0, name=f"{self._slot_name}_ctrl_cons_{pkt_id}"
        )
        # One explicit Bd per chunk, deliberately NOT `BdIteration` (a single
        # packet-tagged Bd repeating over N executions): hardware-verified
        # (Strix) to silently misdirect the packet's embedded address on
        # every execution but the first once packet-tagging and iteration
        # are combined, corrupting exactly the chunks after the first --
        # see test/npu-xrt/tile_sourced_ctrl_pkt_spike/aie.mlir's header
        # comment (lesson 3). A source tile's BD table is a hard, small
        # budget (AIETargetModel::getNumBDs(), 16 on Strix compute tiles),
        # shared by every DMA channel on that tile, so this transport can
        # only place a chunk count that fits it -- refused here, at build
        # time, rather than silently overflowing the BD table.
        num_bds = len(chunks) + 1  # + the trailing done BD
        bd_budget = get_current_device().target_model.get_num_bds(
            src_tile.col, src_tile.row
        )
        if num_bds > bd_budget:
            raise ProgramMemorySlotError(
                f"ProgramMemorySlot '{self._slot_name}': overlay '{overlay.name}' "
                f"needs {len(chunks)} control-packet chunks plus 1 completion "
                f"chunk = {num_bds} DMA descriptors on source tile "
                f"{src_tile}, but that tile's entire BD table holds only "
                f"{bd_budget} (shared across every channel on the tile). "
                f"Shrink the overlay/slot, or free up other DMA usage on "
                f"{src_tile}. (No BdIteration-based workaround exists yet -- "
                f"see this module's tile-sourced transport comments.)"
            )
        # Every BD in a chain that uses a lock at all must have both an
        # acquire and a release (a CDO/hardware encoding rule, not just a
        # synchronization nicety -- aiecc's CDO generator rejects an
        # acquire-only or release-only BD outright), matching
        # test/npu-xrt/tile_sourced_ctrl_pkt_spike/aie.mlir's own two-BD
        # chain. A slot can need more than two BDs (up to the tile's BD
        # budget above), so a *fresh* lock per hop would not scale -- a
        # source tile only has so many hardware locks either. This is a
        # strictly single-pass, one-shot sequence (BD i's release is
        # consumed by BD i+1 and nothing else, ever, in this build), so two
        # locks reused alternately are exactly as safe as a distinct one per
        # hop: whichever hop lock BD i just released is the very next thing
        # BD i+1 acquires, with nothing else in between. The final release
        # targets a lock nothing ever acquires (`sink_lock`), so looping
        # back to BD 0 re-acquires cons_lock -- never released again -- and
        # the channel simply parks there forever, a one-shot burst rather
        # than a repeating one.
        hop_locks = [
            Lock(src_tile, init=0, name=f"{self._slot_name}_ctrl_hop_{pkt_id}_{i}")
            for i in range(2)
        ]
        sink_lock = Lock(src_tile, init=0, name=f"{self._slot_name}_ctrl_sink_{pkt_id}")
        lock_chain = [cons_lock] + [hop_locks[i % 2] for i in range(num_bds - 1)]
        release_targets = lock_chain[1:] + [sink_lock]

        main_bds = [
            Bd(
                payload,
                offset=i * chunk_stride,
                length=chunk_stride,
                acquires=[Acquire(lock_chain[i], 1)],
                releases=[Release(release_targets[i], 1)],
                next=i + 1,
                packet=(1, pkt_id),
            )
            for i in range(len(chunks))
        ]
        done_bd = Bd(
            payload,
            offset=len(wire) - len(done.data) - 1,
            length=len(done.data) + 1,
            acquires=[Acquire(lock_chain[-1], 1)],
            releases=[Release(release_targets[-1], 1)],
            packet=(1, pkt_id),
            next=0,
        )
        channel = DmaChannel(DMAChannelDir.MM2S, 0, bds=[*main_bds, done_bd])
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
            for lk in (cons_lock, *hop_locks, sink_lock):
                lk.resolve()
            payload.resolve()
            flow.resolve()
            tile_dma.resolve()

        # The one real core action lesson #1 needs: nothing else ever
        # touches cons_lock, so without this release (an actual instruction
        # from an independent core context) the DMA channel's hardware queue
        # never starts at all -- see
        # test/npu-xrt/tile_sourced_ctrl_pkt_spike/aie.mlir's header comment.
        cons_lock.release(1)
        self._load_calls.append(overlay.name)
