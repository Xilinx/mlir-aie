# design.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""ProgramMemoryOverlayDesign: the two-pass build that makes overlays just work.

Pass 1 builds the design with every ProgramMemorySlot's payloads
zero-filled, giving each slot's resident a stable, addressable ELF. Every
registered overlay is then linked against that resident at its slot's
address (reusing `iron.overlay._link`, exactly the checks
test/npu-xrt/program_memory_overlay/pmlib/link.py already has: .rodata/ctors
rejected, size and alignment verified, geometry re-validated at the address
the overlay actually landed at). Pass 2 rebuilds the *same* design with the
extracted `.text` bytes now available to `ProgramMemorySlot.load()`, and the
two resident ELFs are compared: any symbol that moved or an overlay's own
import disappearing is a build error, not a warning -- this is a
previously-real failure mode.

Pass 1 also checks that every overlay's deepest stack frame, plus the
resident's own, fits the stack its Worker was linked with (`aiecc`'s Peano
codegen always emits `.stack_sizes`, so this is exact rather than assumed):
the call into a slot is through an absolute address, so nothing about a
normal build would ever catch an overlay overrunning the stack the resident
was sized for. See `_check_stack_budget`, and the same rule pinned against
hand-supplied objects in `test/npu-xrt/program_memory_overlay/nohw/stack_budget.lit`.

A `pingpong()` pair's bootstrap park goes through the same pipeline as any
overlay -- compiled, linked against the resident, its bytes extracted -- the
one difference is its source is generated here rather than supplied by the
caller, and its `ovl_parked`-equivalent Buffer's resolved address (needed for
`ProgramMemorySlot.load()`'s `npu_maskpoll`) is read from the same resident
ELF and injected into the pass-2 build before its Program is resolved.
"""

import subprocess
from pathlib import Path
from typing import Callable

from ...utils import get_current_device
from ...utils.compile.utils import compile_mlir_module
from ._bootstrap import Bootstrap
from ._elf import defined_symbols, find_core_elf, max_stack_frame, peano, text_words
from ._link import OverlayError, link
from .overlay import ProgramMemoryOverlay


class ProgramMemoryOverlayDesignError(Exception):
    """The two-pass overlay build failed in a way specific to this orchestration."""


def _bootstraps_of(overlays: list[ProgramMemoryOverlay]) -> dict[str, Bootstrap]:
    """Every distinct Bootstrap reachable from `overlays`' slots, by name."""
    found: dict[str, Bootstrap] = {}
    for overlay in overlays:
        pingpong = getattr(overlay.slot, "_pingpong", None)
        if pingpong is None:
            continue
        bootstrap = pingpong[2][0]
        if bootstrap is not None:
            found[bootstrap.name] = bootstrap
    return found


def _compile_bootstrap_source(bootstrap: Bootstrap, work_dir: Path) -> str:
    """Compile a Bootstrap's generated stub source, returning the object path."""
    src = work_dir / f"{bootstrap.name}_stub.cc"
    src.write_text(bootstrap.source())
    obj = work_dir / f"{bootstrap.name}_stub.o"
    cmd = [
        peano("clang++"),
        "--target=aie2p-none-unknown-elf",
        "-O2",
        "-std=c++20",
        "-c",
        str(src),
        "-o",
        str(obj),
    ]
    if subprocess.run(cmd).returncode:
        raise ProgramMemoryOverlayDesignError(
            f"compiling the pingpong bootstrap park '{bootstrap.name}' failed"
        )
    return str(obj)


class ProgramMemoryOverlayDesign:
    """Drives the two-pass build a design with `ProgramMemorySlot`s needs.

    A user never invokes `aiecc` or links an overlay by hand: constructing
    this with a `make_design` factory, then calling `.compile()`, does both
    passes, the per-overlay linking, and the resident-stability check.
    """

    def __init__(
        self,
        make_design: Callable[[], "tuple[object, list[ProgramMemoryOverlay]]"],
    ):
        """Construct a ProgramMemoryOverlayDesign.

        Args:
            make_design: A zero-argument factory that builds the *entire*
                design from scratch and returns `(program, overlays)` --
                an unresolved `Program` and every `ProgramMemoryOverlay` used
                anywhere in it. Called twice, once per build pass, and each
                call must construct fresh `Tile`/`Worker`/`ObjectFifo`/
                `ProgramMemorySlot`/`ProgramMemoryOverlay` objects: IRON's
                resolvables are single-use (a `Tile` cannot be bound to a
                second `aie.logical_tile` op), so reusing Python objects a
                second call closed over would fail on the second `resolve()`,
                not silently produce a stale design. Overlays are matched
                between the two calls by `(slot.name, overlay.name)`; a name
                present in one call's result and not the other's is a build
                error naming which pass is missing it.
        """
        self._make_design = make_design

    def compile(
        self,
        work_dir: str | Path,
        xclbin_path: str | Path | None = None,
        insts_path: str | Path | None = None,
        **compile_kwargs,
    ) -> tuple[str, str]:
        """Run both passes and return `(xclbin_path, insts_path)`.

        Args:
            work_dir: Directory for all intermediate and final artifacts.
                Created if it does not exist; pass 1 and pass 2 each get a
                subdirectory.
            xclbin_path: Final xclbin path. Defaults to `work_dir/final.xclbin`.
            insts_path: Final NPU instructions path. Defaults to
                `work_dir/insts.bin`.
            **compile_kwargs: Forwarded to the pass-2 `compile_mlir_module`
                call (e.g. `use_chess=`).
        """
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        xclbin_path = str(xclbin_path or work_dir / "final.xclbin")
        insts_path = str(insts_path or work_dir / "insts.bin")

        device = get_current_device()
        if device is None:
            raise ProgramMemoryOverlayDesignError(
                "ProgramMemoryOverlayDesign.compile(): no current device "
                "(iron.get_current_device() returned None); call "
                "iron.set_current_device(...) first."
            )

        pass1_dir = work_dir / "pass1"
        pass1_dir.mkdir(exist_ok=True)
        pass1_program, pass1_overlays = self._make_design()
        compile_mlir_module(
            pass1_program.resolve_program(),
            xclbin_path=str(pass1_dir / "pass1.xclbin"),
            insts_path=str(pass1_dir / "pass1_insts.bin"),
            work_dir=str(pass1_dir),
            device=device,
        )

        # Link every overlay against pass 1's resident, at its slot's address.
        imports_used: dict[str, set[str]] = {}  # slot name -> resident symbols used
        words_by_key: dict[tuple[str, str], list[int]] = {}
        for overlay in pass1_overlays:
            slot = overlay.slot
            key = (slot.name, overlay.name)
            resident_elf = find_core_elf(str(pass1_dir), slot.tile.col, slot.tile.row)
            linked = str(pass1_dir / f"{slot.name}_{overlay.name}.linked")
            try:
                link(
                    objects=[overlay.object_file_name],
                    resident=resident_elf,
                    slot_base=slot.base,
                    slot_size=slot.size,
                    output=linked,
                    entry=slot.entry_symbol,
                    col=slot.tile.col,
                    row=slot.tile.row,
                    geometry=slot._geometry,
                )
            except OverlayError as exc:
                raise ProgramMemoryOverlayDesignError(
                    f"ProgramMemoryOverlay '{overlay.name}' (slot "
                    f"'{slot.name}'): {exc}"
                ) from exc
            words_by_key[key] = text_words(linked)
            with open(f"{linked}.imports") as f:
                imports_used.setdefault(slot.name, set()).update(
                    s for s in f.read().split("\n") if s
                )
            _check_stack_budget(resident_elf, linked, slot, overlay, device)

        # Tile-sourced slots' flag Buffer ("has the write landed") address:
        # only knowable from the linked resident ELF, same as a pingpong
        # bootstrap's parked Buffer below. Read once per distinct slot (not
        # per overlay -- a tile-sourced slot has exactly one in v1 scope
        # anyway, see ProgramMemorySlot.load's one-call limit).
        ctrl_done_addr: dict[str, int] = {}  # slot name -> masked address
        seen_ctrl_done_slots = set()
        for overlay in pass1_overlays:
            slot = overlay.slot
            if slot._source is None or slot.name in seen_ctrl_done_slots:
                continue
            seen_ctrl_done_slots.add(slot.name)
            resident_elf = find_core_elf(str(pass1_dir), slot.tile.col, slot.tile.row)
            resident_syms = defined_symbols(resident_elf)
            done_sym = slot._ctrl_done_buf.name
            if done_sym not in resident_syms:
                raise ProgramMemoryOverlayDesignError(
                    f"tile-sourced ProgramMemorySlot '{slot.name}': resident "
                    f"has no symbol '{done_sym}' -- its flag Buffer was not "
                    f"materialised. This should not happen if the slot was "
                    f"registered via Worker's fn_args."
                )
            local_mem_size = device.target_model.get_local_memory_size()
            ctrl_done_addr[slot.name] = resident_syms[done_sym] & (
                local_mem_size - 1
            )
            imports_used.setdefault(slot.name, set()).add(done_sym)

        # The pingpong bootstrap park(s), if any: compiled from generated
        # source, linked the same way, and its parked-Buffer address read
        # back for pass 2's npu_maskpoll.
        pass1_bootstraps = _bootstraps_of(pass1_overlays)
        bootstrap_words: dict[str, list[int]] = {}
        bootstrap_parked_addr: dict[str, int] = {}
        bootstrap_imports: dict[str, set[str]] = {}
        for name, bootstrap in pass1_bootstraps.items():
            resident_elf = find_core_elf(
                str(pass1_dir), bootstrap.tile.col, bootstrap.tile.row
            )
            obj = _compile_bootstrap_source(bootstrap, pass1_dir)
            linked = str(pass1_dir / f"{name}_stub.linked")
            try:
                link(
                    objects=[obj],
                    resident=resident_elf,
                    slot_base=bootstrap.base,
                    slot_size=bootstrap.size,
                    output=linked,
                    entry=bootstrap.entry_symbol,
                    col=bootstrap.tile.col,
                    row=bootstrap.tile.row,
                )
            except OverlayError as exc:
                raise ProgramMemoryOverlayDesignError(
                    f"pingpong bootstrap park '{name}': {exc}"
                ) from exc
            bootstrap_words[name] = text_words(linked)
            with open(f"{linked}.imports") as f:
                bootstrap_imports[name] = {s for s in f.read().split("\n") if s}
            resident_syms = defined_symbols(resident_elf)
            parked_sym = bootstrap.parked.name
            if parked_sym not in resident_syms:
                raise ProgramMemoryOverlayDesignError(
                    f"pingpong bootstrap park '{name}': resident has no "
                    f"symbol '{parked_sym}' -- its parked Buffer was not "
                    f"materialised. This should not happen if the Buffer was "
                    f"registered via Worker's fn_args."
                )
            # Tile-relative offset: the core's own view of its data memory
            # starts at a whole multiple of the local memory size (e.g.
            # 0x70000 for a 0x10000-byte local memory on npu2), so masking
            # off that multiple gives what npu.maskpoll actually wants.
            # Matches pmlib/pm.py's identical mask -- getMemInternalBaseAddress()
            # would give this directly but has no Python binding.
            local_mem_size = device.target_model.get_local_memory_size()
            bootstrap_parked_addr[name] = resident_syms[parked_sym] & (
                local_mem_size - 1
            )

        pass2_dir = work_dir / "pass2"
        pass2_dir.mkdir(exist_ok=True)
        pass2_program, pass2_overlays = self._make_design()
        pass2_seen = set()
        for overlay in pass2_overlays:
            key = (overlay.slot.name, overlay.name)
            pass2_seen.add(key)
            if key not in words_by_key:
                raise ProgramMemoryOverlayDesignError(
                    f"ProgramMemoryOverlay '{overlay.name}' (slot "
                    f"'{overlay.slot.name}') was returned by make_design() on "
                    f"the pass-2 call but not pass 1; make_design must build "
                    f"an identical set of overlays on every call."
                )
            overlay.slot._set_pass2_payload(overlay.name, words_by_key[key])
            if overlay.slot.name in ctrl_done_addr:
                overlay.slot._set_ctrl_done_addr(ctrl_done_addr[overlay.slot.name])
        missing_in_pass2 = set(words_by_key) - pass2_seen
        if missing_in_pass2:
            raise ProgramMemoryOverlayDesignError(
                f"make_design() returned overlay(s) {sorted(missing_in_pass2)} "
                f"on the pass-1 call but not pass 2; make_design must build an "
                f"identical set of overlays on every call."
            )

        pass2_bootstraps = _bootstraps_of(pass2_overlays)
        if set(pass2_bootstraps) != set(pass1_bootstraps):
            raise ProgramMemoryOverlayDesignError(
                f"pingpong bootstrap park(s) {sorted(pass1_bootstraps)} on "
                f"pass 1 but {sorted(pass2_bootstraps)} on pass 2; "
                f"make_design must build an identical set of pingpong() "
                f"pairs on every call."
            )
        for name, bootstrap in pass2_bootstraps.items():
            bootstrap.pass2_words = bootstrap_words[name]
            bootstrap.parked_addr = bootstrap_parked_addr[name]

        compile_mlir_module(
            pass2_program.resolve_program(),
            xclbin_path=xclbin_path,
            insts_path=insts_path,
            work_dir=str(pass2_dir),
            device=device,
            **compile_kwargs,
        )

        checked_tiles = set()
        for overlay in pass2_overlays:
            slot = overlay.slot
            tile_key = (slot.tile.col, slot.tile.row)
            if tile_key in checked_tiles:
                continue
            checked_tiles.add(tile_key)
            resident1 = find_core_elf(str(pass1_dir), slot.tile.col, slot.tile.row)
            resident2 = find_core_elf(str(pass2_dir), slot.tile.col, slot.tile.row)
            all_imports = set(imports_used.get(slot.name, set()))
            bootstrap_obj = slot._pingpong[2][0] if slot._pingpong is not None else None
            if bootstrap_obj is not None:
                all_imports |= bootstrap_imports.get(bootstrap_obj.name, set())
            _check_resident_stability(resident1, resident2, all_imports, slot.name)

        return xclbin_path, insts_path


def _check_stack_budget(resident_elf, overlay_elf, slot, overlay, device) -> None:
    """Raise unless `overlay`'s deepest frame fits the stack its Worker was linked with.

    The resident's stack is sized once, when the resident links -- and the
    call into a slot goes through an absolute address, so the compiler cannot
    see the overlay to do interprocedural stack analysis even in principle.
    An overlay that needs more overruns into whatever sits below the stack:
    no fault, just scattered wrong values in another buffer, and in an
    overlay design the damage outlives the phase that caused it. This is the
    same check as `pm.py stack`, against the resident and overlay this build
    actually produced rather than hand-supplied objects.
    """
    worker = slot._worker
    if worker is None:
        raise ProgramMemoryOverlayDesignError(
            f"ProgramMemorySlot '{slot.name}': not registered with any Worker "
            f"(pass it in a Worker's fn_args), so its stack budget is unknown."
        )
    budget = worker.stack_size
    if budget is None:
        budget = device.target_model.get_default_core_stack_size()

    resident_frame = max_stack_frame(resident_elf)
    overlay_frame = max_stack_frame(overlay_elf)
    missing = [
        p for p, f in ((resident_elf, resident_frame), (overlay_elf, overlay_frame)) if f is None
    ]
    if missing:
        raise ProgramMemoryOverlayDesignError(
            f"ProgramMemoryOverlay '{overlay.name}' (slot '{slot.name}'): no "
            f".stack_sizes in {', '.join(missing)}; this should not happen "
            f"for an aiecc-built resident or an overlay compiled with "
            f"-fstack-size-section, and this check silently measures nothing "
            f"without it."
        )

    need = resident_frame + overlay_frame
    if need > budget:
        raise ProgramMemoryOverlayDesignError(
            f"ProgramMemoryOverlay '{overlay.name}' (slot '{slot.name}'): "
            f"needs {need} bytes of stack (resident frame {resident_frame} + "
            f"overlay frame {overlay_frame}) but its Worker was linked with "
            f"a {budget}-byte stack_size. Raise the Worker's stack_size: "
            f"overrunning it corrupts whatever is below the stack, without a "
            f"fault, and in an overlay design the damage outlives the phase "
            f"that caused it."
        )


def _check_resident_stability(
    pass1_elf: str, pass2_elf: str, imports_used: set, slot_name: str
) -> None:
    """Raise if a resident symbol an overlay depends on moved or disappeared.

    The failure mode this guards: pass 2 embeds payload bytes pass 1 did not
    have, which can shift the resident's own layout if anything about the
    design depends on payload size or content -- an overlay linked against
    pass 1's addresses would then be calling into the wrong place, or a
    symbol it imports would no longer exist at all. Neither is something
    aiecc itself would ever flag: both produce a design that builds and
    loads, then behaves oddly at run time.
    """
    syms1 = defined_symbols(pass1_elf)
    syms2 = defined_symbols(pass2_elf)
    for name in imports_used:
        if name not in syms2:
            raise ProgramMemoryOverlayDesignError(
                f"slot '{slot_name}': resident symbol '{name}', which an "
                f"overlay imports, exists in pass 1 but not pass 2. An "
                f"overlay linked against pass 1 would be calling into "
                f"nothing."
            )
        if name in syms1 and syms1[name] != syms2[name]:
            raise ProgramMemoryOverlayDesignError(
                f"slot '{slot_name}': resident symbol '{name}' moved between "
                f"pass 1 (0x{syms1[name]:x}) and pass 2 (0x{syms2[name]:x}). "
                f"An overlay linked against pass 1's address would be calling "
                f"into the wrong place."
            )
