# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The one IRON design every overlay test builds, parameterized by a Config.

One design rather than several because the interesting axes -- how many slots,
which overlay runs in which phase, how big the resident is, which defect is
injected -- are orthogonal, and a test that reads `--recipe one_slot --phases
0,1,2,1` says what it covers more clearly than one that carries six numbers.

Defects are injected at *generation* time (`corrupt`, `skip_write`,
`wrong_address`) rather than patched into the built artifact afterwards. That is
deliberate: locating a payload in a finished instruction stream needs a
multi-word signature, breaks whenever codegen shifts, and can land on padding
that never reaches the output, turning a negative control into a coin flip. Here
the word being corrupted is known exactly.
"""

from dataclasses import dataclass, field

from ml_dtypes import bfloat16
import numpy as np

from aie.dialects.aie import T
import aie.dialects.arith as arith
from aie.dialects.aiex import npu_blockwrite, npu_maskpoll, npu_preempt
import aie.dialects.memref as memref
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.device import AnyShimTile, Tile, from_name
from aie.ir import DenseElementsAttr, InsertionPoint, MemRefType, TypeAttr

from .geometry import Geometry

ENTRY = "overlay_entry"


def entry_name(geometry, i):
    """The symbol the core calls for slot i.

    A slot's address reaches the linker only as a symbol, so two slots need two
    symbols and two call sites. They cannot share one: slot.ld supplies absolute
    addresses, not a table, so `call slots[k % 2]` would need an indirect jump
    through something that does not exist. Unrolling the phase loop by the slot
    count is what keeps both call sites direct.
    """
    return ENTRY if len(geometry.slots) == 1 else f"{ENTRY}_{geometry.slots[i].name}"


# Written into every slot during setup so that a phase whose payload never
# arrives produces this, deterministically, instead of whatever the previous
# xclbin left in program memory. npu-xrt runs serialized, so without it the
# negative controls depend on which test ran last.
POISON_TAG = 0x7BAD


@dataclass
class Config:
    geometry: Geometry
    n_elems: int = 256
    # Real AIE kernels mostly work in bfloat16, and the tile type has to match
    # what they were compiled for. The dummy workloads use int32 because a
    # scalar byte loop is miscompiled on the pinned Peano and int32 stores are
    # not -- see ../peano_scalar_store_canary.
    dtype: str = "i32"
    # Off only for the test that exercises the whole-program-memory overflow.
    reserve: bool = True
    # phase -> overlay index. Permutation, replay and "run one twice" all fall
    # out of this one knob rather than three code paths.
    phases: tuple = (0,)
    # Payload words per overlay index, filled in by the caller from the linked
    # ELFs. Empty means pass 1: build the resident with no payloads at all.
    payloads: tuple = ()
    poison: tuple = ()
    # Defect injection, all generation-time.
    corrupt: tuple = ()  # (phase, word_index)
    skip_write: tuple = ()  # phase indices
    wrong_address: tuple = ()  # (phase, byte_delta)
    # (phase, level): emit aiex.npu.preempt before this phase's release.
    # Whether program memory survives a yield is a firmware property
    # that nothing in this tree documents -- aie-rt only records the
    # level in the transaction -- so it has to be measured.
    preempt: tuple = ()
    # (phase, address, value, mask): block the sequence until the core
    # has written something. The only way the host can wait on a core
    # rather than on a DMA, which is what a second overlay slot needs.
    maskpoll_before: tuple = ()
    maskpoll_after: tuple = ()

    @property
    def slot(self):
        return self.geometry.slots[0]


def _payload_global(index, words, suffix=""):
    """Emit the overlay's bytes as a module-scope memref.global.

    IRON has no verb for this and Program verifies the module before returning
    it, so the global has to be placed while the sequence body is still being
    built: walk out to the enclosing aie.device.
    """
    memref_ty = MemRefType.get([len(words)], T.i32())
    sym = f"overlay_{index}{suffix}"
    device = InsertionPoint.current.block.owner.operation.parent
    with InsertionPoint.at_block_begin(device.regions[0].blocks[0]):
        memref.global_(
            sym,
            TypeAttr.get(memref_ty),
            sym_visibility="private",
            constant=True,
            initial_value=DenseElementsAttr.get(
                np.array(words, dtype=np.uint32).view(np.int32)
            ),
        )
    return memref_ty, sym


def build(cfg):
    """Build the design and return the resolved MLIR module."""
    cfg.geometry.validate()
    g = cfg.geometry
    col, row = g.tile
    n_phases = len(cfg.phases)

    elem = np.dtype[bfloat16] if cfg.dtype == "bf16" else np.dtype[np.int32]
    tile_ty = np.ndarray[(cfg.n_elems,), elem]
    word_ty = np.ndarray[(1,), np.dtype[np.int32]]
    host_in_ty = np.ndarray[(cfg.n_elems,), elem]
    host_out_shape = (n_phases, cfg.n_elems)
    host_out_ty = np.ndarray[host_out_shape, elem]

    compute_tile = Tile(col, row)

    # resident.o supplies the wait loop; slot.ld supplies nothing but the
    # address of ENTRY, so the core's call compiles to a direct jump into the
    # slot and the body turns up at run time.
    ovl_wait = Kernel("ovl_wait", "resident.o", [word_ty])
    entries = [
        Kernel(entry_name(g, i), "slot.ld", [tile_ty, tile_ty])
        for i in range(len(g.slots))
    ]

    flag = Buffer(
        word_ty,
        initial_value=np.array([0], dtype=np.int32),
        name="flag",
        tile=compute_tile,
        use_write_rtp=True,
    )

    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_fn(in_cons, out_prod, flag, ovl_wait, *entries):
        # A plain Python loop, not range_(): range_() emits an scf.for whose
        # size depends on whether LLVM's unroller expands it, which caps the
        # resident around 1.5 KB and then collapses back to ~900 bytes. Emitting
        # the bodies straight-line makes the resident's size linear in the phase
        # count -- about 80 bytes each -- and so a usable knob.
        for phase in range(n_phases):
            ovl_wait(flag)
            a = in_cons.acquire(1)
            c = out_prod.acquire(1)
            # Phases alternate between slots, so with two slots the loop is
            # effectively unrolled by two and each call site is a direct jump to
            # its own slot.
            entries[phase % len(entries)](a, c)
            in_cons.release(1)
            out_prod.release(1)

    # Everything from the lowest slot upward belongs to code written at run
    # time. Declaring it shortens the linker's program region, so a resident
    # that grew into a slot is an ordinary link error naming the section and the
    # overrun -- rather than an ASSERT smuggled in through a link_with fragment,
    # which is what this used to rely on.
    reserved = (
        g.program_memory_size - min(s.base for s in g.slots) if cfg.reserve else None
    )
    worker = Worker(
        core_fn,
        [of_in.cons(), of_out.prod(), flag, ovl_wait, *entries],
        tile=compute_tile,
        while_true=False,
        program_memory_reserved=reserved,
    )

    def write_payload(words, slot_base, index, suffix="", delta=0):
        memref_ty, sym = _payload_global(index, words, suffix)
        npu_blockwrite(
            g.host_offset + slot_base + delta,
            memref.get_global(memref_ty, sym),
            column=col,
            row=row,
        )

    def sequence(host_in, host_out, in_prod, out_cons):
        # Poison every slot before the core can reach one, so a phase whose
        # payload never lands fails the same way every time.
        for i, s in enumerate(g.slots):
            if cfg.poison:
                write_payload(cfg.poison, s.base, i, suffix="_poison")

        for phase, ovl in enumerate(cfg.phases):
            if cfg.payloads and phase not in cfg.skip_write:
                delta = dict(cfg.wrong_address).get(phase, 0)
                words = list(cfg.payloads[ovl])
                for bad_phase, word in cfg.corrupt:
                    if bad_phase == phase:
                        words[word] ^= 0xFFFFFFFF
                slot = g.slots[phase % len(g.slots)]
                write_payload(words, slot.base, ovl, f"_p{phase}", delta)

            for p, addr, val, mask in cfg.maskpoll_before:
                if p == phase:
                    npu_maskpoll(
                        arith.constant(T.i32(), addr),
                        arith.constant(T.i32(), val),
                        arith.constant(T.i32(), mask),
                        column=col,
                        row=row,
                    )

            for p, level in cfg.preempt:
                if p == phase:
                    npu_preempt(level)

            # Only now let the core out of ovl_wait. The blockwrite above is
            # ordered before this in the instruction stream, so the slot holds
            # this phase's kernel by the time the core jumps into it.
            flag[0] = 1

            # After the release: waits on the core having reacted, which is the
            # only kind of wait a runtime sequence cannot otherwise express.
            for p, addr, val, mask in cfg.maskpoll_after:
                if p == phase:
                    npu_maskpoll(
                        arith.constant(T.i32(), addr),
                        arith.constant(T.i32(), val),
                        arith.constant(T.i32(), mask),
                        column=col,
                        row=row,
                    )

            tg = TaskGroup()
            in_prod.fill(host_in, group=tg)
            out_cons.drain(
                host_out,
                TensorAccessPattern(
                    host_out_shape,
                    phase * cfg.n_elems,
                    [1, 1, 1, cfg.n_elems],
                    [0, 0, 0, 1],
                ),
                wait=True,
                group=tg,
            )
            tg.finish()

    rt = Runtime(
        sequence,
        [
            host_in_ty,
            host_out_ty,
            of_in.prod(tile=AnyShimTile),
            of_out.cons(tile=AnyShimTile),
        ],
    )
    return Program(
        from_name(g.dev, n_cols=max(2, col + 1)), rt, workers=[worker]
    ).resolve_program()


def emit_slot_ld(geometry, path, entry=ENTRY, assert_budget=False):
    """The one place a slot's address is stated to the linker.

    Rides into the resident's link as a link_files entry, which the generated ld
    script turns into an INPUT() -- and ld.lld parses an input it does not
    recognise as a linker script.

    Symbol assignments only. The resident-must-not-grow-into-the-slot rule used
    to be an ASSERT in here too, which meant the one thing standing between a
    growing resident and the core overwriting its own code was a side effect of
    how lld treats an unrecognised input. That is now the core's
    program_memory_reserved attribute, so the linker enforces it directly.
    assert_budget re-adds the ASSERT, for the test that shows the two
    mechanisms catching the same overrun.
    """
    with open(path, "w") as f:
        f.write(
            "/* Generated by pmlib/design.py. The resident calls each of these at\n"
            "   a fixed address; the bodies are written there at run time. */\n"
        )
        for i, s in enumerate(geometry.slots):
            f.write(f"{entry_name(geometry, i)} = 0x{s.base:x};\n")
        # Omitted only so a test can reach the *program-memory region* overflow
        # underneath: this ASSERT is the tighter of the two guards and would
        # otherwise always fire first.
        if assert_budget:
            f.write(
                f"ASSERT(SIZEOF(.text) <= 0x{geometry.resident_budget:x},\n"
                f'       "resident .text has grown into the overlay slot")\n'
            )
