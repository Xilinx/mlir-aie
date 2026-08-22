# program_memory_overlay/aie2.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Run three real aie_kernels -- silu, gelu, softmax -- on one core by rewriting
# its program memory between phases, so which kernel the core runs is decided at
# run time rather than at link time.
#
# Program memory is split into a resident half and an overlay slot:
#
#   0x0000  resident   the wait loop, anything overlays call back into, and the
#                      call site -- always present
#   0x2000  slot       whichever overlay has most recently been written here
#
# The split is not arbitrary. A configuration write to the 8 KB half the core is
# fetching from is silently discarded about half the time, while a write to the
# other half always lands (test/npu-xrt/pm_write_while_running). Parking the core
# in the resident half and writing only the slot is therefore always safe, and
# slot.ld's ASSERT fails the build if the resident ever grows into the slot.
#
# Two passes. The first emits the design with no overlay payloads and aiecc
# compiles the resident, which the overlays are then linked against (overlay.py).
# The second emits the same design with each overlay's bytes embedded as a
# memref.global that the runtime sequence writes into the slot.
#
# REQUIRES: ryzen_ai, peano
#
# RUN: %run_on_npu1% %python %S/aie2.py --dev npu1 --emit-slot-ld slot.ld --out design.mlir
# RUN: %run_on_npu2% %python %S/aie2.py --dev npu2 --emit-slot-ld slot.ld --out design.mlir
# RUN: %run_on_npu1% %PEANO_INSTALL_DIR/bin/clang++ --target=aie2-none-unknown-elf -O2 -c %S/resident.cc -o ./resident.o
# RUN: %run_on_npu2% %PEANO_INSTALL_DIR/bin/clang++ --target=aie2p-none-unknown-elf -O2 -c %S/resident.cc -o ./resident.o
# RUN: %aiecc --tmpdir=p1 --get-xclbin --xclbin-name=p1.xclbin --get-npu-insts --npu-insts-name=p1.bin ./design.mlir
#
# Each overlay: compile, then link it at the slot address against the resident.
# RUN: %run_on_npu1% %PEANO_INSTALL_DIR/bin/clang++ --target=aie2-none-unknown-elf -O2 -std=c++20 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ -I %S/../../../include -I %S/../../../third_party/aie_api/include -I %S/../../../aie_kernels -DOVL_ID=0 -c %S/kernels.cc -o ./k0.o
# RUN: %run_on_npu2% %PEANO_INSTALL_DIR/bin/clang++ --target=aie2p-none-unknown-elf -O2 -std=c++20 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ -I %S/../../../include -I %S/../../../third_party/aie_api/include -I %S/../../../aie_kernels -DOVL_ID=0 -c %S/kernels.cc -o ./k0.o
# RUN: %run_on_npu1% %PEANO_INSTALL_DIR/bin/clang++ --target=aie2-none-unknown-elf -O2 -std=c++20 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ -I %S/../../../include -I %S/../../../third_party/aie_api/include -I %S/../../../aie_kernels -DOVL_ID=1 -c %S/kernels.cc -o ./k1.o
# RUN: %run_on_npu2% %PEANO_INSTALL_DIR/bin/clang++ --target=aie2p-none-unknown-elf -O2 -std=c++20 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ -I %S/../../../include -I %S/../../../third_party/aie_api/include -I %S/../../../aie_kernels -DOVL_ID=1 -c %S/kernels.cc -o ./k1.o
# RUN: %run_on_npu1% %PEANO_INSTALL_DIR/bin/clang++ --target=aie2-none-unknown-elf -O2 -std=c++20 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ -I %S/../../../include -I %S/../../../third_party/aie_api/include -I %S/../../../aie_kernels -DOVL_ID=2 -c %S/kernels.cc -o ./k2.o
# RUN: %run_on_npu2% %PEANO_INSTALL_DIR/bin/clang++ --target=aie2p-none-unknown-elf -O2 -std=c++20 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ -I %S/../../../include -I %S/../../../third_party/aie_api/include -I %S/../../../aie_kernels -DOVL_ID=2 -c %S/kernels.cc -o ./k2.o
# RUN: %run_on_npu2% %PEANO_INSTALL_DIR/bin/clang++ --target=aie2p-none-unknown-elf -O2 -std=c++20 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ -I %S/../../../include -I %S/../../../third_party/aie_api/include -I %S/../../../aie_kernels -c %S/../../../aie_kernels/aie2p/silu.cc -o ./silu.o
# RUN: %run_on_npu2% %PEANO_INSTALL_DIR/bin/clang++ --target=aie2p-none-unknown-elf -O2 -std=c++20 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ -I %S/../../../include -I %S/../../../third_party/aie_api/include -I %S/../../../aie_kernels -c %S/../../../aie_kernels/aie2p/gelu.cc -o ./gelu.o
# RUN: %run_on_npu2% %PEANO_INSTALL_DIR/bin/clang++ --target=aie2p-none-unknown-elf -O2 -std=c++20 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ -I %S/../../../include -I %S/../../../third_party/aie_api/include -I %S/../../../aie_kernels -c %S/../../../aie_kernels/aie2p/softmax.cc -o ./softmax.o
# RUN: %python %S/overlay.py link --object k0.o --object silu.o --resident p1 --slot 0x2000 --slot-size 0x2000 --output ovl0.elf
# RUN: %python %S/overlay.py link --object k1.o --object gelu.o --resident p1 --slot 0x2000 --slot-size 0x2000 --output ovl1.elf
# RUN: %python %S/overlay.py link --object k2.o --object softmax.o --resident p1 --slot 0x2000 --slot-size 0x2000 --output ovl2.elf
#
# RUN: %run_on_npu1% %python %S/aie2.py --dev npu1 --overlays ovl0.elf,ovl1.elf,ovl2.elf --out final.mlir
# RUN: %run_on_npu2% %python %S/aie2.py --dev npu2 --overlays ovl0.elf,ovl1.elf,ovl2.elf --out final.mlir
# RUN: %aiecc --tmpdir=p2 --get-xclbin --xclbin-name=aie.xclbin --get-npu-insts --npu-insts-name=insts.bin ./final.mlir
# RUN: %python %S/overlay.py check p1 p2
# RUN: %python %S/overlay.py sizes --resident p1 --overlays ovl0.elf ovl1.elf ovl2.elf
# RUN: %host_clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags %host_link_flags %test_utils_flags
# RUN: %run_on_npu1% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin
# RUN: %run_on_npu2% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin

import argparse

from ml_dtypes import bfloat16
import numpy as np

from aie.dialects.aie import T
from aie.dialects.aiex import npu_blockwrite
import aie.dialects.memref as memref
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    TaskGroup,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import AnyShimTile, Tile, from_name
from aie.ir import DenseElementsAttr, InsertionPoint, MemRefType, TypeAttr

import overlay

CORE_COL, CORE_ROW = 0, 2

# Program memory is 16 KB, split in half. The resident owns the low half and the
# slot the high one, so a write to the slot never touches the half the core is
# fetching from.
PROG_MEM_HOST_OFFSET = 0x20000
SLOT = 0x2000
SLOT_SIZE = 0x2000

# silu_bf16 and gelu_bf16 in aie_kernels/aie2p have this baked in, so it is not
# free to choose.
N_ELEMS = 1024
# Fixed, not derived from how many overlays were passed: both passes must build
# a byte-identical resident, because the overlays are linked against pass 1's
# symbol addresses and pass 2 recompiles the core.
N_PHASES = 3
ENTRY = "overlay_entry"

DESCRIPTION = """\
Emit the program-memory overlay example as MLIR.

Pass 1 (--emit-slot-ld, no --overlays) produces the resident image. Pass 2
(--overlays) embeds each overlay's bytes and writes them into the slot between
phases."""


def build(dev, overlays):
    """Build the design with IRON and return the resolved MLIR module."""
    bf16 = np.dtype[bfloat16]
    tile_ty = np.ndarray[(N_ELEMS,), bf16]
    word_ty = np.ndarray[(1,), np.dtype[np.int32]]
    host_in_ty = np.ndarray[(N_ELEMS,), bf16]
    host_out_shape = (N_PHASES, N_ELEMS)
    host_out_ty = np.ndarray[host_out_shape, bf16]

    compute_tile = Tile(CORE_COL, CORE_ROW)

    # resident.o supplies the wait loop and whatever the overlays call back into.
    # slot.ld supplies nothing but the address of ENTRY: the core's call to it
    # compiles to a direct jump into the slot, and the body turns up at run time.
    ovl_wait = Kernel("ovl_wait", "resident.o", [word_ty])
    overlay_entry = Kernel(ENTRY, "slot.ld", [tile_ty, tile_ty, np.int32])

    flag = Buffer(
        word_ty,
        initial_value=np.array([0], dtype=np.int32),
        name="flag",
        tile=compute_tile,
        use_write_rtp=True,
    )

    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_fn(in_cons, out_prod, flag, ovl_wait, overlay_entry):
        for _ in range_(N_PHASES):
            # Park here, in the resident half, until the host has finished
            # writing the slot.
            ovl_wait(flag)
            a = in_cons.acquire(1)
            c = out_prod.acquire(1)
            overlay_entry(a, c, N_ELEMS)
            in_cons.release(1)
            out_prod.release(1)

    worker = Worker(
        core_fn,
        [of_in.cons(), of_out.prod(), flag, ovl_wait, overlay_entry],
        tile=compute_tile,
        while_true=False,
    )

    def load_overlay(index, words):
        """Write one overlay's code into the slot."""
        memref_ty = MemRefType.get([len(words)], T.i32())
        sym = f"overlay_{index}"

        # IRON has no verb for a module-scope memref.global, and Program verifies
        # the module before returning it, so place the payload while the sequence
        # body is still being built: walk out to the enclosing aie.device.
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
        npu_blockwrite(
            PROG_MEM_HOST_OFFSET + SLOT,
            memref.get_global(memref_ty, sym),
            column=CORE_COL,
            row=CORE_ROW,
        )

    def sequence(host_in, host_out, in_prod, out_cons):
        for phase in range(N_PHASES):
            if overlays:
                load_overlay(phase, overlays[phase])
            # Only now let the core out of ovl_wait: the blockwrite above is
            # ordered before this in the instruction stream, so the slot holds
            # this phase's kernel by the time the core jumps into it.
            flag[0] = 1

            tg = TaskGroup()
            in_prod.fill(host_in, group=tg)
            out_cons.drain(
                host_out,
                TensorAccessPattern(
                    host_out_shape, phase * N_ELEMS, [1, 1, 1, N_ELEMS], [0, 0, 0, 1]
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
    return Program(dev, rt, workers=[worker]).resolve_program()


def emit_slot_ld(path):
    """The one place the slot's address is stated to the linker.

    Rides into the resident's link as a link_files entry, which the generated ld
    script turns into an INPUT() -- and ld.lld parses an input it does not
    recognise as a linker script.
    """
    with open(path, "w") as f:
        f.write(
            f"/* Generated by aie2.py. The resident calls {ENTRY} at this fixed\n"
            f"   address; its body is written there at run time. */\n"
            f"{ENTRY} = 0x{SLOT:x};\n"
            f"ASSERT(SIZEOF(.text) <= 0x{SLOT:x},\n"
            f'       "resident .text has grown into the overlay slot")\n'
        )


def main():
    p = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dev", required=True, help="npu1 or npu2")
    p.add_argument("--emit-slot-ld", help="also write the slot linker fragment here")
    p.add_argument("--overlays", help="comma-separated overlay ELFs, for pass 2")
    p.add_argument("--out", required=True, help="where to write the MLIR")
    args = p.parse_args()

    if args.emit_slot_ld:
        emit_slot_ld(args.emit_slot_ld)

    payloads = None
    if args.overlays:
        payloads = [overlay.text_words(e) for e in args.overlays.split(",")]

    module = build(from_name(args.dev, n_cols=1), payloads)
    with open(args.out, "w") as f:
        print(module, file=f)


if __name__ == "__main__":
    main()
