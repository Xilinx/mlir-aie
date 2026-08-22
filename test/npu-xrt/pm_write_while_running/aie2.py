# pm_write_while_running/aie2.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Can an AIE core's program memory be rewritten while the core is enabled and
# fetching? See README.md for the variant matrix and the measured answer.
#
# Run twice. The first pass (no --elf) emits the design, which aiecc compiles
# into a core ELF. The second pass (--elf <tmpdir>) reads the chosen sel_*_a /
# sel_*_b pair out of that ELF and emits the same design plus a program-memory
# patch aimed at that pair's address. The second build recompiles the core rather than reusing the first
# ELF, so each variant is followed by `overlay_elf.py --check`, which fails if
# the core moved and the patch would land on the wrong address.
#
# Only the variants that are deterministic on hardware are checked here:
#   A  no write                       -> neither half changes
#   C  near write, core debug-halted  -> near half becomes 9
#   D  near write, core disabled      -> near half becomes 9
#   F  near write, core lock-stalled  -> near half becomes 9
#   G  far write, core running        -> far half becomes 9
#   H  far write, core debug-halted   -> far half becomes 9
# B and E write next to the program counter while the core runs, which lands only
# about half the time. Asserting either way would make this test flaky, so they
# are left out -- but --variant still builds them to reproduce the measurement.
#
# The generation steps write via --out rather than shell redirection: the
# %run_on_npuN% guard for the other device expands to "echo", and a redirect
# would clobber the file the real line just produced.
#
# REQUIRES: ryzen_ai, peano
#
# RUN: %run_on_npu1% %PEANO_INSTALL_DIR/bin/clang --target=aie2-none-unknown-elf -O2 -c %S/ovl.cc -o ./ovl.o
# RUN: %run_on_npu2% %PEANO_INSTALL_DIR/bin/clang --target=aie2p-none-unknown-elf -O2 -c %S/ovl.cc -o ./ovl.o
# RUN: %host_clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags %host_link_flags %test_utils_flags
#
# RUN: %run_on_npu1% %python %S/aie2.py --dev npu1 --out design.mlir
# RUN: %run_on_npu2% %python %S/aie2.py --dev npu2 --out design.mlir
# RUN: %aiecc --tmpdir=p1 --get-xclbin --xclbin-name=p1.xclbin --get-npu-insts --npu-insts-name=p1.bin ./design.mlir
#
# RUN: %run_on_npu1% %python %S/aie2.py --dev npu1 --variant A --elf p1 --out final_A.mlir
# RUN: %run_on_npu2% %python %S/aie2.py --dev npu2 --variant A --elf p1 --out final_A.mlir
# RUN: %aiecc --tmpdir=pA --get-xclbin --xclbin-name=aieA.xclbin --get-npu-insts --npu-insts-name=instsA.bin ./final_A.mlir
# RUN: %python %S/overlay_elf.py --check p1 pA
# RUN: %run_on_npu1% env PM_EXPECT_NEAR1=7 PM_EXPECT_FAR1=7 ./test.exe -x aieA.xclbin -k MLIR_AIE -i instsA.bin
# RUN: %run_on_npu2% env PM_EXPECT_NEAR1=7 PM_EXPECT_FAR1=7 ./test.exe -x aieA.xclbin -k MLIR_AIE -i instsA.bin
#
# RUN: %run_on_npu1% %python %S/aie2.py --dev npu1 --variant C --elf p1 --out final_C.mlir
# RUN: %run_on_npu2% %python %S/aie2.py --dev npu2 --variant C --elf p1 --out final_C.mlir
# RUN: %aiecc --tmpdir=pC --get-xclbin --xclbin-name=aieC.xclbin --get-npu-insts --npu-insts-name=instsC.bin ./final_C.mlir
# RUN: %python %S/overlay_elf.py --check p1 pC
# RUN: %run_on_npu1% env PM_EXPECT_NEAR1=9 PM_EXPECT_FAR1=7 ./test.exe -x aieC.xclbin -k MLIR_AIE -i instsC.bin
# RUN: %run_on_npu2% env PM_EXPECT_NEAR1=9 PM_EXPECT_FAR1=7 ./test.exe -x aieC.xclbin -k MLIR_AIE -i instsC.bin
#
# RUN: %run_on_npu1% %python %S/aie2.py --dev npu1 --variant D --elf p1 --out final_D.mlir
# RUN: %run_on_npu2% %python %S/aie2.py --dev npu2 --variant D --elf p1 --out final_D.mlir
# RUN: %aiecc --tmpdir=pD --get-xclbin --xclbin-name=aieD.xclbin --get-npu-insts --npu-insts-name=instsD.bin ./final_D.mlir
# RUN: %python %S/overlay_elf.py --check p1 pD
# RUN: %run_on_npu1% env PM_EXPECT_NEAR1=9 PM_EXPECT_FAR1=7 ./test.exe -x aieD.xclbin -k MLIR_AIE -i instsD.bin
# RUN: %run_on_npu2% env PM_EXPECT_NEAR1=9 PM_EXPECT_FAR1=7 ./test.exe -x aieD.xclbin -k MLIR_AIE -i instsD.bin
#
# RUN: %run_on_npu1% %python %S/aie2.py --dev npu1 --variant F --elf p1 --out final_F.mlir
# RUN: %run_on_npu2% %python %S/aie2.py --dev npu2 --variant F --elf p1 --out final_F.mlir
# RUN: %aiecc --tmpdir=pF --get-xclbin --xclbin-name=aieF.xclbin --get-npu-insts --npu-insts-name=instsF.bin ./final_F.mlir
# RUN: %python %S/overlay_elf.py --check p1 pF
# RUN: %run_on_npu1% env PM_EXPECT_NEAR1=9 PM_EXPECT_FAR1=7 ./test.exe -x aieF.xclbin -k MLIR_AIE -i instsF.bin
# RUN: %run_on_npu2% env PM_EXPECT_NEAR1=9 PM_EXPECT_FAR1=7 ./test.exe -x aieF.xclbin -k MLIR_AIE -i instsF.bin
#
# RUN: %run_on_npu1% %python %S/aie2.py --dev npu1 --variant G --elf p1 --out final_G.mlir
# RUN: %run_on_npu2% %python %S/aie2.py --dev npu2 --variant G --elf p1 --out final_G.mlir
# RUN: %aiecc --tmpdir=pG --get-xclbin --xclbin-name=aieG.xclbin --get-npu-insts --npu-insts-name=instsG.bin ./final_G.mlir
# RUN: %python %S/overlay_elf.py --check p1 pG
# RUN: %run_on_npu1% env PM_EXPECT_NEAR1=7 PM_EXPECT_FAR1=9 ./test.exe -x aieG.xclbin -k MLIR_AIE -i instsG.bin
# RUN: %run_on_npu2% env PM_EXPECT_NEAR1=7 PM_EXPECT_FAR1=9 ./test.exe -x aieG.xclbin -k MLIR_AIE -i instsG.bin
#
# RUN: %run_on_npu1% %python %S/aie2.py --dev npu1 --variant H --elf p1 --out final_H.mlir
# RUN: %run_on_npu2% %python %S/aie2.py --dev npu2 --variant H --elf p1 --out final_H.mlir
# RUN: %aiecc --tmpdir=pH --get-xclbin --xclbin-name=aieH.xclbin --get-npu-insts --npu-insts-name=instsH.bin ./final_H.mlir
# RUN: %python %S/overlay_elf.py --check p1 pH
# RUN: %run_on_npu1% env PM_EXPECT_NEAR1=7 PM_EXPECT_FAR1=9 ./test.exe -x aieH.xclbin -k MLIR_AIE -i instsH.bin
# RUN: %run_on_npu2% env PM_EXPECT_NEAR1=7 PM_EXPECT_FAR1=9 ./test.exe -x aieH.xclbin -k MLIR_AIE -i instsH.bin

import argparse

import numpy as np

from aie.dialects.aie import T
from aie.dialects.aiex import (
    npu_blockwrite,
    npu_maskwrite32,
    set_lock,
)
import aie.dialects.memref as memref
from aie.iron import (
    Buffer,
    Kernel,
    Lock,
    ObjectFifo,
    Program,
    Runtime,
    TaskGroup,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import AnyShimTile, Tile, from_name
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import (
    DenseElementsAttr,
    InsertionPoint,
    MemRefType,
    TypeAttr,
)

from overlay_elf import (
    CORE_CONTROL,
    DEBUG_CONTROL0,
    PROG_MEM_BASE,
    PROG_MEM_ECC_BYPASS_BASE,
    find_core_elf,
    overlay_pair,
)

CORE_COL, CORE_ROW = 0, 2
BATCH = 8  # words collected per round: HALF for the near pair, HALF for the far
HALF = BATCH // 2
ROUNDS = 2
PATCH_SYM = "pm_patch"

VARIANTS = {
    "A": "negative control: no write at all, round 1 must still read 7",
    "B": "write while the core is enabled and spinning in ovl_wait (fetching)",
    "C": "as B, but bracketed by a debug halt (XAie_CoreDebugHalt)",
    "D": "as B, but bracketed by a core disable/enable (CORE_CONTROL bit 0)",
    "E": "as B, but through the ECC-bypass alias at 0x24000",
    "F": "write while the core is enabled but stalled on a lock acquire",
    "G": "as B, but the write lands far from the PC -- real overlay geometry",
    "H": "control for G: same far write, but with the core debug-halted",
}

# B writes next to the spin loop; G writes thousands of bytes away. Comparing
# them separates "a config write contends with fetch anywhere" from "the core
# had already fetched the bytes being overwritten".
FAR_VARIANTS = {"G", "H"}

OVL_OBJ = "ovl.o"  # built from ovl.cc by the RUN lines above

# The lit RUN lines have to be `#` comments, so the module has no docstring for
# argparse to pick up.
DESCRIPTION = """\
Emit the program-memory-write experiment as MLIR.

The core runs two rounds, spinning in ovl_wait() each round until the host
releases it, then calling both sel_near_a() and sel_far_a() and reporting them in
separate halves of the output. Both read 7 unpatched. Between the rounds the
runtime sequence overwrites one pair's program memory with its partner's bytes,
so that half reads 9 if the write took effect and the other half is a control.
The variant selects the core's state and how far the write lands from the
program counter. See README.md."""


def build(dev, variant, elf):
    """Build the design with IRON and return the resolved MLIR module.

    Args:
        dev: the target Device.
        variant: key into VARIANTS, or None for pass 1 (no patch emitted).
        elf: path to the pass-1 core ELF the patch is derived from, or None.
    """
    # The core calls both pairs every round and reports them in separate halves
    # of the output, so one build serves every variant and a single run shows
    # the near and far cases side by side under identical conditions.
    pair = "far" if variant in FAR_VARIANTS else "near"
    patch = overlay_pair(elf, f"sel_{pair}_a", f"sel_{pair}_b") if elf else None
    i32 = np.dtype[np.int32]
    host_shape = (ROUNDS, BATCH)  # one row per round
    host_ty = np.ndarray[host_shape, i32]
    batch_ty = np.ndarray[(BATCH,), i32]
    word_ty = np.ndarray[(1,), i32]

    compute_tile = Tile(CORE_COL, CORE_ROW)

    # Kernel, not ExternalFunction: ExternalFunction's source_file is built by
    # @iron.jit, and this design is handed to aiecc directly. The RUN lines
    # compile ovl.cc to ovl.o first; both symbols live in it.
    ovl_wait = Kernel("ovl_wait", OVL_OBJ, [word_ty])
    sel_near = Kernel("sel_near_a", OVL_OBJ, [word_ty])
    sel_far = Kernel("sel_far_a", OVL_OBJ, [word_ty])

    # Host-driven, one release per round. Holding it back is what parks the core
    # on a lock acquire instead of in ovl_wait's fetch loop.
    gate = Lock(compute_tile, init=0, name="gate")
    # Explicitly zeroed rather than left to .bss: if flag came up non-zero the
    # core would fall straight through round 0's spin and desynchronize from the
    # host's batch boundaries.
    # use_write_rtp makes `flag[0] = 1` in the sequence body emit
    # aiex.npu.rtp_write rather than a core-side store.
    flag = Buffer(
        word_ty,
        initial_value=np.array([0], dtype=np.int32),
        name="flag",
        tile=compute_tile,
        use_write_rtp=True,
    )
    sel_out = Buffer(word_ty, name="sel_out", tile=compute_tile)

    of_out = ObjectFifo(batch_ty, name="out")

    def core_fn(out_prod, gate, flag, sel_out, ovl_wait, sel_near, sel_far):
        for _ in range_(ROUNDS):
            gate.acquire(1)
            ovl_wait(flag)
            elem = out_prod.acquire(1)
            # Lower half reports the near pair, upper half the far pair. The
            # indices are plain Python ints, so these unroll into constant
            # stores rather than needing index arithmetic on an SSA value.
            for sel, base in ((sel_near, 0), (sel_far, HALF)):
                sel(sel_out)
                v = sel_out[0]
                for i in range(HALF):
                    elem[base + i] = v
            out_prod.release(1)

    worker = Worker(
        core_fn,
        [of_out.prod(), gate, flag, sel_out, ovl_wait, sel_near, sel_far],
        tile=compute_tile,
        while_true=False,
    )

    def release_gate():
        """Let the core past the lock acquire and into ovl_wait's fetch loop."""
        set_lock(gate.op, 1)

    def release_flag():
        """Let the core out of ovl_wait and into the sel_* calls.

        This must come after the write in every variant: it is what turns the
        core loose on the patched code, so arming it early lets round 1 run
        the sel_* calls before the write arrives and every variant reports a false
        negative.
        """
        flag[0] = 1

    def blockwrite(base):
        """Overwrite the selected pair's program memory with its partner's bytes.

        Called at most once per design -- a second call would emit a duplicate
        PATCH_SYM global.
        """
        sel_a_addr, words = patch
        memref_ty = MemRefType.get([len(words)], T.i32())

        # IRON has no verb for a module-scope memref.global, and Program verifies
        # the module before handing it back, so the payload has to be placed
        # while the sequence body is still being built. Walk out of the sequence
        # to the enclosing aie.device and put it at the top there.
        device = InsertionPoint.current.block.owner.operation.parent
        with InsertionPoint.at_block_begin(device.regions[0].blocks[0]):
            memref.global_(
                PATCH_SYM,
                TypeAttr.get(memref_ty),
                sym_visibility="private",
                constant=True,
                initial_value=DenseElementsAttr.get(
                    np.array(words, dtype=np.uint32).view(np.int32)
                ),
            )

        data = memref.get_global(memref_ty, PATCH_SYM)
        npu_blockwrite(base + sel_a_addr, data, column=CORE_COL, row=CORE_ROW)

    def set_ctrl_bit0(reg, value):
        """Write bit 0 of one of the core's control registers, preserving the rest."""
        npu_maskwrite32(reg, value, 1, column=CORE_COL, row=CORE_ROW)

    def round_tap(round_idx):
        """The one row of the host tensor this round writes."""
        return TensorAccessPattern(
            host_shape, round_idx * BATCH, [1, 1, 1, BATCH], [0, 0, 0, 1]
        )

    def sequence(host, out_cons):
        # Each round drains under its own TaskGroup so the await lands between
        # the rounds. Left to the Runtime's default group the awaits are flushed
        # at the end of the sequence, and round 1 -- and the patch -- would run
        # before round 0 was ever collected.

        # Round 0: unpatched. Arm the core; both halves must read 7.
        release_gate()
        release_flag()
        tg0 = TaskGroup()
        out_cons.drain(host, round_tap(0), wait=True, group=tg0)
        tg0.finish()

        # The drain above released the ObjectFifo's producer lock, so the core is
        # parked at the top of round 1 waiting on gate. Where release_gate() sits
        # relative to the write is what selects the core's state when the write
        # lands.
        if variant == "F":
            blockwrite(PROG_MEM_BASE)
            release_gate()
        else:
            release_gate()
            if variant in ("B", "G"):
                # Identical sequences; they differ only in which pair `pair`
                # selected, i.e. how far the write lands from the PC.
                blockwrite(PROG_MEM_BASE)
            elif variant == "H":
                # Control for G: same far target, but with the core halted. If
                # this lands and G does not, G's result is about the core's
                # state, not about the far write being malformed.
                set_ctrl_bit0(DEBUG_CONTROL0, 1)
                blockwrite(PROG_MEM_BASE)
                set_ctrl_bit0(DEBUG_CONTROL0, 0)
            elif variant == "E":
                blockwrite(PROG_MEM_ECC_BYPASS_BASE)
            elif variant == "C":
                # Halt preserves the PC and all registers, unlike reset.
                set_ctrl_bit0(DEBUG_CONTROL0, 1)
                blockwrite(PROG_MEM_BASE)
                set_ctrl_bit0(DEBUG_CONTROL0, 0)
            elif variant == "D":
                set_ctrl_bit0(CORE_CONTROL, 0)
                blockwrite(PROG_MEM_BASE)
                set_ctrl_bit0(CORE_CONTROL, 1)
        release_flag()

        # Round 1: the patched half reads 9 if the write landed, 7 if not.
        tg1 = TaskGroup()
        out_cons.drain(host, round_tap(1), wait=True, group=tg1)
        tg1.finish()

    rt = Runtime(sequence, [host_ty, of_out.cons(tile=AnyShimTile)])
    module = Program(dev, rt, workers=[worker]).resolve_program()

    return module


def main():
    """Emit one pass of the design as MLIR.

    Pass 1 takes neither --variant nor --elf; pass 2 takes both.
    """
    p = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="variants:\n" + "\n".join(f"  {k}  {v}" for k, v in VARIANTS.items()),
    )
    p.add_argument("--dev", required=True, help="npu1 or npu2")
    p.add_argument("--variant", choices=sorted(VARIANTS), help="omit for pass 1")
    p.add_argument("--elf", help="pass-1 core ELF, or the aiecc tmpdir holding it")
    p.add_argument("--out", required=True, help="where to write the MLIR")
    args = p.parse_args()

    if bool(args.variant) != bool(args.elf):
        p.error("--variant and --elf go together: both for pass 2, neither for pass 1")

    elf = find_core_elf(args.elf, CORE_COL, CORE_ROW) if args.elf else None
    module = build(from_name(args.dev, n_cols=1), args.variant, elf)
    with open(args.out, "w") as f:
        print(module, file=f)


if __name__ == "__main__":
    main()
