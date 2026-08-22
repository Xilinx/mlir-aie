# pm_write_while_running/aie2.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Can an AIE core's program memory be rewritten while the core is enabled and
# fetching? See README.md for the full matrix and the measured answer.
#
# Two independent axes:
#   --variant  what the core is doing when the write lands (A no write,
#              B running, C debug-halted, D disabled, E ECC-bypass alias,
#              F stalled on a lock)
#   --pair     how far the write lands from the program counter, in bytes
#
# Distance is what actually decides the outcome, so the interesting cases are
# B at the largest distance (a running core, written far away) versus B at the
# smallest. The core calls every pair each round and reports one word each, so
# whichever pairs a run does not patch are controls within that same run.
#
# Run twice. The first pass (no --elf) emits the design, which aiecc compiles
# into a core ELF. The second pass (--elf <tmpdir>) reads the chosen pair out of
# that ELF and emits the same design plus a program-memory patch aimed at that
# pair's address. The second build recompiles the core rather than reusing the
# first ELF, so each case is followed by `overlay_elf.py --check`, which fails if
# the core moved and the patch would land on the wrong address.
#
# Only cases that are deterministic on hardware are asserted here. Writes close
# to the program counter land about half the time; --pair still builds them so
# the sweep in README.md can be reproduced.
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
# RUN: %run_on_npu1% %python %S/aie2.py --dev npu1 --variant A --pair 64 --elf p1 --out final_A.mlir
# RUN: %run_on_npu2% %python %S/aie2.py --dev npu2 --variant A --pair 64 --elf p1 --out final_A.mlir
# RUN: %aiecc --tmpdir=pA --get-xclbin --xclbin-name=aieA.xclbin --get-npu-insts --npu-insts-name=instsA.bin ./final_A.mlir
# RUN: %python %S/overlay_elf.py --check p1 pA
# RUN: %run_on_npu1% env PM_PATCHED_DIST=-1 ./test.exe -x aieA.xclbin -k MLIR_AIE -i instsA.bin
# RUN: %run_on_npu2% env PM_PATCHED_DIST=-1 ./test.exe -x aieA.xclbin -k MLIR_AIE -i instsA.bin
#
# RUN: %run_on_npu1% %python %S/aie2.py --dev npu1 --variant C --pair 64 --elf p1 --out final_Cn.mlir
# RUN: %run_on_npu2% %python %S/aie2.py --dev npu2 --variant C --pair 64 --elf p1 --out final_Cn.mlir
# RUN: %aiecc --tmpdir=pCn --get-xclbin --xclbin-name=aieCn.xclbin --get-npu-insts --npu-insts-name=instsCn.bin ./final_Cn.mlir
# RUN: %python %S/overlay_elf.py --check p1 pCn
# RUN: %run_on_npu1% env PM_PATCHED_DIST=64 ./test.exe -x aieCn.xclbin -k MLIR_AIE -i instsCn.bin
# RUN: %run_on_npu2% env PM_PATCHED_DIST=64 ./test.exe -x aieCn.xclbin -k MLIR_AIE -i instsCn.bin
#
# RUN: %run_on_npu1% %python %S/aie2.py --dev npu1 --variant D --pair 64 --elf p1 --out final_Dn.mlir
# RUN: %run_on_npu2% %python %S/aie2.py --dev npu2 --variant D --pair 64 --elf p1 --out final_Dn.mlir
# RUN: %aiecc --tmpdir=pDn --get-xclbin --xclbin-name=aieDn.xclbin --get-npu-insts --npu-insts-name=instsDn.bin ./final_Dn.mlir
# RUN: %python %S/overlay_elf.py --check p1 pDn
# RUN: %run_on_npu1% env PM_PATCHED_DIST=64 ./test.exe -x aieDn.xclbin -k MLIR_AIE -i instsDn.bin
# RUN: %run_on_npu2% env PM_PATCHED_DIST=64 ./test.exe -x aieDn.xclbin -k MLIR_AIE -i instsDn.bin
#
# RUN: %run_on_npu1% %python %S/aie2.py --dev npu1 --variant F --pair 64 --elf p1 --out final_Fn.mlir
# RUN: %run_on_npu2% %python %S/aie2.py --dev npu2 --variant F --pair 64 --elf p1 --out final_Fn.mlir
# RUN: %aiecc --tmpdir=pFn --get-xclbin --xclbin-name=aieFn.xclbin --get-npu-insts --npu-insts-name=instsFn.bin ./final_Fn.mlir
# RUN: %python %S/overlay_elf.py --check p1 pFn
# RUN: %run_on_npu1% env PM_PATCHED_DIST=64 ./test.exe -x aieFn.xclbin -k MLIR_AIE -i instsFn.bin
# RUN: %run_on_npu2% env PM_PATCHED_DIST=64 ./test.exe -x aieFn.xclbin -k MLIR_AIE -i instsFn.bin
#
# RUN: %run_on_npu1% %python %S/aie2.py --dev npu1 --variant B --pair 8320 --elf p1 --out final_Bf.mlir
# RUN: %run_on_npu2% %python %S/aie2.py --dev npu2 --variant B --pair 8320 --elf p1 --out final_Bf.mlir
# RUN: %aiecc --tmpdir=pBf --get-xclbin --xclbin-name=aieBf.xclbin --get-npu-insts --npu-insts-name=instsBf.bin ./final_Bf.mlir
# RUN: %python %S/overlay_elf.py --check p1 pBf
# RUN: %run_on_npu1% env PM_PATCHED_DIST=8320 ./test.exe -x aieBf.xclbin -k MLIR_AIE -i instsBf.bin
# RUN: %run_on_npu2% env PM_PATCHED_DIST=8320 ./test.exe -x aieBf.xclbin -k MLIR_AIE -i instsBf.bin
#
# RUN: %run_on_npu1% %python %S/aie2.py --dev npu1 --variant C --pair 8320 --elf p1 --out final_Cf.mlir
# RUN: %run_on_npu2% %python %S/aie2.py --dev npu2 --variant C --pair 8320 --elf p1 --out final_Cf.mlir
# RUN: %aiecc --tmpdir=pCf --get-xclbin --xclbin-name=aieCf.xclbin --get-npu-insts --npu-insts-name=instsCf.bin ./final_Cf.mlir
# RUN: %python %S/overlay_elf.py --check p1 pCf
# RUN: %run_on_npu1% env PM_PATCHED_DIST=8320 ./test.exe -x aieCf.xclbin -k MLIR_AIE -i instsCf.bin
# RUN: %run_on_npu2% env PM_PATCHED_DIST=8320 ./test.exe -x aieCf.xclbin -k MLIR_AIE -i instsCf.bin
#
# A realistically sized load: 4 KB written into the other half while the core
# runs, rather than the 32-byte poke every other case uses.
# RUN: %run_on_npu1% %python %S/aie2.py --dev npu1 --variant B --pair 8320 --block 4096 --elf p1 --out final_Blk.mlir
# RUN: %run_on_npu2% %python %S/aie2.py --dev npu2 --variant B --pair 8320 --block 4096 --elf p1 --out final_Blk.mlir
# RUN: %aiecc --tmpdir=pBlk --get-xclbin --xclbin-name=aieBlk.xclbin --get-npu-insts --npu-insts-name=instsBlk.bin ./final_Blk.mlir
# RUN: %python %S/overlay_elf.py --check p1 pBlk
# RUN: %run_on_npu1% env PM_PATCHED_DIST=8320 ./test.exe -x aieBlk.xclbin -k MLIR_AIE -i instsBlk.bin
# RUN: %run_on_npu2% env PM_PATCHED_DIST=8320 ./test.exe -x aieBlk.xclbin -k MLIR_AIE -i instsBlk.bin

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
    PAIR_DISTANCES,
    PROG_MEM_BASE,
    PROG_MEM_ECC_BYPASS_BASE,
    find_core_elf,
    overlay_block,
    overlay_pair,
)

CORE_COL, CORE_ROW = 0, 2
# One output word per overlay pair. A run patches exactly one of them, so the
# rest are controls in every single run.
BATCH = len(PAIR_DISTANCES)
ROUNDS = 2
PATCH_SYM = "pm_patch"

VARIANTS = {
    "A": "negative control: no write at all, round 1 must still read 7",
    "B": "write while the core is enabled and spinning in ovl_wait (fetching)",
    "C": "as B, but bracketed by a debug halt (XAie_CoreDebugHalt)",
    "D": "as B, but bracketed by a core disable/enable (CORE_CONTROL bit 0)",
    "E": "as B, but through the ECC-bypass alias at 0x24000",
    "F": "write while the core is enabled but stalled on a lock acquire",
}

OVL_OBJ = "ovl.o"  # built from ovl.cc by the RUN lines above

# The lit RUN lines have to be `#` comments, so the module has no docstring for
# argparse to pick up.
DESCRIPTION = """\
Emit the program-memory-write experiment as MLIR.

The core runs two rounds, spinning in ovl_wait() each round until the host
releases it, then calling every sel_dN_a() and reporting one word each. All read
7 unpatched. Between the rounds the runtime sequence overwrites one pair's
program memory with its partner's bytes, so that word reads 9 if the write took
effect and the others are controls. --variant selects what the core is doing at
that moment; --pair selects how far the write lands from the program counter.
See README.md."""


def build(dev, variant, pair, spin, block, elf):
    """Build the design with IRON and return the resolved MLIR module.

    Args:
        dev: the target Device.
        variant: key into VARIANTS, or None for pass 1 (no patch emitted).
        pair: which PAIR_DISTANCES entry to patch, i.e. how far the write lands
            from ovl_wait.
        spin: "" for the spin loop at the top of .text, "_lo" for the one at the
            bottom. Moves the program counter without moving any pair.
        block: patch this many bytes of program memory around the pair instead of
            just the pair's 32, to time a realistically sized overlay load. 0
            patches the pair alone.
        elf: path to the pass-1 core ELF the patch is derived from, or None.
    """
    # The core calls every pair each round and reports one word each, so a
    # single build serves every (variant, distance) combination and the pairs
    # that were not patched act as controls within the same run.
    if elf:
        victim, donor = f"sel_d{pair}_a", f"sel_d{pair}_b"
        patch = (
            overlay_block(elf, victim, donor, block)
            if block
            else overlay_pair(elf, victim, donor)
        )
    else:
        patch = None
    i32 = np.dtype[np.int32]
    host_shape = (ROUNDS, BATCH)  # one row per round
    host_ty = np.ndarray[host_shape, i32]
    batch_ty = np.ndarray[(BATCH,), i32]
    word_ty = np.ndarray[(1,), i32]

    compute_tile = Tile(CORE_COL, CORE_ROW)

    # Kernel, not ExternalFunction: ExternalFunction's source_file is built by
    # @iron.jit, and this design is handed to aiecc directly. The RUN lines
    # compile ovl.cc to ovl.o first; both symbols live in it.
    ovl_wait = Kernel(f"ovl_wait{spin}", OVL_OBJ, [word_ty])
    sels = [Kernel(f"sel_d{d}_a", OVL_OBJ, [word_ty]) for d in PAIR_DISTANCES]

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

    def core_fn(out_prod, gate, flag, sel_out, ovl_wait, *sels):
        for _ in range_(ROUNDS):
            gate.acquire(1)
            ovl_wait(flag)
            elem = out_prod.acquire(1)
            # Plain Python enumerate, so these unroll into constant-index
            # stores rather than needing index arithmetic on an SSA value.
            for i, sel in enumerate(sels):
                sel(sel_out)
                elem[i] = sel_out[0]
            out_prod.release(1)

    worker = Worker(
        core_fn,
        [of_out.prod(), gate, flag, sel_out, ovl_wait, *sels],
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
    p.add_argument(
        "--block",
        type=int,
        default=0,
        help="patch this many bytes around the pair (0 = just the pair)",
    )
    p.add_argument(
        "--spin",
        choices=("hi", "lo"),
        default="hi",
        help="which spin loop the core waits in, i.e. where the PC sits",
    )
    p.add_argument(
        "--pair",
        type=int,
        choices=PAIR_DISTANCES,
        default=PAIR_DISTANCES[0],
        help="bytes from the program counter to the write; see README.md",
    )
    p.add_argument("--elf", help="pass-1 core ELF, or the aiecc tmpdir holding it")
    p.add_argument("--out", required=True, help="where to write the MLIR")
    args = p.parse_args()

    if bool(args.variant) != bool(args.elf):
        p.error("--variant and --elf go together: both for pass 2, neither for pass 1")

    elf = find_core_elf(args.elf, CORE_COL, CORE_ROW) if args.elf else None
    spin = "" if args.spin == "hi" else "_lo"
    module = build(
        from_name(args.dev, n_cols=1),
        args.variant,
        args.pair,
        spin,
        args.block,
        elf,
    )
    with open(args.out, "w") as f:
        print(module, file=f)


if __name__ == "__main__":
    main()
