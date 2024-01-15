# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

# RUN: VITIS_DIR=%VITIS WORKDIR=%T XRT_DIR=%XRT_DIR %PYTHON %s | FileCheck %s
# REQUIRES: xrt_python_bindings
# REQUIRES: ryzen_ai

import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from aie.extras.dialects.ext import memref, arith, func
from aie.extras.runtime.passes import run_pipeline

import aie.extras.types as T
from aie.compiler.aiecc.main import (
    generate_cores_list,
    emit_partition,
    emit_design_bif,
    emit_design_kernel_json,
    mem_topology,
    chesshack,
)
from aie.dialects import aie
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
    device,
    generate_bcf,
    generate_cdo_direct,
    ipu_instgen,
    mem,
    memtile_dma,
    tile,
    translate_mlir_to_llvmir,
    aie_llvm_link,
    dma,
)
from aie.dialects.aiex import ipu_sync, ipu_dma_memcpy_nd
from aie.dialects.scf import for_
from aie.dialects.scf import yield_
from aie.xrt import XCLBin
from util import construct_and_print_module

range_ = for_

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


VITIS_DIR = Path(os.getenv("VITIS_DIR", "/opt/tools/Xilinx/Vitis/2023.2")).absolute()
WORKDIR = Path(os.getenv("WORKDIR", str(Path(".").absolute()))).absolute()
XRT_DIR = Path(os.getenv("XRT_DIR", "/opt/xilinx/xrt")).absolute()
XILINXD_LICENSE_FILE = Path(
    os.getenv("XILINXD_LICENSE_FILE", "~/.Xilinx/aie.lic")
).absolute()

# bootgen and xclbinutil
AIETOOLS_DIR = VITIS_DIR / "aietools"
VITIS_BIN_DIR = VITIS_DIR / "bin"

LD_PATH = [
    os.getenv("LD_LIBRARY_PATH"),
    f"{AIETOOLS_DIR}/lib/lnx64.o",
    f"{AIETOOLS_DIR}/lnx64/tools/dot/lib",
    f"{XRT_DIR}/lib",
]
LD_PATH = ":".join(list(filter(None, LD_PATH)))

PATH = [
    os.getenv("PATH"),
    f"{AIETOOLS_DIR}/bin/unwrapped/lnx64.o",
    f"{AIETOOLS_DIR}/tps/lnx64/target/bin/LNa64bin",
    str(VITIS_BIN_DIR),
]
PATH = ":".join(list(filter(None, PATH)))
ENV = {
    "LD_LIBRARY_PATH": LD_PATH,
    "RDI_DATADIR": f"{AIETOOLS_DIR}/data",
    "PATH": PATH,
    "XILINXD_LICENSE_FILE": XILINXD_LICENSE_FILE,
    "XILINX_XRT": XRT_DIR,
}

XCHESS_ARGS = [
    f"{AIETOOLS_DIR}/bin/unwrapped/lnx64.o/xchesscc",
    "+P",
    "4",  # parallel compilation (function + file level)
    "-p",
    "me",  # parallel compilation (function level only)
    "-C",
    "Release_LLVM",  # configuration
    "-D__AIENGINE__",
    "-D__AIE_ARCH__=20",
    "-D__AIEARCH__=20",
    "-Y",
    f"clang={AIETOOLS_DIR}/tps/lnx64/target/bin/LNa64bin/chess-clang",
    "-P",
    f"{AIETOOLS_DIR}/data/aie_ml/lib",  # processor model directory
    "-d",  # disassemble output
    "-f",  # use LLVM frontend
    # "+f", only run LLVM frontend (emits IR)
    "+w",
    str(WORKDIR),
]


def extract_input_files(core_bcf):
    return re.findall(r"^_include _file (.*)", core_bcf, re.MULTILINE)


# CHECK-LABEL: add_256_using_dma_op_no_double_buffering
@construct_and_print_module
def add_256_using_dma_op_no_double_buffering(module):
    RANDOM_NUMBER = random.randint(0, 100)
    LEN = 128
    LOCAL_MEM_SIZE = 32

    @device(AIEDevice.ipu)
    def ipu():
        tile_0_0 = tile(0, 0)
        tile_0_1 = tile(0, 1)
        tile_0_2 = tile(0, 2)

        # in
        buffer_0_2 = aie.buffer(T.memref(LOCAL_MEM_SIZE, T.i32()), tile_0_2)
        # out
        buffer_0_2_1 = aie.buffer(T.memref(LOCAL_MEM_SIZE, T.i32()), tile_0_2)

        lock_0_1_0 = aie.lock(tile_0_1, lock_id=0, init=1)
        lock_0_1_1 = aie.lock(tile_0_1, lock_id=1, init=0)
        lock_0_1_2 = aie.lock(tile_0_1, lock_id=2, init=1)
        lock_0_1_3 = aie.lock(tile_0_1, lock_id=3, init=0)

        lock_0_2_0 = aie.lock(tile_0_2, lock_id=0, init=1)
        lock_0_2_1 = aie.lock(tile_0_2, lock_id=1, init=0)
        lock_0_2_2 = aie.lock(tile_0_2, lock_id=2, init=1)
        lock_0_2_3 = aie.lock(tile_0_2, lock_id=3, init=0)

        # input flow
        aie.flow(tile_0_0, DMA, 0, tile_0_1, DMA, 0)
        aie.flow(tile_0_1, DMA, 0, tile_0_2, DMA, 0)
        # output flow
        aie.flow(tile_0_2, DMA, 0, tile_0_1, DMA, 1)
        aie.flow(tile_0_1, DMA, 1, tile_0_0, DMA, 0)

        @aie.core(tile_0_2)
        def core():
            random_number = arith.constant(RANDOM_NUMBER)
            for _ in range_(0, LEN // LOCAL_MEM_SIZE):
                # wait on both in and out to be ready
                # these have to be acge for some reason...
                aie.use_lock(lock_0_2_1, AcquireGreaterEqual)
                aie.use_lock(lock_0_2_2, AcquireGreaterEqual)

                for arg1 in range_(0, LOCAL_MEM_SIZE):
                    v0 = memref.load(buffer_0_2, [arg1])
                    v1 = arith.addi(v0, random_number)
                    memref.store(v1, buffer_0_2_1, [arg1])
                    yield_([])

                aie.use_lock(lock_0_2_0, Release)
                aie.use_lock(lock_0_2_3, Release)

                yield_([])

        # this is gibberish - everything from here to the end of "bobsyouruncle"
        this_is_meaningless_1 = memref.global_(
            "this_is_meaningless_1",
            T.memref(1, T.f8E4M3B11FNUZ()),
            sym_visibility="public",
        ).opview
        this_is_meaningless_2 = memref.global_(
            "this_is_meaningless_2",
            T.memref(1, T.f8E4M3B11FNUZ()),
            sym_visibility="public",
        ).opview
        aie.shim_dma_allocation(this_is_meaningless_1.sym_name.value, MM2S, 0, 0)
        aie.shim_dma_allocation(this_is_meaningless_2.sym_name.value, S2MM, 0, 0)

        @func.func(emit=True)
        def bobsyouruncle(
            arg0: T.memref(LEN, T.i32()),
            _arg1: T.memref(1, T.i32()),
            arg2: T.memref(LEN, T.i32()),
        ):
            ipu_dma_memcpy_nd(
                this_is_meaningless_1.sym_name.value,
                0,
                arg0,
                [0, 0, 0, 0],
                [1, 1, 1, LEN],
                [0, 0, 0],
            )
            ipu_dma_memcpy_nd(
                this_is_meaningless_2.sym_name.value,
                1,
                arg2,
                [0, 0, 0, 0],
                [1, 1, 1, LEN],
                [0, 0, 0],
            )

            ipu_sync(channel=0, column=0, column_num=1, direction=0, row=0, row_num=1)

        # input flow
        buffer_0_1 = aie.buffer(T.memref(LOCAL_MEM_SIZE, T.i32()), tile_0_1)
        # output flow
        buffer_0_1_0 = aie.buffer(T.memref(LOCAL_MEM_SIZE, T.i32()), tile_0_1)

        @memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            # input flow
            @dma(S2MM, 0)
            def dma1():
                aie.use_lock(lock_0_1_0, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1)
                aie.use_lock(lock_0_1_1, Release)

            @dma(MM2S, 0)
            def dma2():
                aie.use_lock(lock_0_1_1, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1)
                aie.use_lock(lock_0_1_0, Release)

            # output flow
            @dma(S2MM, 1)
            def dma3():
                aie.use_lock(lock_0_1_2, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_0)
                aie.use_lock(lock_0_1_3, Release)

            @dma(MM2S, 1)
            def dma4():
                aie.use_lock(lock_0_1_3, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_0)
                aie.use_lock(lock_0_1_2, Release)

            aie.end()

        @mem(tile_0_2)
        def mem_0_2():
            # input
            @dma(S2MM, 0)
            def dma1():
                aie.use_lock(lock_0_2_0, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_2)
                aie.use_lock(lock_0_2_1, Release)

            # output
            @dma(MM2S, 0)
            def dma2():
                aie.use_lock(lock_0_2_3, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_2_1)
                aie.use_lock(lock_0_2_2, Release)

            aie.end()

    pass_pipeline = ",".join(
        [
            "lower-affine",
            "aie-canonicalize-device",
            "aie.device(aie-assign-buffer-addresses)",
            "convert-scf-to-cf",
        ]
    )
    input_with_addresses = run_pipeline(module, "builtin.module(" + pass_pipeline + ")")

    aie_opt_lower_to_llvm_passes = [
        "canonicalize",
        "cse",
        "convert-vector-to-llvm",
        "expand-strided-metadata",
        "lower-affine",
        "convert-math-to-llvm",
        "convert-arith-to-llvm",
        "finalize-memref-to-llvm",
        "convert-func-to-llvm{ use-bare-ptr-memref-call-conv }",
        "convert-cf-to-llvm",
        "canonicalize",
        "cse",
    ]

    pass_pipeline = ", ".join(
        [
            "aie.device(aie-localize-locks",
            "aie-normalize-address-spaces)",
            "aie-standard-lowering",
            "aiex-standard-lowering",
            *aie_opt_lower_to_llvm_passes,
        ]
    )
    input_opt_with_addresses = run_pipeline(
        input_with_addresses, "builtin.module(" + pass_pipeline + ")"
    )
    input_ll = translate_mlir_to_llvmir(input_opt_with_addresses.operation)
    with open(Path(__file__).parent / "chess_intrinsic_wrapper.ll") as f:
        chess_intrinsic_wrapper = f.read()
        input_llchesslinked_ll = chesshack(
            aie_llvm_link([input_ll, chess_intrinsic_wrapper])
        )
    input_physical = run_pipeline(
        input_with_addresses, "builtin.module(aie.device(aie-create-pathfinder-flows))"
    )

    [(col, row, _)] = generate_cores_list(str(input_with_addresses))
    core_bcf = generate_bcf(input_with_addresses.operation, col, row)

    with open(WORKDIR / "input.llchesslinked.ll", "w") as f:
        f.write(input_llchesslinked_ll)

    # chess compile
    cmd = [
        *XCHESS_ARGS,
        "-c",  # compile/assemble only, do not link
        "input.llchesslinked.ll",
        "-o",
        "input.o",
    ]
    subprocess.run(cmd, check=True, cwd=WORKDIR, env=ENV)
    with open(WORKDIR / f"core_{col}_{row}.bcf", "w") as f:
        f.write(core_bcf)

    cmd = [
        *XCHESS_ARGS,
        "input.o",
        *extract_input_files(core_bcf),
        "+l",  # linker configuration file
        f"core_{col}_{row}.bcf",
        "-o",
        f"core_{col}_{row}.elf",
    ]
    subprocess.run(cmd, check=True, cwd=WORKDIR, env=ENV)

    generate_cdo_direct(input_physical.operation, str(WORKDIR))

    with open(WORKDIR / "mem_topology.json", "w") as f:
        json.dump(mem_topology, f, indent=2)
    with open(WORKDIR / "aie_partition.json", "w") as f:
        json.dump(emit_partition(str(module)), f, indent=2)
    with open(WORKDIR / "kernels.json", "w") as f:
        json.dump(emit_design_kernel_json(), f, indent=2)
    with open(WORKDIR / "design.bif", "w") as f:
        f.write(emit_design_bif(WORKDIR))

    cmd = [
        "bootgen",
        "-arch",
        "versal",
        "-image",
        WORKDIR / "design.bif",
        "-w",  # force overwrite
        "-o",
        WORKDIR / "design.pdi",
    ]
    subprocess.run(cmd, check=True, cwd=WORKDIR, env=ENV)

    cmd = [
        "xclbinutil",
        "--add-replace-section",
        f"MEM_TOPOLOGY:JSON:{WORKDIR / 'mem_topology.json'}",
        "--add-kernel",
        str(WORKDIR / "kernels.json"),
        "--add-replace-section",
        f"AIE_PARTITION:JSON:{WORKDIR / 'aie_partition.json'}",
        "--force",
        "--output",
        f"{WORKDIR / 'final.xclbin'}",
    ]
    subprocess.run(cmd, check=True, cwd=WORKDIR, env=ENV)

    handle = subprocess.run(
        [
            "flock",
            "/tmp/ipu.lock",
            "/opt/xilinx/xrt/amdaie/setup_xclbin_firmware.sh",
            "-dev",
            "Phoenix",
            "-xclbin",
            f"{WORKDIR / 'final.xclbin'}",
        ],
        capture_output=True,
        cwd=WORKDIR,
        env=ENV,
    )
    stderr = handle.stderr.decode("utf-8").strip()
    if len(stderr):
        raise Exception(stderr)

    xclbin = XCLBin(f"{WORKDIR / 'final.xclbin'}", "MLIR_AIE")
    generated_ipu_insts = run_pipeline(
        input_with_addresses, "builtin.module(aie.device(aie-dma-to-ipu))"
    )
    ipu_insts = [int(inst, 16) for inst in ipu_instgen(generated_ipu_insts.operation)]
    xclbin.load_ipu_instructions(ipu_insts)
    inps, outps = xclbin.mmap_buffers([(LEN,), (LEN,)], [(LEN,)], np.int32)

    wrap_A = np.asarray(inps[0])
    wrap_C = np.asarray(outps[0])

    A = np.random.randint(0, 10, LEN, dtype=np.int32)
    C = np.zeros(LEN, dtype=np.int32)

    np.copyto(wrap_A, A, casting="no")
    np.copyto(wrap_C, C, casting="no")

    xclbin.sync_buffers_to_device()
    xclbin.run()
    xclbin.wait()
    xclbin.sync_buffers_from_device()

    assert np.allclose(A + RANDOM_NUMBER, wrap_C)
