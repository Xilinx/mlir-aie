# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: WORKDIR=%T %PYTHON %s | FileCheck %s
# REQUIRES: cdo_direct_generation
# REQUIRES: ryzen_ai

import os
import re
import subprocess
from pathlib import Path

from aie.extras.dialects.ext import memref, arith, func
from aie.extras.runtime.passes import run_pipeline

import aie.extras.types as T
from aie.compiler.aiecc.main import generate_cores_list, chesshack
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
    core,
    device,
    generate_bcf,
    generate_cdo,
    generate_cdo_direct,
    byte_ordering,
    objectfifo,
    objectfifo_link,
    tile,
    translate_mlir_to_llvmir,
    aie_llvm_link,
)
from aie.dialects.aiex import ipu_sync, ipu_dma_memcpy_nd
from aie.dialects.scf import for_
from aie.ir import TypeAttr
from util import construct_and_print_module

range_ = for_

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


def extract_input_files(core_bcf):
    return re.findall(r"^_include _file (.*)", core_bcf, re.MULTILINE)


# CHECK-LABEL: my_passthrough
@construct_and_print_module
def my_passthrough(module):
    N = 4096
    ofifo_mem_ref_ty = T.memref(1024, T.i32())
    tensor_ty = T.memref(N, T.i32())

    @device(AIEDevice.ipu)
    def device_body():
        # Tile declarations
        shim_tile = tile(0, 0)
        compute_tile2 = tile(0, 2)

        # AIE-array data movement with object fifos
        of_in = objectfifo("in", shim_tile, compute_tile2, 2, ofifo_mem_ref_ty)
        of_out = objectfifo("out", compute_tile2, shim_tile, 2, ofifo_mem_ref_ty)
        objectfifo_link(["in"], ["out"])

        @core(compute_tile2)
        def core_body():
            tmp = memref.alloc([1], T.i32())
            v0 = arith.constant(0, T.i32())
            memref.store(v0, tmp, [0])

        @func.func(emit=True)
        def sequence(A: tensor_ty, B: tensor_ty, C: tensor_ty):
            ipu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, N])
            ipu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, sizes=[1, 1, 1, N])
            ipu_sync(column=0, row=0, direction=0, channel=0)

    pass_pipeline = ",".join(
        [
            "lower-affine",
            "aie-canonicalize-device",
            "aie.device(" + "aie-assign-lock-ids",
            "aie-register-objectFifos",
            "aie-objectFifo-stateful-transform",
            "aie-lower-broadcast-packet",
            "aie-create-packet-flows",
            "aie-lower-multicast",
            "aie-assign-buffer-addresses)",
            "convert-scf-to-cf",
        ]
    )
    input_with_addresses = run_pipeline(module, "builtin.module(" + pass_pipeline + ")")
    pass_pipeline = ",".join(
        [
            "aie-create-pathfinder-flows",
            "aie-lower-broadcast-packet",
            "aie-create-packet-flows",
            "aie-lower-multicast",
        ]
    )
    input_physical = run_pipeline(
        input_with_addresses, "builtin.module(aie.device(" + pass_pipeline + "))"
    )

    cores = generate_cores_list(str(input_with_addresses))
    col, row, _ = cores[0]
    core_0_2_bcf = generate_bcf(input_with_addresses.operation, col, row)

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

    with open(Path(__file__).parent / "e2e" / "chess_intrinsic_wrapper.ll") as f:
        chess_intrinsic_wrapper = f.read()
        input_llchesslinked_ll = chesshack(
            aie_llvm_link([input_ll, chess_intrinsic_wrapper])
        )

    WORKDIR = Path(os.getenv("WORKDIR", str(Path(".").absolute()))).absolute()
    AIETOOLS_DIR = Path(os.getenv("AIETOOLS")).absolute()
    # bootgen and xclbinutil
    VITIS_BIN_DIR = AIETOOLS_DIR.parent / "bin"
    RDI_DATADIR = f"{AIETOOLS_DIR}/data"
    XILINXD_LICENSE_FILE = Path(os.getenv("XILINXD_LICENSE_FILE")).absolute()
    LD_PATH = [
        os.getenv("LD_LIBRARY_PATH"),
        f"{AIETOOLS_DIR}/lib/lnx64.o",
        f"{AIETOOLS_DIR}/lnx64/tools/dot/lib",
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
        "RDI_DATADIR": RDI_DATADIR,
        "PATH": PATH,
        "XILINXD_LICENSE_FILE": XILINXD_LICENSE_FILE,
    }

    xchess_args = [
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

    with open(WORKDIR / "input.llchesslinked.ll", "w") as f:
        f.write(input_llchesslinked_ll)
    cmd = [
        *xchess_args,
        "-c",  # compile/assemble only, do not link
        "input.llchesslinked.ll",
        "-o",
        "input.o",
    ]
    subprocess.run(cmd, check=True, cwd=WORKDIR, env=ENV)

    with open(WORKDIR / "core_0_2.bcf", "w") as f:
        f.write(core_0_2_bcf)

    cmd = [
        *xchess_args,
        "input.o",
        *extract_input_files(core_0_2_bcf),
        "+l",  # linker configuration file
        "core_0_2.bcf",
        "-o",
        "core_0_2.elf",
    ]
    r = subprocess.run(cmd, capture_output=True, cwd=WORKDIR, env=ENV)

    aie_control = generate_cdo(input_physical.operation)
    print(aie_control)

    generate_cdo_direct(
        input_physical.operation, str(WORKDIR), byte_ordering.Little_Endian, True
    )

    # CHECK: True
    print(True)
