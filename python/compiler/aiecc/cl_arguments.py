#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

import argparse
import sys

from aie.compiler.aiecc.configure import *


def parse_args(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(prog="aiecc")
    parser.add_argument(
        "--version",
        default=False,
        action="store_true",
        help="Output commit at which the compiler was built and exit.",
    )
    parser.add_argument(
        "filename", nargs="?", metavar="file", default=None, help="MLIR file to compile"
    )
    parser.add_argument(
        "--sysroot", metavar="sysroot", default="", help="sysroot for cross-compilation"
    )
    parser.add_argument(
        "--tmpdir",
        metavar="tmpdir",
        default=None,
        help="directory used for temporary file storage",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        dest="verbose",
        default=False,
        action="store_true",
        help="Trace commands as they are executed",
    )
    parser.add_argument(
        "--vectorize",
        dest="vectorize",
        default=False,
        action="store_true",
        help="Enable MLIR vectorization",
    )
    parser.add_argument(
        "--xbridge",
        dest="xbridge",
        default=aie_link_with_xchesscc,
        action="store_true",
        help="Link using xbridge",
    )
    parser.add_argument(
        "--no-xbridge",
        dest="xbridge",
        default=aie_link_with_xchesscc,
        action="store_false",
        help="Link using peano",
    )
    parser.add_argument(
        "--aiesim",
        dest="aiesim",
        default=False,
        action="store_true",
        help="Generate aiesim Work folder",
    )
    parser.add_argument(
        "--no-aiesim",
        dest="aiesim",
        default=False,
        action="store_false",
        help="Do not generate aiesim Work folder",
    )
    parser.add_argument(
        "--xchesscc",
        dest="xchesscc",
        default=aie_compile_with_xchesscc,
        action="store_true",
        help="Compile using xchesscc",
    )
    parser.add_argument(
        "--no-xchesscc",
        dest="xchesscc",
        default=aie_compile_with_xchesscc,
        action="store_false",
        help="Compile using peano",
    )
    parser.add_argument(
        "--peano",
        dest="peano_install_dir",
        default=peano_install_dir,
        help="Root directory where peano compiler is installed",
    )
    parser.add_argument(
        "--compile",
        dest="compile",
        default=not aie_disable_compile,
        action="store_true",
        help="Enable compiling of AIE code for each core in the array",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile",
        default=not aie_disable_compile,
        action="store_false",
        help="Disable compiling of AIE code for each core in the array",
    )
    parser.add_argument(
        "--host-target",
        dest="host_target",
        default=host_architecture,
        help="Target architecture of the host program",
    )
    parser.add_argument(
        "--compile-host",
        dest="compile_host",
        default=not host_disable_compile,
        action="store_true",
        help="Enable compiling of the host program",
    )
    parser.add_argument(
        "--no-compile-host",
        dest="compile_host",
        default=not host_disable_compile,
        action="store_false",
        help="Disable compiling of the host program",
    )
    parser.add_argument(
        "--link",
        dest="link",
        default=not aie_disable_link,
        action="store_true",
        help="Enable linking of AIE code",
    )
    parser.add_argument(
        "--no-link",
        dest="link",
        default=not aie_disable_link,
        action="store_false",
        help="Disable linking of AIE code",
    )
    parser.add_argument(
        "--alloc-scheme",
        dest="alloc_scheme",
        help="Allocation scheme for AIE buffers: basic-sequential or bank-aware. May be overruled by a tile's specific allocation scheme.",
    )
    parser.add_argument(
        "--generate-ctrl-pkt-overlay",
        dest="ctrl_pkt_overlay",
        default=False,
        action="store_true",
        help="Generate column-wise overlay of control packet routings",
    )
    parser.add_argument(
        "--dynamic-objFifos",
        dest="dynamic_objFifos",
        default=False,
        action="store_true",
        help="Use dynamic object fifos for the for loops",
    )
    parser.add_argument(
        "--packet-sw-objFifos",
        dest="packet_sw_objFifos",
        default=False,
        action="store_true",
        help="Use packet switched flows when lowering object fifos",
    )
    parser.add_argument(
        "host_args",
        action="store",
        help="arguments for host compiler",
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "-j",
        dest="nthreads",
        default=4,
        action="store",
        help="Compile with max n-threads in the machine (default is 4).  An argument of zero corresponds to the maximum number of threads on the machine.",
    )
    parser.add_argument(
        "--profile",
        dest="profiling",
        default=False,
        action="store_true",
        help="Profile commands to find the most expensive executions.",
    )
    parser.add_argument(
        "--unified",
        dest="unified",
        default=aie_unified_compile,
        action="store_true",
        help="Compile all cores together in a single process",
    )
    parser.add_argument(
        "--no-unified",
        dest="unified",
        default=aie_unified_compile,
        action="store_false",
        help="Compile cores independently in separate processes",
    )
    parser.add_argument(
        "-n",
        dest="execute",
        default=True,
        action="store_false",
        help="Disable actually executing any commands.",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        default=False,
        action="store_true",
        help="Show progress visualization",
    )
    parser.add_argument(
        "--aie-generate-npu-insts",
        dest="npu",
        default=False,
        action="store_const",
        const=True,
        help="Generate npu instruction stream from default runtime sequence",
    )
    parser.add_argument(
        "--npu-insts-name",
        dest="insts_name",
        default="{0}_{1}.bin",
        help="Output instructions filename for NPU target. `{0}` is replaced with device name, `{1}` with the selected runtime sequence.",
    )
    parser.add_argument(
        "--aie-generate-cdo",
        dest="cdo",
        default=False,
        action="store_const",
        const=True,
        help="Generate libxaie v2 for CDO",
    )
    parser.add_argument(
        "--aie-generate-txn",
        dest="txn",
        default=False,
        action="store_const",
        const=True,
        help="Generate transaction binary mlir for configuration",
    )
    parser.add_argument(
        "--txn-name",
        dest="txn_name",
        default="{0}_transaction.mlir",
        help="Output filename for transaction binary mlir. `{0}` is replaced with device name.",
    )
    parser.add_argument(
        "--aie-generate-ctrlpkt",
        dest="ctrlpkt",
        default=False,
        action="store_const",
        const=True,
        help="Generate control packets for configuration",
    )
    parser.add_argument(
        "--ctrlpkt-name",
        dest="ctrlpkt_name",
        default="{0}_ctrlpkt.bin",
        help="Output filename for control packet binary data. `{0}` is replaced with the device name.",
    )
    parser.add_argument(
        "--ctrlpkt-dma-seq-name",
        dest="ctrlpkt_dma_seq_name",
        default="{0}_ctrlpkt_dma_seq.bin",
        help="Output filename for control packet DMA sequence. `{0}` is replaced with the device name.",
    )
    parser.add_argument(
        "--ctrlpkt-elf-name",
        dest="ctrlpkt_elf_name",
        default="{0}_ctrlpkt.elf",
        help="Output filename for control packet DMA sequence and control packet binary data in combined ELF file. `{0}` is replaced with the device name.",
    )
    parser.add_argument(
        "--aie-generate-xclbin",
        dest="xcl",
        default=False,
        action="store_const",
        const=True,
        help="Generate xclbin",
    )
    parser.add_argument(
        "--xclbin-input",
        dest="xclbin_input",
        default=None,
        help="Generate kernel into existing xclbin file",
    )
    parser.add_argument(
        "--link_against_hsa",
        dest="link_against_hsa",
        default=False,
        action="store_const",
        const=True,
        help="Link runtime against ROCm runtime HSA interface",
    )
    parser.add_argument(
        "--xclbin-name",
        dest="xclbin_name",
        default="{0}.xclbin",
        help="Output xclbin filename for CDO/XCLBIN target. `{0}` is replaced with device name.",
    )
    parser.add_argument(
        "--aie-generate-pdi",
        dest="pdi",
        default=False,
        action="store_const",
        const=True,
        help="Generate pdi binary for configuration",
    )
    parser.add_argument(
        "--pdi-name",
        dest="pdi_name",
        default="{0}.pdi",
        help="Output pdi filename for PDI/CDO/XCLBIN target. `{0}` is replaced with device name.",
    )
    parser.add_argument(
        "--xclbin-kernel-name",
        dest="kernel_name",
        default="MLIR_AIE",
        help="Kernel name in xclbin file",
    )
    parser.add_argument(
        "--xclbin-instance-name",
        dest="instance_name",
        default="MLIRAIE",
        help="Instance name in xclbin metadata",
    )
    parser.add_argument(
        "--xclbin-kernel-id",
        dest="kernel_id",
        default="0x901",
        help="Kernel id in xclbin file",
    )
    parser.add_argument(
        "--aie-generate-elf",
        dest="elf",
        default=False,
        action="store_const",
        const=True,
        help="Generate elf for AIE control and/or configuration",
    )
    parser.add_argument(
        "--elf-name",
        dest="elf_name",
        default="design_{0}.elf",
        help="Output elf filename for ELF target. `{0}` is replaced with the device name.",
    )
    parser.add_argument(
        "--device-name",
        dest="device_name",
        default="",
        help="Symbol name of the device configuration to compile. If none supplied, all devices are compiled.",
    )
    parser.add_argument(
        "--sequence-name",
        dest="sequence_name",
        default="",
        help="Symbol name of the runtime sequence to compile. If none supplied, all runtime sequences in the selected device(s) are compiled.",
    )
    parser.add_argument(
        "--generate-full-elf",
        dest="generate_full_elf",
        default=False,
        action="store_true",
        help="Generate complete full ELF using aiebu-asm",
    )
    parser.add_argument(
        "--full-elf-name",
        dest="full_elf_name",
        default="aie.elf",
        help="Output filename for full ELF (default: aie.elf)",
    )

    opts = parser.parse_args(args)
    return opts


def strip_host_args_for_aiesim(args):
    parser = argparse.ArgumentParser(prog="aiecc")
    parser.add_argument("-o", metavar="output", default="", help="output file")

    opts = parser.parse_known_args(args)
    return opts[1]


def _positive_int(arg):
    return _int(arg, "positive", lambda i: i > 0)


def _non_negative_int(arg):
    return _int(arg, "non-negative", lambda i: i >= 0)


def _int(arg, kind, pred):
    desc = "requires {} integer, but found '{}'"
    try:
        i = int(arg)
    except ValueError:
        raise _error(desc, kind, arg)
    if not pred(i):
        raise _error(desc, kind, arg)
    return i


def _case_insensitive_regex(arg):
    import re

    try:
        return re.compile(arg, re.IGNORECASE)
    except re.error as reason:
        raise _error("invalid regular expression: '{}', {}", arg, reason)


def _error(desc, *args):
    msg = desc.format(*args)
    return argparse.ArgumentTypeError(msg)
