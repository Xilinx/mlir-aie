# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

"""
aiecc - AIE compiler driver for MLIR tools
"""

import asyncio
import glob
import json
import os
import random
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from textwrap import dedent
import time

from aie.extras.runtime.passes import Pipeline

# this is inside the aie-python-extras (shared) namespace package
from aie.extras.util import find_ops
import aiofiles
import rich.progress as progress

import aie.compiler.aiecc.cl_arguments
import aie.compiler.aiecc.configure
from aie.dialects import aie as aiedialect
from aie.ir import Context, Location, Module
from aie.passmanager import PassManager

INPUT_WITH_ADDRESSES_PIPELINE = (
    Pipeline()
    .lower_affine()
    .add_pass("aie-canonicalize-device")
    .Nested(
        "aie.device",
        Pipeline()
        .add_pass("aie-assign-lock-ids")
        .add_pass("aie-register-objectFifos")
        .add_pass("aie-objectFifo-stateful-transform")
        .add_pass("aie-assign-bd-ids")
        .add_pass("aie-lower-cascade-flows")
        .add_pass("aie-lower-broadcast-packet")
        .add_pass("aie-create-packet-flows")
        .add_pass("aie-lower-multicast")
        .add_pass("aie-assign-buffer-addresses"),
    )
    .convert_scf_to_cf()
)

LOWER_TO_LLVM_PIPELINE = (
    Pipeline()
    .canonicalize()
    .cse()
    .convert_vector_to_llvm()
    .expand_strided_metadata()
    .lower_affine()
    .convert_math_to_llvm()
    .convert_arith_to_llvm()
    .finalize_memref_to_llvm()
    .convert_func_to_llvm(use_bare_ptr_memref_call_conv=True)
    .convert_cf_to_llvm()
    .canonicalize()
    .cse()
)

AIE_LOWER_TO_LLVM = (
    lambda col=None, row=None: (
        Pipeline()
        .Nested(
            "aie.device",
            Pipeline()
            .add_pass("aie-localize-locks")
            .add_pass("aie-normalize-address-spaces"),
        )
        .add_pass("aie-standard-lowering", tilecol=col, tilerow=row)
        .add_pass("aiex-standard-lowering")
    )
    + LOWER_TO_LLVM_PIPELINE
)

CREATE_PATH_FINDER_FLOWS = Pipeline().Nested(
    "aie.device", Pipeline().add_pass("aie-create-pathfinder-flows")
)
DMA_TO_NPU = Pipeline().Nested("aie.device", Pipeline().add_pass("aie-dma-to-npu"))


async def read_file_async(file_path: str) -> str:
    async with aiofiles.open(file_path, mode="r") as f:
        contents = await f.read()
    return contents


async def write_file_async(file_content: str, file_path: str):
    async with aiofiles.open(file_path, mode="w") as f:
        await f.write(file_content)


def emit_design_kernel_json(
    kernel_name="MLIR_AIE",
    kernel_id="0x901",
    instance_name="MLIRAIE",
    buffer_args=None,
):
    if buffer_args is None:
        buffer_args = ["in", "tmp", "out"]

    arguments = [
        {
            "name": "instr",
            "memory-connection": "SRAM",
            "address-qualifier": "GLOBAL",
            "type": "char *",
            "offset": "0x00",
        },
        {
            "name": "ninstr",
            "address-qualifier": "SCALAR",
            "type": "uint64_t",
            "offset": "0x08",
        },
    ]

    offset = 0x10
    for buf in buffer_args:
        arg = {
            "name": buf,
            "memory-connection": "HOST",
            "address-qualifier": "GLOBAL",
            "type": "char *",
            "offset": str(hex(offset)),
        }
        arguments.append(arg)
        offset += 0x8

    return {
        "ps-kernels": {
            "kernels": [
                {
                    "name": kernel_name,
                    "type": "dpu",
                    "extended-data": {
                        "subtype": "DPU",
                        "functional": "1",
                        "dpu_kernel_id": kernel_id,
                    },
                    "arguments": arguments,
                    "instances": [{"name": instance_name}],
                }
            ]
        }
    }


mem_topology = {
    "mem_topology": {
        "m_count": "2",
        "m_mem_data": [
            {
                "m_type": "MEM_DRAM",
                "m_used": "1",
                "m_sizeKB": "0x10000",
                "m_tag": "HOST",
                "m_base_address": "0x4000000",
            },
            {
                "m_type": "MEM_DRAM",
                "m_used": "1",
                "m_sizeKB": "0xc000",
                "m_tag": "SRAM",
                "m_base_address": "0x4000000",
            },
        ],
    }
}


def emit_partition(mlir_module_str, kernel_id="0x901", start_columns=None):
    with Context(), Location.unknown():
        module = Module.parse(mlir_module_str)
        tiles = find_ops(
            module.operation,
            lambda o: isinstance(o.operation.opview, aiedialect.TileOp),
        )
        min_col = min([t.col.value for t in tiles])
        max_col = max([t.col.value for t in tiles])

    num_cols = max_col - min_col + 1
    if start_columns is None:
        start_columns = list(range(1, 6 - num_cols))

    uuid = random.randint(2222, 9999)
    return {
        "aie_partition": {
            "name": "QoS",
            "operations_per_cycle": "2048",
            "inference_fingerprint": "23423",
            "pre_post_fingerprint": "12345",
            "partition": {
                "column_width": num_cols,
                "start_columns": start_columns,
            },
            "PDIs": [
                {
                    "uuid": "00000000-0000-0000-0000-00000000" + str(uuid),
                    "file_name": "./design.pdi",
                    "cdo_groups": [
                        {
                            "name": "DPU",
                            "type": "PRIMARY",
                            "pdi_id": "0x01",
                            "dpu_kernel_ids": [kernel_id],
                            "pre_cdo_groups": ["0xC1"],
                        }
                    ],
                }
            ],
        }
    }


def generate_cores_list(mlir_module_str):
    with Context(), Location.unknown():
        module = Module.parse(mlir_module_str)
        return [
            (
                c.tile.owner.opview.col.value,
                c.tile.owner.opview.row.value,
                c.elf_file.value if c.elf_file is not None else None,
            )
            for c in find_ops(
                module.operation,
                lambda o: isinstance(o.operation.opview, aiedialect.CoreOp),
            )
        ]


def emit_design_bif(root_path, has_cores=True, enable_cores=True):
    elf_file = f"file={root_path}/aie_cdo_elfs.bin" if has_cores else ""
    enable_file = f"file={root_path}/aie_cdo_enable.bin" if enable_cores else ""
    return dedent(
        f"""\
        all:
        {{
          id_code = 0x14ca8093
          extended_id_code = 0x01
          image
          {{
            name=aie_image, id=0x1c000000
            {{ type=cdo
               file={root_path}/aie_cdo_error_handling.bin
               {elf_file}
               file={root_path}/aie_cdo_init.bin
               {enable_file}
            }}
          }}
        }}
        """
    )


# Extract included files from the given Chess linker script.
# We rely on gnu linker scripts to stuff object files into a compile.  However, the Chess compiler doesn't
# do this, so we have to explicitly specify included files on the link line.
async def extract_input_files(file_core_bcf):
    core_bcf = await read_file_async(file_core_bcf)
    return " ".join(re.findall(r"^_include _file (.*)", core_bcf, re.MULTILINE))


def do_run(command, verbose=False):
    if verbose:
        print(" ".join(command))
    m = subprocess.PIPE
    ret = subprocess.run(command, stdout=m, stderr=m, universal_newlines=True)
    return ret


def run_passes(pass_pipeline, mlir_module_str, outputfile=None, verbose=False):
    if verbose:
        print("Running:", pass_pipeline)
    with Context() as ctx, Location.unknown():
        module = Module.parse(mlir_module_str)
        PassManager.parse(pass_pipeline).run(module.operation)
        mlir_module_str = str(module)
        if outputfile:
            with open(outputfile, "w") as g:
                g.write(mlir_module_str)
    return mlir_module_str


def corefile(dirname, core, ext):
    col, row, _ = core
    return os.path.join(dirname, f"core_{col}_{row}.{ext}")


def aie_target_defines(aie_target):
    if aie_target == "AIE2":
        return ["-D__AIEARCH__=20"]
    return ["-D__AIEARCH__=10"]


def chesshack(llvmir_chesslinked):
    llvmir_chesslinked = (
        llvmir_chesslinked.replace("memory(none)", "readnone")
        .replace("memory(read)", "readonly")
        .replace("memory(write)", "writeonly")
        .replace("memory(argmem: readwrite)", "argmemonly")
        .replace("memory(argmem: read)", "argmemonly readonly")
        .replace("memory(argmem: write)", "argmemonly writeonly")
        .replace("memory(inaccessiblemem: readwrite)", "inaccessiblememonly")
        .replace("memory(inaccessiblemem: read)", "inaccessiblememonly readonly")
        .replace("memory(inaccessiblemem: write)", "inaccessiblememonly writeonly")
        .replace(
            "memory(argmem: readwrite, inaccessiblemem: readwrite)",
            "inaccessiblemem_or_argmemonly",
        )
        .replace(
            "memory(argmem: read, inaccessiblemem: read)",
            "inaccessiblemem_or_argmemonly readonly",
        )
        .replace(
            "memory(argmem: write, inaccessiblemem: write)",
            "inaccessiblemem_or_argmemonly writeonly",
        )
    )
    return llvmir_chesslinked


class FlowRunner:
    def __init__(self, mlir_module_str, opts, tmpdirname):
        self.mlir_module_str = mlir_module_str
        self.opts = opts
        self.tmpdirname = tmpdirname
        self.runtimes = dict()
        self.progress_bar = None
        self.maxtasks = 5
        self.stopall = False
        self.peano_clang_path = os.path.join(opts.peano_install_dir, "bin", "clang")
        self.peano_opt_path = os.path.join(opts.peano_install_dir, "bin", "opt")
        self.peano_llc_path = os.path.join(opts.peano_install_dir, "bin", "llc")

    def prepend_tmp(self, x):
        return os.path.join(self.tmpdirname, x)

    async def do_call(self, task, command, force=False):
        if self.stopall:
            return

        commandstr = " ".join(command)
        if task:
            self.progress_bar.update(task, advance=0, command=commandstr[0:30])
        start = time.time()
        if self.opts.verbose:
            print(commandstr)
        if self.opts.execute or force:
            proc = await asyncio.create_subprocess_exec(*command)
            await proc.wait()
            ret = proc.returncode
        else:
            ret = 0
        end = time.time()
        if self.opts.verbose:
            print(f"Done in {end - start:.3f} sec: {commandstr}")
        self.runtimes[commandstr] = end - start
        if task:
            self.progress_bar.update(task, advance=1, command="")
            self.maxtasks = max(self.progress_bar._tasks[task].completed, self.maxtasks)
            self.progress_bar._tasks[task].total = self.maxtasks

        if ret != 0:
            if task:
                self.progress_bar._tasks[task].description = "[red] Error"
            print("Error encountered while running: " + commandstr, file=sys.stderr)
            sys.exit(ret)

    # In order to run xchesscc on modern ll code, we need a bunch of hacks.
    async def chesshack(self, task, llvmir, chess_intrinsic_wrapper_ll_path):
        llvmir_chesshack = llvmir + "chesshack.ll"
        llvmir_chesslinked_path = llvmir + "chesslinked.ll"
        if not self.opts.execute:
            return llvmir_chesslinked_path
        llvmir = await read_file_async(llvmir)

        await write_file_async(llvmir, llvmir_chesshack)
        assert os.path.exists(llvmir_chesshack)
        await self.do_call(
            task,
            [
                "llvm-link",
                llvmir_chesshack,
                chess_intrinsic_wrapper_ll_path,
                "-S",
                "-o",
                llvmir_chesslinked_path,
            ],
        )

        llvmir_chesslinked_ir = await read_file_async(llvmir_chesslinked_path)
        llvmir_chesslinked_ir = chesshack(llvmir_chesslinked_ir)
        await write_file_async(llvmir_chesslinked_ir, llvmir_chesslinked_path)

        return llvmir_chesslinked_path

    async def prepare_for_chesshack(self, task, aie_target):
        if opts.compile and opts.xchesscc:
            install_path = aie.compiler.aiecc.configure.install_path()
            runtime_lib_path = os.path.join(install_path, "aie_runtime_lib")
            chess_intrinsic_wrapper_cpp = os.path.join(
                runtime_lib_path, aie_target.upper(), "chess_intrinsic_wrapper.cpp"
            )

            chess_intrinsic_wrapper_ll_path = self.prepend_tmp(
                "chess_intrinsic_wrapper.ll"
            )

            # fmt: off
            await self.do_call(task, ["xchesscc_wrapper", aie_target.lower(), "+w", self.prepend_tmp("work"), "-c", "-d", "-f", "+f", "+P", "4", chess_intrinsic_wrapper_cpp, "-o", chess_intrinsic_wrapper_ll_path])
            # fmt: on

            # this has to be here and not higher because there are tests that check for the command string for the above do_call
            if not self.opts.execute:
                return
            chess_intrinsic_wrapper = await read_file_async(
                chess_intrinsic_wrapper_ll_path
            )
            chess_intrinsic_wrapper = re.sub(
                r"^target.*", "", chess_intrinsic_wrapper, flags=re.MULTILINE
            )
            await write_file_async(
                chess_intrinsic_wrapper, chess_intrinsic_wrapper_ll_path
            )
            return chess_intrinsic_wrapper_ll_path

    async def process_core(
        self,
        core,
        aie_target,
        aie_peano_target,
        chess_intrinsic_wrapper_ll_path,
        file_with_addresses,
    ):
        async with self.limit:
            if self.stopall:
                return

            install_path = aie.compiler.aiecc.configure.install_path()
            runtime_lib_path = os.path.join(
                install_path, "aie_runtime_lib", aie_target.upper()
            )
            clang_path = os.path.dirname(shutil.which("clang"))
            # The build path for libc can be very different from where it's installed.
            llvmlibc_build_lib_path = os.path.join(
                clang_path,
                "..",
                "runtimes",
                "runtimes-" + aie_target.lower() + "-none-unknown-elf-bins",
                "libc",
                "lib",
                "libc.a",
            )
            llvmlibc_install_lib_path = os.path.join(
                clang_path,
                "..",
                "lib",
                aie_target.lower() + "-none-unknown-elf",
                "libc.a",
            )
            me_basic_o = os.path.join(runtime_lib_path, "me_basic.o")
            if os.path.isfile(llvmlibc_build_lib_path):
                libc = llvmlibc_build_lib_path
            else:
                libc = llvmlibc_install_lib_path

            clang_link_args = [me_basic_o, libc, "-Wl,--gc-sections"]

            if opts.progress:
                task = self.progress_bar.add_task(
                    "[yellow] Core (%d, %d)" % core[0:2],
                    total=self.maxtasks,
                    command="starting",
                )
            else:
                task = None

            # fmt: off
            corecol, corerow, elf_file = core
            if not opts.unified:
                file_core = corefile(self.tmpdirname, core, "mlir")
                await self.do_call(task, ["aie-opt", "--aie-localize-locks", "--aie-normalize-address-spaces", "--aie-standard-lowering=tilecol=%d tilerow=%d" % core[0:2], "--aiex-standard-lowering", file_with_addresses, "-o", file_core])
                file_opt_core = corefile(self.tmpdirname, core, "opt.mlir")
                await self.do_call(task, ["aie-opt", f"--pass-pipeline={LOWER_TO_LLVM_PIPELINE}", file_core, "-o", file_opt_core])
            if self.opts.xbridge:
                file_core_bcf = corefile(self.tmpdirname, core, "bcf")
                await self.do_call(task, ["aie-translate", file_with_addresses, "--aie-generate-bcf", "--tilecol=%d" % corecol, "--tilerow=%d" % corerow, "-o", file_core_bcf])
            else:
                file_core_ldscript = corefile(self.tmpdirname, core, "ld.script")
                await self.do_call(task, ["aie-translate", file_with_addresses, "--aie-generate-ldscript", "--tilecol=%d" % corecol, "--tilerow=%d" % corerow, "-o", file_core_ldscript])
            if not self.opts.unified:
                file_core_llvmir = corefile(self.tmpdirname, core, "ll")
                await self.do_call(task, ["aie-translate", "--mlir-to-llvmir", file_opt_core, "-o", file_core_llvmir])
                file_core_obj = corefile(self.tmpdirname, core, "o")

            file_core_elf = elf_file if elf_file else corefile(".", core, "elf")

            if opts.compile and opts.xchesscc:
                if not opts.unified:
                    file_core_llvmir_chesslinked = await self.chesshack(task, file_core_llvmir, chess_intrinsic_wrapper_ll_path)
                    if self.opts.link and self.opts.xbridge:
                        link_with_obj = await extract_input_files(file_core_bcf)
                        await self.do_call(task, ["xchesscc_wrapper", aie_target.lower(), "+w", self.prepend_tmp("work"), "-d", "-f", "+P", "4", file_core_llvmir_chesslinked, link_with_obj, "+l", file_core_bcf, "-o", file_core_elf])
                    elif self.opts.link:
                        await self.do_call(task, ["xchesscc_wrapper", aie_target.lower(), "+w", self.prepend_tmp("work"), "-c", "-d", "-f", "+P", "4", file_core_llvmir_chesslinked, "-o", file_core_obj])
                        await self.do_call(task, [self.peano_clang_path, "-O2", "--target=" + aie_peano_target, file_core_obj, *clang_link_args, "-Wl,-T," + file_core_ldscript, "-o", file_core_elf])
                else:
                    file_core_obj = self.unified_file_core_obj
                    if opts.link and opts.xbridge:
                        link_with_obj = await extract_input_files(file_core_bcf)
                        await self.do_call(task, ["xchesscc_wrapper", aie_target.lower(), "+w", self.prepend_tmp("work"), "-d", "-f", file_core_obj, link_with_obj, "+l", file_core_bcf, "-o", file_core_elf])
                    elif opts.link:
                        await self.do_call(task, [self.peano_clang_path, "-O2", "--target=" + aie_peano_target, file_core_obj, *clang_link_args, "-Wl,-T," + file_core_ldscript, "-o", file_core_elf])

            elif opts.compile:
                if not opts.unified:
                    file_core_llvmir_stripped = corefile(self.tmpdirname, core, "stripped.ll")
                    await self.do_call(task, [self.peano_opt_path, "--passes=default<O2>,strip", "-S", file_core_llvmir, "-o", file_core_llvmir_stripped])
                    await self.do_call(task, [self.peano_llc_path, file_core_llvmir_stripped, "-O2", "--march=" + aie_target.lower(), "--function-sections", "--filetype=obj", "-o", file_core_obj])
                else:
                    file_core_obj = self.unified_file_core_obj

                if opts.link and opts.xbridge:
                    link_with_obj = await extract_input_files(file_core_bcf)
                    await self.do_call(task, ["xchesscc_wrapper", aie_target.lower(), "+w", self.prepend_tmp("work"), "-d", "-f", file_core_obj, link_with_obj, "+l", file_core_bcf, "-o", file_core_elf])
                elif opts.link:
                    await self.do_call(task, [self.peano_clang_path, "-O2", "--target=" + aie_peano_target, file_core_obj, *clang_link_args, "-Wl,-T," + file_core_ldscript, "-o", file_core_elf])

            self.progress_bar.update(self.progress_bar.task_completed, advance=1)
            if task:
                self.progress_bar.update(task, advance=0, visible=False)
            # fmt: on

    async def process_cdo(self):
        from aie.dialects.aie import generate_cdo

        with Context(), Location.unknown():
            for elf in glob.glob("*.elf"):
                try:
                    shutil.copy(elf, self.tmpdirname)
                except shutil.SameFileError:
                    pass
            for elf_map in glob.glob("*.elf.map"):
                try:
                    shutil.copy(elf_map, self.tmpdirname)
                except shutil.SameFileError:
                    pass
            input_physical = Module.parse(
                await read_file_async(self.prepend_tmp("input_physical.mlir"))
            )
            generate_cdo(input_physical.operation, self.tmpdirname)

    async def process_xclbin_gen(self, has_cores):
        if opts.progress:
            task = self.progress_bar.add_task(
                "[yellow] XCLBIN generation ", total=10, command="starting"
            )
        else:
            task = None

        await write_file_async(
            json.dumps(mem_topology, indent=2),
            self.prepend_tmp("mem_topology.json"),
        )

        await write_file_async(
            json.dumps(emit_partition(self.mlir_module_str, opts.kernel_id), indent=2),
            self.prepend_tmp("aie_partition.json"),
        )

        buffer_arg_names = ["in", "tmp", "out"]
        await write_file_async(
            json.dumps(
                emit_design_kernel_json(
                    opts.kernel_name,
                    opts.kernel_id,
                    opts.instance_name,
                    buffer_arg_names,
                ),
                indent=2,
            ),
            self.prepend_tmp("kernels.json"),
        )

        await write_file_async(
            emit_design_bif(self.tmpdirname, has_cores),
            self.prepend_tmp("design.bif"),
        )

        # fmt: off
        await self.do_call(task, ["bootgen", "-arch", "versal", "-image", self.prepend_tmp("design.bif"), "-o", self.prepend_tmp("design.pdi"), "-w"])
        await self.do_call(task, ["xclbinutil", "--add-replace-section", "MEM_TOPOLOGY:JSON:" + self.prepend_tmp("mem_topology.json"), "--add-kernel", self.prepend_tmp("kernels.json"), "--add-replace-section", "AIE_PARTITION:JSON:" + self.prepend_tmp("aie_partition.json"), "--force", "--output", opts.xclbin_name])
        # fmt: on

    async def process_host_cgen(self, aie_target, file_with_addresses):
        async with self.limit:
            if self.stopall:
                return

            if opts.progress:
                task = self.progress_bar.add_task(
                    "[yellow] Host compilation ", total=10, command="starting"
                )
            else:
                task = None

            # Generate the included host interface
            file_physical = self.prepend_tmp("input_physical.mlir")
            await self.do_call(
                task,
                [
                    "aie-opt",
                    "--aie-create-pathfinder-flows",
                    "--aie-lower-broadcast-packet",
                    "--aie-create-packet-flows",
                    "--aie-lower-multicast",
                    file_with_addresses,
                    "-o",
                    file_physical,
                ],
            )

            if opts.airbin:
                file_airbin = self.prepend_tmp("air.bin")
                await self.do_call(
                    task,
                    [
                        "aie-translate",
                        "--aie-generate-airbin",
                        file_physical,
                        "-o",
                        file_airbin,
                    ],
                )
            else:
                file_inc_cpp = self.prepend_tmp("aie_inc.cpp")
                await self.do_call(
                    task,
                    [
                        "aie-translate",
                        "--aie-generate-xaie",
                        file_physical,
                        "-o",
                        file_inc_cpp,
                    ],
                )

            if opts.link_against_hsa:
                file_inc_cpp = self.prepend_tmp("aie_data_movement.cpp")
                await self.do_call(
                    task,
                    [
                        "aie-translate",
                        "--aie-generate-hsa",
                        file_physical,
                        "-o",
                        file_inc_cpp,
                    ],
                )

            cmd = ["clang++", "-std=c++17"]
            if opts.host_target:
                cmd += ["--target=" + opts.host_target]
                if (
                    opts.aiesim
                    and opts.host_target
                    != aie.compiler.aiecc.configure.host_architecture
                ):
                    sys.exit(
                        "Host cross-compile from "
                        + aie.compiler.aiecc.configure.host_architecture
                        + " to --target="
                        + opts.host_target
                        + " is not supported with --aiesim"
                    )

            if self.opts.sysroot:
                cmd += ["--sysroot=" + opts.sysroot]
                # In order to find the toolchain in the sysroot, we need to have
                # a 'target' that includes 'linux' and for the 'lib/gcc/$target/$version'
                # directory to have a corresponding 'include/gcc/$target/$version'.
                # In some of our sysroots, it seems that we find a lib/gcc, but it
                # doesn't have a corresponding include/gcc directory.  Instead
                # force using '/usr/lib,include/gcc'
                if opts.host_target == "aarch64-linux-gnu":
                    cmd += [f"--gcc-toolchain={opts.sysroot}/usr"]
                    # It looks like the G++ distribution is non standard, so add
                    # an explicit handling of C++ library.
                    # Perhaps related to https://discourse.llvm.org/t/add-gcc-install-dir-deprecate-gcc-toolchain-and-remove-gcc-install-prefix/65091/23
                    cxx_include = glob.glob(f"{opts.sysroot}/usr/include/c++/*.*.*")[0]
                    triple = os.path.basename(opts.sysroot)
                    cmd += [f"-I{cxx_include}", f"-I{cxx_include}/{triple}"]
                    gcc_lib = glob.glob(f"{opts.sysroot}/usr/lib/{triple}/*.*.*")[0]
                    cmd += [f"-B{gcc_lib}", f"-L{gcc_lib}"]
            install_path = aie.compiler.aiecc.configure.install_path()

            # Setting everything up if linking against HSA
            if opts.link_against_hsa:
                cmd += ["-DHSA_RUNTIME"]
                arch_name = opts.host_target.split("-")[0] + "-hsa"
                hsa_path = os.path.join(aie.compiler.aiecc.configure.hsa_dir)
                hsa_include_path = os.path.join(hsa_path, "..", "..", "..", "include")
                hsa_lib_path = os.path.join(hsa_path, "..", "..")
                hsa_so_path = os.path.join(hsa_lib_path, "libhsa-runtime64.so")
            else:
                arch_name = opts.host_target.split("-")[0]

            # Getting a pointer to the libxaie include and library
            runtime_xaiengine_path = os.path.join(
                install_path, "runtime_lib", arch_name, "xaiengine"
            )
            xaiengine_include_path = os.path.join(runtime_xaiengine_path, "include")
            xaiengine_lib_path = os.path.join(runtime_xaiengine_path, "lib")

            # Getting a pointer to the library test_lib
            runtime_testlib_path = os.path.join(
                install_path,
                "runtime_lib",
                arch_name,
                "test_lib",
                "lib",
            )

            # Linking against the correct memory allocator
            if opts.link_against_hsa:
                memory_allocator = os.path.join(
                    runtime_testlib_path, "libmemory_allocator_hsa.a"
                )
            else:
                memory_allocator = os.path.join(
                    runtime_testlib_path, "libmemory_allocator_ion.a"
                )

            cmd += [
                memory_allocator,
                "-I" + xaiengine_include_path,
                "-L" + xaiengine_lib_path,
                "-L" + os.path.join(opts.aietools_path, "lib", "lnx64.o"),
                "-Wl,-R" + xaiengine_lib_path,
                "-I" + self.tmpdirname,
                "-fuse-ld=lld",
                "-lm",
                "-lxaiengine",
            ]
            # Linking against HSA
            if opts.link_against_hsa:
                cmd += [hsa_so_path]
                cmd += ["-I%s" % hsa_include_path]
                cmd += ["-Wl,-rpath,%s" % hsa_lib_path]

            cmd += aie_target_defines(aie_target)

            if len(opts.host_args) > 0:
                await self.do_call(task, cmd + opts.host_args)

            self.progress_bar.update(self.progress_bar.task_completed, advance=1)
            if task:
                self.progress_bar.update(task, advance=0, visible=False)

    async def gen_sim(self, task, aie_target):
        # For simulation, we need to additionally parse the 'remaining' options to avoid things
        # which conflict with the options below (e.g. -o)
        print(opts.host_args)
        host_opts = aie.compiler.aiecc.cl_arguments.strip_host_args_for_aiesim(
            opts.host_args
        )

        sim_dir = self.prepend_tmp("sim")
        shutil.rmtree(sim_dir, ignore_errors=True)
        subdirs = ["arch", "reports", "config", "ps"]

        def make_sim_dir(x):
            dir = os.path.join(sim_dir, x)
            os.makedirs(dir, exist_ok=True)
            return dir

        sim_arch_dir, sim_reports_dir, sim_config_dir, sim_ps_dir = map(
            make_sim_dir, subdirs
        )

        install_path = aie.compiler.aiecc.configure.install_path()

        # Setting everything up if linking against HSA
        if opts.link_against_hsa:
            arch_name = opts.host_target.split("-")[0] + "-hsa"
        else:
            arch_name = opts.host_target.split("-")[0]

        runtime_simlib_path = os.path.join(
            install_path, "aie_runtime_lib", aie_target.upper(), "aiesim"
        )
        runtime_testlib_path = os.path.join(
            install_path,
            "runtime_lib",
            arch_name,
            "test_lib",
            "lib",
        )
        runtime_testlib_include_path = os.path.join(
            install_path,
            "runtime_lib",
            arch_name,
            "test_lib",
            "include",
        )
        sim_makefile = os.path.join(runtime_simlib_path, "Makefile")
        sim_genwrapper = os.path.join(runtime_simlib_path, "genwrapper_for_ps.cpp")
        file_physical = self.prepend_tmp("input_physical.mlir")
        memory_allocator = os.path.join(
            runtime_testlib_path, "libmemory_allocator_sim_aie.a"
        )

        sim_cc_args = [
            "-fPIC",
            "-flto",
            "-fpermissive",
            "-DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR",
            "-Wno-deprecated-declarations",
            "-Wno-enum-constexpr-conversion",
            "-Wno-format-security",
            "-DSC_INCLUDE_DYNAMIC_PROCESSES",
            "-D__AIESIM__",
            "-D__PS_INIT_AIE__",
            "-Og",
            "-Dmain(...)=ps_main(...)",
            "-I" + self.tmpdirname,
            "-I" + opts.aietools_path + "/include",
            "-I" + opts.aietools_path + "/include/drivers/aiengine",
            "-I" + opts.aietools_path + "/data/osci_systemc/include",
            "-I" + opts.aietools_path + "/include/xtlm/include",
            "-I" + opts.aietools_path + "/include/common_cpp/common_cpp_v1_0/include",
            "-I" + runtime_testlib_include_path,
            memory_allocator,
        ]  # clang is picky  # Pickup aie_inc.cpp

        # Don't use shipped version of xaiengine?
        sim_link_args = [
            "-L" + opts.aietools_path + "/lib/lnx64.o",
            "-L" + opts.aietools_path + "/data/osci_systemc/lib/lnx64",
            "-Wl,--as-needed",
            "-lxioutils",
            "-lxaiengine",
            "-ladf_api",
            "-lsystemc",
            "-lxtlm",
            "-flto",
        ]

        processes = []
        processes.append(
            self.do_call(
                task,
                [
                    "aie-translate",
                    "--aie-mlir-to-xpe",
                    file_physical,
                    "-o",
                    os.path.join(sim_reports_dir, "graph.xpe"),
                ],
            )
        )
        processes.append(
            self.do_call(
                task,
                [
                    "aie-translate",
                    "--aie-mlir-to-shim-solution",
                    file_physical,
                    "-o",
                    os.path.join(sim_arch_dir, "aieshim_solution.aiesol"),
                ],
            )
        )
        processes.append(
            self.do_call(
                task,
                [
                    "aie-translate",
                    "--aie-mlir-to-scsim-config",
                    file_physical,
                    "-o",
                    os.path.join(sim_config_dir, "scsim_config.json"),
                ],
            )
        )
        processes.append(
            self.do_call(
                task,
                [
                    "aie-opt",
                    "--aie-find-flows",
                    file_physical,
                    "-o",
                    os.path.join(sim_dir, "flows_physical.mlir"),
                ],
            )
        )
        processes.append(self.do_call(task, ["cp", sim_makefile, sim_dir]))
        processes.append(self.do_call(task, ["cp", sim_genwrapper, sim_ps_dir]))
        processes.append(
            self.do_call(
                task,
                [
                    "clang++",
                    "-O2",
                    "-fuse-ld=lld",
                    "-shared",
                    "-o",
                    os.path.join(sim_ps_dir, "ps.so"),
                    os.path.join(runtime_simlib_path, "genwrapper_for_ps.cpp"),
                    *aie_target_defines(aie_target),
                    *host_opts,
                    *sim_cc_args,
                    *sim_link_args,
                ],
            )
        )
        await asyncio.gather(*processes)
        await self.do_call(
            task,
            [
                "aie-translate",
                "--aie-flows-to-json",
                os.path.join(sim_dir, "flows_physical.mlir"),
                "-o",
                os.path.join(sim_dir, "flows_physical.json"),
            ],
        )

        sim_script = self.prepend_tmp("aiesim.sh")
        sim_script_template = dedent(
            """\
            #!/bin/sh
            prj_name=$(basename $(dirname $(realpath $0)))
            root=$(dirname $(dirname $(realpath $0)))
            vcd_filename=foo
            if [ -n "$1" ]; then
              vcd_filename=$1
            fi
            cd $root
            aiesimulator --pkg-dir=${prj_name}/sim --dump-vcd ${vcd_filename}
            """
        )
        with open(sim_script, "wt") as sim_script_file:
            sim_script_file.write(sim_script_template)
        stats = os.stat(sim_script)
        os.chmod(sim_script, stats.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        target = os.path.join(sim_dir, ".target")
        with open(target, "wt") as target_file:
            target_file.write("hw\n")

        print("Simulation generated...")
        print("To run simulation: " + sim_script)

    async def run_flow(self):
        nworkers = int(opts.nthreads)
        if nworkers == 0:
            nworkers = os.cpu_count()

        self.limit = asyncio.Semaphore(nworkers)
        with progress.Progress(
            *progress.Progress.get_default_columns(),
            progress.TimeElapsedColumn(),
            progress.MofNCompleteColumn(),
            progress.TextColumn("{task.fields[command]}"),
            redirect_stdout=False,
            redirect_stderr=False,
        ) as progress_bar:
            self.progress_bar = progress_bar
            progress_bar.task = progress_bar.add_task(
                "[green] MLIR compilation:", total=1, command="1 Worker"
            )

            file_with_addresses = self.prepend_tmp("input_with_addresses.mlir")
            pass_pipeline = INPUT_WITH_ADDRESSES_PIPELINE.materialize(module=True)
            run_passes(
                pass_pipeline,
                self.mlir_module_str,
                file_with_addresses,
                self.opts.verbose,
            )

            cores = generate_cores_list(await read_file_async(file_with_addresses))
            t = do_run(
                [
                    "aie-translate",
                    "--aie-generate-target-arch",
                    file_with_addresses,
                ],
                self.opts.verbose,
            )
            aie_target = t.stdout.strip()
            if not re.fullmatch("AIE.?", aie_target):
                print(
                    "Unexpected target " + aie_target + ". Exiting...",
                    file=sys.stderr,
                )
                exit(-3)
            aie_peano_target = aie_target.lower() + "-none-elf"

            # Optionally generate insts.txt for NPU instruction stream
            if opts.npu or opts.only_npu:
                generated_insts_mlir = self.prepend_tmp("generated_npu_insts.mlir")
                await self.do_call(
                    progress_bar.task,
                    [
                        "aie-opt",
                        "--aie-dma-to-npu",
                        file_with_addresses,
                        "-o",
                        generated_insts_mlir,
                    ],
                )
                await self.do_call(
                    progress_bar.task,
                    [
                        "aie-translate",
                        "--aie-npu-instgen",
                        generated_insts_mlir,
                        "-o",
                        opts.insts_name,
                    ],
                )
                if opts.only_npu:
                    return

            chess_intrinsic_wrapper_ll_path = await self.prepare_for_chesshack(
                progress_bar.task, aie_target
            )

            # fmt: off
            if opts.unified:
                file_opt_with_addresses = self.prepend_tmp("input_opt_with_addresses.mlir")
                await self.do_call(progress_bar.task, ["aie-opt", f"--pass-pipeline={AIE_LOWER_TO_LLVM()}", file_with_addresses, "-o", file_opt_with_addresses])

                file_llvmir = self.prepend_tmp("input.ll")
                await self.do_call(progress_bar.task, ["aie-translate", "--mlir-to-llvmir", file_opt_with_addresses, "-o", file_llvmir])

                self.unified_file_core_obj = self.prepend_tmp("input.o")
                if opts.compile and opts.xchesscc:
                    file_llvmir_hacked = await self.chesshack(progress_bar.task, file_llvmir, chess_intrinsic_wrapper_ll_path)
                    await self.do_call(progress_bar.task, ["xchesscc_wrapper", aie_target.lower(), "+w", self.prepend_tmp("work"), "-c", "-d", "-f", "+P", "4", file_llvmir_hacked, "-o", self.unified_file_core_obj])
                elif opts.compile:
                    file_llvmir_opt = self.prepend_tmp("input.opt.ll")
                    await self.do_call(progress_bar.task, [self.peano_opt_path, "--passes=default<O2>", "-inline-threshold=10", "-S", file_llvmir, "-o", file_llvmir_opt])
                    await self.do_call(progress_bar.task, [self.peano_llc_path, file_llvmir_opt, "-O2", "--march=" + aie_target.lower(), "--function-sections", "--filetype=obj", "-o", self.unified_file_core_obj])
            # fmt: on

            progress_bar.update(progress_bar.task, advance=0, visible=False)
            progress_bar.task_completed = progress_bar.add_task(
                "[green] AIE Compilation:",
                total=len(cores) + 1,
                command="%d Workers" % nworkers,
            )

            processes = [self.process_host_cgen(aie_target, file_with_addresses)]
            await asyncio.gather(
                *processes
            )  # ensure that process_host_cgen finishes before running gen_sim
            processes = []
            if opts.aiesim:
                processes.append(self.gen_sim(progress_bar.task, aie_target))
            for core in cores:
                processes.append(
                    self.process_core(
                        core,
                        aie_target,
                        aie_peano_target,
                        chess_intrinsic_wrapper_ll_path,
                        file_with_addresses,
                    )
                )
            await asyncio.gather(*processes)

            # Must have elfs, before we build the final binary assembly
            if opts.cdo and opts.execute:
                await self.process_cdo()
            if opts.cdo or opts.xcl:
                await self.process_xclbin_gen(bool(len(cores)))

    def dumpprofile(self):
        sortedruntimes = sorted(
            self.runtimes.items(), key=lambda item: item[1], reverse=True
        )
        for i in range(50):
            if i < len(sortedruntimes):
                s1, s0 = sortedruntimes[i][1], sortedruntimes[i][0]
                print(f"{s1:.4f} sec: {s0}")


def run(mlir_module, args=None):
    global opts
    if args is not None:
        opts = aie.compiler.aiecc.cl_arguments.parse_args(args)

    if "VITIS" not in os.environ:
        # Try to find vitis in the path
        vpp_path = shutil.which("v++")
        if vpp_path:
            vitis_bin_path = os.path.dirname(os.path.realpath(vpp_path))
            vitis_path = os.path.dirname(vitis_bin_path)
            os.environ["VITIS"] = vitis_path
            print("Found Vitis at " + vitis_path)
            os.environ["PATH"] = os.pathsep.join([os.environ["PATH"], vitis_bin_path])

    opts.aietools_path = ""
    if "VITIS" in os.environ:
        vitis_path = os.environ["VITIS"]
        vitis_bin_path = os.path.join(vitis_path, "bin")
        # Find the aietools directory, needed by xchesscc_wrapper

        opts.aietools_path = os.path.join(vitis_path, "aietools")
        if not os.path.exists(opts.aietools_path):
            opts.aietools_path = os.path.join(vitis_path, "cardano")
        os.environ["AIETOOLS"] = opts.aietools_path

        aietools_bin_path = os.path.join(opts.aietools_path, "bin")
        os.environ["PATH"] = os.pathsep.join(
            [os.environ["PATH"], aietools_bin_path, vitis_bin_path]
        )

    else:
        print("Vitis not found...")

    # This path should be generated from cmake
    aie_path = aie.compiler.aiecc.configure.install_path()
    peano_path = os.path.join(opts.peano_install_dir, "bin")
    os.environ["PATH"] = os.pathsep.join([aie_path, os.environ["PATH"]])
    os.environ["PATH"] = os.pathsep.join([peano_path, os.environ["PATH"]])

    if opts.aiesim and not opts.xbridge:
        sys.exit("AIE Simulation (--aiesim) currently requires --xbridge")

    if opts.verbose:
        sys.stderr.write(f"\ncompiling {opts.filename}\n")

    if opts.tmpdir:
        tmpdirname = opts.tmpdir
    elif opts.filename:
        tmpdirname = os.path.basename(opts.filename) + ".prj"
    else:
        tmpdirname = tempfile.mkdtemp()
    tmpdirname = os.path.abspath(tmpdirname)

    try:
        os.mkdir(tmpdirname)
    except FileExistsError:
        pass
    if opts.verbose:
        print("created temporary directory", tmpdirname)

    runner = FlowRunner(str(mlir_module), opts, tmpdirname)
    asyncio.run(runner.run_flow())

    if opts.profiling:
        runner.dumpprofile()


def main():
    global opts
    opts = aie.compiler.aiecc.cl_arguments.parse_args()
    if opts.filename is None:
        print("error: the 'file' positional argument is required.")
        sys.exit(1)

    try:
        with Context() as ctx, Location.unknown():
            with open(opts.filename, "r") as f:
                module = Module.parse(f.read())
            module_str = str(module)
    except Exception as e:
        print(e)
        sys.exit(1)
    run(module_str)
