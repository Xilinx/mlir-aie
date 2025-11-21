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
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from textwrap import dedent
import time
import uuid
import struct

from aie.extras.runtime.passes import Pipeline
from aie.extras.util import find_ops
import aiofiles
import rich.progress as progress

import aie.compiler.aiecc.cl_arguments
import aie.compiler.aiecc.configure
from aie.dialects import aie as aiedialect
from aie.dialects import aiex as aiexdialect
from aie.ir import (
    Context,
    Location,
    Module,
    InsertionPoint,
    IndexType,
    StringAttr,
    IntegerAttr,
    IntegerType,
)
from aie.passmanager import PassManager


def _create_input_with_addresses_pipeline(
    scheme,
    dynamic_objFifos,
    packet_sw_objFifos,
    ctrl_pkt_overlay,
    aie_target,
    opt_level="2",
):
    pipeline = Pipeline()

    # Only add convert-vector-to-aievec for AIE2 and later targets
    # AIE1 ("aie") does not support target_backend="llvmir"
    if aie_target.lower() in ["aie2", "aieml", "aie2p"]:
        # Hoist vector transfer pointers before scf-to-cf conversion (O3 and above only)
        # This runs on the module and walks into aie.core regions
        if int(opt_level) >= 3:
            pipeline.add_pass("aie-hoist-vector-transfer-pointers")
        pipeline.add_pass(
            "convert-vector-to-aievec",
            aie_target=aie_target.lower(),
            target_backend="llvmir",
        )

    # Build nested device pipeline with conditional passes
    device_pipeline = (
        Pipeline()
        .add_pass("aie-trace-to-config")
        .add_pass("aie-trace-pack-reg-writes")
        .add_pass("aiex-inline-trace-config")
        .add_pass("aie-assign-lock-ids")
        .add_pass("aie-register-objectFifos")
        .add_pass(
            "aie-objectFifo-stateful-transform",
            dynamic_objFifos=dynamic_objFifos,
            packet_sw_objFifos=packet_sw_objFifos,
        )
        .add_pass("aie-assign-bd-ids")
        .add_pass("aie-lower-cascade-flows")
        .add_pass("aie-lower-broadcast-packet")
        .add_pass("aie-lower-multicast")
        .add_pass("aie-assign-tile-controller-ids")
        .add_pass(
            "aie-generate-column-control-overlay",
            route_shim_to_tile_ctrl=ctrl_pkt_overlay,
        )
        .add_pass("aie-assign-buffer-addresses", alloc_scheme=scheme)
        .add_pass("aie-vector-transfer-lowering", max_transfer_rank=1)
    )

    # Only add vector-to-pointer-loops for O3 and above
    if int(opt_level) >= 3:
        device_pipeline.add_pass("aie-vector-to-pointer-loops")

    return (
        pipeline.lower_affine()
        .add_pass("aie-canonicalize-device")
        .Nested("aie.device", device_pipeline)
        .convert_scf_to_cf()
    )


INPUT_WITH_ADDRESSES_PIPELINE = _create_input_with_addresses_pipeline

LOWER_TO_LLVM_PIPELINE = (
    Pipeline()
    .canonicalize()
    .cse()
    .expand_strided_metadata()
    .lower_affine()
    .arith_expand()
    .finalize_memref_to_llvm()
    .convert_func_to_llvm(use_bare_ptr_memref_call_conv=True)
    .convert_to_llvm(dynamic=True)
    .canonicalize()
    .cse()
)


def _create_aie_lower_to_llvm_pipeline(
    device_name=None, col=None, row=None, aie_target="aie2", opt_level="2"
):
    pipeline = (
        Pipeline()
        .Nested(
            "aie.device",
            Pipeline()
            .add_pass("aie-localize-locks")
            .add_pass("aie-normalize-address-spaces")
            .add_pass("aie-transform-bfp-types"),
        )
        .add_pass("aie-standard-lowering", device=device_name, tilecol=col, tilerow=row)
        .add_pass("aiex-standard-lowering")
    )

    # Only add aievec-split-load-ups-chains for O3 and above
    if int(opt_level) >= 3:
        pipeline.add_pass("aievec-split-load-ups-chains")

    pipeline.add_pass("convert-aievec-to-llvm", aie_target=aie_target.lower())

    return pipeline + LOWER_TO_LLVM_PIPELINE


AIE_LOWER_TO_LLVM = _create_aie_lower_to_llvm_pipeline

# pipeline to lower and legalize runtime sequence for NPU
NPU_LOWERING_PIPELINE = Pipeline().Nested(
    "aie.device",
    Pipeline()
    .add_pass("aie-materialize-bd-chains")
    .add_pass("aie-substitute-shim-dma-allocations")
    .add_pass("aie-assign-runtime-sequence-bd-ids")
    .add_pass("aie-dma-tasks-to-npu")
    .add_pass("aie-dma-to-npu")
    .add_pass("aie-lower-set-lock"),
)


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
        buffer_args = [f"bo{i}" for i in range(5)]

    arguments = [
        {
            "name": "opcode",
            "address-qualifier": "SCALAR",
            "type": "uint64_t",
            "offset": "0x00",
        },
    ]
    offset = 0x08

    inst_arguments = [
        {
            "name": "instr",
            "memory-connection": "SRAM",
            "address-qualifier": "GLOBAL",
            "type": "char *",
            "offset": str(hex(offset)),
        },
        {
            "name": "ninstr",
            "address-qualifier": "SCALAR",
            "type": "uint32_t",
            "offset": str(hex(offset + 8)),
        },
    ]
    arguments.append(inst_arguments[0])
    arguments.append(inst_arguments[1])
    offset += 12

    for buf in buffer_args:
        arg = {
            "name": buf,
            "memory-connection": "HOST",
            "address-qualifier": "GLOBAL",
            "type": "void*",
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
                        "functional": "0",
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


def emit_partition(mlir_module_str, device_op, design_pdi, kernel_id="0x901"):
    with Context(), Location.unknown():
        module = Module.parse(mlir_module_str)
    device = aiedialect.AIEDevice(int(device_op.device))
    num_cols = aiedialect.get_target_model(device).columns()

    # It's arguable that this should should come from the device model
    # somehow.  Or perhaps that it shouldn't be needed in the
    # XCLbin at all, since it is basically describing information
    # which is already inherent in the CDO.
    # For the time being, we just leave it here.
    if device in [aiedialect.AIEDevice.npu1, aiedialect.AIEDevice.npu2]:
        start_columns = [0]
    else:
        start_columns = list(range(1, 6 - num_cols))

    # Generate a uuid
    pdi_uuid = uuid.uuid4()
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
                    "uuid": str(pdi_uuid),
                    "file_name": design_pdi,
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


def parse_file_as_mlir(mlir_module_str):
    with Context(), Location.unknown():
        return Module.parse(mlir_module_str)


def generate_devices_list(module):
    return [
        (d, d.sym_name.value)
        for d in find_ops(
            module.operation,
            lambda d: isinstance(d.operation.opview, aiedialect.DeviceOp),
        )
        if not opts.device_name or d.sym_name.value == opts.device_name
    ]


def generate_cores_list(device_op):
    return [
        (
            c.tile.owner.opview.col.value,
            c.tile.owner.opview.row.value,
            c.elf_file.value if c.elf_file is not None else None,
        )
        for c in find_ops(
            device_op.operation,
            lambda o: isinstance(o.operation.opview, aiedialect.CoreOp),
        )
    ]


def generate_runtime_sequences_list(device_op):
    return [
        (s, s.sym_name.value)
        for s in find_ops(
            device_op.operation,
            lambda o: isinstance(o.operation.opview, aiexdialect.RuntimeSequenceOp),
        )
        if not opts.sequence_name or s.sym_name.value == opts.sequence_name
    ]


def find_aiebu_asm():
    asm_bin = "aiebu-asm"
    if shutil.which(asm_bin) is None:
        asm_bin = os.path.join("/", "opt", "xilinx", "aiebu", "bin", "aiebu-asm")
        if shutil.which(asm_bin) is None:
            asm_bin = None
    if asm_bin is None:
        print(
            "Error: aiebu-asm not found.",
            file=sys.stderr,
        )
        sys.exit(1)
    return asm_bin


def create_device_id_mapping(devices):
    """Assign an ID to each device in the MLIR; used later to assign IDs for each PDI"""
    device_to_id = {}
    for i, (device_op, device_name) in enumerate(devices, 1):
        device_to_id[device_name] = i
    return device_to_id


def assign_load_pdi_ids(mlir_module_str, device_to_id_mapping):
    """Transform symbolic aiex.npu.load_pdi references to numeric IDs"""
    with Context() as context, Location.unknown():
        module = Module.parse(mlir_module_str)

        for runtime_seq in find_ops(
            module.operation,
            lambda o: isinstance(o.operation.opview, aiexdialect.RuntimeSequenceOp),
        ):
            for load_pdi_op in find_ops(
                runtime_seq.operation,
                lambda o: isinstance(o.operation.opview, aiexdialect.NpuLoadPdiOp)
                and hasattr(o, "device_ref")
                and o.device_ref is not None,
            ):
                device_name = load_pdi_op.device_ref.value
                if device_name not in device_to_id_mapping:
                    print(
                        f"Warning: Device '{device_name}' for load_pdi instruction does not have a matching device PDI."
                    )
                    sys.exit(1)
                pdi_id = device_to_id_mapping[device_name]
                load_pdi_op.id = IntegerAttr.get(
                    IntegerType.get_signless(32, context=context), pdi_id
                )

        return str(module)


def set_elf_file_for_core(core, path):
    with InsertionPoint.at_block_terminator(
        core.parent.regions[0].blocks[0]
    ), Location.unknown():
        result = IndexType.get()
        new_core = aiedialect.CoreOp(result, core.tile)
        for attr in core.attributes:
            new_core.attributes[attr.name] = core.attributes[attr.name]
        new_core.attributes["elf_file"] = StringAttr.get(path)
        new_core_block = new_core.body.blocks.append()
        with InsertionPoint(new_core_block):
            aiedialect.EndOp()
        new_core.move_before(core)
    core.operation.erase()


def emit_design_bif(
    root_path, device_name, has_cores=True, enable_cores=True, unified=False
):
    if unified:
        cdo_unified_file = f"file={root_path}/{device_name}_aie_cdo.bin"
        files = f"{cdo_unified_file}"
    else:
        cdo_elfs_file = f"file={root_path}/{device_name}_aie_cdo_elfs.bin"
        cdo_init_file = f"file={root_path}/{device_name}_aie_cdo_init.bin"
        cdo_enable_file = (
            f"file={root_path}/{device_name}_aie_cdo_enable.bin" if enable_cores else ""
        )
        files = f"{cdo_elfs_file} {cdo_init_file} {cdo_enable_file}"
    return dedent(
        f"""\
        all:
        {{
          id_code = 0x14ca8093
          extended_id_code = 0x01
          image
          {{
            name=aie_image, id=0x1c000000
            {{ type=cdo {files} }}
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
    with Context(), Location.unknown():
        module = Module.parse(mlir_module_str)
        pm = PassManager.parse(pass_pipeline)
        try:
            pm.run(module.operation)
        except Exception as e:
            print("Error running pass pipeline: ", pass_pipeline, e)
            raise e
        mlir_module_str = str(module)
        if outputfile:
            with open(outputfile, "w") as g:
                g.write(mlir_module_str)
    return mlir_module_str


def run_passes_module(pass_pipeline, mlir_module, outputfile=None, verbose=False):
    if verbose:
        print("Running:", pass_pipeline)
    with mlir_module.context, Location.unknown():
        pm = PassManager.parse(pass_pipeline)
        try:
            pm.run(mlir_module.operation)
        except Exception as e:
            print("Error running pass pipeline: ", pass_pipeline, e)
            raise e
        if outputfile:
            mlir_module_str = str(mlir_module)
            with open(outputfile, "w") as g:
                g.write(mlir_module_str)
    return mlir_module


def corefile(dirname, device, core, ext):
    col, row, _ = core
    return os.path.join(dirname, f"{device}_core_{col}_{row}.{ext}")


def aie_target_defines(aie_target):
    if aie_target == "AIE2":
        return ["-D__AIEARCH__=20"]
    return ["-D__AIEARCH__=10"]


def downgrade_ir_for_chess(llvmir_chesslinked):
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
        .replace("captures(none)", "nocapture")
        .replace("getelementptr inbounds nuw", "getelementptr inbounds")
    )
    return llvmir_chesslinked


def downgrade_ir_for_peano(llvmir):
    llvmir = llvmir.replace("getelementptr inbounds nuw", "getelementptr inbounds")
    return llvmir


def drop_alignment_for_peano(llvmir):
    # Remove any ", align <integer>" attribute occurrences
    llvmir = re.sub(r",\s*align\s+\d+", "", llvmir)
    return llvmir


def get_peano_target(aie_target):
    if not re.fullmatch("AIE.?.?", aie_target):
        print(
            "Unexpected target " + aie_target + ". Exiting...",
            file=sys.stderr,
        )
        exit(-3)
    aie_peano_target = aie_target.lower() + "-none-unknown-elf"
    return aie_peano_target


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

    def pdi_file_name(self, device_name):
        return (
            opts.pdi_name.format(device_name)
            if opts.pdi
            else self.prepend_tmp(f"{device_name}.pdi")
        )

    def npu_insts_file_name(self, device_name, seq_name):
        return (
            opts.insts_name.format(device_name, seq_name)
            if opts.npu
            else self.prepend_tmp(f"{device_name}_{seq_name}.bin")
        )

    async def do_call(self, task_id, command, force=False):
        if self.stopall:
            return

        commandstr = " ".join(command)
        if task_id:
            self.progress_bar.update(task_id, advance=0, command=commandstr[0:30])
        start = time.time()
        if self.opts.verbose:
            print(commandstr)
        if self.opts.execute or force:
            proc = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            ret = proc.returncode
            if self.opts.verbose and stdout:
                print(f"{stdout.decode()}")
            if ret != 0 and stderr:
                print(f"{stderr.decode()}", file=sys.stderr)
        else:
            ret = 0
        end = time.time()
        if self.opts.verbose:
            print(f"Done in {end - start:.3f} sec: {commandstr}")
        self.runtimes[commandstr] = end - start
        if task_id:
            self.progress_bar.update(task_id, advance=1, command="")
            self.maxtasks = max(
                self.progress_bar._tasks[task_id].completed, self.maxtasks
            )
            self.progress_bar.update(task_id, total=self.maxtasks)

        if ret != 0:
            if task_id:
                self.progress_bar.update(task_id, description="[red] Error")
            print("Error encountered while running: " + commandstr, file=sys.stderr)
            sys.exit(ret)

    # In order to run xchesscc on modern ll code, we need a bunch of hacks.
    async def chesshack(self, task, llvmir, aie_target):
        llvmir_chesshack = llvmir + "chesshack.ll"
        llvmir_chesslinked_path = llvmir + "chesslinked.ll"
        if not self.opts.execute:
            return llvmir_chesslinked_path

        install_path = aie.compiler.aiecc.configure.install_path()
        runtime_lib_path = os.path.join(install_path, "aie_runtime_lib")
        chess_intrinsic_wrapper_ll_path = os.path.join(
            runtime_lib_path, aie_target.upper(), "chess_intrinsic_wrapper.ll"
        )

        llvmir_ir = await read_file_async(llvmir)
        llvmir_hacked_ir = downgrade_ir_for_chess(llvmir_ir)
        await write_file_async(llvmir_hacked_ir, llvmir_chesshack)

        if aie_target.casefold() == "AIE2".casefold():
            target = "target_aie_ml"
        elif aie_target.casefold() == "AIE2P".casefold():
            target = "target_aie2p"
        else:
            target = "target"
        assert os.path.exists(llvmir_chesshack)
        await self.do_call(
            task,
            [
                # The path below is cheating a bit since it refers directly to the AIE1
                # version of llvm-link, rather than calling the architecture-specific
                # tool version.
                opts.aietools_path
                + "/tps/lnx64/"
                + target
                + "/bin/LNa64bin/chess-llvm-link",
                llvmir_chesshack,
                chess_intrinsic_wrapper_ll_path,
                "-S",
                "-o",
                llvmir_chesslinked_path,
            ],
        )

        return llvmir_chesslinked_path

    # In order to run peano on modern ll code, we need a bunch of hacks.
    async def peanohack(self, llvmir):
        llvmir_peanohack = llvmir + "peanohack.ll"
        if not self.opts.execute:
            return llvmir_peanohack

        llvmir_ir = await read_file_async(llvmir)
        llvmir_hacked_ir = downgrade_ir_for_peano(llvmir_ir)
        llvmir_hacked_ir = drop_alignment_for_peano(llvmir_hacked_ir)
        await write_file_async(llvmir_hacked_ir, llvmir_peanohack)

        return llvmir_peanohack

    async def process_cores(
        self,
        device_op,
        device_name,
        file_with_addresses,
        aie_target,
        aie_peano_target,
        parent_task_id,
    ):
        # If unified compilation is on, we create a single object file that
        # contains the compiled code for all cores. If not, the equivalent
        # of the below is created for each core inside of process_core
        # (singular).

        # fmt: off
        if opts.unified:
            file_opt_with_addresses = self.prepend_tmp(f"{device_name}_input_opt_with_addresses.mlir")
            await self.do_call(parent_task_id, ["aie-opt", f"--pass-pipeline={AIE_LOWER_TO_LLVM(device_name, aie_target=aie_target, opt_level=opts.opt_level)}", file_with_addresses, "-o", file_opt_with_addresses])

            file_llvmir = self.prepend_tmp(f"{device_name}_input.ll")
            await self.do_call(parent_task_id, ["aie-translate", "--mlir-to-llvmir", file_opt_with_addresses, "-o", file_llvmir])

            unified_file_core_obj = self.prepend_tmp(f"{device_name}_input.o")
            if opts.compile and opts.xchesscc:
                file_llvmir_hacked = await self.chesshack(parent_task_id, file_llvmir, aie_target)
                await self.do_call(parent_task_id, ["xchesscc_wrapper", aie_target.lower(), "+w", self.prepend_tmp("work"), "-c", "-d", "+Wclang,-xir", "-f", file_llvmir_hacked, "-o", unified_file_core_obj])
            elif opts.compile:
                file_llvmir_hacked = await self.peanohack(file_llvmir)
                file_llvmir_opt = self.prepend_tmp(f"{device_name}_input.opt.ll")
                opt_level = opts.opt_level
                # Disable loop idiom memset for O3 and above.
                # Rationale: memset is executed as scalar operation, while zeroinitializer will be executed as vector
                opt_flags = [f"--passes=default<O{opt_level}>"]
                if int(opt_level) >= 3:
                    opt_flags.append("-disable-loop-idiom-memset")
                opt_flags.extend(["-inline-threshold=10", "-S", file_llvmir_hacked, "-o", file_llvmir_opt])
                await self.do_call(parent_task_id, [self.peano_opt_path] + opt_flags)
                await self.do_call(parent_task_id, [self.peano_llc_path, file_llvmir_opt, f"-O{opt_level}", "--march=" + aie_target.lower(), "--function-sections", "--filetype=obj", "-o", unified_file_core_obj])
        else:
            unified_file_core_obj = None
        # fmt: on

        # Now, process each individual core.
        processes = []
        cores = generate_cores_list(device_op)
        for core in cores:
            processes.append(
                self.process_core(
                    device_name,
                    core,
                    aie_target,
                    aie_peano_target,
                    file_with_addresses,
                    unified_file_core_obj,
                    parent_task_id,
                )
            )
        device_elf_paths = await asyncio.gather(*processes)
        elf_paths = {}
        for (col, row, _), elf_path in zip(cores, device_elf_paths):
            elf_paths[(col, row)] = elf_path

        # copy the elfs left by proess_core to the tmpdir for process_cdo
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

        return elf_paths

    async def process_core(
        self,
        device_name,
        core,
        aie_target,
        aie_peano_target,
        file_with_addresses,
        unified_file_core_obj,
        parent_task_id,
    ):
        async with self.limit:
            if self.stopall:
                return

            install_path = aie.compiler.aiecc.configure.install_path()
            runtime_lib_path = os.path.join(
                install_path, "aie_runtime_lib", aie_target.upper()
            )

            # --gc-sections to eliminate unneeded code.
            # --orphan-handling=error to ensure that the linker script is as expected.
            # If there are orphaned input sections, then they'd likely end up outside of the normal program memory.
            clang_link_args = ["-Wl,--gc-sections", "-Wl,--orphan-handling=error"]

            task = self.progress_bar.add_task(
                "[yellow] Core (%d, %d)" % core[0:2],
                total=self.maxtasks,
                command="starting",
            )

            # fmt: off
            corecol, corerow, elf_file = core
            if not opts.unified:
                file_core = corefile(self.tmpdirname, device_name, core, "mlir")
                file_opt_core = corefile(self.tmpdirname, device_name, core, "opt.mlir")
                await self.do_call(task, ["aie-opt", f"--pass-pipeline={AIE_LOWER_TO_LLVM(device_name, corecol, corerow, aie_target, opts.opt_level)}", file_with_addresses, "-o", file_opt_core])
            if self.opts.xbridge:
                file_core_bcf = corefile(self.tmpdirname, device_name, core, "bcf")
                await self.do_call(task, ["aie-translate", file_with_addresses, "--aie-generate-bcf", "--aie-device-name", device_name, "--tilecol=%d" % corecol, "--tilerow=%d" % corerow, "-o", file_core_bcf])
            else:
                file_core_ldscript = corefile(self.tmpdirname, device_name, core, "ld.script")
                await self.do_call(task, ["aie-translate", file_with_addresses, "--aie-generate-ldscript", "--aie-device-name", device_name, "--tilecol=%d" % corecol, "--tilerow=%d" % corerow, "-o", file_core_ldscript])
            if not self.opts.unified:
                file_core_llvmir = corefile(self.tmpdirname, device_name, core, "ll")
                await self.do_call(task, ["aie-translate", "--mlir-to-llvmir", file_opt_core, "-o", file_core_llvmir])
                file_core_obj = corefile(self.tmpdirname, device_name, core, "o")

            file_core_elf = elf_file if elf_file else corefile(self.tmpdirname, device_name, core, "elf")

            if opts.compile and opts.xchesscc:
                if not opts.unified:
                    file_core_llvmir_chesslinked = await self.chesshack(task, file_core_llvmir, aie_target)
                    if self.opts.link and self.opts.xbridge:
                        link_with_obj = await extract_input_files(file_core_bcf)
                        await self.do_call(task, ["xchesscc_wrapper", aie_target.lower(), "+w", self.prepend_tmp("work"), "-d", "+Wclang,-xir", "-f", file_core_llvmir_chesslinked, link_with_obj, "+l", file_core_bcf, "-o", file_core_elf])
                    elif self.opts.link:
                        await self.do_call(task, ["xchesscc_wrapper", aie_target.lower(), "+w", self.prepend_tmp("work"), "-c", "-d", "+Wclang,-xir", "-f", file_core_llvmir_chesslinked, "-o", file_core_obj])
                        opt_level = opts.opt_level
                        await self.do_call(task, [self.peano_clang_path, f"-O{opt_level}", "--target=" + aie_peano_target, file_core_obj, *clang_link_args, "-Wl,-T," + file_core_ldscript, "-o", file_core_elf])
                else:
                    file_core_obj = unified_file_core_obj
                    if opts.link and opts.xbridge:
                        link_with_obj = await extract_input_files(file_core_bcf)
                        await self.do_call(task, ["xchesscc_wrapper", aie_target.lower(), "+w", self.prepend_tmp("work"), "-d", "-f", file_core_obj, link_with_obj, "+l", file_core_bcf, "-o", file_core_elf])
                    elif opts.link:
                        opt_level = opts.opt_level
                        await self.do_call(task, [self.peano_clang_path, f"-O{opt_level}", "--target=" + aie_peano_target, file_core_obj, *clang_link_args, "-Wl,-T," + file_core_ldscript, "-o", file_core_elf])

            elif opts.compile:
                if not opts.unified:
                    file_core_llvmir_peanohacked = await self.peanohack(file_core_llvmir)
                    file_core_llvmir_stripped = corefile(self.tmpdirname, device_name, core, "stripped.ll")
                    opt_level = opts.opt_level
                    # Disable loop idiom memset for O3 and above.
                    # Rationale: memset is executed as scalar operation, while zeroinitializer will be executed as vector
                    opt_flags = [f"--passes=default<O{opt_level}>,strip"]
                    if int(opt_level) >= 3:
                        opt_flags.append("-disable-loop-idiom-memset")
                    opt_flags.extend(["-S", file_core_llvmir_peanohacked, "-o", file_core_llvmir_stripped])
                    await self.do_call(task, [self.peano_opt_path] + opt_flags)
                    await self.do_call(task, [self.peano_llc_path, file_core_llvmir_stripped, f"-O{opt_level}", "--march=" + aie_target.lower(), "--function-sections", "--filetype=obj", "-o", file_core_obj])
                else:
                    file_core_obj = unified_file_core_obj

                if opts.link and opts.xbridge:
                    link_with_obj = await extract_input_files(file_core_bcf)
                    await self.do_call(task, ["xchesscc_wrapper", aie_target.lower(), "+w", self.prepend_tmp("work"), "-d", "-f", file_core_obj, link_with_obj, "+l", file_core_bcf, "-o", file_core_elf])
                elif opts.link:
                    opt_level = opts.opt_level
                    await self.do_call(task, [self.peano_clang_path, f"-O{opt_level}", "--target=" + aie_peano_target, file_core_obj, *clang_link_args, "-Wl,-T," + file_core_ldscript, "-o", file_core_elf])

            self.progress_bar.update(parent_task_id, advance=1)
            self.progress_bar.update(task, advance=0, visible=False)
            # fmt: on

            return file_core_elf

    async def write_elf_paths_to_mlir(self, input_physical, elf_paths):
        # After core ELF files are generated, we create a new MLIR file with
        # references to those generated files in place of their IR.
        with Context(), Location.unknown():
            input_physical_with_elfs_module = Module.parse(
                await read_file_async(input_physical)
            )
            for device in find_ops(
                input_physical_with_elfs_module.operation,
                lambda o: isinstance(o.operation.opview, aiedialect.DeviceOp),
            ):
                device_name = device.sym_name.value
                if device_name not in elf_paths:
                    continue

                for core in find_ops(
                    device, lambda o: isinstance(o.operation.opview, aiedialect.CoreOp)
                ):
                    col = core.tile.owner.opview.col.value
                    row = core.tile.owner.opview.row.value
                    if (col, row) not in elf_paths[device_name]:
                        continue

                    set_elf_file_for_core(core, elf_paths[device_name][(col, row)])

            input_physical_with_elfs_str = str(input_physical_with_elfs_module)
            input_physical_with_elfs = self.prepend_tmp("input_physical_with_elfs.mlir")

            with open(input_physical_with_elfs, "w") as f:
                f.write(input_physical_with_elfs_str)
            return input_physical_with_elfs

    async def process_cdo(self, module_str, device_name):
        with Context(), Location.unknown():
            input_physical = Module.parse(module_str)
            aiedialect.generate_cdo(
                input_physical.operation, self.tmpdirname, device_name
            )

    async def process_txn(self, module_str, device_name):
        file_txn = self.prepend_tmp(f"{device_name}_txn.mlir")
        with Context(), Location.unknown():
            run_passes(
                f"builtin.module(aie.device(convert-aie-to-transaction{{device-name={device_name} elf-dir={self.tmpdirname}}}))",
                module_str,
                file_txn,
                self.opts.verbose,
            )
            txn_dest = opts.txn_name.format(device_name)
            if opts.verbose:
                print(f"copy {file_txn} to {txn_dest}")
            shutil.copy(file_txn, txn_dest)
        return file_txn

    async def aiebu_asm(
        self, input_file, output_file, ctrl_packet_file=None, ctrl_packet_idx=0
    ):
        asm_bin = find_aiebu_asm()

        args = [
            asm_bin,
            "-t",
            "aie2txn",
            "-c",
            input_file,
            "-o",
            output_file,
        ]

        if ctrl_packet_file:
            ctrl_packet_size = os.path.getsize(ctrl_packet_file)
            exteral_buffers_json = {
                "external_buffers": {
                    "buffer_ctrl": {
                        "xrt_id": ctrl_packet_idx,
                        "logical_id": -1,
                        "size_in_bytes": ctrl_packet_size,
                        "ctrl_pkt_buffer": 1,
                        "name": "runtime_control_packet",
                    },
                }
            }
            with open(self.prepend_tmp("external_buffers.json"), "w") as f:
                json.dump(exteral_buffers_json, f, indent=2)
            args = args + [
                "-j",
                self.prepend_tmp("external_buffers.json"),
                "-p",
                ctrl_packet_file,
            ]

        await self.do_call(None, args)

    async def generate_full_elf_config_json(
        self, devices, device_to_id_mapping, opts, parent_task=None
    ):
        config = {"xrt-kernels": []}

        for device_op, device_name in devices:
            sequences = generate_runtime_sequences_list(device_op)

            max_arg_count = max(
                len(seq_op.body.blocks[0].arguments) for seq_op, seq_name in sequences
            )
            arguments = [
                {"name": f"arg_{i}", "type": "char *", "offset": hex(i * 8)}
                for i in range(max_arg_count)
            ]

            kernel_entry = {
                "name": device_name,
                "arguments": arguments,
                "instance": [],
                "PDIs": [],
            }

            pdi_id = device_to_id_mapping[device_name]
            pdi_filename = self.pdi_file_name(device_name)
            kernel_entry["PDIs"].append({"id": pdi_id, "PDI_file": pdi_filename})

            for seq_op, seq_name in sequences:
                insts_filename = self.npu_insts_file_name(device_name, seq_name)
                kernel_entry["instance"].append(
                    {"id": seq_name, "TXN_ctrl_code_file": insts_filename}
                )

            config["xrt-kernels"].append(kernel_entry)

        return config

    async def assemble_full_elf(
        self, config_json_path, output_elf_path, parent_task=None
    ):
        asm_bin = find_aiebu_asm()
        args = [
            asm_bin,
            "-t",
            "aie2_config",
            "-j",
            config_json_path,
            "-o",
            output_elf_path,
        ]
        await self.do_call(parent_task, args)
        if self.opts.verbose:
            print(f"Generated full ELF: {output_elf_path}")

    async def generate_full_elf(self, devices, device_to_id_mapping, parent_task=None):
        """Generate config.json and invoke aiebu-asm after all artifacts are ready"""
        if parent_task:
            self.progress_bar.update(
                parent_task, advance=0, command="Generating config.json"
            )
        config = await self.generate_full_elf_config_json(
            devices, device_to_id_mapping, self.opts, parent_task
        )
        config_json_path = self.prepend_tmp("config.json")
        await write_file_async(json.dumps(config, indent=2), config_json_path)
        if self.opts.verbose:
            if self.opts.verbose:
                print(f"Generated config.json: {config_json_path}")
        if parent_task:
            self.progress_bar.update(
                parent_task, advance=1, command="Generating config.json"
            )
        full_elf_path = self.opts.full_elf_name or "aie.elf"
        await self.assemble_full_elf(config_json_path, full_elf_path, parent_task)

    async def process_ctrlpkt(self, module_str, device_op, device_name):
        file_ctrlpkt_mlir = self.prepend_tmp(f"{device_name}_ctrlpkt.mlir")
        file_ctrlpkt_bin = opts.ctrlpkt_name.format(device_name)
        file_ctrlpkt_dma_seq_mlir = self.prepend_tmp(
            f"{device_name}_ctrlpkt_dma_seq.mlir"
        )
        ctrlpkt_mlir_str = run_passes(
            "builtin.module(aie.device(convert-aie-to-transaction{elf-dir="
            + self.tmpdirname
            + "},aie-txn-to-ctrl-packet,aie-legalize-ctrl-packet))",
            module_str,
            file_ctrlpkt_mlir,
            self.opts.verbose,
        )

        # aie-translate --aie-ctrlpkt-to-bin -o ctrlpkt.bin
        with Context(), Location.unknown():
            ctrlpkt_bin = aiedialect.generate_control_packets(
                Module.parse(ctrlpkt_mlir_str).operation, device_name
            )
        with open(file_ctrlpkt_bin, "wb") as f:
            f.write(struct.pack("I" * len(ctrlpkt_bin), *ctrlpkt_bin))

        # aie-opt --aie-ctrl-packet-to-dma -aie-dma-to-npu
        ctrl_seq_str = run_passes(
            "builtin.module(aie.device(aie-ctrl-packet-to-dma,aie-dma-to-npu))",
            ctrlpkt_mlir_str,
            file_ctrlpkt_dma_seq_mlir,
            self.opts.verbose,
        )

        # aie-translate --aie-npu-to-binary -o npu_insts.bin
        with Context(), Location.unknown():
            insts_bin = aiedialect.translate_npu_to_binary(
                Module.parse(ctrl_seq_str).operation, device_name, opts.sequence_name
            )
        with open(opts.insts_name.format(device_name, "seq"), "wb") as f:
            f.write(struct.pack("I" * len(insts_bin), *insts_bin))

        ctrl_idx = 0
        with Context(), Location.unknown():
            # walk the device to find runtime sequence
            seqs = find_ops(
                device_op.operation,
                lambda o: isinstance(o.operation.opview, aiexdialect.RuntimeSequenceOp),
            )
            if seqs:
                ctrl_idx = len(seqs[0].regions[0].blocks[0].arguments.types)
        await self.aiebu_asm(
            opts.insts_name.format(device_name, "seq"),
            opts.elf_name.format(device_name),
            file_ctrlpkt_bin,
            ctrl_idx,
        )

    async def process_elf(self, npu_insts_module, device_name):
        # translate npu instructions to binary and write to file
        npu_insts = aiedialect.translate_npu_to_binary(
            npu_insts_module.operation, device_name, opts.sequence_name
        )

        npu_insts_bin = self.prepend_tmp(f"{device_name}_elf_insts.bin")
        with open(npu_insts_bin, "wb") as f:
            f.write(struct.pack("I" * len(npu_insts), *npu_insts))

        await self.aiebu_asm(npu_insts_bin, opts.elf_name.format(device_name))

    async def process_pdi_gen(self, device_name, file_design_pdi):
        file_design_bif = self.prepend_tmp(f"{device_name}_design.bif")

        await write_file_async(
            emit_design_bif(self.tmpdirname, device_name), file_design_bif
        )

        await self.do_call(
            None,
            [
                "bootgen",
                "-arch",
                "versal",
                "-image",
                file_design_bif,
                "-o",
                file_design_pdi,
                "-w",
            ],
        )

    # generate an xclbin. The inputs are self.mlir_module_str and the cdo
    # binaries from the process_cdo step.
    async def process_xclbin_gen(self, device_op, device_name):
        task = self.progress_bar.add_task(
            "[yellow] XCLBIN generation ", total=10, command="starting"
        )

        file_mem_topology = self.prepend_tmp(f"{device_name}_mem_topology.json")
        file_partition = self.prepend_tmp(f"{device_name}_aie_partition.json")
        file_input_partition = self.prepend_tmp(
            f"{device_name}_aie_input_partition.json"
        )
        file_kernels = self.prepend_tmp(f"{device_name}_kernels.json")
        file_pdi = self.pdi_file_name(device_name)

        # collect the tasks to generate the inputs to xclbinutil
        processes = []

        # generate mem_topology.json
        processes.append(
            write_file_async(json.dumps(mem_topology, indent=2), file_mem_topology)
        )

        # generate aie_partition.json
        processes.append(
            write_file_async(
                json.dumps(
                    emit_partition(
                        self.mlir_module_str, device_op, file_pdi, opts.kernel_id
                    ),
                    indent=2,
                ),
                file_partition,
            )
        )

        # generate kernels.json
        buffer_arg_names = [f"bo{i}" for i in range(5)]
        processes.append(
            write_file_async(
                json.dumps(
                    emit_design_kernel_json(
                        opts.kernel_name,
                        opts.kernel_id,
                        opts.instance_name,
                        buffer_arg_names,
                    ),
                    indent=2,
                ),
                file_kernels,
            )
        )

        # generate pdi
        processes.append(self.process_pdi_gen(device_name, file_pdi))

        # get partition info from input xclbin, if present
        if opts.xclbin_input:
            processes.append(
                self.do_call(
                    task,
                    [
                        "xclbinutil",
                        "--dump-section",
                        f"AIE_PARTITION:JSON:{file_input_partition}",
                        "--force",
                        "--quiet",
                        "--input",
                        opts.xclbin_input,
                    ],
                )
            )

        # wait for all of the above to finish
        await asyncio.gather(*processes)

        # fmt: off
        if opts.xclbin_input:
            # patch the input partition json with the new partition information
            with open(file_input_partition) as f:
                input_partition = json.load(f)
            with open(file_partition) as f:
                new_partition = json.load(f)
            input_partition["aie_partition"]["PDIs"].append(new_partition["aie_partition"]["PDIs"][0])
            with open(file_partition, "w") as f:
                json.dump(input_partition, f, indent=2)
            flag = ['--input', opts.xclbin_input]
        else:
            flag = ["--add-replace-section", "MEM_TOPOLOGY:JSON:" + file_mem_topology]

        # run xclbinutil to generate the xclbin
        await self.do_call(task, ["xclbinutil"] + flag +
                                 ["--add-kernel", file_kernels,
                                  "--add-replace-section", "AIE_PARTITION:JSON:" + file_partition,
                                  "--force", "--quiet", "--output", opts.xclbin_name.format(device_name)])
        # fmt: on

    async def process_host_cgen(self, aie_target, file_physical_with_elfs, device_name):
        async with self.limit:
            if self.stopall:
                return

            task = self.progress_bar.add_task(
                "[yellow] Host compilation ", total=10, command="starting"
            )

            if opts.link_against_hsa:
                file_inc_cpp = self.prepend_tmp("aie_data_movement.cpp")
                await self.do_call(
                    task,
                    [
                        "aie-translate",
                        "--aie-generate-hsa",
                        "--aie-device-name",
                        device_name,
                        file_physical_with_elfs,
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
                "-Wl,-R" + xaiengine_lib_path,
                "-I" + self.tmpdirname,
                "-fuse-ld=lld",
                "-lm",
                "-lxaienginecdo",
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
            self.progress_bar.update(task, advance=0, visible=False)

    async def gen_sim(self, task, aie_target, file_physical, device_name):
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
        sim_genwrapper = os.path.join(runtime_simlib_path, "genwrapper_for_ps.cpp")
        memory_allocator = os.path.join(
            runtime_testlib_path, "libmemory_allocator_sim_aie.a"
        )
        # Getting a pointer to the libxaie include and library
        runtime_xaiengine_path = os.path.join(
            install_path, "runtime_lib", arch_name, "xaiengine"
        )
        xaiengine_include_path = os.path.join(runtime_xaiengine_path, "include")
        xaiengine_lib_path = os.path.join(runtime_xaiengine_path, "lib")
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
            "-I" + xaiengine_include_path,
            "-I" + opts.aietools_path + "/data/osci_systemc/include",
            "-I" + opts.aietools_path + "/include/xtlm/include",
            "-I" + opts.aietools_path + "/include/common_cpp/common_cpp_v1_0/include",
            "-I" + runtime_testlib_include_path,
            memory_allocator,
        ]  # clang is picky  # Pickup aie_inc.cpp

        sim_link_args = [
            "-L" + xaiengine_lib_path,
            "-lxaienginecdo",
            "-L" + opts.aietools_path + "/lib/lnx64.o",
            "-L" + opts.aietools_path + "/lib/lnx64.o/Ubuntu",
            "-L" + opts.aietools_path + "/data/osci_systemc/lib/lnx64",
            "-Wl,--as-needed",
            "-lsystemc",
            "-lxtlm",
        ]

        processes = []
        processes.append(
            self.do_call(
                task,
                [
                    "aie-translate",
                    "--aie-mlir-to-xpe",
                    "--aie-device-name",
                    device_name,
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
                    "--aie-device-name",
                    device_name,
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
                    "--aie-device-name",
                    device_name,
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
                    sim_genwrapper,
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
                "--aie-device-name",
                device_name,
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

    async def get_aie_target_for_device(self, mlir_input_file, device_name):
        t = do_run(
            [
                "aie-translate",
                "--aie-generate-target-arch",
                "--aie-device-name",
                device_name,
                mlir_input_file,
            ],
            self.opts.verbose,
        )
        aie_target = t.stdout.strip()
        return (aie_target, get_peano_target(aie_target))

    async def run_flow(self):
        # First, we run some aie-opt passes that transform the MLIR for every
        # device. Then, we generate the core code for each AIE core tile in
        # every device. The result of this is an ELF file with each core's
        # code; we generate a new MLIR file which referencees those generated
        # ELF files in place of their IR code. We then generate artifacts for
        # each device individually, using this last generated IR.

        nworkers = int(opts.nthreads)
        if nworkers == 0:
            nworkers = os.cpu_count()

        module = parse_file_as_mlir(self.mlir_module_str)

        self.limit = asyncio.Semaphore(nworkers)
        with progress.Progress(
            *progress.Progress.get_default_columns(),
            progress.TimeElapsedColumn(),
            progress.MofNCompleteColumn(),
            progress.TextColumn("{task.fields[command]}"),
            redirect_stdout=False,
            redirect_stderr=False,
            disable=not opts.progress,
        ) as progress_bar:
            self.progress_bar = progress_bar

            # 1.) MLIR transformations

            task1 = progress_bar.add_task(
                "[green] MLIR compilation", total=3, command="1 Worker"
            )

            self.progress_bar.update(task1, advance=1, command="Generating device list")
            devices = generate_devices_list(module)
            if len(devices) == 0:
                print("error: input MLIR must contain at least one aie.device")
                sys.exit(1)
            aie_targets, aie_peano_targets = [], []
            for device_op, device_name in devices:
                aie_target, aie_peano_target = await self.get_aie_target_for_device(
                    opts.filename, device_name
                )
                aie_targets.append(aie_target)
                aie_peano_targets.append(aie_peano_target)

            if len(aie_targets) == 0 or not all(
                aie_target == aie_targets[0] for aie_target in aie_targets
            ):
                print("error: all device targets in the file must be the same")
                # TODO: remove this restriction? currently only needed by AIEVec
                sys.exit(1)
            aie_target, aie_peano_target = aie_targets[0], aie_peano_targets[0]

            # Handle full ELF generation configuration
            if opts.generate_full_elf:
                device_to_id_mapping = create_device_id_mapping(devices)
                self.mlir_module_str = assign_load_pdi_ids(
                    self.mlir_module_str, device_to_id_mapping
                )
                transformed_mlir_path = self.prepend_tmp("input_with_pdi_ids.mlir")
                await write_file_async(self.mlir_module_str, transformed_mlir_path)

            pass_pipeline = INPUT_WITH_ADDRESSES_PIPELINE(
                opts.alloc_scheme,
                opts.dynamic_objFifos,
                opts.packet_sw_objFifos,
                opts.ctrl_pkt_overlay,
                aie_target,
                opts.opt_level,
            ).materialize(module=True)

            self.progress_bar.update(task1, advance=1, command=pass_pipeline[0:30])
            file_with_addresses = self.prepend_tmp("input_with_addresses.mlir")
            run_passes(
                pass_pipeline,
                self.mlir_module_str,
                file_with_addresses,
                self.opts.verbose,
            )

            requires_routing = (
                opts.xcl
                or opts.cdo
                or opts.pdi
                or opts.compile
                or opts.compile_host
                or opts.aiesim
            )
            if requires_routing:
                input_physical = self.prepend_tmp("input_physical.mlir")
                processes = [
                    self.do_call(
                        task1,
                        [
                            "aie-opt",
                            "--aie-create-pathfinder-flows",
                            file_with_addresses,
                            "-o",
                            input_physical,
                        ],
                        force=True,
                    )
                ]
                await asyncio.gather(*processes)
            else:
                input_physical = file_with_addresses

            self.progress_bar.update(task1, advance=1)

            # 2.) Generate code for each core
            requires_core_compilation = (
                opts.xcl
                or opts.cdo
                or opts.pdi
                or opts.compile
                or opts.compile_host
                or opts.aiesim
            )
            if requires_core_compilation:
                task2 = progress_bar.add_task(
                    "[green] Generating code for each core", total=3, command=""
                )

                # create core ELF files for each device and core
                elf_paths = {}
                for i, (device_op, device_name) in enumerate(devices):
                    aie_target, aie_peano_target = aie_targets[i], aie_peano_targets[i]
                    elf_paths[device_name] = await self.process_cores(
                        device_op,
                        device_name,
                        file_with_addresses,
                        aie_target,
                        aie_peano_target,
                        task2,
                    )
                input_physical_with_elfs = await self.write_elf_paths_to_mlir(
                    input_physical, elf_paths
                )
            else:
                input_physical_with_elfs = input_physical

            # 3.) Targets that require the cores to be lowered but apply across all devices

            npu_insts_module = None
            if opts.npu or opts.elf or opts.generate_full_elf and not opts.ctrlpkt:
                task3 = progress_bar.add_task(
                    "[green] Lowering NPU instructions", total=2, command=""
                )
                with Context(), Location.unknown():
                    input_physical_with_elfs_module = Module.parse(
                        await read_file_async(input_physical_with_elfs)
                    )
                    pass_pipeline = NPU_LOWERING_PIPELINE.materialize(module=True)
                    npu_insts_file = self.prepend_tmp(f"npu_insts.mlir")
                    self.progress_bar.update(
                        task3, advance=1, command=pass_pipeline[0:30]
                    )
                    npu_insts_module = run_passes_module(
                        pass_pipeline,
                        input_physical_with_elfs_module,
                        npu_insts_file,
                        self.opts.verbose,
                    )
                    self.progress_bar.update(task3, advance=1)

            # 4.) Generate compilation artifacts for each device

            # create other artifacts for each device
            task4 = progress_bar.add_task(
                "[green] Generating device artifacts", total=len(devices), command=""
            )
            for device_op, device_name in devices:
                aie_target, aie_peano_target = await self.get_aie_target_for_device(
                    input_physical, device_name
                )
                await self.run_flow_for_device(
                    input_physical,
                    input_physical_with_elfs,
                    npu_insts_module,
                    device_op,
                    device_name,
                    aie_target,
                    aie_peano_target,
                    task4,
                )

            self.maxtasks = 2
            task5 = progress_bar.add_task(
                "[green] Creating full ELF", total=2, command=""
            )
            if opts.generate_full_elf:
                await self.generate_full_elf(devices, device_to_id_mapping, task5)

    async def run_flow_for_device(
        self,
        input_physical,
        input_physical_with_elfs,
        npu_insts_module,
        device_op,
        device_name,
        aie_target,
        aie_peano_target,
        parent_task_id,
    ):
        pb = self.progress_bar
        nworkers = int(opts.nthreads)

        # Optionally generate insts.bin for NPU instruction stream
        if opts.npu or opts.generate_full_elf and not opts.ctrlpkt:
            # write each runtime sequence binary into its own file
            runtime_sequences = generate_runtime_sequences_list(device_op)
            for seq_op, seq_name in runtime_sequences:
                pb.update(
                    parent_task_id,
                    description=f"[green] Creating NPU instruction binary",
                )
                npu_insts = aiedialect.translate_npu_to_binary(
                    npu_insts_module.operation, device_name, seq_name
                )
                npu_insts_path = self.npu_insts_file_name(device_name, seq_name)
                with open(npu_insts_path, "wb") as f:
                    f.write(struct.pack("I" * len(npu_insts), *npu_insts))
                pb.update(parent_task_id, advance=1)

        if opts.compile_host or opts.aiesim:
            file_inc_cpp = self.prepend_tmp("aie_inc.cpp")
            await self.do_call(
                parent_task_id,
                [
                    "aie-translate",
                    "--aie-generate-xaie",
                    "--aie-device-name",
                    device_name,
                    input_physical_with_elfs,
                    "-o",
                    file_inc_cpp,
                ],
            )

        if opts.compile_host and len(opts.host_args) > 0:
            await self.process_host_cgen(
                aie_target, input_physical_with_elfs, device_name
            )

        processes = []
        if opts.aiesim:
            processes.append(
                self.gen_sim(parent_task_id, aie_target, input_physical, device_name)
            )

        input_physical_with_elfs_str = await read_file_async(input_physical_with_elfs)

        if (
            opts.cdo or opts.xcl or opts.pdi or opts.generate_full_elf
        ) and opts.execute:
            await self.process_cdo(input_physical_with_elfs_str, device_name)

        if opts.xcl:
            processes.append(self.process_xclbin_gen(device_op, device_name))
        # self.process_pdi_gen is called in process_xclbin_gen,
        # so don't call it again if opts.xcl is set
        elif opts.pdi or opts.generate_full_elf:
            processes.append(
                self.process_pdi_gen(device_name, self.pdi_file_name(device_name))
            )

        if opts.txn and opts.execute:
            input_physical_with_elfs = await self.process_txn(
                input_physical_with_elfs_str, device_name
            )

        if opts.ctrlpkt and opts.execute:
            processes.append(
                self.process_ctrlpkt(
                    input_physical_with_elfs_str, device_op, device_name
                )
            )

        if opts.elf and not opts.ctrlpkt and opts.execute:
            processes.append(self.process_elf(npu_insts_module, device_name))

        await asyncio.gather(*processes)

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

    opts.aietools_path = None

    # If Ryzen AI Software is installed then use it for aietools
    try:
        import ryzen_ai.__about__

        version = ryzen_ai.__about__.__version__
        path = os.path.realpath(ryzen_ai.__path__[0])
        if opts.verbose:
            print(f"Found Ryzen AI software version {version} at {path}")
        # if ryzenai software is pip installed then the path is something like:
        # <workdir>/venv/lib/python3.10/site-packages/
        opts.aietools_path = os.path.realpath(os.path.join(path, ".."))
    except:
        pass

    # Try to find xchesscc in the path
    xchesscc_path = shutil.which("xchesscc")
    if xchesscc_path:
        xchesscc_bin_path = os.path.dirname(os.path.realpath(xchesscc_path))
        xchesscc_path = os.path.dirname(xchesscc_bin_path)
        if opts.verbose:
            print(f"Found xchesscc at {xchesscc_path}")
        os.environ["PATH"] = os.pathsep.join([os.environ["PATH"], xchesscc_bin_path])
        if opts.aietools_path is None:
            opts.aietools_path = xchesscc_path
    else:
        if opts.verbose:
            print("xchesscc not found.")

    if opts.aietools_path is None:
        if opts.verbose:
            print("Could not find aietools from Vitis or Ryzen AI Software.")
        opts.aietools_path = "<aietools not found>"

    os.environ["AIETOOLS"] = opts.aietools_path

    aie_path = aie.compiler.aiecc.configure.install_path()
    peano_path = os.path.join(opts.peano_install_dir, "bin")
    os.environ["PATH"] = os.pathsep.join([aie_path, os.environ["PATH"]])
    os.environ["PATH"] = os.pathsep.join([peano_path, os.environ["PATH"]])

    if opts.aiesim and not opts.xbridge:
        sys.exit("AIE Simulation (--aiesim) currently requires --xbridge")

    if opts.verbose:
        print(f"Compiling {opts.filename}")

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

    # Create a temporary file holding the input ir, if opts.filename is None.
    if opts.filename == None:
        tmpinput_path = os.path.join(tmpdirname, "tmpinput.mlir")
        with open(tmpinput_path, "w") as f:
            f.write(str(mlir_module))
        opts.filename = tmpinput_path

    runner = FlowRunner(str(mlir_module), opts, tmpdirname)
    asyncio.run(runner.run_flow())

    if opts.profiling:
        runner.dumpprofile()


def main():
    global opts
    opts = aie.compiler.aiecc.cl_arguments.parse_args()

    if opts.version:
        print(f"aiecc.py {aie.compiler.aiecc.configure.git_commit}")
        sys.exit(0)

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
