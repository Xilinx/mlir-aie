# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from .aiecc.main import (
    AIE_LOWER_TO_LLVM,
    CREATE_PATH_FINDER_FLOWS,
    INPUT_WITH_ADDRESSES_PIPELINE,
    chesshack,
    emit_design_bif,
    emit_design_kernel_json,
    emit_partition,
    generate_cores_list,
    mem_topology,
)
from .._mlir_libs._mlir.ir import _GlobalDebug
from ..dialects.aie import (
    aie_llvm_link,
    generate_bcf,
    generate_cdo,
    translate_aie_vec_to_cpp,
    translate_mlir_to_llvmir,
)
from ..extras.runtime.passes import Pipeline, run_pipeline

# this is inside the aie-python-extras (shared) namespace package
from ..extras.util import find_ops
from ..ir import Context, Module

VITIS_DIR = Path(os.getenv("VITIS_DIR", "/opt/tools/Xilinx/Vitis/2023.2")).absolute()
XRT_DIR = Path(os.getenv("XRT_DIR", "/opt/xilinx/xrt")).absolute()
XILINXD_LICENSE_FILE = Path(
    os.getenv("XILINXD_LICENSE_FILE", Path.home() / ".Xilinx/aie.lic")
).absolute()

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


XCHESS_ARGS = lambda workdir: [
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
    "+w",
    str(workdir),
]

# https://github.com/amd/xdna-driver/blob/d8ff9afc5c202c2bee22e6d36d1fc24dcdb6ea71/src/shim/ipu/hwctx.cpp#L58
os.environ["XRT_HACK_UNSECURE_LOADING_XCLBIN"] = "1"


def extract_input_files(core_bcf):
    return re.findall(r"^_include _file (.*)", core_bcf, re.MULTILINE)


def _run_command(cmd, workdir, *, debug=False):
    if debug:
        print("shelling out command:")
        print(" ".join([f"{k}={v}" for k, v in ENV.items()]) + " " + " ".join(cmd))
    handle = subprocess.run(cmd, capture_output=True, cwd=workdir, env=ENV)
    stderr = handle.stderr.decode("utf-8").strip()
    if handle.returncode != 0:
        print(handle.stdout, file=sys.stderr)
        raise Exception(stderr)
    if debug:
        stdout = handle.stdout.decode("utf-8").strip()
        print(stdout)
        print(stderr)


def aie_llvm_link_with_chess_intrinsic_wrapper(input_ll):
    return chesshack(aie_llvm_link([input_ll, _CHESS_INTRINSIC_WRAPPER_LL]))


def chess_compile(input_ll, workdir, output_filename="input", debug=False):
    if (
        Path(workdir / f"{output_filename}.ll").exists()
        and Path(workdir / f"{output_filename}.o").exists()
    ):
        with open(workdir / f"{output_filename}.ll", "r") as f:
            if (
                hashlib.sha256(f.read().encode("utf-8")).hexdigest()
                == hashlib.sha256(input_ll.encode("utf-8")).hexdigest()
            ):
                if debug:
                    print(f"using cached {output_filename}.o")
                return
    with open(workdir / f"{output_filename}.ll", "w") as f:
        f.write(input_ll)

    # chess compile
    cmd = [
        *XCHESS_ARGS(workdir),
        "-c",  # compile/assemble only, do not link
        f"{output_filename}.ll",
        "-o",
        f"{output_filename}.o",
    ]
    _run_command(cmd, workdir, debug=debug)


def chess_compile_cpp_to_ll(cpp, workdir, prefix="aievec", debug=False):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", prefix=f"{workdir}/{prefix}_", suffix=".cpp"
    ) as temp_xchess_input:
        temp_xchess_input.write(cpp)
        temp_xchess_input.flush()

        output_path = temp_xchess_input.name + ".ll"
        cmd = [
            *XCHESS_ARGS(workdir),
            "-c",
            "-f",
            "+f",  # only run LLVM frontend (emits IR)
            "+P",
            "4",
            temp_xchess_input.name,
            "-o",
            output_path,
        ]
        _run_command(cmd, workdir, debug=debug)
    with open(output_path, "r") as temp_xchess_output:
        aievec_ll = temp_xchess_output.read()
    return aievec_ll


def chess_llvm_link(
    file_strs,
    workdir,
    prefix="chess_llvm_link_output",
    input_prefixes=None,
    debug=False,
):
    if input_prefixes is None:
        input_prefixes = [f"chess_llvm_link_input_{i}" for i in range(len(file_strs))]
    else:
        assert len(input_prefixes) == len(
            file_strs
        ), "need as many input prefixes as file strings"
    files = []
    for i, f_str in enumerate(file_strs):
        t = tempfile.NamedTemporaryFile(
            delete=False,
            mode="w",
            prefix=f"{workdir}/{input_prefixes[i]}_",
            suffix=".ll",
        )
        t.write(f_str)
        t.flush()
        files.append(t)

    with tempfile.NamedTemporaryFile(
        delete=False,
        mode="w",
        prefix=f"{workdir}/{prefix}_",
        suffix=".fullylinked.ll",
    ) as temp_xchess_llvm_link_aie_output:
        cmd = [
            f"{AIETOOLS_DIR}/tps/lnx64/target_aie_ml/bin/LNa64bin/chess-llvm-link",
            *[f.name for f in files],
            "--opaque-pointers=1",
            "-S",
            "-o",
            temp_xchess_llvm_link_aie_output.name,
        ]
        _run_command(cmd, workdir, debug=debug)

        temp_xchess_llvm_link_aie_output.flush()
    with open(temp_xchess_llvm_link_aie_output.name) as f:
        output_str = f.read()
    return output_str


def make_core_elf(core_bcf, workdir, object_filename="input", debug=False):
    input_files = extract_input_files(core_bcf)
    core_name = re.findall(r"_symbol (.*?) _after _main_init", core_bcf, re.MULTILINE)
    assert len(core_name) == 1
    core_name = core_name[0]

    with open(workdir / f"{core_name}.bcf", "w") as f:
        f.write(core_bcf)

    cmd = [
        *XCHESS_ARGS(workdir),
        f"{object_filename}.o",
        *input_files,
        "+l",  # linker configuration file
        f"{core_name}.bcf",
        "-o",
        f"{core_name}.elf",
    ]
    _run_command(cmd, workdir, debug=debug)


def make_design_pdi(workdir, *, enable_cores=True, debug=False):
    with open(workdir / "design.bif", "w") as f:
        f.write(emit_design_bif(workdir, enable_cores=enable_cores))

    cmd = [
        "bootgen",
        "-arch",
        "versal",
        "-image",
        workdir / "design.bif",
        "-w",  # force overwrite
        "-o",
        workdir / "design.pdi",
    ]
    _run_command(cmd, workdir, debug=debug)


def make_xclbin(
    module, workdir, name="final", kernel_json=None, start_columns=None, debug=False
):
    with open(workdir / "mem_topology.json", "w") as f:
        json.dump(mem_topology, f, indent=2)
    with open(workdir / "aie_partition.json", "w") as f:
        json.dump(
            emit_partition(str(module), start_columns=start_columns),
            f,
            indent=2,
        )
    if kernel_json is None:
        kernel_json = emit_design_kernel_json()
    with open(workdir / "kernels.json", "w") as f:
        json.dump(kernel_json, f, indent=2)
    xclbin_path = str(workdir / f"{name}.xclbin")
    cmd = [
        "xclbinutil",
        "--add-replace-section",
        f"MEM_TOPOLOGY:JSON:{workdir / 'mem_topology.json'}",
        "--add-kernel",
        str(workdir / "kernels.json"),
        "--add-replace-section",
        f"AIE_PARTITION:JSON:{workdir / 'aie_partition.json'}",
        "--force",
        "--output",
        xclbin_path,
    ]
    _run_command(cmd, workdir, debug=debug)
    return xclbin_path


@contextlib.contextmanager
def _global_debug(debug):
    _GlobalDebug.flag = debug
    yield
    _GlobalDebug.flag = False


def compile_with_vectorization(
    mod_aie,
    mod_aievec,
    workdir,
    *,
    debug=False,
    xaie_debug=False,
    cdo_debug=False,
    partition_start_col=1,
    enable_cores=True,
    basicAlloc=True,
):
    debug = debug or xaie_debug or cdo_debug
    input_with_addresses = run_pipeline(
        mod_aie,
        Pipeline().convert_linalg_to_affine_loops() + INPUT_WITH_ADDRESSES_PIPELINE(basicAlloc),
        enable_ir_printing=debug,
    )

    aievec_cpp = translate_aie_vec_to_cpp(mod_aievec.operation, aieml=True)
    aievec_cpp = aievec_cpp.replace("void", 'extern "C" void')
    aievec_ll = chess_compile_cpp_to_ll(aievec_cpp, workdir, debug=debug)
    # TODO(max) connect each core to its own kernel...
    kernel = find_ops(mod_aievec, lambda o: "aie_kernel" in o.attributes, single=True)
    if kernel:
        print("compiling kernel")
        chess_compile(
            aievec_ll, workdir, output_filename=f"{kernel.sym_name.value}", debug=debug
        )

    for col, row, _ in generate_cores_list(str(mod_aie)):
        print(f"compiling core {col} {row}")
        with Context():
            core_mod = Module.parse(str(input_with_addresses))
            core_bcf = generate_bcf(core_mod.operation, col, row)
            core_lowered_to_llvm_dialect = run_pipeline(
                core_mod, AIE_LOWER_TO_LLVM(col, row), enable_ir_printing=debug
            )
            core_input_ll = translate_mlir_to_llvmir(
                core_lowered_to_llvm_dialect.operation
            )
            # TODO(max) connect each core to its own kernel...
            if kernel:
                chess_compile(
                    aie_llvm_link_with_chess_intrinsic_wrapper(core_input_ll),
                    workdir,
                    output_filename=f"core_{col}_{row}",
                    debug=debug,
                )
            else:
                fullylinked_ll = chess_llvm_link(
                    [chesshack(core_input_ll), aievec_ll, _CHESS_INTRINSIC_WRAPPER_LL],
                    workdir,
                    prefix=f"core_{col}_{row}_chess_llvm_link_output",
                    input_prefixes=[
                        f"core_{col}_{row}_aie_input",
                        f"core_{col}_{row}_aievec_input",
                        f"core_{col}_{row}_chess_intrinsic_wrapper",
                    ],
                    debug=debug,
                )

                chess_compile(
                    fullylinked_ll,
                    workdir,
                    output_filename=f"core_{col}_{row}",
                    debug=debug,
                )
        make_core_elf(
            core_bcf, workdir, object_filename=f"core_{col}_{row}", debug=debug
        )

    input_physical = run_pipeline(
        mod_aie,
        CREATE_PATH_FINDER_FLOWS
        + Pipeline().convert_linalg_to_affine_loops()
        + INPUT_WITH_ADDRESSES_PIPELINE(basicAlloc),
        enable_ir_printing=debug,
    )
    with _global_debug(debug):
        generate_cdo(
            input_physical.operation,
            str(workdir),
            partition_start_col=partition_start_col,
            cdo_debug=cdo_debug,
            xaie_debug=xaie_debug,
            enable_cores=enable_cores,
        )

    make_design_pdi(workdir, enable_cores=enable_cores)


def compile_without_vectorization(
    module,
    workdir,
    *,
    debug=False,
    xaie_debug=False,
    cdo_debug=False,
    partition_start_col=1,
    enable_cores=True,
    basicAlloc=True,
):
    debug = debug or xaie_debug or cdo_debug
    module = run_pipeline(module, Pipeline().canonicalize())
    lowered_linalg = run_pipeline(
        module,
        Pipeline().convert_linalg_to_loops().fold_memref_alias_ops(),
        enable_ir_printing=debug,
    )
    input_with_addresses = run_pipeline(
        lowered_linalg, INPUT_WITH_ADDRESSES_PIPELINE(basicAlloc), enable_ir_printing=debug
    )

    for col, row, _ in generate_cores_list(str(module)):
        print(f"compiling core {col} {row}")
        with Context():
            core_mod = Module.parse(str(input_with_addresses))
            core_bcf = generate_bcf(core_mod.operation, col, row)
            core_lowered_to_llvm_dialect = run_pipeline(
                core_mod, AIE_LOWER_TO_LLVM(col, row), enable_ir_printing=debug
            )
            core_input_ll = translate_mlir_to_llvmir(
                core_lowered_to_llvm_dialect.operation
            )
            chess_compile(
                aie_llvm_link_with_chess_intrinsic_wrapper(core_input_ll),
                workdir,
                output_filename=f"core_{col}_{row}",
                debug=debug,
            )
        make_core_elf(
            core_bcf, workdir, object_filename=f"core_{col}_{row}", debug=debug
        )

    input_physical = run_pipeline(
        module,
        CREATE_PATH_FINDER_FLOWS + INPUT_WITH_ADDRESSES_PIPELINE(basicAlloc),
        enable_ir_printing=debug,
    )
    with _global_debug(debug):
        generate_cdo(
            input_physical.operation,
            str(workdir),
            partition_start_col=partition_start_col,
            cdo_debug=cdo_debug,
            xaie_debug=xaie_debug,
            enable_cores=enable_cores,
        )

    make_design_pdi(workdir, enable_cores=enable_cores)


_CHESS_INTRINSIC_WRAPPER_LL = """
; ModuleID = 'aie_runtime_lib/AIE2/chess_intrinsic_wrapper.cpp'
source_filename = "aie_runtime_lib/AIE2/chess_intrinsic_wrapper.cpp"

%struct.ipd.custom_type.uint2_t.uint2_t = type { i2 }

; Function Attrs: mustprogress nounwind
define dso_local void @llvm___aie2___acquire(i32 noundef %0, i32 noundef %1) local_unnamed_addr addrspace(1) #0 {
  tail call addrspace(1) void @llvm.chess_memory_fence()
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #4
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_acquire_guarded___uint___uint(i32 zeroext %0, i32 zeroext %1) #4
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #4
  tail call addrspace(1) void @llvm.chess_memory_fence()
  ret void
}

; Function Attrs: mustprogress nounwind
define dso_local void @llvm___aie2___release(i32 noundef %0, i32 noundef %1) local_unnamed_addr addrspace(1) #0 {
  tail call addrspace(1) void @llvm.chess_memory_fence()
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #4
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_release_guarded___uint___sint(i32 zeroext %0, i32 signext %1) #4
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #4
  tail call addrspace(1) void @llvm.chess_memory_fence()
  ret void
}

; Function Attrs: nounwind
define dso_local void @llvm___aie___event0() local_unnamed_addr addrspace(1) #1 {
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t.uint2_t zeroinitializer) #4
  ret void
}

; Function Attrs: nounwind
define dso_local void @llvm___aie___event1() local_unnamed_addr addrspace(1) #1 {
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t.uint2_t { i2 1 }) #4
  ret void
}

; Function Attrs: mustprogress nounwind willreturn
declare void @llvm.chess_memory_fence() addrspace(1) #2

; Function Attrs: inaccessiblememonly nounwind
declare dso_local void @_Z25chess_separator_schedulerv() local_unnamed_addr addrspace(1) #3

; Function Attrs: inaccessiblememonly nounwind
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_acquire_guarded___uint___uint(i32 zeroext, i32 zeroext) local_unnamed_addr addrspace(1) #3

; Function Attrs: inaccessiblememonly nounwind
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_release_guarded___uint___sint(i32 zeroext, i32 signext) local_unnamed_addr addrspace(1) #3

; Function Attrs: inaccessiblememonly nounwind
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t.uint2_t) local_unnamed_addr addrspace(1) #3

attributes #0 = { mustprogress nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { mustprogress nounwind willreturn }
attributes #3 = { inaccessiblememonly nounwind "frame-pointer"="all" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { inaccessiblememonly nounwind "no-builtin-memcpy" }

!llvm.linker.options = !{}
!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 15.0.5 (/u/sgasip/ipd/repositories/llvm_ipd 3a25925e0239306412dac02da5e4c8c51ae722e8)"}
"""
