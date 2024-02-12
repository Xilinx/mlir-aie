import __main__
import contextlib
import inspect
from itertools import zip_longest
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from aie._mlir_libs._mlir.ir import _GlobalDebug
from aie.extras.runtime.passes import Pipeline, run_pipeline

from aie.compiler.aiecc.main import (
    AIE_LOWER_TO_LLVM,
    CREATE_PATH_FINDER_FLOWS,
    DMA_TO_IPU,
    INPUT_WITH_ADDRESSES_PIPELINE,
    chesshack,
    emit_design_bif,
    emit_design_kernel_json,
    emit_partition,
    generate_cores_list,
    mem_topology,
)
from aie.dialects.aie import (
    aie_llvm_link,
    generate_bcf,
    generate_cdo,
    ipu_instgen,
    translate_aie_vec_to_cpp,
    translate_mlir_to_llvmir,
)
from aie.ir import Context, InsertionPoint, Location, Module

WORKDIR = os.getenv("WORKDIR")
if WORKDIR is None:
    WORKDIR = Path(__main__.__file__).parent.absolute() / (
        __main__.__file__[:-3] + "_workdir"
    )
else:
    WORKDIR = Path(WORKDIR).absolute()

WORKDIR.mkdir(exist_ok=True)

VITIS_DIR = Path(os.getenv("VITIS_DIR", "/opt/tools/Xilinx/Vitis/2023.2")).absolute()
XRT_DIR = Path(os.getenv("XRT_DIR", "/opt/xilinx/xrt")).absolute()
XILINXD_LICENSE_FILE = Path(
    os.getenv("XILINXD_LICENSE_FILE", Path.home() / ".Xilinx/aie.lic")
).absolute()
assert XILINXD_LICENSE_FILE.exists() and XRT_DIR.exists() and VITIS_DIR.exists()

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

XCHESS_ARGS = lambda: [
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
    str(WORKDIR),
]

# https://github.com/amd/xdna-driver/blob/d8ff9afc5c202c2bee22e6d36d1fc24dcdb6ea71/src/shim/ipu/hwctx.cpp#L58
os.environ["XRT_HACK_UNSECURE_LOADING_XCLBIN"] = "1"


def construct_and_print_module(f):
    global WORKDIR
    assert WORKDIR is not None and WORKDIR.exists()
    WORKDIR = WORKDIR / (f.__name__ + "_workdir")
    WORKDIR.mkdir(exist_ok=True)

    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            args = inspect.getfullargspec(f).args
            if args:
                if args[0] in {"module", "_module"}:
                    module = f(module)
                else:
                    raise Exception(f"only `module` arg supported {args=}")
            else:
                f()
        if module is not None:
            assert module.operation.verify()
            print(module)


def extract_input_files(core_bcf):
    return re.findall(r"^_include _file (.*)", core_bcf, re.MULTILINE)


def _run_command(cmd, debug=False):
    handle = subprocess.run(cmd, capture_output=True, cwd=WORKDIR, env=ENV)
    stderr = handle.stderr.decode("utf-8").strip()
    if handle.returncode != 0:
        print(handle.stdout, file=sys.stderr)
        raise Exception(stderr)
    if debug:
        stdout = handle.stdout.decode("utf-8").strip()
        print(stdout)
        print(stderr)


def link_with_chess_intrinsic_wrapper(input_ll):
    with open(Path(__file__).parent / "chess_intrinsic_wrapper.ll") as f:
        chess_intrinsic_wrapper = f.read()
    return chesshack(aie_llvm_link([input_ll, chess_intrinsic_wrapper]))


def chess_compile(input_ll, output_filename="input.o"):
    with open(WORKDIR / "input.ll", "w") as f:
        f.write(input_ll)

    # chess compile
    cmd = [
        *XCHESS_ARGS(),
        "-c",  # compile/assemble only, do not link
        "input.ll",
        "-o",
        output_filename,
    ]
    _run_command(cmd)


def chess_compile_cpp_to_ll(cpp, prefix="aievec", debug=False):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", prefix=f"{WORKDIR}/{prefix}_", suffix=".cpp"
    ) as temp_xchess_input:
        temp_xchess_input.write(cpp)
        temp_xchess_input.flush()

        output_path = temp_xchess_input.name + ".ll"
        cmd = [
            *XCHESS_ARGS(),
            "-c",
            "-f",
            "+f",  # only run LLVM frontend (emits IR)
            "+P",
            "4",
            temp_xchess_input.name,
            "-o",
            output_path,
        ]
        _run_command(cmd, debug)
    with open(output_path, "r") as temp_xchess_output:
        aievec_ll = temp_xchess_output.read()
    return aievec_ll


def chess_llvm_link(*file_strs, prefix="chess_llvm_link_output", input_prefixes=None):
    if input_prefixes:
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
            prefix=f"{WORKDIR}/{input_prefixes[i]}_",
            suffix=".ll",
        )
        t.write(f_str)
        t.flush()
        files.append(t)

    with tempfile.NamedTemporaryFile(
        delete=False,
        mode="w",
        prefix=f"{WORKDIR}/{prefix}_",
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
        _run_command(cmd)

        temp_xchess_llvm_link_aie_output.flush()
    with open(temp_xchess_llvm_link_aie_output.name) as f:
        output_str = f.read()
    return output_str


def make_core_elf(core_bcf, input_object_file="input.o"):
    input_files = extract_input_files(core_bcf)
    core_name = re.findall(r"_symbol (.*?) _after _main_init", core_bcf, re.MULTILINE)
    assert len(core_name) == 1
    core_name = core_name[0]

    with open(WORKDIR / f"{core_name}.bcf", "w") as f:
        f.write(core_bcf)

    cmd = [
        *XCHESS_ARGS(),
        input_object_file,
        *input_files,
        "+l",  # linker configuration file
        f"{core_name}.bcf",
        "-o",
        f"{core_name}.elf",
    ]
    _run_command(cmd)


def make_design_pdi():
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
    _run_command(cmd)


def make_xclbin(module, name="final", kernel_json=None, start_columns=None):
    with open(WORKDIR / "mem_topology.json", "w") as f:
        json.dump(mem_topology, f, indent=2)
    with open(WORKDIR / "aie_partition.json", "w") as f:
        json.dump(
            emit_partition(str(module), start_columns=start_columns),
            f,
            indent=2,
        )
    if kernel_json is None:
        kernel_json = emit_design_kernel_json()
    with open(WORKDIR / "kernels.json", "w") as f:
        json.dump(kernel_json, f, indent=2)
    xclbin_path = str(WORKDIR / f"{name}.xclbin")
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
        xclbin_path,
    ]
    _run_command(cmd)
    return xclbin_path


def setup_xclbin_firmware(xclbin_path):
    cmd = [
        str(XRT_DIR / "amdxdna/setup_xclbin_firmware.sh"),
        "-dev",
        "Phoenix",
        "-xclbin",
        xclbin_path,
    ]
    _run_command(cmd)


@contextlib.contextmanager
def _global_debug(debug):
    _GlobalDebug.flag = debug
    yield
    _GlobalDebug.flag = False


def compile_with_vectorization(mod_aie, mod_aievec, debug=False, partition_start_col=1):
    aievec_cpp = translate_aie_vec_to_cpp(mod_aievec.operation, aieml=True)
    aievec_cpp = aievec_cpp.replace("void", 'extern "C" void')

    input_with_addresses = run_pipeline(mod_aie, INPUT_WITH_ADDRESSES_PIPELINE)
    input_physical = run_pipeline(input_with_addresses, CREATE_PATH_FINDER_FLOWS)
    input_opt_with_addresses = run_pipeline(input_physical, AIE_LOWER_TO_LLVM)
    aie_ll = translate_mlir_to_llvmir(input_opt_with_addresses.operation)

    aievec_ll = chess_compile_cpp_to_ll(aievec_cpp, debug=True)
    # this is wonky because it's already on disk but oh well...
    with open(
        Path(__file__).parent / "chess_intrinsic_wrapper.ll"
    ) as chess_intrinsic_wrapper_ll:
        fullylinked_ll = chess_llvm_link(
            chesshack(aie_ll),
            aievec_ll,
            chess_intrinsic_wrapper_ll.read(),
            input_prefixes=["aie_input", "aievec_input", "chess_intrinsic_wrapper"],
        )

    chess_compile(fullylinked_ll)

    for col, row, _ in generate_cores_list(str(input_with_addresses)):
        core_bcf = generate_bcf(input_with_addresses.operation, col, row)
        make_core_elf(core_bcf)

    with _global_debug(debug):
        generate_cdo(
            input_physical.operation,
            str(WORKDIR),
            partition_start_col=partition_start_col,
        )

    make_design_pdi()

    generated_ipu_insts = run_pipeline(input_with_addresses, DMA_TO_IPU)
    return [int(inst, 16) for inst in ipu_instgen(generated_ipu_insts.operation)]


def compile_without_vectorization(module, debug=False, partition_start_col=1):
    module = run_pipeline(module, Pipeline().canonicalize())
    lowered_linalg = run_pipeline(
        module, Pipeline().convert_linalg_to_loops().fold_memref_alias_ops()
    )

    input_with_addresses = run_pipeline(lowered_linalg, INPUT_WITH_ADDRESSES_PIPELINE)
    input_opt_with_addresses = run_pipeline(input_with_addresses, AIE_LOWER_TO_LLVM)
    input_physical = run_pipeline(input_with_addresses, CREATE_PATH_FINDER_FLOWS)
    chess_compile(
        link_with_chess_intrinsic_wrapper(
            translate_mlir_to_llvmir(input_opt_with_addresses.operation)
        )
    )

    for col, row, _ in generate_cores_list(str(input_with_addresses)):
        core_bcf = generate_bcf(input_with_addresses.operation, col, row)
        make_core_elf(core_bcf)

    with _global_debug(debug):
        generate_cdo(
            input_physical.operation,
            str(WORKDIR),
            partition_start_col=partition_start_col,
        )

    make_design_pdi()

    generated_ipu_insts = run_pipeline(input_with_addresses, DMA_TO_IPU)
    return [int(inst, 16) for inst in ipu_instgen(generated_ipu_insts.operation)]


def grouper(iterable, n, *, incomplete="fill", fill_value=None):
    args = [iter(iterable)] * n
    match incomplete:
        case "fill":
            return zip_longest(*args, fillvalue=fill_value)
        case "strict":
            return zip(*args, strict=True)
        case "ignore":
            return zip(*args)
        case _:
            raise ValueError("Expected fill, strict, or ignore")
