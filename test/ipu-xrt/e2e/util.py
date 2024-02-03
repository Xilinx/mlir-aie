import __main__
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


from aie.compiler.aiecc.main import (
    chesshack,
    emit_design_bif,
    mem_topology,
    emit_partition,
    emit_design_kernel_json,
)
from aie.dialects.aie import aie_llvm_link
from aie.ir import Context, Location, Module, InsertionPoint

WORKDIR = Path(
    os.getenv(
        "WORKDIR",
        Path(__main__.__file__).parent.absolute()
        / (__main__.__file__[:-3] + "_workdir"),
    )
).absolute()

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


def maybe_make_workdir():
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir()


maybe_make_workdir()


def construct_and_print_module(f):
    global WORKDIR
    # fmt: off
    WORKDIR = (
        Path(
            os.getenv(
                "WORKDIR",
                Path(__main__.__file__).parent.absolute()
                / (__main__.__file__[:-3] + "_workdir"),
            )
        ).absolute()
        / (f.__name__ + "_workdir")
    )
    # fmt: on
    maybe_make_workdir()
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
    with open(WORKDIR / f"core.bcf", "w") as f:
        f.write(core_bcf)

    input_files = extract_input_files(core_bcf)
    core_name = re.findall(r"_symbol (.*?) _after _main_init", core_bcf, re.MULTILINE)
    assert len(core_name) == 1
    core_name = core_name[0]

    cmd = [
        *XCHESS_ARGS(),
        input_object_file,
        *input_files,
        "+l",  # linker configuration file
        f"core.bcf",
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


def make_xclbin(module, name="final"):
    with open(WORKDIR / "mem_topology.json", "w") as f:
        json.dump(mem_topology, f, indent=2)
    with open(WORKDIR / "aie_partition.json", "w") as f:
        json.dump(emit_partition(str(module)), f, indent=2)
    with open(WORKDIR / "kernels.json", "w") as f:
        json.dump(emit_design_kernel_json(), f, indent=2)
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
