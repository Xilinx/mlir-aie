import __main__
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from aie.extras.runtime.passes import Pipeline

from aie.compiler.aiecc.main import (
    chesshack,
    emit_design_bif,
    mem_topology,
    emit_partition,
    emit_design_kernel_json,
)
from aie.dialects import aiex, aie
from aie.dialects.aie import aie_llvm_link, DMAChannelDir, WireBundle, LockAction
from aie.ir import Context, Location, Module, InsertionPoint

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release

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

INPUT_WITH_ADDRESSES_PIPELINE = (
    Pipeline()
    .convert_linalg_to_affine_loops()
    .lower_affine()
    .add_pass("aie-canonicalize-device")
    .add_pass("aie.device(aie-assign-lock-ids,aie-assign-buffer-addresses)")
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
    Pipeline()
    .add_pass("aie.device(aie-localize-locks,aie-normalize-address-spaces)")
    .add_pass("aie-standard-lowering")
    .add_pass("aiex-standard-lowering")
)
AIE_LOWER_TO_LLVM += LOWER_TO_LLVM_PIPELINE

CREATE_PATH_FINDER_FLOWS = Pipeline().add_pass(
    "aie.device(aie-create-pathfinder-flows)"
)
DMA_TO_IPU = Pipeline().add_pass("aie.device(aie-dma-to-ipu)")


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
        "/opt/xilinx/xrt/amdaie/setup_xclbin_firmware.sh",
        "-dev",
        "Phoenix",
        "-xclbin",
        xclbin_path,
    ]
    _run_command(cmd)


# from runtime_lib/xaiengine/aie-rt/driver/src/global/xaiemlgbl_params.h
# these aren't completely correct - right values but not necessarily the right names?
XAIEMLGBL_NOC_MODULE_DMA_MM2S_0_TASK_QUEUE = 0x0001D214
XAIEMLGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE = 0x0001D204
XAIEMLGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE_ENABLE_TOKEN_ISSUE_MASK = 0x80000000
XAIEMLGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE_START_BD_ID_MASK = 0x0000000F


def ipu_write32(channel_dir, channel_index, col, bd_id, repeats=0):
    if channel_dir == DMAChannelDir.MM2S:
        address = XAIEMLGBL_NOC_MODULE_DMA_MM2S_0_TASK_QUEUE
    else:
        address = XAIEMLGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE
    if channel_index == 1:
        address += 0x8
    value = bd_id & XAIEMLGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE_START_BD_ID_MASK
    value |= (repeats & 0xFF) << 16
    if channel_dir == DMAChannelDir.S2MM:
        # issue token
        value |= XAIEMLGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE_ENABLE_TOKEN_ISSUE_MASK
    aiex.ipu_write32(address=address, column=col, row=0, value=value)


def ipu_writebd_shimtile(
    bd_id,
    buffer_length,
    offset,
    ddr_id,
    d2_stride=1,
    d1_size=None,
    d1_stride=1,
    d0_size=None,
    d0_stride=1,
    iteration_size=0,
    iteration_stride=0,
    iteration_current=0,
    lock_acq_enable=0,
    lock_acq_id=0,
    lock_acq_val=0,
    lock_rel_id=0,
    lock_rel_val=0,
    next_bd=0,
    use_next_bd=0,
    data_width=32,
):
    d2_stride -= 1
    d1_stride -= 1
    d0_stride -= 1
    assert d2_stride >= 0 and d1_stride >= 0 and d0_stride >= 0
    # byte offset
    offset *= data_width // 8
    # None means do not wrap which is 0 on the arch
    if d1_size is None:
        d1_size = 0
    # None means do not wrap which is 0 on the arch
    if d0_size is None:
        d0_size = 0

    return aiex.ipu_writebd_shimtile(
        bd_id=bd_id,
        buffer_length=buffer_length,
        buffer_offset=offset,
        column=0,
        column_num=1,
        d0_size=d0_size,
        d0_stride=d0_stride,
        d1_size=d1_size,
        d1_stride=d1_stride,
        d2_stride=d2_stride,
        ddr_id=ddr_id,
        enable_packet=0,
        iteration_current=iteration_current,
        iteration_size=iteration_size,
        iteration_stride=iteration_stride,
        lock_acq_enable=lock_acq_enable,
        lock_acq_id=lock_acq_id,
        lock_acq_val=lock_acq_val,
        lock_rel_id=lock_rel_id,
        lock_rel_val=lock_rel_val,
        next_bd=next_bd,
        out_of_order_id=0,
        packet_id=0,
        packet_type=0,
        use_next_bd=use_next_bd,
        valid_bd=1,
    )


def process_bd(
    acq_lock,
    buffer,
    rel_lock,
    acq_action=aie.LockAction.AcquireGreaterEqual,
    rel_action=aie.LockAction.Release,
    acq_val=None,
    rel_val=None,
):
    aie.use_lock(acq_lock, acq_action, value=acq_val)
    aie.dma_bd(buffer)
    aie.use_lock(rel_lock, rel_action, value=rel_val)


def forward_bd(tile, channel_idx, buffer, read_in_lock=None, write_out_lock=None):
    if read_in_lock is None:
        read_in_lock = aie.lock(tile, init=1)
    if write_out_lock is None:
        write_out_lock = aie.lock(tile, init=0)

    @aie.dma(S2MM, channel_idx)
    def dma_incoming():
        process_bd(read_in_lock, buffer, write_out_lock)

    @aie.dma(MM2S, channel_idx)
    def dma_outgoing():
        process_bd(write_out_lock, buffer, read_in_lock)


@contextmanager
def hold_lock(acq_lock, rel_lock):
    aie.use_lock(acq_lock, AcquireGreaterEqual)
    try:
        yield
    finally:
        aie.use_lock(rel_lock, Release)
