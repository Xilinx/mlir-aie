# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest --noconftest %s
# REQUIRES: peano

"""How compile_mlir_module and _run_aiecc drive the real aiecc."""

import importlib.util
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

import aie.utils.config as config
from aie.iron import ExternalFunction
from aie.iron.device import NPU1Col1

# One core storing a constant: a whole build, with no kernel to link.
DESIGN = """module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    %buf = aie.buffer(%core) {sym_name = "buf"} : memref<16xi32>
    aie.core(%core) {
      %c7 = arith.constant 7 : i32
      %c0 = arith.constant 0 : index
      memref.store %c7, %buf[%c0] : memref<16xi32>
      aie.end
    }
    aie.runtime_sequence(%a: memref<16xi32>) {
    }
  }
}
"""

# One core calling a kernel it links from `fill.o`.
LINKED_DESIGN = """module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    %buf = aie.buffer(%core) {sym_name = "buf"} : memref<16xi32>
    func.func private @fill(memref<16xi32>) attributes {link_with = "fill.o"}
    aie.core(%core) {
      func.call @fill(%buf) : (memref<16xi32>) -> ()
      aie.end
    }
    aie.runtime_sequence(%a: memref<16xi32>) {
    }
  }
}
"""

KERNEL = 'extern "C" void fill(int *out) { for (int i = 0; i < 16; i++) out[i] = 7; }\n'


@pytest.fixture
def compile_utils():
    source = Path(__file__).resolve().parents[2] / "python/utils/compile/utils.py"
    spec = importlib.util.spec_from_file_location("compile_utils", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_kernel(directory: Path) -> None:
    """Build `KERNEL` for npu1 into `directory`/fill.o."""
    source = directory / "fill.cc"
    source.write_text(KERNEL)
    subprocess.run(
        [
            config.peano_cxx_path(),
            "--target=aie2-none-unknown-elf",
            "-O2",
            "-c",
            str(source),
            "-o",
            str(directory / "fill.o"),
        ],
        check=True,
    )


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("full_elf", [False, True])
def test_compile_writes_outputs_through_work_dir(
    compile_utils, monkeypatch, tmp_path, relative, full_elf
):
    monkeypatch.chdir(tmp_path)
    work_dir = tmp_path / "build dir"
    work_dir.mkdir()
    output = tmp_path / "output.bin"
    paths = {"insts_path": output, "xclbin_path": "design.xclbin"}
    if full_elf:
        paths["full_elf_path"] = output

    compile_utils.compile_mlir_module(
        DESIGN,
        work_dir=os.path.relpath(work_dir) if relative else work_dir,
        pdi_path="design.pdi",
        elf_path="design.elf",
        **paths,
    )

    assert Path.cwd() == tmp_path
    assert (work_dir / "aie.mlir").read_text() == DESIGN
    assert (work_dir / "input_with_addresses.mlir").is_file()
    for name in ("design.pdi", "design.elf"):
        assert (work_dir / name).stat().st_size > 0
    assert (output.read_bytes()[:4] == b"\x7fELF") == full_elf
    assert (work_dir / "design.xclbin").is_file() != full_elf


@pytest.mark.skipif(shutil.which("xchesscc") is None, reason="xchesscc")
def test_chess_compile_drives_xchesscc(compile_utils, caplog, tmp_path):
    with caplog.at_level(logging.DEBUG, logger=compile_utils.logger.name):
        compile_utils.compile_mlir_module(
            DESIGN,
            work_dir=tmp_path,
            use_chess=True,
            insts_path=tmp_path / "insts.bin",
            xclbin_path=tmp_path / "design.xclbin",
            options=["-n", "--verbose"],
        )
    assert "xchesscc" in caplog.text


@pytest.mark.parametrize("broken", [False, True])
def test_no_work_dir_preserves_cwd_and_cleans_up(
    compile_utils, monkeypatch, tmp_path, broken
):
    monkeypatch.chdir(tmp_path)
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    if broken:
        with pytest.raises(RuntimeError, match="exit code 1:\n.*error: "):
            compile_utils.compile_mlir_module(
                DESIGN.replace("aie.end", "aie.bogus"), insts_path="insts.bin"
            )
    else:
        compile_utils.compile_mlir_module(DESIGN, insts_path="insts.bin")
        assert (tmp_path / "insts.bin").stat().st_size > 0
    assert Path.cwd() == tmp_path
    assert not list(scratch.iterdir())


def test_link_files_are_built_while_aiecc_waits(compile_utils, tmp_path):
    """aiecc reads none of the files its cores link until they are built."""
    work_dir = tmp_path / "build"
    work_dir.mkdir()
    insts = tmp_path / "insts.bin"
    xclbin = tmp_path / "design.xclbin"

    compile_utils.compile_mlir_module(
        LINKED_DESIGN,
        insts_path=insts,
        xclbin_path=xclbin,
        work_dir=work_dir,
        build_link_files=lambda: compile_kernel(work_dir),
    )

    assert insts.stat().st_size > 0
    assert xclbin.stat().st_size > 0


def test_link_files_failure_stops_aiecc(compile_utils, tmp_path):
    work_dir = tmp_path / "build"
    work_dir.mkdir()
    insts = tmp_path / "insts.bin"
    xclbin = tmp_path / "design.xclbin"

    def build_link_files():
        raise ValueError("kernel failed to compile")

    with pytest.raises(ValueError, match="kernel failed to compile"):
        compile_utils.compile_mlir_module(
            LINKED_DESIGN,
            insts_path=insts,
            xclbin_path=xclbin,
            work_dir=work_dir,
            build_link_files=build_link_files,
        )
    assert not insts.exists()
    assert not xclbin.exists()


def test_link_files_need_work_dir(compile_utils):
    with pytest.raises(ValueError, match="build_link_files requires work_dir"):
        compile_utils.compile_mlir_module(
            DESIGN, insts_path="insts.bin", build_link_files=lambda: None
        )


def test_run_aiecc_child_resolves_kernel_in_work_dir(
    compile_utils, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    work_dir = tmp_path / "build dir"
    work_dir.mkdir()
    (tmp_path / "fill.o").write_text("stale caller object")
    compile_kernel(work_dir)
    design = work_dir / "aie.mlir"
    design.write_text(LINKED_DESIGN)

    compile_utils._run_aiecc(
        os.path.relpath(design),
        [
            f"--peano={config.peano_install_dir()}",
            "--get-npu-insts",
            f"--npu-insts-name={tmp_path / 'insts.bin'}",
            f"--tmpdir={work_dir}",
        ],
        cwd=work_dir,
    )

    assert (tmp_path / "insts.bin").stat().st_size > 0
    assert Path.cwd() == tmp_path


@pytest.mark.parametrize("relative", [False, True])
def test_copy_object_files_refreshes_work_dir(
    compile_utils, monkeypatch, tmp_path, relative
):
    monkeypatch.chdir(tmp_path)
    work_dir = tmp_path / "build dir"
    work_dir.mkdir()
    sources = [tmp_path / name for name in ("kernel.o", "helper.o")]
    for source in sources:
        source.write_bytes(b"current object")
        (work_dir / source.name).write_bytes(b"stale object")

    compile_utils._copy_object_files(
        [os.path.relpath(source) if relative else source for source in sources],
        work_dir,
    )

    for source in sources:
        assert (work_dir / source.name).read_bytes() == source.read_bytes()


def test_copy_object_files_already_in_work_dir(compile_utils, tmp_path):
    source = tmp_path / "kernel.o"
    source.write_bytes(b"current object")

    compile_utils._copy_object_files([source], tmp_path)

    assert source.read_bytes() == b"current object"


def test_copy_object_files_missing_source(compile_utils, tmp_path):
    source = tmp_path / "missing" / "kernel.o"
    dest = tmp_path / source.name
    dest.write_bytes(b"stale object")

    with pytest.raises(FileNotFoundError):
        compile_utils._copy_object_files([source], tmp_path)

    assert dest.read_bytes() == b"stale object"


def test_compile_mlir_module_ignores_stale_external_functions(compile_utils, tmp_path):
    """Only kernels the current module declares reach the auto-build and its
    arch check: `ExternalFunction._instances` also holds kernels left over from
    an earlier, unrelated compile in the same process."""
    work_dir = tmp_path / "build"
    work_dir.mkdir()
    referenced_source = tmp_path / "fill.cc"
    referenced_source.write_text(KERNEL)
    stale_source = tmp_path / "stale.cc"
    stale_source.write_text("#error a stale kernel is not built\n")
    ExternalFunction._instances.clear()
    referenced = ExternalFunction(
        "fill",
        source_file=str(referenced_source),
        arg_types=[],
        object_file_name="fill.o",
    )
    stale = ExternalFunction(
        "stale_kernel_from_earlier_compile",
        source_file=str(stale_source),
        arg_types=[],
    )
    stale.built_for_arch = "aie2p"

    compile_utils.compile_mlir_module(
        LINKED_DESIGN,
        insts_path=tmp_path / "insts.bin",
        work_dir=work_dir,
        device=NPU1Col1(),
    )

    assert (work_dir / referenced.object_file_name).is_file()
    assert not (work_dir / stale.object_file_name).exists()
    assert (tmp_path / "insts.bin").stat().st_size > 0


def test_declared_link_with_picks_among_kernels_sharing_a_symbol(compile_utils):
    """The declaration's ``link_with`` picks among kernels sharing a symbol.

    An inline kernel keeps its bare symbol on every arch. A symbol no instance
    matches keeps them all, so the arch check can still report it.
    """
    setup = 'extern "C" void setup() {}'
    aie2 = ExternalFunction(
        "setup", source_string=setup, object_file_name="setup_aaaa.ll"
    )
    aie2p = ExternalFunction(
        "setup", source_string=setup, object_file_name="setup_bbbb.ll"
    )
    other = ExternalFunction(
        "kernel",
        source_string='extern "C" void kernel(int) {}',
        object_file_name="kernel.o",
    )

    def select(funcs, text):
        declared = compile_utils._declared_objects(text)
        return compile_utils._select_declared_kernels(funcs, declared)

    text = """module {
      func.func private @setup() attributes {link_with = "setup_bbbb.ll"}
      func.func private @kernel(%arg0: i32)
    }"""
    assert select([aie2, aie2p, other], text) == [aie2p, other]
    stale = """module {
      func.func private @setup() attributes {link_with = "setup_cccc.ll"}
    }"""
    assert select([aie2, aie2p], stale) == [aie2, aie2p]
    both = """module {
      module @a {
        func.func private @setup() attributes {link_with = "setup_aaaa.ll"}
      }
      module @b {
        func.func private @setup() attributes {link_with = "setup_bbbb.ll"}
        func.func private @kernel(%arg0: i32)
      }
    }"""
    assert select([aie2, aie2p, other], both) == [aie2, aie2p, other]
    assert select([aie2, other], "module {}") == []
