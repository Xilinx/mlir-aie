# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
# REQUIRES: peano
"""A design's kernels are built while aiecc lowers it (--await-link-files).

The objects its cores link are written into the build directory only once
aiecc is running, so what aiecc builds must match a build whose objects were
there from the start, and a kernel that fails to compile must stop aiecc and
report its own error. That holds for a CompilableDesign and for
compile_mlir_module's device=.
"""

import logging
import subprocess

import numpy as np
import pytest

import aie.utils.compile.utils as compile_utils
import aie.utils.config as config
from aie.iron import ExternalFunction, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2Col1
from aie.utils import set_current_device
from aie.utils.compile.jit.compilabledesign import CompilableDesign
from aie.utils.compile.utils import compile_external_kernels, compile_mlir_module

TILE = 64
tile_ty = np.ndarray[(TILE,), np.dtype[np.int32]]


@pytest.fixture(autouse=True)
def _device():
    set_current_device(NPU2Col1())
    ExternalFunction._instances.clear()
    yield
    ExternalFunction._instances.clear()


def fill(kernel: ExternalFunction):
    """The module of a design whose one core runs `kernel` on its output."""
    of_out = ObjectFifo(tile_ty, name="out")

    def core(of_out, fn):
        for _ in range_(1):
            elem = of_out.acquire(1)
            fn(elem)
            of_out.release(1)

    worker = Worker(core, [of_out.prod(), kernel])

    def sequence(out, out_h):
        out_h.drain(out, wait=True)

    rt = Runtime(sequence, [tile_ty, of_out.cons()])
    return Program(NPU2Col1(), rt, workers=[worker]).resolve_program()


@pytest.fixture
def prebuilt(tmp_path):
    """A design's MLIR, and the object its core links, built by Peano."""
    source = tmp_path / "fill_seven.cc"
    source.write_text(
        'extern "C" void fill_seven(int *out) {\n'
        f"  for (int i = 0; i < {TILE}; i++) out[i] = 7;\n"
        "}\n"
    )
    kernel = ExternalFunction(
        "fill_seven", source_file=str(source), arg_types=[tile_ty]
    )
    obj = tmp_path / "objects" / kernel.object_file_name
    obj.parent.mkdir()
    subprocess.run(
        [
            config.peano_cxx_path(),
            "--target=aie2p-none-unknown-elf",
            "-O2",
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        check=True,
    )
    mlir = tmp_path / "design.mlir"
    mlir.write_text(str(fill(kernel)))
    ExternalFunction._instances.clear()
    return mlir, obj


@pytest.mark.parametrize("full_elf", [False, True])
def test_objects_linked_while_aiecc_lowers(tmp_path, prebuilt, full_elf):
    mlir, obj = prebuilt
    design = CompilableDesign(mlir, object_files=[obj], use_cache=False)
    out = tmp_path / "out"
    if full_elf:
        design.compile(full_elf_path=out / "design.elf")
        assert (out / "design.elf").stat().st_size > 0
    else:
        xclbin, insts = design.compile(
            xclbin_path=out / "final.xclbin", inst_path=out / "insts.bin"
        )
        assert xclbin.stat().st_size > 0 and insts.stat().st_size > 0


def test_overlapped_build_matches_serial(tmp_path, prebuilt):
    mlir, obj = prebuilt
    overlapped = tmp_path / "overlapped"
    CompilableDesign(mlir, object_files=[obj], use_cache=False).compile(
        full_elf_path=overlapped / "design.elf"
    )

    serial = tmp_path / "serial"
    serial.mkdir()
    (serial / obj.name).write_bytes(obj.read_bytes())
    compile_mlir_module(
        mlir.read_text(),
        work_dir=serial,
        full_elf_path=serial / "design.elf",
    )

    assert (overlapped / "design.elf").read_bytes() == (
        serial / "design.elf"
    ).read_bytes()


def test_kernel_error_stops_aiecc(tmp_path):
    def generator():
        return fill(
            ExternalFunction(
                "fill_broken",
                source_string='#error "fill_broken does not compile"\n',
                arg_types=[tile_ty],
            )
        )

    design = CompilableDesign(generator, use_cache=False)
    with pytest.raises(Exception, match="fill_broken does not compile") as raised:
        design.compile(full_elf_path=tmp_path / "out" / "design.elf")
    assert "stdin closed" not in str(raised.value)


def test_kernels_build_with_the_designs_include_paths(tmp_path):
    include = tmp_path / "include"
    include.mkdir()
    (include / "seven.h").write_text("#define SEVEN 7\n")

    def generator():
        return fill(
            ExternalFunction(
                "fill_included",
                source_string=(
                    '#include "seven.h"\n'
                    'extern "C" void fill_included(int *out) {\n'
                    f"  for (int i = 0; i < {TILE}; i++) out[i] = SEVEN;\n"
                    "}\n"
                ),
                arg_types=[tile_ty],
            )
        )

    design = CompilableDesign(generator, include_paths=[include], use_cache=False)
    design.compile(full_elf_path=tmp_path / "out" / "design.elf")
    assert (tmp_path / "out" / "design.elf").stat().st_size > 0


def test_device_builds_kernels_while_aiecc_lowers(tmp_path, caplog):
    source = tmp_path / "fill_seven.cc"
    source.write_text(
        'extern "C" void fill_seven(int *out) {\n'
        f"  for (int i = 0; i < {TILE}; i++) out[i] = 7;\n"
        "}\n"
    )
    kernel = ExternalFunction(
        "fill_seven", source_file=str(source), arg_types=[tile_ty]
    )
    text = str(fill(kernel))

    overlapped = tmp_path / "overlapped"
    overlapped.mkdir()
    with caplog.at_level(logging.DEBUG, logger=compile_utils.__name__):
        compile_mlir_module(
            text,
            work_dir=overlapped,
            full_elf_path=overlapped / "design.elf",
            device=NPU2Col1(),
        )
    assert any("--await-link-files" in record.getMessage() for record in caplog.records)

    serial = tmp_path / "serial"
    serial.mkdir()
    compile_external_kernels([kernel], str(serial), "aie2p")
    compile_mlir_module(text, work_dir=serial, full_elf_path=serial / "design.elf")

    assert (overlapped / "design.elf").read_bytes() == (
        serial / "design.elf"
    ).read_bytes()


def test_device_kernel_error_stops_aiecc(tmp_path):
    source = tmp_path / "fill_broken.cc"
    source.write_text('#error "fill_broken does not compile"\n')
    text = str(
        fill(
            ExternalFunction(
                "fill_broken", source_file=str(source), arg_types=[tile_ty]
            )
        )
    )
    build = tmp_path / "build"
    build.mkdir()
    with pytest.raises(Exception, match="fill_broken does not compile") as raised:
        compile_mlir_module(
            text,
            work_dir=build,
            full_elf_path=build / "design.elf",
            device=NPU2Col1(),
        )
    assert "stdin closed" not in str(raised.value)


def test_device_and_build_link_files_are_one_or_the_other(tmp_path):
    with pytest.raises(ValueError, match="pass one of them"):
        compile_mlir_module(
            "module {}",
            work_dir=tmp_path,
            device=NPU2Col1(),
            build_link_files=lambda: None,
        )
