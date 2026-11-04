# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
# REQUIRES: peano

import json
import re
import subprocess

import aie
from aie.iron import Program, Runtime, Worker, WorkerRuntimeBarrier
from aie.iron.device import NPU2Col1
from aie.utils import config


def test_iron_program_stamps_elf(tmp_path, monkeypatch):
    # Give source builds the same package fields a wheel provides.
    monkeypatch.setattr(aie, "__version__", "1.2.3", raising=False)
    monkeypatch.setattr(aie, "__commit__", "abcdef0", raising=False)
    barrier = WorkerRuntimeBarrier()

    def task(barrier):
        barrier.wait_for_value(1)
        barrier.release_with_value(1)

    def sequence():
        barrier.set(1)

    program = Program(
        NPU2Col1(), Runtime(sequence, []), workers=[Worker(task, fn_args=[barrier])]
    )
    source = tmp_path / "design.mlir"
    source.write_text(str(program.resolve_program()))
    compiler = config.aiecc_path()
    subprocess.run(
        [compiler, str(source), "--get-full-elf", "--full-elf-name=design.elf"],
        cwd=tmp_path,
        check=True,
    )
    dump = subprocess.check_output(
        [
            config.readobj_path(),
            "--string-dump=.note.mlir-aie.provenance",
            tmp_path / "design.elf",
        ],
        text=True,
    )
    info = json.loads(re.search(r"^\[.*\] (\{.*\})$", dump, re.MULTILINE)[1])
    banner = subprocess.check_output([compiler, "--version"], text=True)
    assert info == {
        "revision": re.search(r"git SHA:\s+(\S+)", banner)[1],
        "iron": {"version": "1.2.3", "revision": "abcdef0"},
    }
