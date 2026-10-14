# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %pytest %s

"""Check diagnostic forwarding from successful and failed aiecc builds."""

from pathlib import Path

import pytest
from aie.utils.compile import utils

TESTS = Path(__file__).resolve().parents[1]
# Its DMA queue overflows on a target with no pollable occupancy register, which
# aiecc reports as a warning with notes and builds anyway.
QUEUE_DEPTH = (
    TESTS / "bd-chains-and-dma-tasks/assign-runtime-sequence-bd-ids"
    "/queue-depth-unsupported-target.mlir"
)
# Its flows can deadlock however they are routed, which fails the build.
DEADLOCK_PRONE = TESTS / "aiecc/allow_deadlock_prone_routing.mlir"


@pytest.mark.parametrize("located", [True, False])
def test_successful_diagnostics(tmp_path, capsys, located):
    source = QUEUE_DEPTH.read_text()
    if not located:
        push = "    aiex.dma_start_task(%t4)\n"
        assert source.count(push) == 1
        source = source.replace(push, push.replace(")\n", ") loc(unknown)\n"))
    design = tmp_path / "design.mlir"
    design.write_text(source)

    utils._run_aiecc(
        str(design),
        [
            "-n",
            f"--tmpdir={tmp_path / 'prj'}",
            "--get-npu-insts",
            f"--npu-insts-name={tmp_path / 'insts.bin'}",
        ],
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    where = f"{design}:" if located else "<unknown>:0:"
    lines = captured.err.splitlines()
    assert [line.split(": ")[1] for line in lines] == ["warning", "note", "note"]
    assert all(line.startswith(f"[aiecc] {where}") for line in lines)
    assert "whose task queue is only 4 deep" in lines[0]
    assert "the compiler would have waited for a free slot here" in lines[2]


def test_failed_diagnostics(tmp_path, capsys):
    with pytest.raises(
        RuntimeError,
        match="exit code 1:\n.*error: Flows can deadlock however they are routed",
    ):
        utils._run_aiecc(
            str(DEADLOCK_PRONE), ["-n", f"--tmpdir={tmp_path}", "--get-xclbin"]
        )

    assert capsys.readouterr().err == ""
