# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %pytest %s
# REQUIRES: peano

"""What IRON shows of an aiecc build that placed the design more than once."""

from pathlib import Path

import pytest
from aie.utils import config
from aie.utils.compile import utils

AIECC_TESTS = Path(__file__).resolve().parents[1] / "aiecc"
# Its first placement does not route, and the one with three cores a column
# does; every placement warns about an uncalled function.
ROUTES_SECOND = AIECC_TESTS / "Inputs/placement_retry_npu2_decode.mlir"
# No placement routes; placement 1's failure stands.
NEVER_ROUTES = AIECC_TESTS / "placement_retry.test"
# Placement 1 has no room for its walked pools; placement 2 allocates, then
# cannot link a missing kernel object, and stands.
FAILS_ELSEWHERE = AIECC_TESTS / "placement_retry_failed_elsewhere.test"


def test_success_shows_the_standing_placement(tmp_path, capsys):
    utils._run_aiecc(
        str(ROUTES_SECOND),
        ["-n", f"--tmpdir={tmp_path / 'prj'}", "--get=input_physical.mlir"],
        cwd=tmp_path,
    )

    err = capsys.readouterr().err
    assert "Unable to find a legal routing" not in err, err
    assert "[aiecc] aiecc: placement 2 (--cores-per-col=3) stands" in err, err
    assert err.count("never called from any core") == 1, err


def test_failure_summarises_the_first_placement_when_it_stands(tmp_path):
    with pytest.raises(RuntimeError) as failure:
        utils._run_aiecc(
            str(NEVER_ROUTES),
            ["-n", f"--tmpdir={tmp_path / 'prj'}", "--get=input_physical.mlir"],
            cwd=tmp_path,
        )

    summary = str(failure.value).split("--- full aiecc log ---")[0]
    assert summary.count("Unable to find a legal routing") == 1, summary
    assert "the messages above are from placement 3" in summary, summary
    assert "placement 1's failure stands" in summary, summary


def test_failure_locates_the_standing_placement(tmp_path):
    with pytest.raises(RuntimeError) as failure:
        utils._run_aiecc(
            str(FAILS_ELSEWHERE),
            [
                f"--peano={config.peano_install_dir()}",
                f"--tmpdir={tmp_path / 'prj'}",
                "--placer=sa_placer",
                "--get=input_with_addresses.mlir",
            ],
            cwd=tmp_path,
        )

    summary = str(failure.value).split("--- full aiecc log ---")[0]
    assert "could not place buffers" not in summary, summary
    assert "missing_kernel.o" in summary, summary
    assert "placement 2 (--sa-walked-fifos=main:b_in,main:a_in) stands" in summary
    # Placement 2's error has no location in the design: no frame points at
    # placement 1's allocation error there.
    frames = []
    tb = failure.value.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame.f_code.co_filename)
        tb = tb.tb_next
    assert str(FAILS_ELSEWHERE) not in frames, frames
