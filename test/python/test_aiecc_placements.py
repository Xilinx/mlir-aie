# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %pytest %s
# REQUIRES: peano

"""What IRON shows of an aiecc build that placed the design more than once."""

import traceback
from pathlib import Path

import pytest
from aie.utils import config
from aie.utils.compile import utils

AIECC_TESTS = Path(__file__).resolve().parents[1] / "aiecc"
# Its first placement does not route, and the one with three cores a column
# does; every placement warns about an uncalled function.
ROUTES_SECOND = AIECC_TESTS / "Inputs/placement_retry_npu2_decode.mlir"
# No placement routes; the last, placement 4, stands.
NEVER_ROUTES = AIECC_TESTS / "placement_retry.test"
# Placement 1 has no room for its walked pools; placement 2 allocates, then
# cannot link a missing kernel object, and stands.
FAILS_ELSEWHERE = AIECC_TESTS / "placement_retry_failed_elsewhere.test"
# Placement 1 has no room for its walked pools; placement 2 allocates them
# and has no room beside a core's sections; placement 3 fails as placement 2
# did, and stands.
MIXED = AIECC_TESTS / "placement_retry_mixed.test"


def _failure(design: Path, tmp_path: Path) -> RuntimeError:
    with pytest.raises(RuntimeError) as failure:
        utils._run_aiecc(
            str(design),
            [
                f"--peano={config.peano_install_dir()}",
                f"--tmpdir={tmp_path / 'prj'}",
                "--placer=sa_placer",
                "--get=input_with_addresses.mlir",
            ],
            cwd=tmp_path,
        )
    return failure.value


@pytest.fixture(scope="module")
def fails_elsewhere(tmp_path_factory) -> RuntimeError:
    return _failure(FAILS_ELSEWHERE, tmp_path_factory.mktemp("elsewhere"))


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


def test_failure_summarises_the_placement_that_stands(tmp_path):
    with pytest.raises(RuntimeError) as failure:
        utils._run_aiecc(
            str(NEVER_ROUTES),
            ["-n", f"--tmpdir={tmp_path / 'prj'}", "--get=input_physical.mlir"],
            cwd=tmp_path,
        )

    summary = str(failure.value).split("--- full aiecc log ---")[0]
    assert summary.count("Unable to find a legal routing") == 1, summary
    assert "the messages above are from placement 3" in summary, summary
    assert "placement 4's failure stands" in summary, summary


def test_failure_locates_the_standing_placement(fails_elsewhere):
    summary = str(fails_elsewhere).split("--- full aiecc log ---")[0]
    assert "could not place buffers" not in summary, summary
    assert "missing_kernel.o" in summary, summary
    assert "placement 2 (--sa-walked-fifos=main:a_in,main:b_in) stands" in summary
    # Placement 2's error has no location in the design: no frame points at
    # placement 1's allocation error there.
    frames = [f.filename for f in traceback.extract_tb(fails_elsewhere.__traceback__)]
    assert str(FAILS_ELSEWHERE) not in frames, frames


def test_failure_summary_keeps_log_order(fails_elsewhere):
    summary = str(fails_elsewhere).split("--- full aiecc log ---")[0]
    marker = summary.index("the messages above are from placement 1")
    error = summary.index("missing_kernel.o")
    assert marker < error < summary.index(") stands"), summary


def test_failure_summary_keeps_log_order_when_a_later_placement_stands(tmp_path):
    summary = str(_failure(MIXED, tmp_path)).split("--- full aiecc log ---")[0]
    marker = summary.index("the messages above are from placement 2")
    error = summary.index("could not be placed", marker)
    assert error < summary.index("placement 3's failure stands"), summary


def test_summary_limit_keeps_what_says_which_placement_stands(fails_elsewhere):
    log = str(fails_elsewhere).split("--- full aiecc log ---\n")[1]
    marker = next(
        line for line in log.splitlines() if line.startswith("aiecc: placing again")
    )
    # Placement 2 printing 25 distinct errors before its link fails.
    errors = "".join(f"design.mlir:{n}:1: error: error {n}\n" for n in range(25))
    summary = utils.aiecc_diagnostics(log.replace(marker, marker + "\n" + errors))

    assert summary[0] == marker, summary
    assert summary[-1].startswith("aiecc: "), summary
    assert any(line.endswith(") stands") for line in summary), summary
    assert any(line.startswith("... truncated") for line in summary), summary
