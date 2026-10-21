# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""The YOLO26n component check's parsers and gates; no NPU required."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "utils/component_checks"

# What yolo26n.py --synthetic 7 --n 8 --time printed on a Strix npu2, at
# batch 1 and at batch 8.
YOLO_IMAGES = "".join(
    f"image random{i}: PASS (0/1800 differ)\n" for i in range(8)
) + "yolo26n: 8/8 images PASS\n"
YOLO_TIMED_OUTPUT = YOLO_IMAGES + """\
NPU time     (avg/min/max us): 11569.1 / 11255.0 / 11950.5   [median 11461.5, MAD 107.9, p95 11893.7]
End-to-end   (avg/min/max us): 12163.4 / 11386.1 / 12739.0   [median 12115.9, MAD 331.1, p95 12677.8]
per-image: 11386.1 us (87.83 fps) at batch 1, fastest of 20 after 10 warmups; median 12115.9 us, MAD 331.1 us
timed dispatch: PASS
"""
YOLO_BATCHED_OUTPUT = YOLO_IMAGES + """\
NPU time     (avg/min/max us): 47910.0 / 47656.1 / 48316.0   [median 47898.6, MAD 174.8, p95 48128.4]
End-to-end   (avg/min/max us): 48656.1 / 48370.6 / 49110.0   [median 48624.9, MAD 172.8, p95 48945.5]
per-image: 6046.3 us (165.39 fps) at batch 8, fastest of 20 after 10 warmups; median 6078.1 us, MAD 21.6 us
timed dispatch: PASS
"""
ORT_LINE = "ort: 8/8 images match onnxruntime\n"


@pytest.fixture(scope="module")
def yolo():
    spec = importlib.util.spec_from_file_location(
        "yolo26n_hw", CHECKS / "yolo26n_hw.py"
    )
    module = importlib.util.module_from_spec(spec)
    # The check imports its siblings, as when run as a script.
    sys.path.insert(0, str(CHECKS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(CHECKS))
    return module


def test_a_run_that_matches_every_image_passes(yolo):
    assert yolo.parse_correctness(YOLO_TIMED_OUTPUT + ORT_LINE, 0) == {
        "passed": True,
        "images_passed": 8,
        "images_total": 8,
        "ort_matches": 8,
        "failed_images": [],
    }


def test_an_image_that_differs_fails_the_run_by_name(yolo):
    out = YOLO_TIMED_OUTPUT.replace(
        "random1: PASS (0/1800 differ)", "random1: FAIL (7/1800 differ)"
    ).replace("8/8 images PASS", "7/8 images PASS")
    got = yolo.parse_correctness(out, 1)
    assert not got["passed"]
    assert (got["images_passed"], got["images_total"]) == (7, 8)
    assert got["failed_images"] == ["random1"]


@pytest.mark.parametrize(
    "out, returncode",
    [
        # A run that stops before its summary counts its image lines.
        (YOLO_IMAGES.split("yolo26n:")[0], 0),
        (YOLO_TIMED_OUTPUT, 1),
        ("", 0),
    ],
    ids=["no-summary", "nonzero-exit", "no-output"],
)
def test_a_run_without_a_clean_summary_fails(yolo, out, returncode):
    got = yolo.parse_correctness(out, returncode)
    assert not got["passed"]
    assert got["images_passed"] == out.count(": PASS (")


def test_a_run_without_onnxruntime_has_no_ort_row(yolo):
    got = yolo.parse_correctness(YOLO_TIMED_OUTPUT, 0)
    assert got["passed"] and got["ort_matches"] is None
    rows = yolo.aggregate(got, {}, {}, {})
    assert [r["name"] for r in rows] == ["yolo26n/640/images_passed"]


def test_timed_runs_report_the_fastest_call_with_its_spread(yolo):
    assert yolo.parse_timing(YOLO_IMAGES) == {}
    timing = yolo.parse_timing(YOLO_TIMED_OUTPUT) | yolo.parse_timing(
        YOLO_BATCHED_OUTPUT
    )
    assert timing == {
        "us_per_image_b1": 11386.1,
        "fps_b1": 87.83,
        "timed_calls_b1": 20,
        "median_us_per_image_b1": 12115.9,
        "mad_us_per_image_b1": 331.1,
        "us_per_image_b8": 6046.3,
        "fps_b8": 165.39,
        "timed_calls_b8": 20,
        "median_us_per_image_b8": 6078.1,
        "mad_us_per_image_b8": 21.6,
    }
    rows = yolo.aggregate(
        yolo.parse_correctness(YOLO_TIMED_OUTPUT + ORT_LINE, 0),
        timing,
        {"elf_bytes": 1_234_567},
        {"compile_s": 95.1, "compile_peak_rss_mb": 3100.0},
    )
    assert [(r["name"], r["unit"], r["value"], r.get("range")) for r in rows] == [
        ("yolo26n/640/images_passed", "images", 8, "of 8"),
        ("yolo26n/640/ort_matches", "images", 8, "of 8"),
        (
            "yolo26n/640/us_per_image_b1",
            "us",
            11386.1,
            "fastest of 20; median 12115.9, MAD 331.1",
        ),
        ("yolo26n/640/fps_b1", "fps", 87.83, None),
        (
            "yolo26n/640/us_per_image_b8",
            "us",
            6046.3,
            "fastest of 20; median 6078.1, MAD 21.6",
        ),
        ("yolo26n/640/fps_b8", "fps", 165.39, None),
        ("yolo26n/640/elf_bytes", "bytes", 1_234_567, None),
        ("yolo26n/640/compile_s", "s", 95.1, None),
        ("yolo26n/640/compile_peak_rss_mb", "MB", 3100.0, None),
    ]
    # Every metric the check publishes has a threshold the page and the PR
    # report share.
    thresholds = json.loads(
        (ROOT / "utils/kernel_checks/thresholds.json").read_text(encoding="utf-8")
    )
    assert {r["name"].rsplit("/", 1)[1] for r in rows} <= thresholds.keys()


def test_a_run_of_the_earlier_format_publishes_no_time(yolo):
    old = "per-image: 78365.1 us (12.76 fps) at batch 1\n"
    assert yolo.parse_timing(YOLO_IMAGES + old) == {}


def test_a_passing_run_timed_in_full_is_sane(yolo):
    correctness = yolo.parse_correctness(YOLO_BATCHED_OUTPUT, 0)
    timing = yolo.parse_timing(YOLO_TIMED_OUTPUT) | yolo.parse_timing(
        YOLO_BATCHED_OUTPUT
    )
    assert yolo.measurement_sane(correctness, timing, [1, 8])


@pytest.mark.parametrize(
    "edit, returncode",
    [
        # A failed image withholds every row, so it never becomes the baseline.
        (("random3: PASS (0/1800", "random3: FAIL (9/1800"), 1),
        # Fewer timed calls than the check asked for.
        (("fastest of 20 after 10", "fastest of 5 after 2"), 0),
        # A fastest call slower than the median cannot come from one sample set.
        (("median 6078.1 us", "median 6000.0 us"), 0),
        # A batch that printed no time.
        (("per-image: 6046.3", "per image: 6046.3"), 0),
    ],
    ids=["failed-image", "too-few-calls", "min-above-median", "untimed-batch"],
)
def test_a_run_whose_numbers_cannot_stand_is_not_sane(yolo, edit, returncode):
    batched = YOLO_BATCHED_OUTPUT.replace(*edit)
    if edit[0].startswith("random3"):
        batched = batched.replace("8/8 images PASS", "7/8 images PASS")
    correctness = yolo.parse_correctness(batched, returncode)
    timing = yolo.parse_timing(YOLO_TIMED_OUTPUT) | yolo.parse_timing(batched)
    assert not yolo.measurement_sane(correctness, timing, [1, 8])


def test_the_elf_size_is_the_design_elfs(yolo, tmp_path):
    (tmp_path / "design.elf").write_bytes(b"\0" * 300)
    # aiecc's per-core ELFs share the directory and sort first.
    (tmp_path / "core_0_2.elf").write_bytes(b"\0" * 7)
    (tmp_path / "elfs_0.elf").write_bytes(b"\0" * 9)
    assert yolo.artifact_stats(str(tmp_path)) == {"elf_bytes": 300}


def test_a_build_without_a_design_elf_reports_its_xclbin(yolo, tmp_path):
    (tmp_path / "elfs_0.elf").write_bytes(b"\0" * 9)
    (tmp_path / "final.xclbin").write_bytes(b"\0" * 40)
    assert yolo.artifact_stats(str(tmp_path)) == {"xclbin_bytes": 40}


@pytest.mark.parametrize("n_images", ["0", "12", "-8"])
def test_n_images_must_be_a_positive_multiple_of_the_batch(
    yolo, tmp_path, n_images, capsys
):
    with pytest.raises(SystemExit) as exited:
        yolo.main(
            [
                "--n-images",
                n_images,
                "--out",
                str(tmp_path / "perf.json"),
                "--meta",
                str(tmp_path / "meta.json"),
            ]
        )
    assert exited.value.code == 2
    assert "positive multiple of 8" in capsys.readouterr().err
    assert not (tmp_path / "meta.json").exists()
