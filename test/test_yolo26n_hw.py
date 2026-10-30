# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""The YOLO26n component check's parsers and gates; no NPU required."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "utils/component_checks"
YOLO = ROOT / "programming_examples/ml/yolo26n"

# What yolo26n.py --synthetic 7 --n 8 --time printed on a Strix npu2, at
# batch 1 and at batch 8, its full ELF carrying the empty runtime sequence.
YOLO_IMAGES = (
    "".join(f"image random{i}: PASS (0/1800 differ)\n" for i in range(8))
    + "yolo26n: 8/8 images PASS\n"
)
YOLO_TIMED_OUTPUT = YOLO_IMAGES + """\
Dispatch          (avg/min/max us): 11482.0 / 11278.5 / 12073.8   [median 11390.8, MAD 77.8, p95 11945.8]
Dispatch overhead (avg/min/max us): 103.7 / 56.0 / 228.2   [median 88.0, MAD 18.5, p95 196.9]
NPU               (avg/min/max us): 11394.0 / 11190.5 / 11985.8   [median 11302.8, MAD 77.8, p95 11857.9]
End-to-end        (avg/min/max us): 11966.6 / 11505.7 / 12419.0   [median 11909.0, MAD 203.4, p95 12334.8]
per-image: 11505.7 us (86.91 fps) at batch 1, fastest of 20 after 10 warmups; median 11909.0 us, MAD 203.4 us
timed dispatch: PASS
"""
YOLO_BATCHED_OUTPUT = YOLO_IMAGES + """\
Dispatch          (avg/min/max us): 47662.6 / 47275.5 / 48074.2   [median 47690.9, MAD 132.8, p95 48013.8]
Dispatch overhead (avg/min/max us): 181.4 / 125.2 / 228.3   [median 180.2, MAD 11.6, p95 222.0]
NPU               (avg/min/max us): 47482.5 / 47095.3 / 47894.0   [median 47510.7, MAD 132.8, p95 47833.6]
End-to-end        (avg/min/max us): 48662.1 / 48260.2 / 49056.5   [median 48661.4, MAD 175.1, p95 48991.8]
per-image: 6032.5 us (165.77 fps) at batch 8, fastest of 20 after 10 warmups; median 6082.7 us, MAD 21.9 us
timed dispatch: PASS
"""
# The same run of a full ELF built without the empty runtime sequence: no
# baseline to take from the dispatch.
YOLO_NO_BASELINE_OUTPUT = YOLO_IMAGES + """\
Dispatch          (avg/min/max us): 11367.0 / 11156.0 / 12927.1   [median 11236.2, MAD 67.5, p95 12296.9]
End-to-end        (avg/min/max us): 11634.2 / 11312.9 / 13317.1   [median 11497.0, MAD 174.7, p95 12447.8]
per-image: 11312.9 us (88.39 fps) at batch 1, fastest of 20 after 10 warmups; median 11497.0 us, MAD 174.7 us
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


@pytest.fixture(scope="module")
def arbiter_class():
    sys.path.insert(0, str(YOLO))
    try:
        module = importlib.import_module("yolo26n_numpy")
    finally:
        sys.path.remove(str(YOLO))
    return module.SoftmaxArbiter


def test_a_run_that_matches_every_image_passes(yolo):
    assert yolo.parse_correctness(YOLO_TIMED_OUTPUT + ORT_LINE, 0) == {
        "passed": True,
        "ort_passed": True,
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


def test_a_run_without_onnxruntime_fails_and_has_no_ort_row(yolo):
    got = yolo.parse_correctness(YOLO_TIMED_OUTPUT, 0)
    assert not got["passed"] and not got["ort_passed"]
    assert got["ort_matches"] is None
    rows = yolo.aggregate(got, {}, {}, {})
    assert [r["name"] for r in rows] == ["yolo26n/640/images_passed"]


@pytest.mark.parametrize(
    "matches, known, passed",
    [(8, 0, True), (7, 0, False), (6, 2, True), (5, 2, False)],
)
def test_onnxruntime_below_the_known_count_fails_the_run(yolo, matches, known, passed):
    out = YOLO_TIMED_OUTPUT + f"ort: {matches}/8 images match onnxruntime\n"
    got = yolo.parse_correctness(out, 0, known)
    assert (got["passed"], got["ort_passed"], got["ort_matches"]) == (
        passed,
        passed,
        matches,
    )
    ort = yolo.aggregate(got, {}, {}, {})[1]
    assert (ort["name"], ort["value"], ort["range"]) == (
        "yolo26n/640/ort_matches",
        matches,
        "of 8",
    )


def test_the_parameters_key_follows_the_model_extractor_and_packages(yolo, tmp_path):
    model = tmp_path / "yolo26n_xint8.onnx"
    model.write_bytes(b"model")
    sources = []
    for name in yolo.EXTRACTOR_SOURCES:
        (tmp_path / name).write_text(f"# {name}\n")
        sources.append(tmp_path / name)
    packages = {"onnx": "1.17.0", "onnxruntime": "1.20.1"}
    key = yolo.params_key(model, sources, packages)
    assert len(key) == 64
    assert yolo.params_key(model, sources, dict(packages)) == key
    keys = {key}
    model.write_bytes(b"modem")
    keys.add(yolo.params_key(model, sources, packages))
    model.write_bytes(b"model")
    sources[1].write_text("# edited\n")
    keys.add(yolo.params_key(model, sources, packages))
    sources[1].write_text(f"# {sources[1].name}\n")
    keys.add(yolo.params_key(model, sources, {**packages, "onnxruntime": "1.21.0"}))
    # The same bytes under another file's name are another extractor.
    sources[0].write_text(f"# {sources[1].name}\n")
    sources[1].write_text(f"# {yolo.EXTRACTOR_SOURCES[0]}\n")
    keys.add(yolo.params_key(model, sources, packages))
    assert len(keys) == 5


def test_every_extractor_source_is_in_the_example(yolo):
    assert all((YOLO / name).is_file() for name in yolo.EXTRACTOR_SOURCES)


@pytest.mark.parametrize(
    "scores, index, expected",
    [
        # 128 / 4 = 32 exactly.
        ([5, 5, 5, 5], 0, 32),
        # 128 / 3 = 42.67.
        ([-7, -7, -7], 2, 43),
        # 128 / 256 = 0.5, a tie, rounds to even.
        ([0] * 256, 0, 0),
        # 128 / 1 saturates to int8.
        ([3], 0, 127),
        # 128 / (1 + e^-1) = 93.58 and 128 e^-1 / (1 + e^-1) = 34.42.
        ([0, -8], 0, 94),
        ([0, -8], 1, 34),
    ],
)
def test_the_arbiter_rounds_the_exact_probability(
    arbiter_class, scores, index, expected
):
    arbiter = arbiter_class(score_scale=-3, prob_scale=-7)
    assert arbiter.exact(np.array(scores), index) == expected


def test_the_arbiter_agrees_with_float64_away_from_a_boundary(arbiter_class):
    arbiter = arbiter_class(score_scale=-3, prob_scale=-7)
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(20):
        scores = rng.integers(-128, 128, 64)
        e = np.exp((scores - scores.max()) * 2.0**-3)
        grid = e / e.sum() * 128
        for j in np.nonzero(np.abs(grid - np.floor(grid) - 0.5) > 1e-6)[0]:
            assert arbiter.exact(scores, int(j)) == min(int(np.rint(grid[j])), 127)
            checked += 1
    assert checked > 1000


@pytest.mark.parametrize(
    "oracle, ort, verdict",
    [
        (94, 93, "ORT off-boundary"),
        (93, 94, "oracle wrong"),
        (95, 93, "both wrong"),
    ],
)
def test_the_arbiter_says_who_is_off(arbiter_class, oracle, ort, verdict):
    arbiter = arbiter_class(score_scale=-3, prob_scale=-7)
    assert arbiter.verdict(np.array([0, -8]), 0, oracle, ort) == verdict


def test_the_arbiter_will_not_decide_inside_its_error(arbiter_class):
    # 93.58 is 0.08 above the boundary; three digits leave an error of about 3.7.
    arbiter = arbiter_class(score_scale=-3, prob_scale=-7, digits=3)
    assert arbiter.exact(np.array([0, -8]), 0) is None
    assert arbiter.verdict(np.array([0, -8]), 0, 94, 93) == "undecidable"


def test_timed_runs_report_the_fastest_call_with_its_spread(yolo):
    assert yolo.parse_timing(YOLO_IMAGES) == {}
    timing = yolo.parse_timing(YOLO_TIMED_OUTPUT) | yolo.parse_timing(
        YOLO_BATCHED_OUTPUT
    )
    # The NPU time is the dispatch less the empty dispatch's median; the host
    # overhead is the fastest call less the fastest dispatch, per image.
    assert timing == {
        "us_per_image_b1": 11505.7,
        "fps_b1": 86.91,
        "timed_calls_b1": 20,
        "median_us_per_image_b1": 11909.0,
        "mad_us_per_image_b1": 203.4,
        "dispatch_overhead_us_b1": 56.0,
        "dispatch_us_per_image_b1": 11278.5,
        "median_dispatch_us_per_image_b1": 11390.8,
        "mad_dispatch_us_per_image_b1": 77.8,
        "npu_us_per_image_b1": 11190.5,
        "median_npu_us_per_image_b1": 11302.8,
        "mad_npu_us_per_image_b1": 77.8,
        "host_overhead_us_per_image_b1": 227.2,
        "median_host_overhead_us_per_image_b1": 518.2,
        "mad_host_overhead_us_per_image_b1": 281.2,
        "us_per_image_b8": 6032.5,
        "fps_b8": 165.77,
        "timed_calls_b8": 20,
        "median_us_per_image_b8": 6082.7,
        "mad_us_per_image_b8": 21.9,
        "dispatch_overhead_us_b8": 125.2,
        "dispatch_us_per_image_b8": 5909.44,
        "median_dispatch_us_per_image_b8": 5961.36,
        "mad_dispatch_us_per_image_b8": 16.6,
        "npu_us_per_image_b8": 5886.91,
        "median_npu_us_per_image_b8": 5938.84,
        "mad_npu_us_per_image_b8": 16.6,
        "host_overhead_us_per_image_b8": 123.09,
        "median_host_overhead_us_per_image_b8": 121.31,
        "mad_host_overhead_us_per_image_b8": 38.49,
    }
    rows = yolo.aggregate(
        yolo.parse_correctness(YOLO_TIMED_OUTPUT + ORT_LINE, 0),
        timing,
        {"elf_bytes": 1_234_567},
        {"compile_s": 95.1, "compile_peak_mib": 4200.0, "compile_peak_rss_mb": 3100.0},
    )
    spread = "fastest of 20; median {}, MAD {}".format
    assert [(r["name"], r["unit"], r["value"], r.get("range")) for r in rows] == [
        ("yolo26n/640/images_passed", "images", 8, "of 8"),
        ("yolo26n/640/ort_matches", "images", 8, "of 8"),
        ("yolo26n/640/us_per_image_b1", "us", 11505.7, spread(11909.0, 203.4)),
        ("yolo26n/640/fps_b1", "fps", 86.91, None),
        ("yolo26n/640/dispatch_us_per_image_b1", "us", 11278.5, spread(11390.8, 77.8)),
        ("yolo26n/640/npu_us_per_image_b1", "us", 11190.5, spread(11302.8, 77.8)),
        (
            "yolo26n/640/host_overhead_us_per_image_b1",
            "us",
            227.2,
            spread(518.2, 281.2),
        ),
        ("yolo26n/640/us_per_image_b8", "us", 6032.5, spread(6082.7, 21.9)),
        ("yolo26n/640/fps_b8", "fps", 165.77, None),
        ("yolo26n/640/dispatch_us_per_image_b8", "us", 5909.44, spread(5961.4, 16.6)),
        ("yolo26n/640/npu_us_per_image_b8", "us", 5886.91, spread(5938.8, 16.6)),
        (
            "yolo26n/640/host_overhead_us_per_image_b8",
            "us",
            123.09,
            spread(121.3, 38.5),
        ),
        ("yolo26n/640/elf_bytes", "bytes", 1_234_567, None),
        ("yolo26n/640/compile_s", "s", 95.1, None),
        ("yolo26n/640/compile_peak_mib", "MiB", 4200.0, None),
        ("yolo26n/640/compile_peak_rss_mb", "MiB", 3100.0, None),
    ]
    # Every metric the check publishes has a threshold the page and the PR
    # report share.
    thresholds = json.loads(
        (ROOT / "utils/kernel_checks/thresholds.json").read_text(encoding="utf-8")
    )
    assert {r["name"].rsplit("/", 1)[1] for r in rows} <= thresholds.keys()


def test_an_elf_without_the_empty_sequence_publishes_no_npu_time(yolo):
    timing = yolo.parse_timing(YOLO_NO_BASELINE_OUTPUT)
    assert timing["dispatch_overhead_us_b1"] is None
    assert timing["dispatch_us_per_image_b1"] == 11156.0
    assert not [m for m in timing if m.startswith(("npu_", "median_npu_", "mad_npu_"))]
    # Still a sound measurement: the dispatch and the call were both timed.
    correctness = yolo.parse_correctness(YOLO_NO_BASELINE_OUTPUT + ORT_LINE, 0)
    assert yolo.measurement_sane(correctness, timing, [1])


def test_a_run_of_the_earlier_format_publishes_no_time(yolo):
    old = "per-image: 78365.1 us (12.76 fps) at batch 1\n"
    assert yolo.parse_timing(YOLO_IMAGES + old) == {}


def test_map50_is_read_from_the_runs_own_line(yolo):
    out = "map50: 0.5859 over 128 images (929 labelled objects)\n"
    assert yolo.parse_map50(out) == {
        "map50_coco128": 0.5859,
        "map50_coco128_objects": 929,
    }


def test_a_run_whose_weights_dir_has_no_labels_publishes_no_map50(yolo):
    assert yolo.parse_map50(YOLO_TIMED_OUTPUT) == {}


def test_map50_is_aggregated_as_one_row_with_its_object_count_in_range(yolo):
    timing = {"map50_coco128": 0.5859, "map50_coco128_objects": 929}
    correctness = yolo.parse_correctness(YOLO_IMAGES, 0)
    rows = yolo.aggregate(correctness, timing, {}, {})
    assert [(r["name"], r["unit"], r["value"], r.get("range")) for r in rows] == [
        ("yolo26n/640/images_passed", "images", 8, "of 8"),
        ("yolo26n/640/map50_coco128", "mAP", 0.5859, "929 labelled objects"),
    ]


def test_batch_1_and_batch_8_runs_must_agree_on_map50(yolo):
    correctness = yolo.parse_correctness(YOLO_BATCHED_OUTPUT + ORT_LINE, 0)
    timing = yolo.parse_timing(YOLO_TIMED_OUTPUT) | yolo.parse_timing(
        YOLO_BATCHED_OUTPUT
    )
    sane_timing = timing | {"map50_coco128_disagrees": False}
    assert yolo.measurement_sane(correctness, sane_timing, [1, 8])
    insane_timing = timing | {"map50_coco128_disagrees": True}
    assert not yolo.measurement_sane(correctness, insane_timing, [1, 8])


def test_a_passing_run_timed_in_full_is_sane(yolo):
    correctness = yolo.parse_correctness(YOLO_BATCHED_OUTPUT + ORT_LINE, 0)
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
        (("median 6082.7 us", "median 6000.0 us"), 0),
        # A batch that printed no time.
        (("per-image: 6032.5", "per image: 6032.5"), 0),
        # A batch whose dispatch was not timed apart from the call.
        (("Dispatch          (avg", "NPU time          (avg"), 0),
    ],
    ids=[
        "failed-image",
        "too-few-calls",
        "min-above-median",
        "untimed-batch",
        "untimed-dispatch",
    ],
)
def test_a_run_whose_numbers_cannot_stand_is_not_sane(yolo, edit, returncode):
    batched = YOLO_BATCHED_OUTPUT.replace(*edit)
    if edit[0].startswith("random3"):
        batched = batched.replace("8/8 images PASS", "7/8 images PASS")
    correctness = yolo.parse_correctness(batched + ORT_LINE, returncode)
    timing = yolo.parse_timing(YOLO_TIMED_OUTPUT) | yolo.parse_timing(batched)
    assert not yolo.measurement_sane(correctness, timing, [1, 8])


def test_a_sweep_that_passes_every_batch_times_every_batch(yolo):
    # The recorded batch-8 run, as it prints at each batch.
    sweep = yolo.BatchSweep(0)
    assert all(
        sweep.add(YOLO_BATCHED_OUTPUT.replace("batch 8", f"batch {b}") + ORT_LINE, 0)
        for b in yolo.BATCHES
    )
    timing = sweep.timing()
    assert {k for k in timing if k.startswith("us_per_image_b")} == {
        f"us_per_image_b{b}" for b in yolo.BATCHES
    }
    assert yolo.measurement_sane(sweep.correctness, timing, list(yolo.BATCHES))


@pytest.mark.parametrize("failing", [1, 2, 4, 16])
def test_the_first_failing_batch_ends_the_sweep_and_withholds_all_timing(yolo, failing):
    sweep = yolo.BatchSweep(0)
    went_on = []
    for batch in yolo.BATCHES:
        out = YOLO_BATCHED_OUTPUT.replace("batch 8", f"batch {batch}") + ORT_LINE
        if batch == failing:
            out = out.replace("random3: PASS (0/1800", "random3: FAIL (9/1800")
            out = out.replace("8/8 images PASS", "7/8 images PASS")
        went_on.append(sweep.add(out, 1 if batch == failing else 0))
        if not went_on[-1]:
            break
    assert went_on[-1] is False
    assert len(went_on) == yolo.BATCHES.index(failing) + 1
    assert sweep.correctness["failed_images"] == ["random3"]
    assert sweep.timing() == {}
    assert not yolo.measurement_sane(
        sweep.correctness, sweep.timing(), list(yolo.BATCHES)
    )


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
    assert "positive multiple of 16" in capsys.readouterr().err
    assert not (tmp_path / "meta.json").exists()


@pytest.fixture(scope="module")
def yolo_modules():
    sys.path.insert(0, str(YOLO))
    try:
        spec = importlib.import_module("yolo_spec")
        design = importlib.import_module("yolo26n_design")
        params = importlib.import_module("yolo_params")
    finally:
        sys.path.remove(str(YOLO))
    return spec, design, params


def test_a_span_is_named_by_its_first_and_last_ops(yolo_modules):
    spec, _, _ = yolo_modules
    network = spec.Network.yolo26n()
    start, stop = network.span("/model.9/cv1/conv/Conv", "/model.9/Add")
    assert network.ops[start].name == "/model.9/cv1/conv/Conv"
    assert network.ops[stop - 1].name == "/model.9/Add"
    with pytest.raises(KeyError, match="/model.9/Ad"):
        network.span("/model.9/cv1/conv/Conv", "/model.9/Ad")
    with pytest.raises(ValueError, match="comes before"):
        network.span("/model.9/Add", "/model.9/cv1/conv/Conv")


def test_every_fused_run_names_ops_in_order(yolo_modules):
    spec, design, _ = yolo_modules
    network = spec.Network.yolo26n()
    spans = [network.span(first, last) for first, last in design.FUSED]
    assert all(a < b for a, b in spans)
    assert all(b <= c for (_, b), (c, _) in zip(spans, spans[1:]))


def test_a_run_that_cuts_a_fused_run_is_refused(yolo_modules):
    spec, design, params = yolo_modules
    network = spec.Network.yolo26n()
    first, last = design.FUSED[0]
    start, stop = network.span(first, last)
    with pytest.raises(ValueError, match="cuts the fused run"):
        design.SegmentDesign(
            network, params.Params(), network.ops[: stop - 1], [network.ops[0].dst]
        )


@pytest.fixture(scope="module")
def coco_map():
    sys.path.insert(0, str(YOLO))
    try:
        module = importlib.import_module("coco_map")
    finally:
        sys.path.remove(str(YOLO))
    return module


def test_labels_are_read_into_the_letterboxed_frame(coco_map, tmp_path):
    # A 1280x720 image letterboxed to 640: ratio 0.5, top padding 140, no
    # left padding. One normalized box at its centre, a quarter wide and half
    # tall, lands at (240, 230, 400, 410) by hand.
    path = tmp_path / "labels.txt"
    path.write_text("0 0.5 0.5 0.25 0.5\n")
    labels = coco_map.Labels.read(
        path, height=720, width=1280, ratio=0.5, left=0, top=140
    )
    assert labels.classes.tolist() == [0]
    np.testing.assert_allclose(labels.boxes, [[240.0, 230.0, 400.0, 410.0]])


def test_an_image_with_no_objects_has_no_label_file(coco_map, tmp_path):
    # The YOLO convention: a zero-object image has no file, not an empty one.
    labels = coco_map.Labels.read(
        tmp_path / "missing.txt", height=720, width=1280, ratio=0.5, left=0, top=140
    )
    assert labels.classes.tolist() == [] and labels.boxes.shape == (0, 4)


def test_multiple_objects_and_classes_are_read_in_file_order(coco_map, tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text("0 0.5 0.5 0.25 0.5\n3 0.25 0.25 0.5 0.5\n")
    labels = coco_map.Labels.read(path, height=100, width=100, ratio=1.0, left=0, top=0)
    assert labels.classes.tolist() == [0, 3]
    np.testing.assert_allclose(
        labels.boxes, [[37.5, 25.0, 62.5, 75.0], [0.0, 0.0, 50.0, 50.0]]
    )


def _detection(x1, y1, x2, y2, score, cls):
    return [x1, y1, x2, y2, score, cls]


def test_a_perfect_detector_scores_near_one(coco_map):
    # Two labels, both matched exactly (IoU 1) in descending confidence: the
    # 101-point interpolation's last sentinel (recall 1.0 -> precision 0, by
    # construction) costs the curve its last half-step, 0.005 of the 0.01
    # spacing -- a known property of this AP formula, not a bug here.
    scored = coco_map.MeanAP50()
    labels = coco_map.Labels(
        classes=np.array([0, 0]),
        boxes=np.array([[0, 0, 10, 10], [20, 20, 30, 30]], np.float32),
    )
    detections = np.array(
        [
            _detection(0, 0, 10, 10, 0.9, 0),
            _detection(20, 20, 30, 30, 0.8, 0),
        ],
        np.float32,
    )
    scored.add(detections, labels)
    assert scored.value() == pytest.approx(0.995)
    assert scored.images == 1 and scored.objects == 2


def test_a_detection_with_no_overlap_scores_zero(coco_map):
    scored = coco_map.MeanAP50()
    labels = coco_map.Labels(
        classes=np.array([0]), boxes=np.array([[0, 0, 10, 10]], np.float32)
    )
    detections = np.array([_detection(100, 100, 110, 110, 0.9, 0)], np.float32)
    scored.add(detections, labels)
    assert scored.value() == 0.0


def test_a_detection_of_the_wrong_class_cannot_match(coco_map):
    # Same box, same confidence, a different class: no match, so no TP.
    scored = coco_map.MeanAP50()
    labels = coco_map.Labels(
        classes=np.array([0]), boxes=np.array([[0, 0, 10, 10]], np.float32)
    )
    detections = np.array([_detection(0, 0, 10, 10, 0.9, 1)], np.float32)
    scored.add(detections, labels)
    assert scored.value() == 0.0


def test_the_first_detection_in_model_order_claims_an_overlapped_label(coco_map):
    # Two detections overlap the same one label, in descending confidence as
    # the model emits them (add()'s documented order): the first claims it,
    # the second is a false positive, not a second match.
    scored = coco_map.MeanAP50()
    labels = coco_map.Labels(
        classes=np.array([0]), boxes=np.array([[0, 0, 10, 10]], np.float32)
    )
    detections = np.array(
        [
            _detection(0, 0, 10, 10, 0.9, 0),
            _detection(0, 0, 10, 10, 0.5, 0),
        ],
        np.float32,
    )
    scored.add(detections, labels)
    # recall=[1,1], precision=[1,0.5]: the envelope holds precision at 1
    # until recall reaches 1, so the AP is the same 0.995 as a single match.
    assert scored.value() == pytest.approx(0.995)


def test_detections_at_or_below_the_confidence_floor_are_dropped(coco_map):
    scored = coco_map.MeanAP50()
    labels = coco_map.Labels(
        classes=np.array([0]), boxes=np.array([[0, 0, 10, 10]], np.float32)
    )
    detections = np.array([_detection(0, 0, 10, 10, coco_map.CONF, 0)], np.float32)
    scored.add(detections, labels)
    assert scored.value() == 0.0


def test_mean_ap50_accumulates_over_several_images(coco_map):
    scored = coco_map.MeanAP50()
    labels_a = coco_map.Labels(
        classes=np.array([0]), boxes=np.array([[0, 0, 10, 10]], np.float32)
    )
    labels_b = coco_map.Labels(
        classes=np.array([0]), boxes=np.array([[0, 0, 10, 10]], np.float32)
    )
    scored.add(np.array([_detection(0, 0, 10, 10, 0.9, 0)], np.float32), labels_a)
    scored.add(np.array([_detection(100, 100, 110, 110, 0.8, 0)], np.float32), labels_b)
    assert scored.images == 2 and scored.objects == 2
    # One of two labels matched: recall tops out at 0.5.
    assert 0 < scored.value() < 1
