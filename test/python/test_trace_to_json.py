# test_trace_to_json.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python -m pytest %s -v

"""TraceConfig.trace_to_json and parse_trace on a single-design trace.

The inputs are parse-trace/test1's: a captured trace and the design that wrote
it. The per-slice output of a shared buffer is covered on hardware by
npu-xrt/trace_slices_configure_run.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from aie.utils.trace import TraceConfig
from aie.utils.trace.parse import parse_trace

INPUTS = Path(__file__).resolve().parent.parent / "parse-trace" / "test1"
MLIR = INPUTS / "aie_test1.mlir"
# Without the file's zero padding, so that a buffer of exactly this size is full.
WORDS = np.trim_zeros(
    np.array(
        [int(line, 16) for line in (INPUTS / "trace_test1.txt").read_text().split()],
        dtype=np.uint32,
    ),
    "b",
)


def write(tmp_path, trace_size):
    config = TraceConfig(trace_size=trace_size, trace_file=str(tmp_path / "t.txt"))
    config.write_trace(WORDS)
    return config


def test_whole_buffer_goes_to_output_name(tmp_path):
    config = write(tmp_path, 2 * WORDS.nbytes)
    output = tmp_path / "trace.json"
    assert config.trace_to_json(str(MLIR), str(output)) == [str(output)]
    assert json.loads(output.read_text()) == parse_trace(
        config.read_trace(), MLIR.read_text()
    )


def test_a_full_buffer_is_reported_truncated(tmp_path, caplog):
    config = write(tmp_path, WORDS.nbytes)
    with caplog.at_level(logging.WARNING, logger="aie.utils.trace.config"):
        config.trace_to_json(str(MLIR), str(tmp_path / "trace.json"))
    assert "likely truncated" in caplog.text


def test_a_buffer_with_room_left_is_not(tmp_path, caplog):
    config = write(tmp_path, 2 * WORDS.nbytes)
    with caplog.at_level(logging.WARNING, logger="aie.utils.trace.config"):
        config.trace_to_json(str(MLIR), str(tmp_path / "trace.json"))
    assert "truncated" not in caplog.text


def test_data_from_an_untraced_tile_raises(tmp_path):
    # Shifting the design's columns moves its traced tile away from the one
    # that wrote the data. That used to exit the interpreter.
    with pytest.raises(ValueError, match="does not trace"):
        parse_trace(WORDS, MLIR.read_text(), colshift=2)
