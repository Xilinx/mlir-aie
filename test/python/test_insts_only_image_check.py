# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
# REQUIRES: peano
"""An instructions-only build given the image it runs on is checked against
it: a stream whose static design is the image's builds, and one whose runtime
sequence placed something elsewhere is refused, naming what to pin.

The designs are test/check-against-image/Inputs: one static design, with a
light runtime sequence that lets its two shim ends share a shim and a heavy
one that does not.
"""

from pathlib import Path

import pytest

from aie.iron.device import NPU2
from aie.utils import set_current_device
from aie.utils.compile.jit.compilabledesign import CompilableDesign

INPUTS = Path(__file__).resolve().parent.parent / "check-against-image" / "Inputs"


@pytest.fixture(autouse=True)
def _device():
    with set_current_device(NPU2()):
        yield


@pytest.fixture
def image(tmp_path):
    design = CompilableDesign(INPUTS / "light.mlir", use_cache=False)
    design.compile(
        xclbin_path=tmp_path / "image" / "final.xclbin",
        inst_path=tmp_path / "image" / "insts.bin",
    )
    return design


def test_a_stream_of_the_images_design_builds(tmp_path, image):
    stream = CompilableDesign(INPUTS / "light.mlir", use_cache=False, insts_only=True)
    _, insts = stream.compile(inst_path=tmp_path / "stream" / "insts.bin", image=image)
    assert insts.read_bytes() == (tmp_path / "image" / "insts.bin").read_bytes()


def test_a_stream_placed_elsewhere_is_refused_with_what_to_pin(tmp_path, image):
    stream = CompilableDesign(INPUTS / "heavy.mlir", use_cache=False, insts_only=True)
    with pytest.raises(Exception) as refused:
        stream.compile(
            inst_path=tmp_path / "stream" / "insts.bin",
            image=image.get_cache_entry().lowered_mlir,
        )
    assert (
        "in_b's shim end: image (0, 0) MM2S 1, this stream (1, 0) MM2S 0; "
        "pin it to keep it stable across shapes" in str(refused.value)
    )
