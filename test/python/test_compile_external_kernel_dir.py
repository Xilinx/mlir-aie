# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
# REQUIRES: peano
"""compile_external_kernel builds into a kernel_dir given relative to the cwd."""

import numpy as np
import pytest

from aie.iron import ExternalFunction
from aie.utils.compile.utils import compile_external_kernel

_SOURCE = """
extern "C" void add_one(int *a, int *b, int n) {
  for (int i = 0; i < n; i++)
    b[i] = a[i] + 1;
}
"""


@pytest.mark.parametrize("staged", ["source_string", "source_file"])
def test_a_relative_kernel_dir_builds(tmp_path, monkeypatch, staged):
    source = tmp_path / "src" / "add_one.cc"
    source.parent.mkdir()
    source.write_text(_SOURCE)
    given = (
        dict(source_string=_SOURCE)
        if staged == "source_string"
        else dict(source_file=str(source))
    )
    kernel = ExternalFunction(
        "add_one",
        arg_types=[
            np.ndarray[(64,), np.dtype[np.int32]],
            np.ndarray[(64,), np.dtype[np.int32]],
            np.int32,
        ],
        **given,
    )
    (tmp_path / "kernels").mkdir()
    monkeypatch.chdir(tmp_path)
    assert compile_external_kernel(kernel, "kernels", "aie2p")
    assert (tmp_path / "kernels" / kernel.object_file_name).stat().st_size
