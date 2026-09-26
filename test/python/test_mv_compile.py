# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
# REQUIRES: peano
"""Compile bf16 matvec boundary shapes with the real AIE compiler (no NPU)."""

from pathlib import Path

import pytest
from aie.utils.compile.utils import compile_cxx_core_function

_SOURCE = Path(__file__).resolve().parents[2] / "aie_kernels" / "linalg" / "mv_bf16.cc"


@pytest.mark.parametrize("arch", ["aie2", "aie2p"])
@pytest.mark.parametrize("chunks", [1, 2, 4])
def test_matvec_complete_chunks_compile(tmp_path, arch, chunks):
    output = tmp_path / "mv.o"
    compile_cxx_core_function(
        str(_SOURCE),
        arch,
        str(output),
        compile_args=[f"-DDIM_K={64 * chunks}", "-DVEC_SIZE=64"],
    )
    assert output.stat().st_size > 0


@pytest.mark.parametrize("dim_k", [0, 32, 65])
def test_matvec_invalid_chunks_fail_compilation(tmp_path, dim_k):
    with pytest.raises(RuntimeError, match="static assertion failed"):
        compile_cxx_core_function(
            str(_SOURCE),
            "aie2",
            str(tmp_path / "mv.o"),
            compile_args=[f"-DDIM_K={dim_k}", "-DVEC_SIZE=64"],
        )
