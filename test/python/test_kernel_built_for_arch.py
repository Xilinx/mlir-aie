# test_kernel_built_for_arch.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""A library kernel compiles only for the architecture its factory chose.

A factory reads the bound device when it is called and picks its source,
flags and contract for that architecture (aie2 when none is bound). A
script that calls it before binding, then runs a design on an NPU2, used
to hand Peano the aie2 LUT headers for aie2p ("call to 'mul_elem_16_2' is
ambiguous").
"""

import os

import pytest
from aie.iron import kernels
from aie.utils import config, get_current_device
from aie.utils.compile.utils import compile_external_kernels
from aie.utils.hostruntime import set_current_device


def _peano_available() -> bool:
    try:
        return os.path.isfile(config.peano_cxx_path())
    except RuntimeError:
        return False


@pytest.fixture
def no_device():
    previous = get_current_device(probe_runtime=False)
    set_current_device(None)
    try:
        yield
    finally:
        set_current_device(previous)


def test_a_kernel_built_unbound_refuses_an_aie2p_compile(no_device, tmp_path):
    exp = kernels.bf16_exp(tile_size=1024)
    assert exp.built_for_arch == "aie2"
    with pytest.raises(ValueError, match="built for aie2 but .* for aie2p.*or none"):
        compile_external_kernels([exp], str(tmp_path), "aie2p")
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
def test_a_kernel_built_bound_compiles_for_its_arch(npu2_device, tmp_path):
    exp = kernels.bf16_exp(tile_size=1024)
    assert exp.built_for_arch == "aie2p"
    compile_external_kernels([exp], str(tmp_path), "aie2p")
    assert (tmp_path / exp.object_file_name).is_file()
