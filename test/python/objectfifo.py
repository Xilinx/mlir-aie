# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

import pytest
import numpy as np

from aie.iron.device import (
    NPU1Col1,
    NPU1,
    NPU2,
    Tile,
    AnyMemTile,
    AnyComputeTile,
    AnyShimTile,
)
from aie.iron.dataflow.objectfifo import ObjectFifo
from aie.iron.dataflow.endpoint import ObjectFifoEndpoint


@pytest.fixture(params=[NPU1Col1, NPU1, NPU2])
def device(request):
    return request.param()


def test_has_legal_mem_affinity(device):
    n_ty = np.ndarray[(1024,), np.dtype[np.int32]]

    # Legal affinity
    of_legal = ObjectFifo(n_ty)
    of_legal.prod().endpoint = ObjectFifoEndpoint(Tile(1, 2))
    of_legal.cons().endpoint = ObjectFifoEndpoint(Tile(1, 3))
    assert of_legal.has_legal_mem_affinity(device)

    # Illegal affinity
    of_illegal = ObjectFifo(n_ty)
    of_illegal.prod().endpoint = ObjectFifoEndpoint(Tile(0, 0))
    of_illegal.cons().endpoint = ObjectFifoEndpoint(Tile(1, 2))
    assert not of_illegal.has_legal_mem_affinity(device)

    # Multiple consumers, legal
    of_mult_cons_legal = ObjectFifo(n_ty)
    of_mult_cons_legal.prod().endpoint = ObjectFifoEndpoint(Tile(1, 2))
    of_mult_cons_legal.cons().endpoint = ObjectFifoEndpoint(Tile(1, 3))
    of_mult_cons_legal.cons().endpoint = ObjectFifoEndpoint(Tile(1, 4))
    assert of_mult_cons_legal.has_legal_mem_affinity(device)

    # Multiple consumers, illegal
    of_mult_cons_illegal = ObjectFifo(n_ty)
    of_mult_cons_illegal.prod().endpoint = ObjectFifoEndpoint(Tile(1, 2))
    of_mult_cons_illegal.cons().endpoint = ObjectFifoEndpoint(Tile(1, 3))
    of_mult_cons_illegal.cons().endpoint = ObjectFifoEndpoint(Tile(0, 0))
    assert not of_mult_cons_illegal.has_legal_mem_affinity(device)

    # Forwarded ObjectFifo
    of_forward = ObjectFifo(n_ty)
    of_forward.prod().endpoint = ObjectFifoEndpoint(Tile(1, 2))
    forwarded = of_forward.cons().forward(placement=AnyMemTile)
    forwarded.cons().endpoint = ObjectFifoEndpoint(Tile(1, 3))
    with pytest.raises(ValueError):
        of_forward.has_legal_mem_affinity(device)
    with pytest.raises(ValueError):
        forwarded.has_legal_mem_affinity(device)

    # AnyComputeTile
    of_any_compute = ObjectFifo(n_ty)
    of_any_compute.prod().endpoint = ObjectFifoEndpoint(AnyComputeTile)
    of_any_compute.cons().endpoint = ObjectFifoEndpoint(Tile(1, 3))
    with pytest.raises(ValueError):
        of_any_compute.has_legal_mem_affinity(device)

    # AnyShimTile
    of_any_shim = ObjectFifo(n_ty)
    of_any_shim.prod().endpoint = ObjectFifoEndpoint(AnyShimTile)
    of_any_shim.cons().endpoint = ObjectFifoEndpoint(Tile(1, 3))
    with pytest.raises(ValueError):
        of_any_shim.has_legal_mem_affinity(device)


def test_set_iter_count():
    n_ty = np.ndarray[(1024,), np.dtype[np.int32]]
    of = ObjectFifo(n_ty)
    of.set_iter_count(10)
    assert of._iter_count == 10
    with pytest.raises(ValueError):
        of.set_iter_count(0)
    with pytest.raises(ValueError):
        of.set_iter_count(257)


def test_acquire_release():
    n_ty = np.ndarray[(1024,), np.dtype[np.int32]]
    of = ObjectFifo(n_ty, depth=5)
    prod = of.prod()
    cons = of.cons()
    with pytest.raises(ValueError):
        prod.acquire(6)
    with pytest.raises(ValueError):
        cons.acquire(6)
    with pytest.raises(ValueError):
        prod.release(6)
    with pytest.raises(ValueError):
        cons.release(6)


def test_join():
    n_ty = np.ndarray[(1024,), np.dtype[np.int32]]
    of = ObjectFifo(n_ty)
    prod = of.prod()
    with pytest.raises(ValueError):
        of.cons().join([0])
    sub_fifos = prod.join([0, 1024])
    assert len(sub_fifos) == 2
    assert isinstance(sub_fifos[0], ObjectFifo)
    assert isinstance(sub_fifos[1], ObjectFifo)
    assert sub_fifos[0].name == of.name + "_join0"
    assert sub_fifos[1].name == of.name + "_join1"


def test_split():
    n_ty = np.ndarray[(1024,), np.dtype[np.int32]]
    of = ObjectFifo(n_ty)
    cons = of.cons()
    with pytest.raises(ValueError):
        of.prod().split([0])
    sub_fifos = cons.split([0, 1024])
    assert len(sub_fifos) == 2
    assert isinstance(sub_fifos[0], ObjectFifo)
    assert isinstance(sub_fifos[1], ObjectFifo)
    assert sub_fifos[0].name == of.name + "_split0"
    assert sub_fifos[1].name == of.name + "_split1"
