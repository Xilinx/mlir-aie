# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

import pytest
import numpy as np

from aie.dialects._aie_enum_gen import AIETileType
from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker
from aie.iron.dataflow.objectfifo import ObjectFifoLink
from aie.iron.dataflow.endpoint import ObjectFifoEndpoint
from aie.iron.device import (
    NPU1Col1,
    NPU1,
    NPU2,
    Tile,
    AnyMemTile,
    AnyComputeTile,
    AnyShimTile,
)
from aie.iron.runtime.endpoint import RuntimeEndpoint


@pytest.fixture(params=[NPU1Col1, NPU1, NPU2])
def device(request):
    return request.param()


def test_can_used_shared_mem(device):
    n_ty = np.ndarray[(1024,), np.dtype[np.int32]]

    # Legal affinity
    of_legal = ObjectFifo(n_ty)
    of_legal.prod().endpoint = ObjectFifoEndpoint(Tile(1, 2))
    of_legal.cons().endpoint = ObjectFifoEndpoint(Tile(1, 3))
    assert of_legal.can_used_shared_mem(device)
    assert of_legal.can_used_shared_mem(device, cons_only=True)

    # Illegal affinity
    of_illegal = ObjectFifo(n_ty)
    of_illegal.prod().endpoint = ObjectFifoEndpoint(Tile(0, 0))
    of_illegal.cons().endpoint = ObjectFifoEndpoint(Tile(1, 2))
    assert not of_illegal.can_used_shared_mem(device)
    assert of_illegal.can_used_shared_mem(device, cons_only=True)

    # Multiple consumers, legal
    of_mult_cons_legal = ObjectFifo(n_ty)
    of_mult_cons_legal.prod().endpoint = ObjectFifoEndpoint(Tile(1, 2))
    of_mult_cons_legal.cons().endpoint = ObjectFifoEndpoint(Tile(1, 3))
    of_mult_cons_legal.cons().endpoint = ObjectFifoEndpoint(Tile(1, 4))
    assert of_mult_cons_legal.can_used_shared_mem(device)
    assert of_mult_cons_legal.can_used_shared_mem(device, cons_only=True)

    # Multiple consumers, illegal
    of_mult_cons_illegal = ObjectFifo(n_ty)
    of_mult_cons_illegal.prod().endpoint = ObjectFifoEndpoint(Tile(1, 2))
    of_mult_cons_illegal.cons().endpoint = ObjectFifoEndpoint(Tile(1, 3))
    of_mult_cons_illegal.cons().endpoint = ObjectFifoEndpoint(Tile(0, 0))
    assert not of_mult_cons_illegal.can_used_shared_mem(device)
    assert not of_mult_cons_illegal.can_used_shared_mem(device, cons_only=True)

    # Illegal producer, legal consumer
    of_illegal_prod = ObjectFifo(n_ty)
    of_illegal_prod.prod().endpoint = ObjectFifoEndpoint(Tile(0, 0))
    of_illegal_prod.cons().endpoint = ObjectFifoEndpoint(Tile(1, 2))
    assert not of_illegal_prod.can_used_shared_mem(device)
    assert of_illegal_prod.can_used_shared_mem(device, cons_only=True)

    # Forwarded ObjectFifo
    of_forward = ObjectFifo(n_ty)
    of_forward.prod().endpoint = ObjectFifoEndpoint(Tile(1, 2))
    forwarded = of_forward.cons().forward(tile=AnyMemTile)
    forwarded.cons().endpoint = ObjectFifoEndpoint(Tile(1, 3))
    with pytest.raises(ValueError):
        of_forward.can_used_shared_mem(device)
    with pytest.raises(ValueError):
        forwarded.can_used_shared_mem(device)

    # AnyComputeTile
    of_any_compute = ObjectFifo(n_ty)
    of_any_compute.prod().endpoint = ObjectFifoEndpoint(AnyComputeTile)
    of_any_compute.cons().endpoint = ObjectFifoEndpoint(Tile(1, 3))
    with pytest.raises(ValueError):
        of_any_compute.can_used_shared_mem(device)

    # AnyShimTile
    of_any_shim = ObjectFifo(n_ty)
    of_any_shim.prod().endpoint = ObjectFifoEndpoint(AnyShimTile)
    of_any_shim.cons().endpoint = ObjectFifoEndpoint(Tile(1, 3))
    with pytest.raises(ValueError):
        of_any_shim.can_used_shared_mem(device)


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


def test_worker_tile_type_validation():
    """Worker must use CoreTile; other tile types are rejected."""
    with pytest.raises(ValueError, match="Worker requires a compute tile"):
        Worker(None, tile=AnyMemTile)
    with pytest.raises(ValueError, match="Worker requires a compute tile"):
        Worker(None, tile=AnyShimTile)
    # CoreTile is accepted via sentinel
    w = Worker(None, tile=AnyComputeTile)
    assert w.tile.tile_type == AIETileType.CoreTile
    # Default tile_type is CoreTile
    w2 = Worker(None)
    assert w2.tile.tile_type == AIETileType.CoreTile


def test_runtime_endpoint_tile_type_validation():
    """RuntimeEndpoint must use ShimNOCTile; other tile types are rejected."""
    with pytest.raises(ValueError, match="RuntimeEndpoint requires a shim tile"):
        RuntimeEndpoint(tile=AnyComputeTile)
    with pytest.raises(ValueError, match="RuntimeEndpoint requires a shim tile"):
        RuntimeEndpoint(tile=AnyMemTile)
    # ShimNOCTile is accepted via sentinel
    ep = RuntimeEndpoint(tile=AnyShimTile)
    assert ep.tile.tile_type == AIETileType.ShimNOCTile
    # Default tile_type is ShimNOCTile
    ep2 = RuntimeEndpoint()
    assert ep2.tile.tile_type == AIETileType.ShimNOCTile


def test_workers_cannot_share_tile():
    """Two workers placed on the same coordinates must error."""
    n_ty = np.ndarray[(1024,), np.dtype[np.int32]]
    shared_tile = Tile(0, 2)
    of1 = ObjectFifo(n_ty, name="shared_of1")
    of2 = ObjectFifo(n_ty, name="shared_of2")
    w1 = Worker(None, [of1.cons()], tile=shared_tile)
    w2 = Worker(None, [of2.cons()], tile=shared_tile)
    rt = Runtime()
    with rt.sequence(n_ty, n_ty) as (A, B):
        rt.start(w1, w2)
        rt.fill(of1.prod(), A)
        rt.fill(of2.prod(), B)
    with pytest.raises(ValueError, match="Multiple workers cannot share the same tile"):
        Program(NPU2(), rt).resolve_program()


def test_workers_cannot_share_tile_by_coordinates():
    """Two workers with different Tile objects but same coordinates must error."""
    n_ty = np.ndarray[(1024,), np.dtype[np.int32]]
    tile_1 = Tile(0, 2)
    tile_2 = Tile(0, 2)
    of1 = ObjectFifo(n_ty, name="coord_of1")
    of2 = ObjectFifo(n_ty, name="coord_of2")
    w1 = Worker(None, [of1.cons()], tile=tile_1)
    w2 = Worker(None, [of2.cons()], tile=tile_2)
    rt = Runtime()
    with rt.sequence(n_ty, n_ty) as (A, B):
        rt.start(w1, w2)
        rt.fill(of1.prod(), A)
        rt.fill(of2.prod(), B)
    with pytest.raises(ValueError, match="Multiple workers cannot share the same tile"):
        Program(NPU2(), rt).resolve_program()


def test_buffer_cannot_be_shared_across_workers():
    """A Buffer passed to two Workers must error on the second assignment."""
    n_ty = np.ndarray[(1024,), np.dtype[np.int32]]
    buf_ty = np.ndarray[(16,), np.dtype[np.int32]]
    buf = Buffer(type=buf_ty, name="shared_buf")
    of1 = ObjectFifo(n_ty, name="buf_of1")
    _ = Worker(None, [of1.cons(), buf])
    of2 = ObjectFifo(n_ty, name="buf_of2")
    with pytest.raises(ValueError, match="already placed on"):
        Worker(None, [of2.cons(), buf])


def test_forward_shares_link_tile():
    """forward() must link both ObjectFifos through the same MemTile logical tile."""
    n_ty = np.ndarray[(1024,), np.dtype[np.int32]]
    of_in = ObjectFifo(n_ty, name="fwd_in")
    cons = of_in.cons()
    of_out = cons.forward(name="fwd_out")

    # The consumer of of_in and the producer of of_out should share the same
    # ObjectFifoLink endpoint, which holds a single MemTile.
    link = cons.endpoint
    assert isinstance(link, ObjectFifoLink)
    assert link is of_out.prod().endpoint
    assert link.tile.tile_type == AIETileType.MemTile

    # Both sides of the link resolve to the same Python Tile object.
    assert cons.endpoint.tile is of_out.prod().endpoint.tile


def test_fill_conflicting_tiles_errors():
    """Calling fill() twice on the same handle with different tile coordinates must error."""
    n_ty = np.ndarray[(1024,), np.dtype[np.int32]]
    of = ObjectFifo(n_ty, name="conflict_of")
    prod = of.prod()
    rt = Runtime()
    with rt.sequence(n_ty, n_ty) as (A, _):
        rt.start(Worker(None, [of.cons()]))
        rt.fill(prod, A, tile=Tile(0, 0))
        with pytest.raises(ValueError, match="Endpoint already set"):
            rt.fill(prod, A, tile=Tile(1, 0))


def test_fill_same_tile_allowed():
    """Calling fill() twice on the same handle with the same coordinates is allowed (tiling loop pattern)."""
    n_ty = np.ndarray[(1024,), np.dtype[np.int32]]
    of = ObjectFifo(n_ty, name="same_of")
    prod = of.prod()
    rt = Runtime()
    with rt.sequence(n_ty, n_ty) as (A, _):
        rt.start(Worker(None, [of.cons()]))
        rt.fill(prod, A, tile=Tile(0, 0))
        rt.fill(prod, A, tile=Tile(0, 0))  # same coords — no error


def test_fill_unplaced_tile_allowed():
    """Calling fill() twice with default (unplaced) tiles is allowed (tiling loop pattern)."""
    n_ty = np.ndarray[(1024,), np.dtype[np.int32]]
    of = ObjectFifo(n_ty, name="unplaced_of")
    prod = of.prod()
    rt = Runtime()
    with rt.sequence(n_ty, n_ty) as (A, _):
        rt.start(Worker(None, [of.cons()]))
        rt.fill(prod, A)
        rt.fill(prod, A)  # both default AnyShimTile — no error
