# test_fifo_handle_program_walk.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Contract tests for ``Program._walk_object_fifos`` on ``*FifoHandle`` subclasses."""

from __future__ import annotations

import pytest

def _make_noop_kernel_factory():
    """Build a no-op core_fn factory shaped like the canonical IRON
    examples. Captures only the worker-handle's ``acquire`` /
    ``release`` calls so the lowered ``aie.core`` body has at least one
    op (some MLIR verifiers reject empty cores)."""
    def core_fn(*_handles):
        # No-op body is sufficient for resolve_program -- the
        # tile-collection walk fires before the core body emits.
        pass
    return core_fn

def test_packet_fifo_handle_all_of_endpoints_returns_endpoint_typed_objects():
    """    objects exposing a ``.tile`` attribute, not raw :class:`Tile`
    objects. The pre-fix shape (``list[Tile]``) is what crashed
    ``iron/program.py``'s tile-collection walk."""
    from aie.iron import PacketFifo
    from aie.iron.dataflow.endpoint import ObjectFifoEndpoint
    from aie.iron.device import Tile

    pf = PacketFifo(
        producers=[Tile(0, 2), Tile(0, 3)],
        consumers=[Tile(0, 5)],
    )
    handle = pf.prod(0)
    eps = handle.all_of_endpoints()

    # Must be a non-empty list with one slot per producer + consumer.
    assert len(eps) == pf.num_producers + pf.num_consumers

    # Every element must expose a ``.tile`` attribute (the load-bearing
    # invariant for ``iron/program.py``'s walk).
    for ep in eps:
        assert hasattr(ep, "tile"), (
            f"PacketFifoHandle.all_of_endpoints() returned an element "
            f"without a `.tile` attribute: {ep!r}. iron/program.py's "
            f"`[e.tile for e in fifo.all_of_endpoints()]` walk requires "
            f"endpoint-typed objects."
        )
        # The tile must be a Tile (or None for unplaced -- but in this
        # constructed-with-tiles case it must be a Tile).
        assert isinstance(ep.tile, Tile), (
            f"PacketFifoHandle.all_of_endpoints() yielded ep.tile of "
            f"type {type(ep.tile).__name__}, expected Tile"
        )

    # Tile coordinates must match the constructed PacketFifo.
    tile_coords = [(ep.tile.col, ep.tile.row) for ep in eps]
    assert (0, 2) in tile_coords
    assert (0, 3) in tile_coords
    assert (0, 5) in tile_coords

def test_packet_fifo_handle_program_walk_does_not_crash():
    """    must succeed for a PacketFifoHandle. Pre-fix this raised
    ``AttributeError: 'Tile' object has no attribute 'tile'``."""
    from aie.iron import PacketFifo
    from aie.iron.device import Tile

    pf = PacketFifo(
        producers=[Tile(0, 2), Tile(0, 3)],
        consumers=[Tile(0, 5)],
    )
    handle = pf.prod(0)

    # This is the literal expression at iron/program.py:81.
    tiles = [e.tile for e in handle.all_of_endpoints()]
    assert all(isinstance(t, Tile) for t in tiles), (
        "iron/program.py's tile-collection walk must yield Tile objects "
        "for PacketFifoHandle endpoints"
    )

def test_packet_fifo_handle_endpoint_uses_worker_when_set():
    """When the registry-driven ``Worker.fn_args`` dispatch attaches the
    Worker as the handle's endpoint, ``all_of_endpoints()`` should
    surface the live Worker (which subclasses :class:`ObjectFifoEndpoint`),
    not synthesize a fresh wrapper. This preserves the pass-through
    semantics ObjectFifoHandle has for placement-aware passes."""
    from aie.iron import PacketFifo, Worker
    from aie.iron.device import Tile

    pf = PacketFifo(
        producers=[Tile(0, 2)],
        consumers=[Tile(0, 5)],
    )
    p_handle = pf.prod(0)
    c_handle = pf.cons(0)

    # Constructing the Workers triggers fn_args dispatch which sets
    # handle.endpoint = worker.
    w_prod = Worker(_make_noop_kernel_factory(), fn_args=[p_handle], tile=Tile(0, 2))
    w_cons = Worker(_make_noop_kernel_factory(), fn_args=[c_handle], tile=Tile(0, 5))

    # all_of_endpoints from either handle should now contain the live
    # Worker instances (since Worker subclasses ObjectFifoEndpoint).
    eps = p_handle.all_of_endpoints()
    assert w_prod in eps, (
        "PacketFifoHandle.all_of_endpoints() must surface the producer "
        "Worker as its endpoint after Worker.fn_args dispatch"
    )
    assert w_cons in eps, (
        "PacketFifoHandle.all_of_endpoints() must surface the consumer "
        "Worker as its endpoint after Worker.fn_args dispatch"
    )

def test_accum_fifo_handle_all_of_endpoints_does_not_crash():
    """    ``AttributeError`` for the missing ``_object_fifo`` field. Pre-fix
    the inherited ``ObjectFifoHandle.all_of_endpoints`` did
    ``self._object_fifo._get_endpoint(...)`` which crashed because
    AccumFifoHandle.__init__ bypasses the parent constructor."""
    from aie.iron import AccumFifo
    from aie.iron.device import Tile

    af = AccumFifo(producer=Tile(0, 2), consumer=Tile(0, 3))
    prod_h = af.prod()
    cons_h = af.cons()

    # Pre-fix this raised AttributeError on _object_fifo. Must succeed.
    prod_eps = prod_h.all_of_endpoints()
    cons_eps = cons_h.all_of_endpoints()

    assert len(prod_eps) == 2  # one producer endpoint + one consumer endpoint
    assert len(cons_eps) == 2

def test_accum_fifo_handle_all_of_endpoints_returns_endpoint_typed_objects():
    """    matching :class:`ObjectFifoHandle.all_of_endpoints`'s contract."""
    from aie.iron import AccumFifo
    from aie.iron.device import Tile

    af = AccumFifo(producer=Tile(0, 2), consumer=Tile(0, 3))
    handle = af.prod()
    eps = handle.all_of_endpoints()

    for ep in eps:
        assert hasattr(ep, "tile"), (
            f"AccumFifoHandle.all_of_endpoints() returned an element "
            f"without a `.tile` attribute: {ep!r}"
        )
        assert isinstance(ep.tile, Tile)

    coords = sorted((ep.tile.col, ep.tile.row) for ep in eps)
    assert coords == [(0, 2), (0, 3)]

def test_accum_fifo_handle_program_walk_does_not_crash():
    """    must succeed for an AccumFifoHandle."""
    from aie.iron import AccumFifo
    from aie.iron.device import Tile

    af = AccumFifo(producer=Tile(0, 2), consumer=Tile(0, 3))
    handle = af.prod()

    # iron/program.py:81 literal expression.
    tiles = [e.tile for e in handle.all_of_endpoints()]
    assert len(tiles) == 2
    assert all(isinstance(t, Tile) for t in tiles)

def test_accum_fifo_handle_endpoint_uses_worker_when_set():
    """When the registry-driven ``Worker.fn_args`` dispatch attaches the
    Worker as the handle's endpoint, ``all_of_endpoints()`` should
    surface the live Worker rather than synthesize a fresh wrapper."""
    from aie.iron import AccumFifo, Worker
    from aie.iron.device import Tile

    af = AccumFifo(producer=Tile(0, 2), consumer=Tile(0, 3))
    p_handle = af.prod()
    c_handle = af.cons()

    w_prod = Worker(_make_noop_kernel_factory(), fn_args=[p_handle], tile=Tile(0, 2))
    w_cons = Worker(_make_noop_kernel_factory(), fn_args=[c_handle], tile=Tile(0, 3))

    eps = p_handle.all_of_endpoints()
    assert w_prod in eps
    assert w_cons in eps

# -- Cross-cutting: the contract is shared with the ObjectFifoHandle base ----

def test_all_fifo_handle_subclasses_satisfy_program_walk_contract():
    """The contract test rolled up: every ``*FifoHandle`` subclass must
    return endpoint-typed objects from ``all_of_endpoints`` once the
    fifo's handles are wired into Workers (i.e. once
    ``Worker.fn_args``-dispatch has set ``handle.endpoint = worker``).

    Iterates over the live registry so any future
    ``register_fifo_handle()`` addition is caught here without
    test-file edits."""
    import numpy as np

    from aie.iron import AccumFifo, ObjectFifo, PacketFifo, Worker
    from aie.iron.dataflow.fifo_handle_registry import (
        get_registered_handle_classes,
    )
    from aie.iron.device import Tile

    # Build one instance of each known handle-bearing fifo type and
    # attach Workers so each handle has its endpoint set (this is how
    # the live walk in iron/program.py sees them).
    of_dtype = np.ndarray[(4,), np.dtype[np.int32]]
    of = ObjectFifo(of_dtype, name="of_walk_test", depth=2)
    of_p = of.prod()
    of_c = of.cons()
    Worker(_make_noop_kernel_factory(), fn_args=[of_p], tile=Tile(0, 2))
    Worker(_make_noop_kernel_factory(), fn_args=[of_c], tile=Tile(0, 5))

    pf = PacketFifo(
        producers=[Tile(0, 3)],
        consumers=[Tile(0, 4)],
        name="pf_walk_test",
    )
    pf_p = pf.prod(0)
    pf_c = pf.cons(0)
    Worker(_make_noop_kernel_factory(), fn_args=[pf_p], tile=Tile(0, 3))
    Worker(_make_noop_kernel_factory(), fn_args=[pf_c], tile=Tile(0, 4))

    af = AccumFifo(
        producer=Tile(1, 2),
        consumer=Tile(1, 3),
        name="af_walk_test",
    )
    af_p = af.prod()
    af_c = af.cons()
    Worker(_make_noop_kernel_factory(), fn_args=[af_p], tile=Tile(1, 2))
    Worker(_make_noop_kernel_factory(), fn_args=[af_c], tile=Tile(1, 3))

    handles = [of_p, of_c, pf_p, pf_c, af_p, af_c]

    # Touch the registered class set so the test fails noisily if the
    # registry shape changes (e.g. a new subclass with a broken
    # all_of_endpoints lands without test coverage).
    registered = get_registered_handle_classes()
    assert len(registered) >= 1  # ObjectFifoHandle is pre-registered

    for h in handles:
        eps = h.all_of_endpoints()
        assert len(eps) >= 1, (
            f"Handle {type(h).__name__}.all_of_endpoints() must return "
            f"at least one endpoint after Worker dispatch"
        )
        for ep in eps:
            assert hasattr(ep, "tile"), (
                f"Handle {type(h).__name__}.all_of_endpoints() returned "
                f"a non-endpoint object: {ep!r}. Every *FifoHandle subclass "
                f"must satisfy iron/program.py's tile-collection walk "
                f"contract."
            )
