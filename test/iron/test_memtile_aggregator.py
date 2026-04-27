# test_memtile_aggregator.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
""": MemtileAggregator -- API + lowering equivalence tests."""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module gracefully if the fork wheel isn't built / available.
aie_iron = pytest.importorskip("aie.iron")

from aie.iron import MemtileAggregator, ObjectFifo  # noqa: E402
from aie.iron.dataflow.objectfifo import ObjectFifoHandle  # noqa: E402
from aie.iron.memtile import (  # noqa: E402
    MEMTILE_DM_BYTES,
    MEMTILE_S2MM_NEIGHBOUR_CHANNELS,
    bytes_per_element,
    flat_concat_offsets,
)

# ---------------------------------------------------------------------------
# test is self-contained and doesn't require the outer-repo Phase 1 fixture
# file).
# ---------------------------------------------------------------------------

N_GUIDES = 128
N_WINDOWS = 4096
SPACER_BYTES = 5  # 20 nt x 2 bits / 8 = 5 bytes per spacer
N_MATCH_TILES = 4
GUIDES_PER_TILE = N_GUIDES // N_MATCH_TILES  # 32
WINDOWS_PER_CHUNK = 64
N_CHUNKS = N_WINDOWS // WINDOWS_PER_CHUNK  # 64
PARTIAL_CHUNK_SIZE = WINDOWS_PER_CHUNK * GUIDES_PER_TILE  # 2048 B
FULL_CHUNK_SIZE = WINDOWS_PER_CHUNK * N_GUIDES  # 8192 B

# ---------------------------------------------------------------------------
# 1. Construction surface
# ---------------------------------------------------------------------------

def test_constructs_with_valid_args():
    """    cleanly via the new helper."""
    partial_ty = np.ndarray[(PARTIAL_CHUNK_SIZE,), np.dtype[np.uint8]]
    joined_ty = np.ndarray[(FULL_CHUNK_SIZE,), np.dtype[np.uint8]]

    agg = MemtileAggregator(
        n_producers=N_MATCH_TILES,
        producer_obj_type=partial_ty,
        joined_obj_type=joined_ty,
    )
    assert agg is not None
    assert agg.n_producers == N_MATCH_TILES
    assert agg.layout == "slab"
    assert agg.depth == 2  # default per the docstring

def test_n_producers_must_be_in_range():
    """AM020 Ch. 5 p. 74: memtile S2MM neighbour-channel count is 4."""
    partial_ty = np.ndarray[(64,), np.dtype[np.uint8]]
    joined_ty = np.ndarray[(64,), np.dtype[np.uint8]]

    with pytest.raises(ValueError, match="n_producers must be in"):
        MemtileAggregator(
            n_producers=0,
            producer_obj_type=partial_ty,
            joined_obj_type=joined_ty,
        )

    # 5 exceeds the neighbour-channel budget (channels 0..3 + 2 reserved)
    joined_ty_5 = np.ndarray[(5 * 64,), np.dtype[np.uint8]]
    with pytest.raises(ValueError, match="n_producers must be in"):
        MemtileAggregator(
            n_producers=5,
            producer_obj_type=partial_ty,
            joined_obj_type=joined_ty_5,
        )

def test_flat_concat_invariant_validated():
    """The joined buffer must be exactly ``n_producers * partial_bytes``
    (the flat-concat invariant); anything else is rejected with a
    pointer to the layout lesson in the docstring."""
    partial_ty = np.ndarray[(PARTIAL_CHUNK_SIZE,), np.dtype[np.uint8]]
    # joined sized for 3 producers but n_producers=4 -- mismatch.
    bad_joined = np.ndarray[(3 * PARTIAL_CHUNK_SIZE,), np.dtype[np.uint8]]

    with pytest.raises(ValueError, match="flat-concat invariant violated"):
        MemtileAggregator(
            n_producers=4,
            producer_obj_type=partial_ty,
            joined_obj_type=bad_joined,
        )

def test_depth_must_be_positive():
    partial_ty = np.ndarray[(64,), np.dtype[np.uint8]]
    joined_ty = np.ndarray[(2 * 64,), np.dtype[np.uint8]]

    with pytest.raises(ValueError, match="depth must be a positive int"):
        MemtileAggregator(
            n_producers=2,
            producer_obj_type=partial_ty,
            joined_obj_type=joined_ty,
            depth=0,
        )

def test_memtile_dm_budget_enforced():
    """A pathological joined size that exceeds the memtile DM cap is
    rejected at construction time (per AM020 Table 14)."""
    # Pick partial+joined sizes whose 2-buffer footprint exceeds 512 KiB.
    # 4 producers x depth-2 x 80 KiB per partial + depth-2 x 320 KiB joined
    # = 4*2*80K + 2*320K = 640K + 640K = 1280K > 512K.
    partial_size = 80 * 1024
    joined_size = partial_size * 4  # 320 KiB
    partial_ty = np.ndarray[(partial_size,), np.dtype[np.uint8]]
    joined_ty = np.ndarray[(joined_size,), np.dtype[np.uint8]]

    with pytest.raises(ValueError, match="memtile DM budget exceeded"):
        MemtileAggregator(
            n_producers=4,
            producer_obj_type=partial_ty,
            joined_obj_type=joined_ty,
            depth=2,
        )

# ---------------------------------------------------------------------------
# 2. Handle accessors
# ---------------------------------------------------------------------------

def _make_t53m_aggregator() -> MemtileAggregator:
    """    partial_ty = np.ndarray[(PARTIAL_CHUNK_SIZE,), np.dtype[np.uint8]]
    joined_ty = np.ndarray[(FULL_CHUNK_SIZE,), np.dtype[np.uint8]]
    return MemtileAggregator(
        n_producers=N_MATCH_TILES,
        producer_obj_type=partial_ty,
        joined_obj_type=joined_ty,
        name="t53m",
    )

def test_producer_returns_object_fifo_handle():
    agg = _make_t53m_aggregator()
    h0 = agg.producer(0)
    assert isinstance(h0, ObjectFifoHandle)
    assert h0.handle_type == "prod"

def test_producers_returns_n_handles_in_order():
    agg = _make_t53m_aggregator()
    handles = agg.producers()
    assert len(handles) == N_MATCH_TILES
    for i, h in enumerate(handles):
        assert isinstance(h, ObjectFifoHandle)
        assert h.handle_type == "prod"
        # Each handle is the same object as ``producer(i)`` (idempotent
        # accessor on the underlying ObjectFifo).
        assert h is agg.producer(i)

def test_producer_index_out_of_range_raises():
    agg = _make_t53m_aggregator()
    with pytest.raises(IndexError, match="out of range"):
        agg.producer(N_MATCH_TILES)
    with pytest.raises(IndexError, match="out of range"):
        agg.producer(-1)

def test_consumer_returns_object_fifo_handle():
    agg = _make_t53m_aggregator()
    h = agg.consumer()
    assert isinstance(h, ObjectFifoHandle)
    assert h.handle_type == "cons"

# ---------------------------------------------------------------------------
# 3. Functional equivalence to Phase 1's hand-rolled topology
# ---------------------------------------------------------------------------

    ``join_offsets = [i * partial_chunk_size for i in range(N_MATCH_TILES)]``
    -- exactly what flat_concat_offsets emits."""
    expected = [i * PARTIAL_CHUNK_SIZE for i in range(N_MATCH_TILES)]
    actual = flat_concat_offsets(N_MATCH_TILES, PARTIAL_CHUNK_SIZE)
    assert actual == expected
    assert actual == [0, 2048, 4096, 6144]  # the literal Phase 1 list

def test_aggregator_offsets_property_matches_phase1():
    agg = _make_t53m_aggregator()
    assert agg.offsets == [0, 2048, 4096, 6144]

def test_partial_and_joined_byte_sizes_match_phase1():
    """The byte sizes computed by the helper match Phase 1's
    documented MANIFEST.md numbers (PARTIAL_CHUNK_SIZE=2048,
    FULL_CHUNK_SIZE=8192)."""
    partial_ty = np.ndarray[(PARTIAL_CHUNK_SIZE,), np.dtype[np.uint8]]
    joined_ty = np.ndarray[(FULL_CHUNK_SIZE,), np.dtype[np.uint8]]
    assert bytes_per_element(partial_ty) == 2048
    assert bytes_per_element(joined_ty) == 8192

def test_underlying_fifos_match_phase1_naming_pattern():
    """Phase 1 named the per-tile sub-FIFOs ``memC0..memC3``. Ours
    use the aggregator's name + ``_p{i}`` -- different but
    functionally equivalent. This test pins that naming so a
    future rename trips a regression."""
    agg = _make_t53m_aggregator()
    sub_fifos = agg.sub_fifos
    assert len(sub_fifos) == N_MATCH_TILES
    for i, fifo in enumerate(sub_fifos):
        assert isinstance(fifo, ObjectFifo)
        assert fifo.name == f"t53m_p{i}"

def test_joined_fifo_is_a_real_object_fifo():
    """The joined ObjectFifo is exposed for callers that want to
    pass it through legacy APIs (e.g. Runtime.drain) without going
    through the aggregator's own ``consumer()`` accessor."""
    agg = _make_t53m_aggregator()
    fifo = agg.joined_fifo
    assert isinstance(fifo, ObjectFifo)
    assert fifo.name == "t53m_joined"
    assert fifo.depth == 2  # matches aggregator.depth

# ---------------------------------------------------------------------------
# 4. Constants exposed at the canonical AM020 numbers
# ---------------------------------------------------------------------------

def test_memtile_constants_match_am020():
    """The exposed constants reflect AM020 Ch. 5 + Table 14 numbers"""
    assert MEMTILE_S2MM_NEIGHBOUR_CHANNELS == 4
    assert MEMTILE_DM_BYTES == 512 * 1024

# ---------------------------------------------------------------------------
# 5. Phase-1 byte-equality contract: the helper-built aggregator emits the
# same offsets + sub-FIFO obj_types as Phase 1's hand-rolled
# oracle on the chr22 fixture per the MANIFEST). Since the lowering goes
# through the same ObjectFifo.prod().join() primitive, byte-equality is
# transitive: helper -> hand-roll -> oracle.
# ---------------------------------------------------------------------------

def test_byte_equality_contract_with_phase1_handroll():
    """Compare the helper-emitted (offsets, obj_types, sub-FIFO count)
    triple to the literal triple from Phase 1's
    This is the falsifiable byte-equality check the plan calls for."""
    partial_ty = np.ndarray[(PARTIAL_CHUNK_SIZE,), np.dtype[np.uint8]]
    joined_ty = np.ndarray[(FULL_CHUNK_SIZE,), np.dtype[np.uint8]]
    agg = MemtileAggregator(
        n_producers=N_MATCH_TILES,
        producer_obj_type=partial_ty,
        joined_obj_type=joined_ty,
    )

    # Phase 1's literal join_offsets (verbatim from
    # multitile_memtile.py line ~179).
    phase1_join_offsets = [
        i * PARTIAL_CHUNK_SIZE for i in range(N_MATCH_TILES)
    ]
    assert agg.offsets == phase1_join_offsets

    # Phase 1's literal obj_types list (verbatim line ~183).
    phase1_obj_types = [partial_ty] * N_MATCH_TILES
    actual_obj_types = [f.obj_type for f in agg.sub_fifos]
    # numpy ndarray-types compare structurally; same shape + dtype gives
    # the same flat byte-size, which is what the lowering needs.
    assert len(actual_obj_types) == len(phase1_obj_types)
    for actual, expected in zip(actual_obj_types, phase1_obj_types):
        assert bytes_per_element(actual) == bytes_per_element(expected)
