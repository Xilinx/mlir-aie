# test_sparse_fifo.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Surface, pattern, lowering, and registry tests for IRON SparseFifo.

- Surface: import + class shape + sparsity (N, M) validation.
- Pattern correctness: 2:4 / 1:4 / 1:2 weight tensors satisfy the
  structural N-nonzeros-per-M rule and decompress to a dense
  reference bit-equal (numpy reference for the compression format).
- Lowering: ``resolve()`` emits ``aie.objectfifo`` (the underlying
  ObjectFifo lowering) AND attaches ``aie.compress_mm2s`` /
  ``aie.decompress_s2mm`` / sparsity-pattern attributes that the
  BD-emit pass consumes.
- Registry: SparseFifoHandle is dispatched via the FifoHandle
  registry from ``Worker.fn_args``.

Tests are skipped (not failed) if ``import aie.iron`` fails so the
suite is hermetic against an un-built tree.
"""

from __future__ import annotations

import numpy as np
import pytest

# These imports are ordered to fail-fast: if `aie` isn't built, all tests
# fork tests (test_cascade_fifo.py, test_worker_fifo_handle_extension.py).
aie_iron = pytest.importorskip("aie.iron")
aie_dialects_aie = pytest.importorskip("aie.dialects.aie")

# ---------------------------------------------------------------------------
# class must construct cleanly and report N/M correctly.
# ---------------------------------------------------------------------------

def test_sparse_fifo_imports_from_aie_iron():
    """`from aie.iron import SparseFifo` resolves to the real class
    """
    from aie.iron import SparseFifo

    # Real impl is the sparse module's class; stub was an inline class
    # in __init__.py. Discriminate via __module__.
    assert SparseFifo.__module__.endswith(".sparse"), (
        f"SparseFifo.__module__ = {SparseFifo.__module__!r}; "
    )

def test_sparse_fifo_real_construction_does_not_raise():
    """Construct cleanly given valid arguments. Default 2:4 pattern."""
    from aie.iron import SparseFifo
    from aie.iron.device import Tile

    sf = SparseFifo(
        producer=Tile(0, 0),
        consumer=Tile(0, 2),
        obj_type=np.ndarray[(64, 64), np.dtype(np.int8)],
    )
    assert sf is not None
    assert sf.sparsity_pattern == "N:M"
    assert sf.N == 2
    assert sf.M == 4
    assert sf.compression_ratio == 0.5

def test_sparse_fifo_handle_subclasses_object_fifo_handle():
    """SparseFifoHandle subclasses ObjectFifoHandle so Worker.fn_args
    """
    from aie.iron import SparseFifo
    from aie.iron.dataflow import ObjectFifoHandle
    from aie.iron.device import Tile
    from aie.iron.sparse import SparseFifoHandle

    sf = SparseFifo(
        producer=Tile(0, 0),
        consumer=Tile(0, 2),
        obj_type=np.ndarray[(64, 64), np.dtype(np.int8)],
    )
    h = sf.prod()
    assert isinstance(h, SparseFifoHandle)
    assert isinstance(h, ObjectFifoHandle)

def test_sparse_fifo_constants():
    """Module-level constants match the AM020 spec."""
    from aie.iron import sparse

    assert sparse.SPARSE_CHANNELS_PER_DIRECTION == 2
    assert sparse.SPARSE_ATTR_COMPRESS_MM2S == "aie.compress_mm2s"
    assert sparse.SPARSE_ATTR_DECOMPRESS_S2MM == "aie.decompress_s2mm"
    assert sparse.SPARSE_ATTR_PATTERN == "aie.sparsity_pattern"

# ---------------------------------------------------------------------------
# Validation tests -- (N, M) rules per AM020 + escape hatch.
# ---------------------------------------------------------------------------

def test_validation_rejects_unsupported_pattern_tag():
    """Only ``"N:M"`` is accepted today; ``"block"`` / ``"COO"`` etc.
    raise ValueError so future extensions don't silently degrade.
    """
    from aie.iron import SparseFifo
    from aie.iron.device import Tile

    with pytest.raises(ValueError, match="sparsity_pattern"):
        SparseFifo(
            producer=Tile(0, 0),
            consumer=Tile(0, 2),
            obj_type=np.ndarray[(64, 64), np.dtype(np.int8)],
            sparsity_pattern="block",
            N=2,
            M=4,
        )

def test_validation_rejects_M_lt_2():
    """M==1 is degenerate (no zeros possible); use ObjectFifo instead."""
    from aie.iron import SparseFifo
    from aie.iron.device import Tile

    with pytest.raises(ValueError, match="M must be >= 2"):
        SparseFifo(
            producer=Tile(0, 0),
            consumer=Tile(0, 2),
            obj_type=np.ndarray[(64, 64), np.dtype(np.int8)],
            N=1,
            M=1,
        )

def test_validation_rejects_N_eq_M():
    """N == M means dense; should use ObjectFifo."""
    from aie.iron import SparseFifo
    from aie.iron.device import Tile

    with pytest.raises(ValueError, match="0 < N < M"):
        SparseFifo(
            producer=Tile(0, 0),
            consumer=Tile(0, 2),
            obj_type=np.ndarray[(64, 64), np.dtype(np.int8)],
            N=4,
            M=4,
        )

def test_validation_rejects_N_zero():
    from aie.iron import SparseFifo
    from aie.iron.device import Tile

    with pytest.raises(ValueError, match="0 < N < M"):
        SparseFifo(
            producer=Tile(0, 0),
            consumer=Tile(0, 2),
            obj_type=np.ndarray[(64, 64), np.dtype(np.int8)],
            N=0,
            M=4,
        )

def test_validation_rejects_unverified_pattern_by_default():
    """(3, 8) is not in the AM020-verified set; default must reject."""
    from aie.iron import SparseFifo
    from aie.iron.device import Tile

    with pytest.raises(ValueError, match="not in the AM020-verified set"):
        SparseFifo(
            producer=Tile(0, 0),
            consumer=Tile(0, 2),
            obj_type=np.ndarray[(64, 64), np.dtype(np.int8)],
            N=3,
            M=8,
        )

def test_validation_allows_unverified_with_escape_hatch():
    """``allow_unverified=True`` accepts AIE2P-investigation patterns."""
    from aie.iron import SparseFifo
    from aie.iron.device import Tile

    sf = SparseFifo(
        producer=Tile(0, 0),
        consumer=Tile(0, 2),
        obj_type=np.ndarray[(64, 64), np.dtype(np.int8)],
        N=3,
        M=8,
        allow_unverified=True,
    )
    assert sf.N == 3
    assert sf.M == 8
    assert not sf.is_pattern_am020_verified

def test_non_int_N_M_rejected():
    from aie.iron import SparseFifo
    from aie.iron.device import Tile

    with pytest.raises(TypeError, match="must be int"):
        SparseFifo(
            producer=Tile(0, 0),
            consumer=Tile(0, 2),
            obj_type=np.ndarray[(64, 64), np.dtype(np.int8)],
            N=2.0,  # type: ignore[arg-type]
            M=4,
        )

def test_invalid_producer_type_rejected():
    from aie.iron import SparseFifo
    from aie.iron.device import Tile

    with pytest.raises(TypeError, match="producer must be a Tile"):
        SparseFifo(
            producer="not_a_tile",
            consumer=Tile(0, 2),
            obj_type=np.ndarray[(64, 64), np.dtype(np.int8)],
        )

# ---------------------------------------------------------------------------
# N:M sparse compression-format property tests (numpy reference).
#
# These exercise the structural N:M sparsity rule the on-tile
# decompression hardware assumes — they are NOT tests of SparseFifo's
# lowering or silicon behaviour. Lowering is exercised in the
# "Lowering" section below, and silicon-side correctness is the
# responsibility of integration tests against the BD-emit pass.
# ---------------------------------------------------------------------------

def _make_nm_sparse(dense: np.ndarray, N: int, M: int) -> np.ndarray:
    """Return a copy of ``dense`` zeroed to satisfy N:M structured sparsity.

    For each contiguous group of ``M`` elements along the inner-most
    axis, keep the ``N`` largest-magnitude elements and zero the rest.
    This is the canonical structured-sparsity pruning rule the BD-emit
    pass + on-tile decompression assume.
    """
    if dense.shape[-1] % M != 0:
        raise ValueError(
            f"_make_nm_sparse: inner dim {dense.shape[-1]} must be a "
            f"multiple of M={M}"
        )
    sparse = dense.copy()
    flat_shape = (-1, dense.shape[-1] // M, M)
    flat = sparse.reshape(flat_shape)
    for outer in range(flat.shape[0]):
        for grp in range(flat.shape[1]):
            block = flat[outer, grp]
            # Top-N by absolute value, zero the rest.
            keep_idx = np.argsort(-np.abs(block))[:N]
            mask = np.zeros_like(block, dtype=bool)
            mask[keep_idx] = True
            block[~mask] = 0
            flat[outer, grp] = block
    return sparse

def _count_zeros_per_group(sparse: np.ndarray, M: int) -> np.ndarray:
    """Count zero entries per contiguous group of M along the last axis.

    Returns an integer array of shape ``sparse.shape[:-1] + (sparse.shape[-1] // M,)``.
    """
    flat = sparse.reshape(*sparse.shape[:-1], sparse.shape[-1] // M, M)
    return np.sum(flat == 0, axis=-1)

@pytest.mark.parametrize(
    "N, M",
    [
        (1, 2),  # 50 % sparsity, simplest pattern
        (1, 4),  # 75 % sparsity
    ],
)
def test_synthetic_pattern_has_expected_zero_count(N, M):
    """Each group of M entries has exactly (M - N) zeros after pruning."""
    rng = np.random.default_rng(seed=0xC0FFEE)
    dense = rng.standard_normal((8, 64)).astype(np.float32)
    sparse = _make_nm_sparse(dense, N=N, M=M)
    zeros = _count_zeros_per_group(sparse, M=M)
    expected_zeros = M - N
    assert (zeros == expected_zeros).all(), (
        f"N:M=({N},{M}) sparsity should have exactly {expected_zeros} "
        f"zeros per group of {M}; got max={zeros.max()}, min={zeros.min()}"
    )

@pytest.mark.parametrize("N, M", [(1, 2), (2, 4)])
def test_nm_compression_roundtrip_is_lossless_on_compliant_input(N, M):
    """Property test for the N:M compression format itself (not for
    SparseFifo).

    When the input weight matrix already satisfies the N:M pattern
    (at most M-N zeros per group of M elements along the inner axis),
    a numpy model of compress->decompress preserves the matrix
    bit-equal. This is what the on-tile S2MM decompressor's contract
    will rely on; SparseFifo's resolve() simply attaches the
    compression metadata and the BD-emit pass flips the
    Enable_Compression bit. SparseFifo behaviour proper is exercised
    in the "Lowering" section below.
    """
    rng = np.random.default_rng(seed=0xBEEF)
    W_dense_pruned = _make_nm_sparse(
        rng.standard_normal((32, 64)).astype(np.float32), N=N, M=M
    )
    x = rng.standard_normal((16, 32)).astype(np.float32)

    # Reference: dense matmul on the pruned tensor.
    ref = x @ W_dense_pruned

    # Simulated on-tile decompression: compress -> store nonzero positions
    # only -> decompress by re-injecting zeros at the position-map gaps.
    # If the round-trip matches bit-for-bit, the SparseFifo lowering
    # contract is satisfiable on AIE-ML.
    flat = W_dense_pruned.reshape(-1, W_dense_pruned.shape[-1] // M, M)
    decompressed = np.zeros_like(flat)
    for outer in range(flat.shape[0]):
        for grp in range(flat.shape[1]):
            block = flat[outer, grp]
            nonzero_idx = np.flatnonzero(block != 0)
            assert len(nonzero_idx) <= N, (
                f"group {(outer, grp)} has {len(nonzero_idx)} nonzeros, "
                f"more than N={N}; pruning rule violated"
            )
            # Compressed = (positions, values); decompress via scatter.
            decompressed[outer, grp, nonzero_idx] = block[nonzero_idx]
    decompressed = decompressed.reshape(W_dense_pruned.shape)

    np.testing.assert_array_equal(
        decompressed,
        W_dense_pruned,
        err_msg="N:M decompression round-trip is not bit-equal",
    )

    out = x @ decompressed
    np.testing.assert_array_equal(
        out,
        ref,
        err_msg="matmul on decompressed weights diverges from dense reference",
    )

# ---------------------------------------------------------------------------
# Lowering -- resolve() must emit aie.objectfifo AND attach
# compression / sparsity attributes the BD-emit pass consumes.
# ---------------------------------------------------------------------------

def test_resolve_emits_objectfifo_with_compression_attrs():
    """SparseFifo.resolve() builds the ObjectFifoCreateOp and pins the
    ``aie.compress_mm2s`` / ``aie.decompress_s2mm`` attributes on it.
    """
    from aie.iron import SparseFifo
    from aie.iron.device import Tile
    from aie.iron.sparse import (
        SPARSE_ATTR_COMPRESS_MM2S,
        SPARSE_ATTR_DECOMPRESS_S2MM,
        SPARSE_ATTR_M,
        SPARSE_ATTR_N,
        SPARSE_ATTR_PATTERN,
    )
    from aie.dialects.aie import device, AIEDevice
    from aie import ir

    with ir.Context() as ctx, ir.Location.unknown():
        from aie.dialects.aie import register_dialect  # type: ignore

        register_dialect(ctx)
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):

            @device(AIEDevice.npu1_1col)
            def device_body():
                prod = Tile(0, 0)
                cons = Tile(0, 2)
                # Trigger Tile.resolve to build aie.tile ops.
                prod.resolve()
                cons.resolve()

                sf = SparseFifo(
                    producer=prod,
                    consumer=cons,
                    obj_type=np.ndarray[(64, 64), np.dtype(np.int8)],
                    N=2,
                    M=4,
                    name="t25_sf",
                )
                # Build the producer / consumer endpoints so the
                # underlying ObjectFifo can resolve.
                sf.prod()
                sf.cons()
                # Attach endpoints to placement so the underlying
                # ObjectFifo's _prod_tile_op / _cons_tiles_ops succeed.
                # We don't have a Worker here so wire endpoints directly.
                from aie.iron.dataflow.endpoint import ObjectFifoEndpoint

                sf._prod._endpoint = ObjectFifoEndpoint(prod)
                sf._cons[0]._endpoint = ObjectFifoEndpoint(cons)

                sf.resolve()

                # Inspect the lowered op for the sparsity attrs.
                op = sf.op
                attrs = op.operation.attributes

                names = {a.name for a in attrs}
                assert SPARSE_ATTR_COMPRESS_MM2S in names, (
                    f"compress_mm2s attr missing from lowered op; "
                    f"present attrs = {names}"
                )
                assert SPARSE_ATTR_DECOMPRESS_S2MM in names
                assert SPARSE_ATTR_PATTERN in names
                assert SPARSE_ATTR_N in names
                assert SPARSE_ATTR_M in names

                # Pattern attr is the string "N:M".
                pat_attr = attrs[SPARSE_ATTR_PATTERN]
                assert ir.StringAttr(pat_attr).value == "N:M"
                # N / M attrs carry the integer values.
                assert ir.IntegerAttr(attrs[SPARSE_ATTR_N]).value == 2
                assert ir.IntegerAttr(attrs[SPARSE_ATTR_M]).value == 4

def test_resolve_idempotent_on_double_call():
    """Calling resolve() twice must not double-attach attributes or
    re-emit the underlying ObjectFifoCreateOp.
    """
    from aie.iron import SparseFifo
    from aie.iron.device import Tile
    from aie.dialects.aie import device, AIEDevice
    from aie import ir

    with ir.Context() as ctx, ir.Location.unknown():
        from aie.dialects.aie import register_dialect  # type: ignore

        register_dialect(ctx)
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):

            @device(AIEDevice.npu1_1col)
            def device_body():
                prod = Tile(0, 0)
                cons = Tile(0, 2)
                prod.resolve()
                cons.resolve()

                sf = SparseFifo(
                    producer=prod,
                    consumer=cons,
                    obj_type=np.ndarray[(64, 64), np.dtype(np.int8)],
                    N=1,
                    M=2,
                )
                sf.prod()
                sf.cons()
                from aie.iron.dataflow.endpoint import ObjectFifoEndpoint

                sf._prod._endpoint = ObjectFifoEndpoint(prod)
                sf._cons[0]._endpoint = ObjectFifoEndpoint(cons)
                sf.resolve()
                op_first = sf.op
                sf.resolve()
                op_second = sf.op
                # Same op instance both times (no re-emit).
                assert op_first is op_second

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def test_sparse_fifo_handle_is_registered():
    """SparseFifoHandle is registered with the FifoHandle registry at
    """
    from aie.iron.dataflow.fifo_handle_registry import (
        get_registered_handle_classes,
    )
    from aie.iron.sparse import SparseFifoHandle

    classes = get_registered_handle_classes()
    assert SparseFifoHandle in classes, (
        f"SparseFifoHandle not registered with the FifoHandle registry; "
        f"registered = {[c.__name__ for c in classes]}"
    )

def test_sparse_fifo_handle_dispatches_via_registry():
    """``dispatch_fn_arg`` recognizes a SparseFifoHandle and runs the
    SparseFifo handler (which mirrors ObjectFifoHandle bookkeeping).
    """
    from aie.iron import SparseFifo
    from aie.iron.dataflow.fifo_handle_registry import dispatch_fn_arg
    from aie.iron.device import Tile

    sf = SparseFifo(
        producer=Tile(0, 0),
        consumer=Tile(0, 2),
        obj_type=np.ndarray[(64, 64), np.dtype(np.int8)],
    )
    h = sf.prod()

    class _StubWorker:
        def __init__(self):
            self._fifos = []

    stub = _StubWorker()
    matched = dispatch_fn_arg(h, stub)
    assert matched is True
    assert h in stub._fifos
    assert h.endpoint is stub

# ---------------------------------------------------------------------------
# Diagnostic / introspection surface stability.
# ---------------------------------------------------------------------------

def test_sparse_fifo_handle_diagnostic_properties():
    """SparseFifoHandle exposes (N, M, compression_ratio, sparsity_pattern)
    so downstream debug code can branch on the sparse-channel case
    without re-walking the parent fifo by hand.
    """
    from aie.iron import SparseFifo
    from aie.iron.device import Tile

    sf = SparseFifo(
        producer=Tile(0, 0),
        consumer=Tile(0, 2),
        obj_type=np.ndarray[(64, 64), np.dtype(np.int8)],
        N=2,
        M=4,
    )
    h = sf.prod()
    assert h.N == 2
    assert h.M == 4
    assert h.sparsity_pattern == "N:M"
    assert h.compression_ratio == 0.5
    assert h.sparse_fifo is sf

def test_str_returns_formatted_summary():
    """Defensive smoke test: __str__ should not raise and should
    include the sparsity pattern + N/M for debug-dump readability.
    """
    from aie.iron import SparseFifo
    from aie.iron.device import Tile

    sf = SparseFifo(
        producer=Tile(0, 0),
        consumer=Tile(0, 2),
        obj_type=np.ndarray[(64, 64), np.dtype(np.int8)],
        N=2,
        M=4,
        name="dbg_sf",
    )
    s = str(sf)
    assert "N=2" in s
    assert "M=4" in s
    assert "N:M" in s

def test_module_all_exports():
    """Pin the public surface: ``aie.iron.sparse.__all__`` must include
    SparseFifo + SparseFifoHandle plus the BD attribute name constants
    the BD-emit pass keys off.
    """
    from aie.iron import sparse

    expected = {
        "SparseFifo",
        "SparseFifoHandle",
        "SPARSE_CHANNELS_PER_DIRECTION",
        "SPARSE_ATTR_COMPRESS_MM2S",
        "SPARSE_ATTR_DECOMPRESS_S2MM",
        "SPARSE_ATTR_PATTERN",
        "SPARSE_ATTR_N",
        "SPARSE_ATTR_M",
    }
    actual = set(sparse.__all__)
    missing = expected - actual
    assert not missing, f"public surface drift: missing {missing} from __all__"
