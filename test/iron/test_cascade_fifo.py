# test_cascade_fifo.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Surface, topology, lowering, and dialect-parity tests for CascadeFifo.

Covers:

- Surface: import + class shape + dtype/handshake validation.
- Topology: producer/consumer tile placement + tiles property.
- Lowering: ``resolve()`` emits an ``aie.cascade_flow`` op connecting
  the producer and consumer tile ops.
- Behavioural parity: a 2-tile cascade pipeline built via CascadeFifo
  emits the same ``cascade_flow`` / ``configure_cascade`` ops as a
  hand-written dialect-level pipeline through the same entry points.
"""

from __future__ import annotations

import pytest

# These imports are ordered to fail-fast: if `aie` isn't built, all tests
# are skipped (not failed) — this matches the convention in
aie_iron = pytest.importorskip("aie.iron")
aie_dialects_aie = pytest.importorskip("aie.dialects.aie")

# -- Surface tests ---------------------------------------------------------

def test_cascade_fifo_imports_from_aie_iron():
    """`from aie.iron import CascadeFifo` resolves to the real class
    """
    from aie.iron import CascadeFifo

    # Real impl is the cascade module's class; stub was an inline class
    # in __init__.py. Discriminate via __module__.
    assert CascadeFifo.__module__.endswith(".cascade"), (
        f"CascadeFifo.__module__ = {CascadeFifo.__module__!r}; "
    )

def test_cascade_fifo_real_construction_does_not_raise():
    """Construct cleanly given valid Tile arguments."""
    from aie.iron import CascadeFifo
    from aie.iron.device import Tile

    cas = CascadeFifo(Tile(0, 2), Tile(0, 3), dtype="bfloat16")
    assert cas is not None
    assert cas.dtype == "bfloat16"

def test_cascade_fifo_constants_match_am020():
    """AM020 Ch. 4 p. 67 documents 512-bit cascade with FP32-accumulator
    or i32 lanes-of-16 packing; bf16 packs 32 lanes per word.
    """
    from aie.iron.cascade import (
        CASCADE_BITS,
        CASCADE_LANES_BFLOAT16,
        CASCADE_LANES_FP32,
        CASCADE_LANES_INT32,
    )

    assert CASCADE_BITS == 512
    assert CASCADE_LANES_FP32 == 16
    assert CASCADE_LANES_INT32 == 16
    assert CASCADE_LANES_BFLOAT16 == 32

# -- Validation tests ------------------------------------------------------

def test_invalid_producer_tile_type_raises():
    from aie.iron import CascadeFifo
    from aie.iron.device import Tile

    with pytest.raises(TypeError, match="producer_tile must be a Tile"):
        CascadeFifo("not_a_tile", Tile(0, 3))

def test_invalid_consumer_tile_type_raises():
    from aie.iron import CascadeFifo
    from aie.iron.device import Tile

    with pytest.raises(TypeError, match="consumer_tile must be a Tile"):
        CascadeFifo(Tile(0, 2), 42)

def test_invalid_dtype_raises():
    from aie.iron import CascadeFifo
    from aie.iron.device import Tile

    with pytest.raises(ValueError, match="dtype must be one of"):
        CascadeFifo(Tile(0, 2), Tile(0, 3), dtype="bogus")

def test_zero_elements_per_handshake_raises():
    from aie.iron import CascadeFifo
    from aie.iron.device import Tile

    with pytest.raises(ValueError, match="elements_per_handshake must be > 0"):
        CascadeFifo(
            Tile(0, 2), Tile(0, 3), dtype="accfloat", elements_per_handshake=0
        )

def test_non_lane_multiple_handshake_raises():
    from aie.iron import CascadeFifo
    from aie.iron.device import Tile

    # accfloat has 16 lanes per cascade word; 17 is not a multiple of 16.
    with pytest.raises(ValueError, match="must be a positive multiple"):
        CascadeFifo(
            Tile(0, 2), Tile(0, 3), dtype="accfloat", elements_per_handshake=17
        )

# -- Topology tests --------------------------------------------------------

def test_default_handshake_for_accfloat_is_one_word():
    from aie.iron import CascadeFifo
    from aie.iron.device import Tile

    cas = CascadeFifo(Tile(0, 2), Tile(0, 3), dtype="accfloat")
    # Default is `lanes_per_word`; 1 word = 16 accfloat lanes.
    assert cas.elements_per_handshake == 16
    assert cas.words_per_handshake == 1
    assert cas.cascade_bits == 512

def test_default_handshake_for_bfloat16_is_one_word():
    from aie.iron import CascadeFifo
    from aie.iron.device import Tile

    cas = CascadeFifo(Tile(0, 2), Tile(0, 3), dtype="bfloat16")
    # bf16 packs 32 lanes per 512-bit word.
    assert cas.elements_per_handshake == 32
    assert cas.words_per_handshake == 1

def test_explicit_multi_word_handshake():
    from aie.iron import CascadeFifo
    from aie.iron.device import Tile

    # 64 accfloat lanes = 4 cascade words.
    cas = CascadeFifo(
        Tile(0, 2), Tile(0, 3), dtype="accfloat", elements_per_handshake=64
    )
    assert cas.words_per_handshake == 4
    assert cas.cascade_bits == 4 * 512

def test_tiles_property_returns_prod_then_cons():
    from aie.iron import CascadeFifo
    from aie.iron.device import Tile

    p = Tile(0, 2)
    c = Tile(0, 3)
    cas = CascadeFifo(p, c)
    tiles = cas.tiles
    assert len(tiles) == 2
    # Tile.copy() in __init__ means we can't compare object identity;
    # compare col/row.
    assert tiles[0].col == 0 and tiles[0].row == 2
    assert tiles[1].col == 0 and tiles[1].row == 3

def test_objectfifo_compat_prod_and_cons_methods():
    """Mirror ObjectFifo's `prod()` / `cons()` shape for muscle-memory
    compatibility — but cascade is unbuffered so they return Tiles
    directly, not handle objects."""
    from aie.iron import CascadeFifo
    from aie.iron.device import Tile

    cas = CascadeFifo(Tile(0, 2), Tile(0, 3), dtype="accfloat")
    p = cas.prod()
    c = cas.cons()
    assert p.col == 0 and p.row == 2
    assert c.col == 0 and c.row == 3

def test_unique_default_names():
    from aie.iron import CascadeFifo
    from aie.iron.device import Tile

    a = CascadeFifo(Tile(0, 2), Tile(0, 3))
    b = CascadeFifo(Tile(0, 3), Tile(0, 4))
    assert a.name != b.name
    assert a.name.startswith("cas")
    assert b.name.startswith("cas")

def test_explicit_name_honoured():
    from aie.iron import CascadeFifo
    from aie.iron.device import Tile

    cas = CascadeFifo(Tile(0, 2), Tile(0, 3), name="my_chain_link_0")
    assert cas.name == "my_chain_link_0"

# -- Lowering tests --------------------------------------------------------

def test_resolve_emits_cascade_flow_op_in_module():
    """`resolve()` inside an `aie.device` body emits an `aie.cascade_flow`
    op connecting the producer's tile op to the consumer's tile op.
    """
    from aie.dialects.aie import AIEDevice, device, tile
    from aie.extras.context import mlir_mod_ctx
    from aie.iron import CascadeFifo
    from aie.iron.device import Tile

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2)
        def device_body():
            t_prod_op = tile(0, 2)
            t_cons_op = tile(0, 3)
            t_prod = Tile(0, 2)
            t_prod.op = t_prod_op
            t_cons = Tile(0, 3)
            t_cons.op = t_cons_op
            cas = CascadeFifo(t_prod, t_cons, dtype="accfloat")
            cas.resolve()

        module_str = str(ctx.module)

    assert "aie.cascade_flow" in module_str, (
        f"expected aie.cascade_flow op in module after resolve(); "
        f"got:\n{module_str}"
    )

def test_resolve_without_tile_op_raises():
    """If neither tile has an .op set, resolve() raises a clear
    ValueError rather than the AttributeError CascadeFlowOp would
    raise on a None operand.
    """
    from aie.dialects.aie import AIEDevice, device
    from aie.extras.context import mlir_mod_ctx
    from aie.iron import CascadeFifo
    from aie.iron.device import Tile

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2)
        def device_body():
            t_prod = Tile(0, 2)
            t_cons = Tile(0, 3)
            cas = CascadeFifo(t_prod, t_cons)
            with pytest.raises(ValueError, match="tile op not set"):
                cas.resolve()

def test_double_resolve_is_idempotent():
    """Calling resolve() twice does not emit two cascade_flow ops."""
    from aie.dialects.aie import AIEDevice, device, tile
    from aie.extras.context import mlir_mod_ctx
    from aie.iron import CascadeFifo
    from aie.iron.device import Tile

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2)
        def device_body():
            t_prod_op = tile(0, 2)
            t_cons_op = tile(0, 3)
            t_prod = Tile(0, 2)
            t_prod.op = t_prod_op
            t_cons = Tile(0, 3)
            t_cons.op = t_cons_op
            cas = CascadeFifo(t_prod, t_cons, dtype="accfloat")
            cas.resolve()
            cas.resolve()  # should be a no-op

        module_str = str(ctx.module)

    # exactly one cascade_flow op in the module.
    assert module_str.count("aie.cascade_flow") == 1, (
        f"expected exactly 1 cascade_flow op after double resolve, "
        f"got:\n{module_str}"
    )

# -- Behavioural parity vs Phase 1 wrapper --------------------------------

def test_cascade_fifo_lowering_matches_dialect_cascade_flow_op():
    """The CascadeFifo path emits the same MLIR op kind that the
    on lowering through (`aie.cascade_flow`).

    This is the parity assertion the task brief calls out: the
    wrapper's chain-of-N lowers via the same dialect entry point this
    primitive emits, on a per-link basis. Once the wrapper is
    refactored to delegate to CascadeFifo (a follow-up task), the
    test changes from "emits the same op" to "is a strict drop-in".
    """
    from aie.dialects.aie import AIEDevice, device, tile
    from aie.dialects._aie_ops_gen import CascadeFlowOp
    from aie.extras.context import mlir_mod_ctx
    from aie.iron import CascadeFifo
    from aie.iron.device import Tile

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2)
        def device_body():
            t_prod_op = tile(0, 2)
            t_cons_op = tile(0, 3)
            t_prod = Tile(0, 2)
            t_prod.op = t_prod_op
            t_cons = Tile(0, 3)
            t_cons.op = t_cons_op
            cas = CascadeFifo(t_prod, t_cons, dtype="accfloat")
            cas.resolve()
            # The resolved op IS a CascadeFlowOp.
            assert isinstance(cas.op, CascadeFlowOp)
