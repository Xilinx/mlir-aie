# cascade.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""CascadeFifo: first-class IRON cascade-stream primitive.

Wraps the AIE-ML/AIE2P **cascade stream** (AM020 Ch. 4 p. 67): a 512-bit,
accumulator-to-accumulator inter-tile transfer between vertically- (or
horizontally-) adjacent CoreTiles. The MLIR ops `aie.put_cascade`,
`aie.get_cascade`, `aie.cascade_flow`, and `aie.configure_cascade` are
already exported by the dialect; this class promotes them to a
first-class IRON primitive that mirrors the surface of
:class:`aie.iron.ObjectFifo` so users can reach for "cascade" the same
way they reach for "object fifo".

A :class:`CascadeFifo` represents the cascade connection between exactly
two CoreTiles: the producer (which calls `put_mcd` / `aie.put_cascade`
inside its core_fn) and the consumer (which calls `get_scd_v16int32` /
`aie.get_cascade`). Multi-stage cascade chains are built by composing
several CascadeFifos head-to-tail. See
:func:`cascade_stream_chain` in `bionpu.iron_extensions.cascade_stream`
for the higher-level chain helper that this primitive supersedes for
canonical use cases.

Architectural reference
-----------------------

- AM020 Ch. 4 p. 67: cascade stream is a 512-bit physical channel
  between adjacent CoreTiles.
- AM020 Appendix A p. 80 Figure 45: vertical+horizontal cascade
  topology on AIE-ML; AIE2P inherits the same routing.
- ``aie.put_cascade`` / ``aie.get_cascade``: the cascade write/read MLIR
  ops; the operand type must match the cascade size (AIE2: i512 or
  vector<16xi32>; AIE2P inherits this).
- ``aie.cascade_flow``: declarative connection between two tiles that
  the placer lowers to per-tile ``aie.configure_cascade`` ops.

Design notes
------------

CascadeFifo is **not** a subclass of ObjectFifo. ObjectFifo lowers to
shared-memory + lock synchronization; cascade lowers to a dedicated
physical wire between adjacent compute tiles. Both share the *user-facing
shape* (producer/consumer endpoints, tile placement, dtype/lanes) so
this class duplicates that shape but resolves through a different MLIR
path. The two-class split keeps `ObjectFifo`'s shared-memory semantics
intact while letting `CascadeFifo` opt into the cascade physical
channel.

The C++ kernel side still owns the actual cascade reads/writes via the
``put_mcd`` / ``get_scd_v16int32`` intrinsics inside the core function
body — see ``aie_kernels/aie2/cascade_mm.cc`` for a reference. The IRON
layer here only emits the placement (tiles + cascade_flow) and the
runtime metadata; the kernel-author is responsible for the per-element
math.

Phase 1's wrapper at ``bionpu/iron_extensions/cascade_stream.py``
remains usable for callers that need the chain-of-N abstraction; the
wrapper's underlying primitive is now this CascadeFifo class.
"""

from __future__ import annotations
from typing import Sequence

import numpy as np

from .. import ir  # type: ignore
from ..dialects._aie_ops_gen import (  # type: ignore
    CascadeFlowOp,
)
from .device import Tile
from .resolvable import Resolvable, NotResolvedError


# Cascade-stream wire width on AIE-ML / AIE2P (per AM020 Ch. 4 p. 67).
# Identical to one accumulator register slice. The canonical kernel-side
# views are ``v16int32`` (16 lanes of int32) and ``v16 accfloat``
# (16 lanes of FP32 accumulator). See aie_api/adf/stream.hpp.
CASCADE_BITS: int = 512
CASCADE_LANES_INT32: int = 16
CASCADE_LANES_FP32: int = 16
CASCADE_LANES_BFLOAT16: int = 32  # bf16 cascade transfers pack 32 lanes per 512-bit word


# Mapping of accepted dtype name -> (numpy dtype if applicable, lanes per cascade word).
# Kept narrow on purpose: only the accumulator-typed cascade variants the
# AIE2P dialect supports today. Adding a new dtype here requires confirming
# the C++ intrinsic exists (put_mcd / get_scd) and the MLIR op accepts the
# vector type at verification time.
_CASCADE_DTYPES: dict[str, tuple[np.dtype | None, int]] = {
    "accfloat": (None, CASCADE_LANES_FP32),  # FP32 accumulator (BM register)
    "acc32": (np.dtype("int32"), CASCADE_LANES_INT32),
    "int32": (np.dtype("int32"), CASCADE_LANES_INT32),
    "i32": (np.dtype("int32"), CASCADE_LANES_INT32),
    "bfloat16": (None, CASCADE_LANES_BFLOAT16),
    "bf16": (None, CASCADE_LANES_BFLOAT16),
}


class CascadeFifo(Resolvable):
    """Cascade-stream FIFO between two adjacent CoreTiles.

    Mirrors :class:`aie.iron.ObjectFifo`'s constructor surface for
    muscle-memory compatibility, but lowers to a cascade physical
    channel (``aie.cascade_flow`` + per-tile ``aie.configure_cascade``)
    rather than shared-memory + locks.

    The C++ kernel running on each tile is responsible for emitting the
    actual ``put_mcd`` / ``get_scd_v16int32`` intrinsics inside its core
    body; this class only declares the producer/consumer endpoints, the
    cascade dtype, and the per-handshake element count so the placer
    knows how to route the cascade wire.

    Args:
        producer_tile: CoreTile that writes the cascade output.
        consumer_tile: CoreTile that reads the cascade input. Must be
            cascade-adjacent to ``producer_tile`` (vertically- or
            horizontally-neighbouring per AM020 Appendix A p. 80).
        dtype: Cascade element dtype, one of ``"accfloat"`` (FP32
            accumulator; the canonical cross-walk path), ``"acc32"`` /
            ``"int32"`` (i32 accumulator), or ``"bfloat16"`` (bf16
            multiplier-input). Defaults to ``"bfloat16"`` matching the
            T2.1 task brief.
        elements_per_handshake: Number of cascade-typed elements
            transferred per cascade handshake. Defaults to the number
            of lanes per 512-bit cascade word for the chosen dtype
            (16 for accfloat/acc32, 32 for bfloat16). Must be a
            positive integer multiple of the per-word lane count.
        name: Optional symbolic name. Auto-generated if omitted.

    Raises:
        TypeError: If a tile argument is not a :class:`Tile`.
        ValueError: If the dtype is not in
            :data:`_CASCADE_DTYPES`, or if
            ``elements_per_handshake`` is non-positive or not a
            multiple of the dtype's lanes-per-word.

    Notes:
        - The producer/consumer terminology mirrors ObjectFifo. The
          underlying cascade ops are unidirectional (put -> get); this
          class does not support multi-consumer broadcast (unlike
          ObjectFifo) because the cascade wire is point-to-point.
        - Tile-adjacency validation is performed by the placer pass
          (``--aie-place-tiles``), not at construction time, because
          unplaced tiles (``Tile()`` without col/row) are common in
          IRON designs.

    Example:

    .. code-block:: python

        from aie.iron import CascadeFifo, Worker, Kernel
        from aie.iron.device import Tile

        prod = Tile(0, 2)
        cons = Tile(0, 3)

        cas = CascadeFifo(prod, cons, dtype="accfloat",
                          elements_per_handshake=16)

        k_put = Kernel("matmul_put_only", "mm.o", [...])
        k_get = Kernel("matmul_get_only", "mm.o", [...])

        w_prod = Worker(producer_core_fn,
                        fn_args=[..., k_put], tile=prod)
        w_cons = Worker(consumer_core_fn,
                        fn_args=[..., k_get], tile=cons)
    """

    # Used to generate unique CascadeFifo names when none is provided.
    __cf_index = 0

    def __init__(
        self,
        producer_tile: Tile,
        consumer_tile: Tile,
        dtype: str = "bfloat16",
        elements_per_handshake: int | None = None,
        name: str | None = None,
    ):
        if not isinstance(producer_tile, Tile):
            raise TypeError(
                f"CascadeFifo: producer_tile must be a Tile, "
                f"got {type(producer_tile).__name__}"
            )
        if not isinstance(consumer_tile, Tile):
            raise TypeError(
                f"CascadeFifo: consumer_tile must be a Tile, "
                f"got {type(consumer_tile).__name__}"
            )

        if dtype not in _CASCADE_DTYPES:
            raise ValueError(
                f"CascadeFifo: dtype must be one of "
                f"{sorted(_CASCADE_DTYPES.keys())}, got {dtype!r}"
            )
        np_dtype, lanes_per_word = _CASCADE_DTYPES[dtype]

        if elements_per_handshake is None:
            elements_per_handshake = lanes_per_word
        if not isinstance(elements_per_handshake, int):
            raise ValueError(
                f"CascadeFifo: elements_per_handshake must be int, "
                f"got {type(elements_per_handshake).__name__}"
            )
        if elements_per_handshake <= 0:
            raise ValueError(
                f"CascadeFifo: elements_per_handshake must be > 0, "
                f"got {elements_per_handshake}"
            )
        if elements_per_handshake % lanes_per_word != 0:
            raise ValueError(
                f"CascadeFifo: elements_per_handshake "
                f"({elements_per_handshake}) must be a positive multiple "
                f"of the cascade lane count for dtype={dtype!r} "
                f"({lanes_per_word})"
            )

        self._producer_tile = producer_tile.copy()
        self._consumer_tile = consumer_tile.copy()
        self._dtype = dtype
        self._np_dtype = np_dtype
        self._lanes_per_word = lanes_per_word
        self._elements_per_handshake = elements_per_handshake
        if name is None:
            self.name = f"cas{CascadeFifo.__get_index()}"
        else:
            self.name = name
        self._op: CascadeFlowOp | None = None
        self._resolving = False

    @classmethod
    def __get_index(cls) -> int:
        idx = cls.__cf_index
        cls.__cf_index += 1
        return idx

    @property
    def producer_tile(self) -> Tile:
        """The producer-side CoreTile (emits cascade words via ``put_mcd``)."""
        return self._producer_tile

    @property
    def consumer_tile(self) -> Tile:
        """The consumer-side CoreTile (reads cascade words via ``get_scd``)."""
        return self._consumer_tile

    @property
    def dtype(self) -> str:
        """The cascade element dtype name (e.g. ``"accfloat"``)."""
        return self._dtype

    @property
    def lanes_per_word(self) -> int:
        """The number of dtype-typed lanes per 512-bit cascade word."""
        return self._lanes_per_word

    @property
    def elements_per_handshake(self) -> int:
        """The number of cascade-typed elements per handshake."""
        return self._elements_per_handshake

    @property
    def words_per_handshake(self) -> int:
        """The number of 512-bit cascade words per handshake."""
        return self._elements_per_handshake // self._lanes_per_word

    @property
    def cascade_bits(self) -> int:
        """Total bit-width transferred per handshake."""
        return self.words_per_handshake * CASCADE_BITS

    @property
    def tiles(self) -> list[Tile]:
        """Endpoint tiles, in (producer, consumer) order."""
        return [self._producer_tile, self._consumer_tile]

    @property
    def op(self) -> CascadeFlowOp:
        """The lowered ``aie.cascade_flow`` MLIR op.

        Raises :class:`NotResolvedError` if accessed before
        :meth:`resolve` has been called.
        """
        if self._op is None:
            raise NotResolvedError()
        return self._op

    def __str__(self) -> str:
        return (
            f"CascadeFifo(name='{self.name}', "
            f"dtype={self._dtype!r}, "
            f"elements_per_handshake={self._elements_per_handshake}, "
            f"prod_tile={self._producer_tile}, "
            f"cons_tile={self._consumer_tile})"
        )

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        """Lower this CascadeFifo to its MLIR ops.

        Emits a single ``aie.cascade_flow`` op connecting the producer
        tile to the consumer tile. The placer pass (``--aie-place-tiles``)
        validates cascade-adjacency and lowers the cascade_flow to
        per-tile ``aie.configure_cascade`` ops based on the relative
        physical placement (West/East for horizontal cascade,
        North/South for vertical).

        The actual ``aie.put_cascade`` / ``aie.get_cascade`` ops are
        emitted by the C++ kernel's ``put_mcd`` / ``get_scd_v16int32``
        intrinsics inside each Worker's core_fn — they are not emitted
        here.
        """
        if self._resolving:
            return
        self._resolving = True

        # Access ._op directly because Tile.op raises ValueError ("Cannot
        # get op before it is set.") when ._op is None. We want to give a
        # CascadeFifo-aware diagnostic instead.
        if self._producer_tile._op is None:
            raise ValueError(
                f"CascadeFifo {self.name}: producer tile op not set; "
                f"resolve the tile (or attach the producer Worker to a "
                f"Program) before resolving the cascade."
            )
        if self._consumer_tile._op is None:
            raise ValueError(
                f"CascadeFifo {self.name}: consumer tile op not set; "
                f"resolve the tile (or attach the consumer Worker to a "
                f"Program) before resolving the cascade."
            )
        prod_op = self._producer_tile.op
        cons_op = self._consumer_tile.op
        self._op = CascadeFlowOp(source_tile=prod_op, dest_tile=cons_op)

    # -----------------------------------------------------------------
    # Compatibility shims for callers that mirror ObjectFifo's
    # endpoint API. Cascade is point-to-point so we don't return real
    # handle objects (no acquire/release semantics — the cascade wire
    # is unbuffered); these helpers exist purely to give muscle-memory
    # compatibility for callers that build placement collections.
    # -----------------------------------------------------------------

    def prod(self) -> Tile:
        """Return the producer-side tile.

        Mirrors :meth:`aie.iron.ObjectFifo.prod` for muscle-memory
        compatibility but returns the :class:`Tile` directly (cascade
        has no acquire/release-style handle since it's an unbuffered
        physical wire, not a synchronized circular buffer).
        """
        return self._producer_tile

    def cons(self) -> Tile:
        """Return the consumer-side tile.

        Mirrors :meth:`aie.iron.ObjectFifo.cons` for muscle-memory
        compatibility but returns the :class:`Tile` directly. See
        :meth:`prod` for the rationale.
        """
        return self._consumer_tile
