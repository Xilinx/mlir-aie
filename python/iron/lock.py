# lock.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Iron-level Lock primitive — a named ``aie.lock`` on a specific tile.

Pairs with :class:`Buffer` for designs that wire DMA / compute
synchronization explicitly (via :class:`TileDma` and :class:`Flow`)
instead of letting :class:`ObjectFifo` manage it.
"""

import itertools

from .. import ir  # pyright: ignore[reportMissingImports]
from ..dialects._aie_enum_gen import LockAction  # pyright: ignore[reportMissingImports]
from ..dialects.aie import (
    lock as _lock_op,
    use_lock as _use_lock,  # pyright: ignore[reportAttributeAccessIssue]
)
from .device import Tile
from .resolvable import NotResolvedError, Resolvable


class Lock(Resolvable):
    """A named hardware lock on a specific tile."""

    _glock_index = itertools.count()

    def __init__(
        self,
        tile: Tile,
        lock_id: int | None = None,
        init: int = 0,
        name: str | None = None,
    ):
        """Construct a Lock.

        Args:
            tile (Tile): The tile that owns this lock.
            lock_id (int | None): Hardware lock ID; passed straight through to
                the underlying ``aie.lock`` op.  If ``None`` (the default),
                the lowering pass picks one.
            init (int): Initial lock value at design startup.  Defaults to 0.
            name (str | None): Symbol name for the lock.  A unique name is
                generated if not provided.
        """
        self._tile = tile
        self._lock_id = lock_id
        self._init = init
        self._name = name or f"lock_{next(Lock._glock_index)}"
        self._op = None

    @property
    def tile(self) -> Tile:
        return self._tile

    @property
    def name(self) -> str:
        return self._name

    @property
    def op(self):
        if self._op is None:
            raise NotResolvedError()
        return self._op

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self._op is None:
            if self._tile is None:
                raise ValueError("Cannot resolve Lock until it has been placed.")
            self._op = _lock_op(
                self._tile.op,
                lock_id=self._lock_id,
                init=self._init,
                sym_name=self._name,
            )

    # ------------------------------------------------------------------
    # Emit-time helpers — call these from inside a Worker body to emit
    # ``aie.use_lock`` ops without reaching into ``aie.dialects.aie``.
    # ------------------------------------------------------------------

    def acquire(self, value: int = 1) -> None:
        """Emit ``aie.use_lock(self, AcquireGreaterEqual, value=value)``.

        The default ``AcquireGreaterEqual`` mode matches what almost every
        ObjectFifo / DMA-driven design wants; use :meth:`acquire_exact` for
        the rarer ``Acquire`` (exact-equality) mode.
        """
        _use_lock(self.op, LockAction.AcquireGreaterEqual, value=value)

    def acquire_exact(self, value: int = 1) -> None:
        """Emit ``aie.use_lock(self, Acquire, value=value)`` (exact match)."""
        _use_lock(self.op, LockAction.Acquire, value=value)

    def release(self, value: int = 1) -> None:
        """Emit ``aie.use_lock(self, Release, value=value)``."""
        _use_lock(self.op, LockAction.Release, value=value)
