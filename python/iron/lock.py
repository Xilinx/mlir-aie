# lock.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""IRON-level Lock primitive — a named `aie.lock` on a specific tile.

Pairs with [`Buffer`][iron.Buffer] for designs that wire DMA / compute
synchronization explicitly (via [`TileDma`][iron.TileDma] and
[`Flow`][iron.Flow]) instead of letting [`ObjectFifo`][iron.ObjectFifo]
manage it.
"""

from .. import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ..dialects._aie_enum_gen import LockAction  # pyright: ignore[reportMissingImports]
from ..dialects.aie import (
    lock as _lock_op,
)
from ..dialects.aie import (
    use_lock as _use_lock,  # pyright: ignore[reportAttributeAccessIssue]
)
from ..dialects.aiex import set_lock_value as _set_lock_value
from .device import Tile
from .resolvable import NotResolvedError, Resolvable


class Lock(Resolvable):
    """A named hardware lock on a specific tile."""

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
                the underlying `aie.lock` op. If `None` (the default),
                the lowering pass picks one.
            init (int): Initial lock value at design startup. Defaults to 0.
            name (str | None): Symbol name for the lock. Defaults to None
                (unnamed).
        """
        self._tile = tile
        self._lock_id = lock_id
        self._init = init
        self._name = name
        self._op = None

    @property
    def tile(self) -> Tile:
        return self._tile

    @property
    def name(self) -> str | None:
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
        """Emit `aie.use_lock(self, AcquireGreaterEqual, value=value)`.

        The default `AcquireGreaterEqual` mode matches what almost every
        ObjectFifo / DMA-driven design wants; use
        [`acquire_exact`][iron.lock.Lock.acquire_exact] for the rarer
        `Acquire` (exact-equality) mode.
        """
        _use_lock(self.op, LockAction.AcquireGreaterEqual, value=value)

    def acquire_exact(self, value: int = 1) -> None:
        """Emit `aie.use_lock(self, Acquire, value=value)` (exact match)."""
        _use_lock(self.op, LockAction.Acquire, value=value)

    def release(self, value: int = 1) -> None:
        """Emit `aie.use_lock(self, Release, value=value)`."""
        _use_lock(self.op, LockAction.Release, value=value)

    def set(self, value: int) -> None:
        """Emit `aiex.set_lock(self, value)` from a runtime sequence body.

        Overwrites the lock's value from the host side, e.g. to re-arm a
        producer lock before a runtime-sequence DMA chain starts reusing a
        buffer. The write is not ordered against anything the array is doing,
        so pair it with a blocking op (an await, or a lock the core waits on)
        that makes it safe.

        A core cannot assign a lock: its lock instructions only add to or
        subtract from the value, which is what `acquire`/`release` emit. The
        `aiex.set_lock` verifier rejects a call outside a runtime sequence and
        a value outside `[0, Device.max_lock_value]`.
        """
        _set_lock_value(self.op, value)
