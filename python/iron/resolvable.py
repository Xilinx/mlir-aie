# resolvable.py -*- Python -*-
#
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Structural protocol for objects that lower to MLIR operations."""

import functools
from typing import Protocol, runtime_checkable

from .. import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ..helpers.sourceloc import capture_source_site, site_location
from ..helpers.errors import filter_internal_frames


def _site_name(obj) -> str | None:
    """Best-effort symbol name, used to label this object's location."""
    for attr in ("_name", "name"):
        try:
            value = getattr(obj, attr, None)
        except Exception:
            continue  # a property that needs resolution we haven't done yet
        if isinstance(value, str) and value:
            return value
    return None


def _attach_source_site(cls) -> None:
    """Give `cls` automatic user-source attribution.

    Wraps `__init__` to record where the user declared the object, and
    `resolve` to hand that site to MLIR whenever the caller did not pass an
    explicit `loc`. Doing this on the protocol rather than in each subclass is
    what keeps attribution from silently rotting: a `resolve` that forgets to
    thread `loc` would otherwise fall back to `get_user_code_loc`, which at
    resolve time points into IRON's own source rather than the user's.
    """
    init = cls.__dict__.get("__init__")
    if init is not None and not getattr(init, "_iron_located", False):

        @functools.wraps(init)
        def __init__(self, *args, **kwargs):
            self._source_site = capture_source_site()
            try:
                return init(self, *args, **kwargs)
            except Exception as exc:
                # A constructor rejects a design long before resolve_program
                # gets a boundary around it, so its frames are filtered here or
                # not at all.
                raise filter_internal_frames(exc) from None

        __init__._iron_located = True
        cls.__init__ = __init__

    resolve = cls.__dict__.get("resolve")
    if resolve is not None and not getattr(resolve, "_iron_located", False):

        @functools.wraps(resolve)
        def resolve_wrapper(self, loc=None, *args, **kwargs):
            if loc is None:
                loc = site_location(
                    getattr(self, "_source_site", None), _site_name(self)
                )
            if loc is None:
                return resolve(self, loc, *args, **kwargs)
            # Make the site ambient as well as passing it: ops built by helpers
            # several layers down (a shim DMA task expands into a cluster of
            # them) then inherit it without every builder in between having to
            # forward a loc. A nested resolve overrides this with its own site,
            # and an explicit loc= at any builder still wins.
            with loc:
                return resolve(self, loc, *args, **kwargs)

        resolve_wrapper._iron_located = True
        cls.resolve = resolve_wrapper


# Structural typing via @runtime_checkable Protocol: any class with both
# .resolve() and .tiles() passes isinstance(x, Resolvable).  The two-method
# requirement is the safeguard against false positives from classes that
# happen to define an unrelated .resolve() (e.g. pathlib.Path).
@runtime_checkable
class Resolvable(Protocol):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _attach_source_site(cls)

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        """Resolve the current object into one or more MLIR operations.

        Should only be called within an MLIR context.

        Args:
            loc (ir.Location | None, optional): Location is used by MLIR object during construction in some cases. Defaults to None.
            ip (ir.InsertionPoint | None, optional): InsertionPoint is used by MLIR object during construction in some cases. Defaults to None.
        """
        ...

    def tiles(self) -> list:
        """Tiles this Resolvable depends on for code generation.

        Override this in user-side Resolvable subclasses that reference tiles
        which aren't already discoverable via Workers or ObjectFifos. The
        Program will resolve these tiles before calling `resolve`, so
        `tile.op` is valid by then. Default: empty list.
        """
        return []


class NotResolvedError(Exception):
    """Raised when a property or operation is accessed on a `Resolvable` object before `resolve` has been called."""

    def __init__(self, message="Cannot get operation; class not resolved."):
        self.message = message
        super().__init__(self.message)
