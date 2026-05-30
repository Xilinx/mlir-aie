# fifo_handle_registry.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""FifoHandle registry: extensible type-dispatch for ``Worker.fn_args``.

``Worker.__init__``'s ``fn_args`` resolution as the single biggest
blocker for promoting ObjectFifo subclasses (PacketFifo, CascadeFifo,
AccumFifo, SparseFifo) to first-class IRON primitives. The original
implementation hard-coded the type-dispatch chain to recognize only
``ObjectFifoHandle`` (plus ``Buffer``, ``ObjectFifo``, and
``WorkerRuntimeBarrier``), so any new FifoHandle subclass would have
to either fork ``worker.py`` or be silently treated as a "metaprogramming
value" -- with no chance to register itself for placement / resolution.

This module provides a lightweight registry of FifoHandle classes and
SparseFifoHandle) registers its handle class via
:func:`register_fifo_handle` (or the ``@register_fifo_handle`` decorator)
without modifying ``worker.py`` at all.

Backward-compat: ``ObjectFifoHandle`` is pre-registered by
``dataflow/__init__.py`` with a handler that reproduces the original
``Worker.__init__`` behavior bit-for-bit (set ``arg.endpoint = worker``;
append to ``worker._fifos``).

Public surface:

- :func:`register_fifo_handle` -- register a FifoHandle class (callable
  form OR decorator form).
- :func:`unregister_fifo_handle` -- remove a registration (test-only).
- :func:`get_registered_handle_classes` -- enumerate registered classes
  (read-only snapshot).
- :func:`dispatch_fn_arg` -- consulted by ``Worker.__init__`` to attempt
  registry-driven dispatch for one argument.
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover -- type-checking only
    from ..worker import Worker

# Mapping: handle class -> handler callable ``(arg, worker) -> None``.
#
# The handler receives the argument and the Worker being constructed. It
# may mutate the Worker's bookkeeping lists (``_fifos``, ``_buffers``,
# ``_barriers``, etc.) and/or the argument itself (e.g. setting
# ``arg.endpoint = worker``). A handler returning normally signals
# "argument was recognized and bookkept"; raising propagates to the
# caller.
#
# Registration order is preserved (Python dict insertion order); when
# two registered classes are in a parent/child relationship, the more
# specific class should be registered LAST so it wins isinstance()
# checks. ``dispatch_fn_arg`` walks the registry in reverse-insertion
# order to honor that.
_REGISTRY: "dict[type, Callable[[object, Worker], None]]" = {}

def register_fifo_handle(
    handle_cls: type,
    handler: "Callable[[object, Worker], None] | None" = None,
):
    """Register a FifoHandle class with its Worker.fn_args handler.

    Two call forms are supported:

    1. **Function call** (most explicit)::

           register_fifo_handle(MyFifoHandle, my_handler)

    2. **Decorator** (preferred for new fork-internal subclasses)::

           @register_fifo_handle(MyFifoHandle)
           def _my_fifo_handle_handler(arg, worker):
               worker._fifos.append(arg)
               arg.endpoint = worker

    Args:
        handle_cls: The FifoHandle subclass to register. Must be a class
            (anything ``isinstance`` can use as the second argument).
        handler: A callable ``(arg, worker) -> None`` that bookkeeps the
            argument on the Worker. If omitted, the function returns a
            decorator that registers the wrapped callable.

    Returns:
        Either ``None`` (function-call form) or a decorator that
        captures the handler and returns it unchanged (decorator form).

    Raises:
        TypeError: ``handle_cls`` is not a class, or ``handler`` is not
            callable.
        ValueError: A *different* handler is already registered for
            ``handle_cls``. Re-registering the same callable is
            idempotent (no-op) so module reloads / repeated imports
            stay safe; use :func:`unregister_fifo_handle` first to
            install a different handler.
    """
    if not isinstance(handle_cls, type):
        raise TypeError(
            f"register_fifo_handle: handle_cls must be a class, got "
            f"{type(handle_cls).__name__}"
        )

    def _do_register(h: "Callable[[object, Worker], None]"):
        if not callable(h):
            raise TypeError(
                f"register_fifo_handle: handler must be callable, got "
                f"{type(h).__name__}"
            )
        existing = _REGISTRY.get(handle_cls)
        if existing is not None:
            if existing is h:
                # Idempotent re-registration (e.g. module reload, repeat
                # import in tests). Safe no-op.
                return h
            raise ValueError(
                f"register_fifo_handle: {handle_cls.__name__} is already "
                f"registered with a different handler. Call "
                f"unregister_fifo_handle() first if override is intentional."
            )
        _REGISTRY[handle_cls] = h
        return h

    if handler is None:
        # Decorator form: @register_fifo_handle(MyFifoHandle)
        return _do_register
    # Function-call form: register_fifo_handle(MyFifoHandle, my_handler)
    _do_register(handler)
    return None

def unregister_fifo_handle(handle_cls: type) -> bool:
    """Remove a previously-registered handle class.

    Primarily intended for tests that register a custom handle class
    inside a fixture and want to clean up afterward. Returns ``True`` if
    a registration was removed; ``False`` if no registration existed.
    """
    return _REGISTRY.pop(handle_cls, None) is not None

def get_registered_handle_classes() -> "tuple[type, ...]":
    """Return a snapshot of registered handle classes (insertion order).

    The returned tuple is a copy; mutating the registry during iteration
    is therefore safe.
    """
    return tuple(_REGISTRY.keys())

def dispatch_fn_arg(arg: object, worker: "Worker") -> bool:
    """Attempt to dispatch ``arg`` through the registry.

    Walks the registered classes in **reverse-insertion order** so that
    later registrations of more-specific subclasses take precedence over
    earlier registrations of base classes (``ObjectFifoHandle`` is
    pre-registered first by ``dataflow/__init__.py`` and acts as the
    fallback for anything that subclasses it without registering its
    own handler).

    Args:
        arg: The candidate Worker.fn_args entry.
        worker: The Worker being constructed.

    Returns:
        True if a handler matched and was invoked; False if no
        registered class is an ``isinstance`` of ``arg``.
    """
    # reversed() preserves dict ordering semantics; later registrations
    # win. We don't materialize a list -- the registry is small.
    for cls in reversed(_REGISTRY):
        if isinstance(arg, cls):
            _REGISTRY[cls](arg, worker)
            return True
    return False

# ---------------------------------------------------------------------------
# Test-support helper. Production code should use the registration API
# above directly; this exists so tests can save/restore registry state
# without poking at the private dict.
# ---------------------------------------------------------------------------

class _RegistrySnapshot:
    """Internal context manager that saves and restores registry state.

    Test-only utility: ``with _RegistrySnapshot(): ...`` saves the
    registry on entry and restores it on exit. Not part of the public
    API; the leading underscore signals "test-internal use only".

    Usage in tests::

        with _RegistrySnapshot():
            register_fifo_handle(MyTestHandle, _my_handler)
            ...
        # registry restored here -- MyTestHandle no longer registered.
    """

    def __enter__(self):
        self._saved = dict(_REGISTRY)
        return self

    def __exit__(self, *exc):
        _REGISTRY.clear()
        _REGISTRY.update(self._saved)
        return False

__all__ = [
    "register_fifo_handle",
    "unregister_fifo_handle",
    "get_registered_handle_classes",
    "dispatch_fn_arg",
]
