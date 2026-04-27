# test_worker_fifo_handle_extension.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Tests for the FifoHandle registry that drives ``Worker.fn_args`` dispatch.

These tests cover the registry-driven type dispatch added in
``aie/iron/dataflow/fifo_handle_registry.py``:

1. **Backward-compat (regression guard)**: ``ObjectFifoHandle`` is
   pre-registered by ``dataflow/__init__.py`` and is still recognized
   by ``Worker.__init__`` exactly as before.
2. **Custom FifoHandle subclass**: a class that subclasses
   :class:`ObjectFifoHandle` and registers its own handler is
   dispatched via the registry; the handler runs and the Worker's
   bookkeeping reflects the custom registration.
3. **Runtime registration is reflected**: registering a brand-new
   handle class at runtime makes ``Worker.__init__`` recognize
   instances of it without any other code changes.
4. **Reverse-order precedence**: when a class and its subclass are
   both registered, the most-recently-registered (subclass) handler
   wins.
5. **Decorator form** of :func:`register_fifo_handle` works the same
   as the function-call form.
6. **Snapshot context manager** correctly restores registry state.
7. **Error handling**: double-registration raises; non-class /
   non-callable arguments are rejected.

These tests are pure-Python and do not require an MLIR context (no
``Program.compile()`` is invoked); they exercise the dispatch logic
inside ``Worker.__init__`` directly.
"""

from __future__ import annotations

import pytest

# Skip the whole module gracefully if the fork wheel isn't built / available.
aie_iron = pytest.importorskip("aie.iron")
from aie.iron import ObjectFifo, Worker  # noqa: E402
from aie.iron.device import Tile  # noqa: E402
from aie.iron.dataflow import ObjectFifoHandle  # noqa: E402
from aie.iron.dataflow.fifo_handle_registry import (  # noqa: E402
    _RegistrySnapshot,
    dispatch_fn_arg,
    get_registered_handle_classes,
    register_fifo_handle,
    unregister_fifo_handle,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_registry():
    """Save and restore the registry around each test that mutates it."""
    with _RegistrySnapshot():
        yield

@pytest.fixture
def fifo():
    """A vanilla ObjectFifo for handle construction. Uses int32, depth 2."""
    import numpy as np

    return ObjectFifo(np.ndarray[(8,), np.dtype[np.int32]], depth=2, name="t24_fifo")

def _noop_core(*_args):
    """Trivial core_fn for Worker construction. Never actually executed in
    these tests because Worker.resolve() is not called (no MLIR context)."""

# ---------------------------------------------------------------------------
# 1. Backward-compat: ObjectFifoHandle still works exactly as before.
# ---------------------------------------------------------------------------

def test_object_fifo_handle_is_pre_registered():
    """``dataflow/__init__.py`` pre-registers ObjectFifoHandle so that
    Phase 1 designs keep working without any other code change."""
    classes = get_registered_handle_classes()
    assert ObjectFifoHandle in classes, (
        f"ObjectFifoHandle missing from registry -- backward-compat broken. "
        f"Registered: {[c.__name__ for c in classes]}"
    )

def test_worker_recognizes_object_fifo_handle_via_registry(fifo):
    """Regression guard: Worker.__init__ still records ObjectFifoHandle in
    ``_fifos`` and sets the handle's endpoint, just like Phase 1."""
    handle = fifo.prod()
    worker = Worker(_noop_core, fn_args=[handle], tile=Tile(0, 2))

    assert handle in worker.fifos, (
        "ObjectFifoHandle not recorded on Worker.fifos -- registry "
        "did not invoke the pre-registered handler"
    )
    assert handle.endpoint is worker, (
        "ObjectFifoHandle.endpoint was not set to the Worker -- "
        "the pre-registered handler did not run its bookkeeping"
    )

def test_worker_still_rejects_raw_object_fifo(fifo):
    """Non-FifoHandle paths (Buffer/ObjectFifo/Barrier branches in
    Worker.__init__) are unchanged. Passing a raw ObjectFifo is still
    a hard error."""
    with pytest.raises(ValueError, match="Cannot give an ObjectFifo"):
        Worker(_noop_core, fn_args=[fifo], tile=Tile(0, 2))

# ---------------------------------------------------------------------------
# 2. Custom FifoHandle subclass dispatches via the registry.
# ---------------------------------------------------------------------------

def test_custom_subclass_dispatches_via_registry(clean_registry, fifo):
    """A subclass with its own registered handler wins isinstance() against
    ObjectFifoHandle's registration (registered later -> reverse-order walk
    in dispatch_fn_arg picks subclass first)."""

    class _MyCustomHandle(ObjectFifoHandle):
        pass

    captured: list[tuple[object, object]] = []

    def _custom_handler(arg, worker):
        captured.append((arg, worker))
        # Mirror the ObjectFifoHandle bookkeeping so Worker's fifos
        # property still reports the handle (otherwise the Worker would
        # silently lose the FIFO).
        arg.endpoint = worker
        worker._fifos.append(arg)

    register_fifo_handle(_MyCustomHandle, _custom_handler)

    handle = fifo.prod()
    # __class__-swap to fake a subclass instance without touching
    # ObjectFifoHandle.__init__ semantics. This is the cleanest way to
    # construct a subclass instance for these tests because
    # ObjectFifoHandle's constructor side-effects (claiming the producer
    # slot on the parent fifo) only fire once.
    handle.__class__ = _MyCustomHandle

    worker = Worker(_noop_core, fn_args=[handle], tile=Tile(0, 2))

    assert len(captured) == 1, (
        f"custom handler did not run exactly once: {len(captured)} calls"
    )
    assert captured[0][0] is handle and captured[0][1] is worker
    assert handle in worker.fifos

def test_runtime_registration_reflected_in_worker(clean_registry, fifo):
    """Registering a new handle class after the wheel is loaded is
    reflected in subsequent Worker constructions -- the registry is
    consulted dynamically, not snapshotted at import time."""

    class _LateRegisteredHandle(ObjectFifoHandle):
        pass

    handle = fifo.prod()
    handle.__class__ = _LateRegisteredHandle

    # Before registration: registry doesn't know _LateRegisteredHandle by
    # itself, but the parent ObjectFifoHandle handler still fires (since
    # _LateRegisteredHandle is-a ObjectFifoHandle). Use the dispatch
    # function directly with a stub Worker to confirm.
    class _StubWorker:
        def __init__(self):
            self._fifos = []

    stub = _StubWorker()
    assert dispatch_fn_arg(handle, stub) is True
    # ObjectFifoHandle handler fired (parent-class fallback).
    assert stub._fifos == [handle]

    # Now register a custom handler that does something distinguishable
    # (records the worker into a local list rather than _fifos).
    custom_record: list[object] = []

    def _custom(arg, worker):
        custom_record.append((arg, worker))

    register_fifo_handle(_LateRegisteredHandle, _custom)

    stub2 = _StubWorker()
    assert dispatch_fn_arg(handle, stub2) is True
    assert custom_record == [(handle, stub2)]
    # Custom handler did NOT do the parent's _fifos bookkeeping (it's a
    # different handler entirely) -- confirms reverse-order precedence.
    assert stub2._fifos == []

def test_reverse_order_precedence(clean_registry, fifo):
    """Later registrations of more-specific subclasses win over earlier
    parent registrations in dispatch_fn_arg's reverse-order walk."""

    class _Mid(ObjectFifoHandle):
        pass

    class _Leaf(_Mid):
        pass

    fired: list[str] = []

    register_fifo_handle(_Mid, lambda a, w: fired.append("mid"))
    register_fifo_handle(_Leaf, lambda a, w: fired.append("leaf"))

    handle = fifo.prod()
    handle.__class__ = _Leaf

    class _StubWorker:
        def __init__(self):
            self._fifos = []

    stub = _StubWorker()
    dispatch_fn_arg(handle, stub)

    # Most-specific subclass (registered LAST) wins.
    assert fired == ["leaf"], (
        f"reverse-order walk picked the wrong handler: fired={fired}"
    )

# ---------------------------------------------------------------------------
# 3. Decorator form vs function-call form.
# ---------------------------------------------------------------------------

def test_decorator_form_registers(clean_registry, fifo):
    """``@register_fifo_handle(MyHandle)`` registers the wrapped callable
    and returns it unchanged for chaining or further decoration."""

    class _DecoratedHandle(ObjectFifoHandle):
        pass

    @register_fifo_handle(_DecoratedHandle)
    def _handler(arg, worker):
        worker._fifos.append(arg)

    assert _DecoratedHandle in get_registered_handle_classes()
    # Returned callable is the original (decorator does not wrap it).
    assert callable(_handler)

def test_function_call_form_registers(clean_registry, fifo):
    """``register_fifo_handle(MyHandle, my_handler)`` registers without
    needing decorator syntax."""

    class _CallFormHandle(ObjectFifoHandle):
        pass

    def _handler(arg, worker):
        worker._fifos.append(arg)

    ret = register_fifo_handle(_CallFormHandle, _handler)
    assert ret is None  # function-call form has no return value
    assert _CallFormHandle in get_registered_handle_classes()

# ---------------------------------------------------------------------------
# 4. Snapshot context manager + unregister helper.
# ---------------------------------------------------------------------------

def test_registry_snapshot_restores_state(fifo):
    """The snapshot context manager restores the registry to its pre-entry
    state on exit -- registrations made inside the ``with`` block are
    discarded."""

    class _ScopedHandle(ObjectFifoHandle):
        pass

    classes_before = set(get_registered_handle_classes())

    with _RegistrySnapshot():
        register_fifo_handle(_ScopedHandle, lambda a, w: None)
        assert _ScopedHandle in get_registered_handle_classes()

    classes_after = set(get_registered_handle_classes())
    assert classes_after == classes_before
    assert _ScopedHandle not in classes_after

def test_unregister_fifo_handle_returns_bool(clean_registry):
    """``unregister_fifo_handle`` returns True if a registration was
    removed, False otherwise. Test-only helper but worth a smoke test."""

    class _UnregHandle(ObjectFifoHandle):
        pass

    assert unregister_fifo_handle(_UnregHandle) is False
    register_fifo_handle(_UnregHandle, lambda a, w: None)
    assert unregister_fifo_handle(_UnregHandle) is True
    # Idempotent removal.
    assert unregister_fifo_handle(_UnregHandle) is False

# ---------------------------------------------------------------------------
# 5. Error handling.
# ---------------------------------------------------------------------------

def test_double_registration_raises(clean_registry):
    """Registering the same class twice raises ValueError to catch
    accidental double-registration in fork-internal code."""

    class _DupHandle(ObjectFifoHandle):
        pass

    register_fifo_handle(_DupHandle, lambda a, w: None)
    with pytest.raises(ValueError, match="already registered"):
        register_fifo_handle(_DupHandle, lambda a, w: None)

def test_non_class_handle_cls_rejected(clean_registry):
    """``register_fifo_handle`` rejects non-class first arguments."""
    with pytest.raises(TypeError, match="must be a class"):
        register_fifo_handle("not a class", lambda a, w: None)  # type: ignore[arg-type]

def test_non_callable_handler_rejected(clean_registry):
    """``register_fifo_handle`` rejects non-callable handlers."""

    class _BadHandlerHandle(ObjectFifoHandle):
        pass

    with pytest.raises(TypeError, match="must be callable"):
        register_fifo_handle(_BadHandlerHandle, "not callable")  # type: ignore[arg-type]

def test_dispatch_returns_false_for_unregistered(clean_registry):
    """``dispatch_fn_arg`` returns False when no registered class matches."""

    class _NotAHandle:
        pass

    class _StubWorker:
        def __init__(self):
            self._fifos = []

    assert dispatch_fn_arg(_NotAHandle(), _StubWorker()) is False

# ---------------------------------------------------------------------------
# 6. Surface stability (the public registry API is what fork-PR consumers
# rely on; pin it down so accidental rename / removal trips a test).
# ---------------------------------------------------------------------------

def test_registry_module_public_surface():
    """Pin the public registry API down so accidental renames trip a test."""
    from aie.iron.dataflow import fifo_handle_registry as reg

    expected = {
        "register_fifo_handle",
        "unregister_fifo_handle",
        "get_registered_handle_classes",
        "dispatch_fn_arg",
    }
    actual = set(reg.__all__)
    missing = expected - actual
    assert not missing, f"public surface drift: missing {missing} from __all__"
