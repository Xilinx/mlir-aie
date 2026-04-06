# test_compile_context.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Unit tests for compile_context and get_compile_arg — no NPU required."""

import threading

import pytest

from aie.iron.compile.context import (
    compile_context,
    get_compile_arg,
)

# ---------------------------------------------------------------------------
# Baseline: outside any context
# ---------------------------------------------------------------------------


def test_get_compile_arg_outside_context_returns_none():
    assert get_compile_arg("M") is None


def test_get_compile_arg_outside_context_custom_default():
    assert get_compile_arg("M", default=42) == 42


def test_get_compile_arg_outside_context_falsy_default():
    """Falsy defaults (0, False, "") are returned correctly, not confused with None."""
    assert get_compile_arg("x", default=0) == 0
    assert get_compile_arg("x", default=False) is False
    assert get_compile_arg("x", default="") == ""


# ---------------------------------------------------------------------------
# Basic single-level injection
# ---------------------------------------------------------------------------


def test_single_key_injection():
    with compile_context(M=512):
        assert get_compile_arg("M") == 512


def test_multiple_key_injection():
    with compile_context(M=512, K=256, N=128):
        assert get_compile_arg("M") == 512
        assert get_compile_arg("K") == 256
        assert get_compile_arg("N") == 128


def test_absent_key_returns_none_inside_context():
    with compile_context(M=512):
        assert get_compile_arg("N") is None


def test_absent_key_returns_custom_default_inside_context():
    with compile_context(M=512):
        assert get_compile_arg("N", default=99) == 99


def test_non_integer_value_types():
    """Context accepts any Python value: floats, strings, booleans, lists."""
    import numpy as np

    with compile_context(dtype=np.float32, label="gemm", flag=True, dims=[64, 64]):
        assert get_compile_arg("dtype") is np.float32
        assert get_compile_arg("label") == "gemm"
        assert get_compile_arg("flag") is True
        assert get_compile_arg("dims") == [64, 64]


def test_empty_context_injects_nothing():
    with compile_context():
        assert get_compile_arg("anything") is None


# ---------------------------------------------------------------------------
# Cleanup after context exit
# ---------------------------------------------------------------------------


def test_context_exits_cleanly_normal():
    with compile_context(M=512):
        pass
    assert get_compile_arg("M") is None


def test_context_exits_cleanly_after_exception():
    with pytest.raises(ValueError):
        with compile_context(M=512):
            raise ValueError("deliberate")
    assert get_compile_arg("M") is None


def test_context_exits_cleanly_after_runtime_error():
    with pytest.raises(RuntimeError):
        with compile_context(x=1, y=2):
            raise RuntimeError("boom")
    assert get_compile_arg("x") is None
    assert get_compile_arg("y") is None


# ---------------------------------------------------------------------------
# Nesting: inner shadows outer; outer restored after inner exits
# ---------------------------------------------------------------------------


def test_nested_inner_overrides_outer_key():
    with compile_context(M=512, K=128):
        with compile_context(M=1024):
            assert get_compile_arg("M") == 1024
            assert get_compile_arg("K") == 128  # outer still visible
        assert get_compile_arg("M") == 512
        assert get_compile_arg("K") == 128


def test_nested_inner_adds_new_key():
    with compile_context(M=512):
        with compile_context(N=256):
            assert get_compile_arg("M") == 512
            assert get_compile_arg("N") == 256
        assert get_compile_arg("N") is None


def test_three_level_nesting():
    with compile_context(x=1):
        with compile_context(x=2, y=10):
            with compile_context(x=3):
                assert get_compile_arg("x") == 3
                assert get_compile_arg("y") == 10
            assert get_compile_arg("x") == 2
            assert get_compile_arg("y") == 10
        assert get_compile_arg("x") == 1
        assert get_compile_arg("y") is None


def test_sibling_contexts_are_independent():
    with compile_context(a=1):
        assert get_compile_arg("a") == 1
    with compile_context(b=2):
        assert get_compile_arg("a") is None
        assert get_compile_arg("b") == 2
    assert get_compile_arg("b") is None


def test_nested_exception_in_inner_restores_outer():
    with compile_context(outer=True):
        with pytest.raises(RuntimeError):
            with compile_context(inner=True):
                raise RuntimeError("inner failure")
        assert get_compile_arg("outer") is True
        assert get_compile_arg("inner") is None


# ---------------------------------------------------------------------------
# Context manager yield value
# ---------------------------------------------------------------------------


def test_context_yields_injected_dict():
    with compile_context(M=512, K=256) as ctx:
        assert ctx == {"M": 512, "K": 256}


def test_nested_context_yields_merged_dict():
    with compile_context(a=1) as outer:
        with compile_context(b=2) as inner:
            assert inner == {"a": 1, "b": 2}
        assert outer == {"a": 1}


def test_outer_dict_not_mutated_by_inner():
    with compile_context(a=1) as outer:
        outer_id = id(outer)
        with compile_context(a=99) as inner:
            pass
        # outer dict object unchanged; inner override was temporary
        assert id(outer) == outer_id
        assert outer == {"a": 1}


# ---------------------------------------------------------------------------
# Transitive visibility: functions called inside the context see the values
# ---------------------------------------------------------------------------


def _read_m():
    return get_compile_arg("M")


def _read_nested(key):
    return get_compile_arg(key)


def test_transitive_call_sees_context():
    with compile_context(M=77):
        assert _read_m() == 77
    assert _read_m() is None


def test_transitive_nested_call_sees_outer_context():
    with compile_context(dtype="float32"):
        assert _read_nested("dtype") == "float32"


def test_recursive_call_chain_sees_context():
    def depth_reader(n):
        if n == 0:
            return get_compile_arg("depth")
        return depth_reader(n - 1)

    with compile_context(depth=99):
        assert depth_reader(5) == 99


# ---------------------------------------------------------------------------
# Thread isolation via contextvars
# ---------------------------------------------------------------------------


def test_thread_isolation():
    """compile_context values are NOT visible to child threads.

    CPython's ``threading.Thread`` does not propagate ``contextvars`` mutations
    from the spawning thread — each thread runs in an independent copy of the
    context as it existed at module import time (the default).  Changes made via
    ``_compile_context_var.set()`` inside a ``compile_context`` block are local
    to the current thread only.  A child thread therefore always sees the
    default empty dict, regardless of what the parent thread has set.
    """
    results = {}

    def worker(key, out_key):
        results[out_key] = get_compile_arg(key)

    with compile_context(secret=42):
        t = threading.Thread(target=worker, args=("secret", "child"))
        t.start()
        t.join()
        results["parent"] = get_compile_arg("secret")

    # Parent sees the value while inside the context.
    assert results["parent"] == 42
    # Child thread runs in an isolated context; it sees the default (None).
    assert results["child"] is None


# ---------------------------------------------------------------------------
# Interaction with _compile_context_var internals
# ---------------------------------------------------------------------------


def test_default_outside_context_is_none():
    assert get_compile_arg("__nonexistent__") is None


def test_active_context_returns_injected_values():
    with compile_context(foo=1):
        assert get_compile_arg("foo") == 1
    assert get_compile_arg("foo") is None
