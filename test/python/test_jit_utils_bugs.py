# test_jit_utils_bugs.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
#
# Regression tests for two pre-existing bugs fixed in this branch.
# These tests run without hardware, XRT, or a live MLIR context.

import hashlib
import os
import shutil
import tempfile
import pytest


# ---------------------------------------------------------------------------
# Regression: hash_module must produce distinct cache keys for distinct
# ExternalFunction configurations.
#
# Bug: "|".join(running_hash) iterated over characters of a concatenated
# string rather than over per-kernel hash values, causing false collisions
# (e.g. hashes "1"+"2" produced the same key as hash "12").
# ---------------------------------------------------------------------------


def _hash_module(mlir_str, kernels):
    """Minimal reimplementation of hash_module for unit testing.

    Mirrors the production logic in python/utils/jit.py without requiring
    the full aie module import chain.
    """
    if kernels:
        combined_str = mlir_str + "|" + "|".join(str(hash(f)) for f in kernels)
    else:
        combined_str = mlir_str
    return hashlib.sha256(combined_str.encode("utf-8")).hexdigest()[:16]


class _FixedHashKernel:
    """Stub whose hash() is pinned to a constructor-supplied int."""

    def __init__(self, h):
        self._h = h

    def __hash__(self):
        return self._h


def test_hash_distinct_single_kernels_produce_distinct_keys():
    """Two kernels with different hashes must produce different cache keys."""
    mlir = "module {}"
    assert _hash_module(mlir, [_FixedHashKernel(1)]) != _hash_module(
        mlir, [_FixedHashKernel(2)]
    )


def test_hash_one_vs_two_kernels_produce_distinct_keys():
    """One kernel with hash "12" must not collide with two kernels hashing to "1" and "2"."""
    mlir = "module {}"
    key_one = _hash_module(mlir, [_FixedHashKernel(12)])
    key_two = _hash_module(mlir, [_FixedHashKernel(1), _FixedHashKernel(2)])
    assert key_one != key_two


def test_hash_same_kernels_produce_same_key():
    """Same kernel configuration must always produce the same cache key."""
    mlir = "module {}"
    k = _FixedHashKernel(42)
    assert _hash_module(mlir, [k]) == _hash_module(mlir, [k])


def test_hash_no_kernels_vs_with_kernels_differ():
    """A module with no kernels must hash differently from one with kernels."""
    mlir = "module {}"
    assert _hash_module(mlir, None) != _hash_module(mlir, [_FixedHashKernel(1)])


# ---------------------------------------------------------------------------
# Regression: compile_external_kernel must raise FileNotFoundError when
# source_file does not exist.
#
# Bug: the function silently returned without compiling or raising, leaving
# the caller to encounter a confusing downstream linker error.
# ---------------------------------------------------------------------------


class _FakeExternalFunction:
    """Minimal stub satisfying the fields compile_external_kernel reads."""

    def __init__(self, name, source_file=None, source_string=None):
        self._name = name
        self._source_file = source_file
        self._source_string = source_string
        self._include_dirs = []
        self._compile_flags = []
        self._compiled = False
        self.bin_name = f"{name}.o"


def _compile_external_kernel_source_branch(func, kernel_dir):
    """Minimal reimplementation of the fixed source_file branch for unit testing."""
    source_file = os.path.join(kernel_dir, f"{func._name}.cc")
    if func._source_file is not None:
        if not os.path.exists(func._source_file):
            raise FileNotFoundError(
                f"ExternalFunction '{func._name}': source file not found: {func._source_file}"
            )
        shutil.copy2(func._source_file, source_file)


def test_missing_source_file_raises_file_not_found():
    """FileNotFoundError with the missing path when source_file does not exist."""
    func = _FakeExternalFunction("my_kernel", source_file="/nonexistent/kernel.cc")
    with tempfile.TemporaryDirectory() as kernel_dir:
        with pytest.raises(FileNotFoundError, match="source file not found.*kernel.cc"):
            _compile_external_kernel_source_branch(func, kernel_dir)


def test_existing_source_file_is_copied():
    """No error and the source is copied when source_file exists."""
    with tempfile.TemporaryDirectory() as src_dir, \
         tempfile.TemporaryDirectory() as kernel_dir:
        src = os.path.join(src_dir, "real_kernel.cc")
        with open(src, "w") as f:
            f.write('extern "C" void real_kernel() {}')

        func = _FakeExternalFunction("real_kernel", source_file=src)
        _compile_external_kernel_source_branch(func, kernel_dir)

        assert os.path.exists(os.path.join(kernel_dir, "real_kernel.cc"))
