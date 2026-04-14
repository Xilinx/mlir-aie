# test_symbol_prefix.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Unit tests for ExternalFunction symbol_prefix parameter."""

import pytest
from aie.iron.kernel import ExternalFunction


def _make_ef(name, symbol_prefix=None, source_string="void f(){}"):
    return ExternalFunction(
        name,
        source_string=source_string,
        symbol_prefix=symbol_prefix,
    )


def test_symbol_prefix_sets_effective_name():
    ef = _make_ef("mm", symbol_prefix="op_a")
    assert ef._name == "op_a_mm"


def test_symbol_prefix_sets_object_file_name():
    ef = _make_ef("mm", symbol_prefix="op_a")
    assert ef.object_file_name == "op_a_mm.o"


def test_different_prefixes_produce_different_hashes():
    ef_a = _make_ef("mm", symbol_prefix="op_a")
    ef_b = _make_ef("mm", symbol_prefix="op_b")
    assert hash(ef_a) != hash(ef_b)


def test_no_prefix_differs_from_prefixed():
    ef_plain = _make_ef("mm")
    ef_prefixed = _make_ef("mm", symbol_prefix="op_a")
    assert hash(ef_plain) != hash(ef_prefixed)


def test_same_prefix_and_source_produce_equal_hashes():
    source = "void mm(){}"
    ef1 = _make_ef("mm", symbol_prefix="op_a", source_string=source)
    ef2 = _make_ef("mm", symbol_prefix="op_a", source_string=source)
    assert hash(ef1) == hash(ef2)


def test_no_prefix_preserves_original_name_in_name():
    ef = _make_ef("mm")
    assert ef._name == "mm"
    assert ef.object_file_name == "mm.o"


def test_original_name_stored_when_prefix_set():
    ef = _make_ef("mm", symbol_prefix="op_a")
    assert ef._original_name == "mm"


def test_original_name_stored_when_no_prefix():
    ef = _make_ef("mm")
    assert ef._original_name == "mm"
