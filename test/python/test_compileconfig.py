# test_compileconfig.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Unit tests for the @compileconfig decorator — no NPU required."""

from __future__ import annotations

import functools
from pathlib import Path

import pytest

from aie.iron.compile.compilabledesign import CompilableDesign
from aie.iron.compile.compileconfig import compileconfig
from aie.iron.compile.markers import Compile, In, Out

# ---------------------------------------------------------------------------
# Bare decorator: @compileconfig (no parentheses)
# ---------------------------------------------------------------------------


def test_bare_decorator_returns_compilable_design():
    @compileconfig
    def gen(a: In, M: Compile[int]):
        pass

    assert isinstance(gen, CompilableDesign)


def test_bare_decorator_default_use_cache_is_true():
    @compileconfig
    def gen(a: In):
        pass

    assert gen.use_cache is True


def test_bare_decorator_default_flags_are_empty():
    @compileconfig
    def gen(a: In):
        pass

    assert gen.compile_flags == []
    assert gen.aiecc_flags == []
    assert gen.source_files == []
    assert gen.include_paths == []
    assert gen.object_files == []


def test_bare_decorator_does_not_bind_compile_kwargs():
    """@compileconfig must not pre-bind compile_kwargs — those come from jit/CompilableDesign."""

    @compileconfig
    def gen(a: In, M: Compile[int]):
        pass

    assert gen.compile_kwargs == {}


def test_bare_decorator_preserves_generator():
    @compileconfig
    def my_generator(a: In, M: Compile[int]):
        pass

    assert my_generator.mlir_generator.__name__ == "my_generator"


# ---------------------------------------------------------------------------
# Keyword-argument decorator: @compileconfig(...)
# ---------------------------------------------------------------------------


def test_kwargs_decorator_returns_compilable_design():
    @compileconfig(use_cache=False)
    def gen(a: In):
        pass

    assert isinstance(gen, CompilableDesign)


def test_kwargs_decorator_propagates_use_cache_false():
    @compileconfig(use_cache=False)
    def gen(a: In):
        pass

    assert gen.use_cache is False


def test_kwargs_decorator_propagates_source_files():
    @compileconfig(source_files=["kernel.cc", "helper.cc"])
    def gen(a: In):
        pass

    names = [sf.name for sf in gen.source_files]
    assert "kernel.cc" in names
    assert "helper.cc" in names


def test_kwargs_decorator_propagates_aiecc_flags():
    @compileconfig(aiecc_flags=["--verbose", "--no-xchesscc"])
    def gen(a: In):
        pass

    assert "--verbose" in gen.aiecc_flags
    assert "--no-xchesscc" in gen.aiecc_flags


def test_kwargs_decorator_propagates_compile_flags():
    @compileconfig(compile_flags=["-O3", "-DNDEBUG"])
    def gen(a: In):
        pass

    assert "-O3" in gen.compile_flags
    assert "-DNDEBUG" in gen.compile_flags


def test_kwargs_decorator_propagates_include_paths():
    @compileconfig(include_paths=["/opt/aie/include", "/usr/local/include"])
    def gen(a: In):
        pass

    path_strs = [str(p) for p in gen.include_paths]
    assert any("/opt/aie/include" in s for s in path_strs)


def test_kwargs_decorator_propagates_object_files():
    @compileconfig(object_files=["add.o", "mul.o"])
    def gen(a: In):
        pass

    names = [of.name for of in gen.object_files]
    assert "add.o" in names
    assert "mul.o" in names


def test_kwargs_decorator_all_options_together():
    @compileconfig(
        use_cache=False,
        source_files=["k.cc"],
        aiecc_flags=["--verbose"],
        compile_flags=["-O2"],
        include_paths=["/inc"],
        object_files=["a.o"],
    )
    def gen(a: In, M: Compile[int]):
        pass

    assert gen.use_cache is False
    assert gen.source_files[0].name == "k.cc"
    assert "--verbose" in gen.aiecc_flags
    assert "-O2" in gen.compile_flags
    assert any("/inc" in str(p) for p in gen.include_paths)
    assert gen.object_files[0].name == "a.o"


# ---------------------------------------------------------------------------
# Regression: functools.partial bug fix
# The original erika-vibe-coding code called functools.partial without
# providing a callable as its first argument.  We verify:
#   1. compileconfig(use_cache=False) returns a callable (partial),
#      not a CompilableDesign.
#   2. Applying that callable to a function produces a CompilableDesign.
# ---------------------------------------------------------------------------


def test_partial_application_is_callable():
    decorator = compileconfig(use_cache=False)
    assert callable(decorator)


def test_partial_application_is_not_compilable_design():
    decorator = compileconfig(use_cache=False)
    assert not isinstance(decorator, CompilableDesign)


def test_partial_application_produces_compilable_design_when_called():
    decorator = compileconfig(use_cache=False)

    def my_gen(a: In, M: Compile[int]):
        pass

    result = decorator(my_gen)
    assert isinstance(result, CompilableDesign)


def test_partial_preserves_config_through_application():
    decorator = compileconfig(use_cache=False, aiecc_flags=["--verbose"])

    def gen(a: In):
        pass

    result = decorator(gen)
    assert result.use_cache is False
    assert "--verbose" in result.aiecc_flags


def test_partial_can_be_used_multiple_times():
    """A partial decorator must be reusable across multiple generator functions."""
    decorator = compileconfig(use_cache=False)

    def gen_a(a: In):
        pass

    def gen_b(b: In):
        pass

    result_a = decorator(gen_a)
    result_b = decorator(gen_b)
    assert isinstance(result_a, CompilableDesign)
    assert isinstance(result_b, CompilableDesign)
    # Each is a separate CompilableDesign wrapping its own generator.
    assert result_a.mlir_generator is gen_a
    assert result_b.mlir_generator is gen_b


# ---------------------------------------------------------------------------
# Source files: list vs. tuple vs. Path objects
# ---------------------------------------------------------------------------


def test_source_files_as_paths():
    @compileconfig(source_files=[Path("kernel.cc")])
    def gen(a: In):
        pass

    assert gen.source_files[0] == Path("kernel.cc")


def test_source_files_as_tuple():
    @compileconfig(source_files=("kernel.cc",))
    def gen(a: In):
        pass

    assert gen.source_files[0].name == "kernel.cc"


def test_source_files_empty_list():
    @compileconfig(source_files=[])
    def gen(a: In):
        pass

    assert gen.source_files == []


# ---------------------------------------------------------------------------
# Interaction: @compileconfig + CompilableDesign(compile_kwargs=...)
# ---------------------------------------------------------------------------


def test_compileconfig_design_accepts_compile_kwargs_later():
    """A CompilableDesign from @compileconfig can receive compile_kwargs
    by constructing a new CompilableDesign with the generator."""

    @compileconfig(use_cache=False)
    def gemm_design(a: In, b: In, c: Out, M: Compile[int], N: Compile[int]):
        pass

    # The @compileconfig result is itself a CompilableDesign.  To bind
    # compile_kwargs, create a new one from the underlying generator.
    bound = CompilableDesign(
        gemm_design.mlir_generator,
        compile_kwargs={"M": 512, "N": 512},
        use_cache=False,
    )
    assert bound.compile_kwargs == {"M": 512, "N": 512}
    assert bound.use_cache is False


# ---------------------------------------------------------------------------
# Keyword-only enforcement: positional misuse raises TypeError
# ---------------------------------------------------------------------------


def test_compileconfig_keyword_only_enforcement():
    """All config options are keyword-only; positional use raises TypeError."""
    with pytest.raises(TypeError):
        # Passing True positionally (where the function expects mlir_generator=None)
        # to a kwarg-only param should fail.
        compileconfig(None, True)  # True is positional for 'use_cache'
