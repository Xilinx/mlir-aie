# test_kernels_memoization.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# RUN: %pytest %s
"""Memoization, collision protection, .zero attribute, and auto-prefix
on symbol collision for the aie.iron.kernels factory functions.

Sibling files:
  test_kernels_specs.py   — spec-table-driven coverage of every factory
  test_kernels_chess.py   — use_chess + emulated bf16 plumbing
"""

import numpy as np
import pytest

from aie.iron import kernels
from aie.iron.kernel import ExternalFunction, Kernel

# ---------------------------------------------------------------------------
# Memoization + collision protection (kernels.X helpers and ExternalFunction)
#
# Background: a real bug caught while porting whole_array — calling
# kernels.mm() twice with different parameters produced two ExternalFunctions
# with the same default `<name>.o` filename.  Both were registered with the
# JIT, both got compiled, and the .o files silently overwrote each other.
# The wrong-flag kernel won the race, producing wrong hardware output.
#
# These tests pin down the two defenses:
#   (A) _make_extern memoizes on the full input parameter set, so identical
#       helper calls return the SAME instance.  Different parameterizations
#       get distinct instances AND distinct object_file_names (auto-suffixed
#       with a digest of compile_flags).
#   (B) ExternalFunction.__init__ refuses to register two instances with
#       the same (name, object_file_name) but a different content digest —
#       a backstop for code that bypasses the helper (constructs
#       ExternalFunction directly).
# ---------------------------------------------------------------------------


def test_kernels_mm_memoized_same_params_returns_same_instance():
    """Defense A: identical kernels.mm() calls return the exact same instance."""
    ef1 = kernels.mm(
        dim_m=64, dim_k=64, dim_n=32, input_dtype=np.int16, output_dtype=np.int16
    )
    ef2 = kernels.mm(
        dim_m=64, dim_k=64, dim_n=32, input_dtype=np.int16, output_dtype=np.int16
    )
    assert ef1 is ef2


def test_kernels_mm_different_params_returns_different_instances():
    """Defense A: different params get distinct instances (no spurious sharing)."""
    ef_plain = kernels.mm(dim_m=64, dim_k=64, dim_n=32, c_col_maj=False)
    ef_ccm = kernels.mm(dim_m=64, dim_k=64, dim_n=32, c_col_maj=True)
    assert ef_plain is not ef_ccm


def test_kernels_mm_different_params_have_distinct_object_files():
    """Defense A: distinct parameterizations get distinct object_file_names so
    JIT compilation outputs cannot overwrite each other on disk.
    This is the precise bug that broke whole_array c_col_maj support."""
    ef_plain = kernels.mm(dim_m=64, dim_k=64, dim_n=32, c_col_maj=False)
    ef_ccm = kernels.mm(dim_m=64, dim_k=64, dim_n=32, c_col_maj=True)
    assert ef_plain.object_file_name != ef_ccm.object_file_name


def test_kernels_mm_b_and_c_col_maj_independent():
    """Defense A: (b_col_maj, c_col_maj) tuples produce four distinct instances."""
    instances = {
        (b, c): kernels.mm(dim_m=64, dim_k=64, dim_n=32, b_col_maj=b, c_col_maj=c)
        for b in (False, True)
        for c in (False, True)
    }
    # All four are distinct ExternalFunction instances
    assert len({id(v) for v in instances.values()}) == 4
    # And all four have distinct .o filenames
    assert len({v.object_file_name for v in instances.values()}) == 4


def test_external_function_collision_check_fires():
    """Defense B: directly constructing two ExternalFunctions with the same
    (name, object_file_name) but different compile_flags is rejected."""
    # Use a unique source_string so we don't collide with real kernels.X
    # registrations from other tests in this session.
    src = "/* dummy */ int sentinel_for_collision_test() { return 0; }"
    name = "sentinel_for_collision_test"
    obj = "sentinel_for_collision_test.o"
    ExternalFunction(
        name=name,
        object_file_name=obj,
        source_string=src,
        arg_types=[],
        compile_flags=["-DA"],
    )
    with pytest.raises(ValueError, match="would collide"):
        ExternalFunction(
            name=name,
            object_file_name=obj,
            source_string=src,
            arg_types=[],
            compile_flags=["-DB"],  # different flags → different content digest
        )


def test_external_function_collision_check_allows_identical_redeclaration():
    """Defense B: re-registering an EXACT duplicate is fine (set semantics).

    Only differing-content collisions are rejected; identical re-instantiation
    might happen if a helper happens to be called twice without the memoization
    layer (e.g. by tests).  It must not raise.
    """
    src = "/* dummy */ int sentinel_redeclare_ok() { return 0; }"
    name = "sentinel_redeclare_ok"
    obj = "sentinel_redeclare_ok.o"
    ExternalFunction(
        name=name,
        object_file_name=obj,
        source_string=src,
        arg_types=[],
        compile_flags=["-DA"],
    )
    # Same content → same digest → no collision.
    ExternalFunction(
        name=name,
        object_file_name=obj,
        source_string=src,
        arg_types=[],
        compile_flags=["-DA"],
    )


# ---------------------------------------------------------------------------
# kernels.mm and kernels.mv expose a `.zero` Kernel attribute pointing at
# the same .o file (mm.cc / mv.cc emit both matmul_*/matvec_* and zero_*
# symbols natively).  Designs use `matmul = kernels.mm(...); zero = matmul.zero`
# — one mm.cc compile, both bindings, no duplicate-symbol footgun.
# ---------------------------------------------------------------------------


def test_mm_zero_attribute_is_kernel():
    """kernels.mm(...).zero is a Kernel binding the zero symbol."""
    ef = kernels.mm(
        dim_m=64, dim_k=64, dim_n=32, input_dtype=np.int16, output_dtype=np.int16
    )
    assert isinstance(ef.zero, Kernel)
    assert ef.zero._name == "zero_i16"


def test_mm_zero_attribute_shares_object_file():
    """ef.zero must point at the same .o the mm ExternalFunction will produce —
    that's the whole point of the attribute pattern (one compile, two bindings)."""
    ef = kernels.mm(
        dim_m=64, dim_k=64, dim_n=32, input_dtype=np.int16, output_dtype=np.int16
    )
    assert ef.zero.object_file_name == ef.object_file_name


def test_mm_zero_attribute_arg_count():
    """zero kernel takes one arg (the output buffer to zero)."""
    ef = kernels.mm(
        dim_m=64, dim_k=64, dim_n=32, input_dtype=np.int16, output_dtype=np.int16
    )
    assert len(ef.zero._arg_types) == 1


def test_mm_zero_attribute_scalar_variant():
    """vectorized=False picks zero_scalar_* instead of zero_*."""
    ef = kernels.mm(
        dim_m=64,
        dim_k=64,
        dim_n=32,
        input_dtype=np.int16,
        output_dtype=np.int16,
        vectorized=False,
    )
    assert ef.zero._name == "zero_scalar_i16"


def test_mv_zero_attribute_is_kernel():
    """kernels.mv(...).zero is a Kernel binding the zero symbol against the
    same mv.cc-built .o."""
    ef = kernels.mv(dim_m=32, dim_k=32, vectorized=False)
    assert isinstance(ef.zero, Kernel)
    assert ef.zero._name == "zero_scalar_i32"
    assert ef.zero.object_file_name == ef.object_file_name


def test_mm_no_longer_carries_only_flags():
    """Sanity: dropping the MATMUL_ONLY/ZERO_ONLY gating means kernels.mm
    no longer adds those flags to its compile_flags."""
    ef = kernels.mm(
        dim_m=64, dim_k=64, dim_n=32, input_dtype=np.int16, output_dtype=np.int16
    )
    assert "-DMATMUL_ONLY" not in ef._compile_flags
    assert "-DZERO_ONLY" not in ef._compile_flags


# ---------------------------------------------------------------------------
# Auto-prefix on symbol collision (Defense C — see _make_extern in
# python/iron/kernels/_common.py).
#
# Two kernels.X() calls with different parameterizations would otherwise
# produce two ExternalFunctions with the same C symbol name and trip
# duplicate-symbol errors at MLIR-verify and link time.  _make_extern now
# auto-prefixes the second-and-later instance's symbol with the same digest
# already used for object_file_name disambiguation.  The first call keeps
# the unprefixed canonical name (preserves byte-identity for existing
# single-version designs).
# ---------------------------------------------------------------------------


def test_kernels_mm_parameterized_symbol_is_digest_prefixed():
    """A parameterized variant's symbol is digest-prefixed and the digest is a
    pure function of the kernel's identity — never of registration order, so a
    separate build (e.g. the full-model .o cache vs. a per-block design)
    computes the SAME symbol and .o filename and can reuse the object."""
    ef = kernels.mm(
        dim_m=64, dim_k=64, dim_n=32, input_dtype=np.int16, output_dtype=np.int16
    )
    assert ef._original_name == "matmul_i16_i16"
    assert ef._symbol_prefix is not None and len(ef._symbol_prefix) == 8
    assert ef._name == f"{ef._symbol_prefix}_matmul_i16_i16"


def test_kernels_mm_two_versions_get_distinct_symbol_names():
    """Two parameterizations get two distinct digest-prefixed symbols so MLIR +
    linker don't see two `func.func @matmul_i16_i16` declarations."""
    ef1 = kernels.mm(dim_m=64, dim_k=64, dim_n=32, c_col_maj=False)
    ef2 = kernels.mm(dim_m=64, dim_k=64, dim_n=32, c_col_maj=True)
    assert ef1._name != ef2._name
    # Both are digest-prefixed; the prefix is order-independent.
    for ef in (ef1, ef2):
        assert ef._symbol_prefix is not None and len(ef._symbol_prefix) == 8
        assert ef._name == f"{ef._symbol_prefix}_matmul_i16_i16"


def test_kernels_mm_three_versions_all_distinct_names():
    """Three distinct parameterizations get three distinct effective names,
    each digest-prefixed."""
    efs = [
        kernels.mm(dim_m=64, dim_k=64, dim_n=32, b_col_maj=False, c_col_maj=False),
        kernels.mm(dim_m=64, dim_k=64, dim_n=32, b_col_maj=True, c_col_maj=False),
        kernels.mm(dim_m=64, dim_k=64, dim_n=32, b_col_maj=False, c_col_maj=True),
    ]
    names = [e._name for e in efs]
    assert len(set(names)) == 3
    assert all(e._symbol_prefix is not None for e in efs)


def test_kernels_mm_object_file_name_is_order_independent():
    """object_file_name is the suffix form ``{name}_{digest}.o`` and embeds the
    same digest as the symbol prefix — so the on-disk filename a separate build
    reuses by name is identical regardless of registration order."""
    ef = kernels.mm(dim_m=64, dim_k=64, dim_n=32, c_col_maj=True)
    assert ef._object_file_name == f"matmul_i16_i16_{ef._symbol_prefix}.o"
