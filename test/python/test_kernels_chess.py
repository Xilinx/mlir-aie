# test_kernels_chess.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""use_chess opt-in plumbing + emulate_bf16_mmul_with_bfp16 macro plumbing
for the aie.iron.kernels factory functions.

Sibling files:
  test_kernels_specs.py        — spec-table-driven coverage of every factory
  test_kernels_memoization.py  — memoization, independent zero, auto-prefix-on-collision

The npu2_device fixture used by the bf16-emulated tests comes from
conftest.py at this directory level.
"""

from pathlib import Path

import numpy as np
import pytest
from ml_dtypes import bfloat16

from aie.dialects.aie import AIEDevice, device
from aie.extras.context import mlir_mod_ctx
from aie.iron import kernels
from aie.iron.kernel import ExternalFunction, Kernel
from aie.utils.compile.jit.compilabledesign import _compute_hash

# ---------------------------------------------------------------------------
# use_chess=True opt-in plumbing through kernels.X → _make_extern →
# ExternalFunction → JIT compile orchestration (chess infra PR).
#
# These tests exercise only the Python plumbing — the actual xchesscc_wrapper
# invocation is hardware-verified manually (see the chess infra plan).
# ---------------------------------------------------------------------------


def test_kernels_mm_use_chess_carries_flag():
    """kernels.mm(use_chess=True) propagates the flag onto the ExternalFunction.

    The JIT compile orchestration reads ``ef._use_chess`` to decide between
    invoking ``xchesscc_wrapper`` and ``clang++`` (and to pick the matching
    aiecc front-end).  If the flag doesn't make it onto the EF, the kernel
    silently builds with peano even when the user asked for chess.
    """
    ef_chess = kernels.mm(
        dim_m=64,
        dim_k=64,
        dim_n=32,
        input_dtype=np.int16,
        output_dtype=np.int16,
        use_chess=True,
    )
    ef_peano = kernels.mm(
        dim_m=64,
        dim_k=64,
        dim_n=32,
        input_dtype=np.int16,
        output_dtype=np.int16,
        use_chess=False,
    )
    assert ef_chess._use_chess is True
    assert ef_peano._use_chess is False


def test_kernels_mm_default_use_chess_is_false():
    """Omitting use_chess defaults to peano — chess must be opt-in."""
    ef = kernels.mm(
        dim_m=64, dim_k=64, dim_n=32, input_dtype=np.int16, output_dtype=np.int16
    )
    assert ef._use_chess is False


@pytest.mark.parametrize("arch", ["aie2", "aie2p"])
@pytest.mark.parametrize("use_chess", [False, True])
@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize(
    "input_dtype,output_dtype,suffix,zero_suffix",
    [
        (np.int8, np.int8, "i8_i8", "i8"),
        (np.int8, np.int16, "i8_i16", "i16"),
        (np.int8, np.int32, "i8_i32", "i32"),
        (np.int16, np.int16, "i16_i16", "i16"),
        (np.int16, np.int32, "i16_i32", "i32"),
        (bfloat16, bfloat16, "bf16_bf16", "bf16"),
        (bfloat16, np.float32, "bf16_f32", "f32"),
    ],
)
def test_kernels_mm_selects_combo_with_independent_zero(
    monkeypatch,
    arch,
    use_chess,
    vectorized,
    input_dtype,
    output_dtype,
    suffix,
    zero_suffix,
):
    monkeypatch.setattr(kernels.linalg, "_detect_arch", lambda: arch)
    ef = kernels.mm(
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        vectorized=vectorized,
        use_chess=use_chess,
    )
    assert [flag for flag in ef._compile_flags if flag.endswith("_ONLY")] == [
        f"-D{suffix}_ONLY"
    ]
    matmul = "matmul" if vectorized else "matmul_scalar"
    assert ef._name == ef.object_file.resolve_symbol(f"{matmul}_{suffix}")
    zero = ef.contract.initializers[0][1](ef)
    assert zero.use_chess == use_chess
    assert zero.object_file is not ef.object_file
    assert Path(zero.source_file).parts[-2:] == ("zero", "zero.cc")


def test_external_function_rejects_inline_with_chess():
    """Inline IR is a Peano-only path, so reject Chess before compilation."""
    with pytest.raises(ValueError, match="inline=True requires the Peano toolchain"):
        ExternalFunction(
            "inline_chess",
            source_string='extern "C" void inline_chess() {}',
            inline=True,
            use_chess=True,
        )


def test_kernels_mm_chess_distinct_digest_from_peano():
    """Same params + different toolchain → different content digest.

    Without this distinction, the per-EF .o cache would treat chess- and
    peano-built objects as interchangeable; the second compile would skip
    rebuilding even though the toolchain changed, and the wrong .o would
    end up linked into the xclbin.
    """
    ef_chess = kernels.mm(dim_m=64, dim_k=64, dim_n=32, use_chess=True)
    ef_peano = kernels.mm(dim_m=64, dim_k=64, dim_n=32, use_chess=False)
    assert ef_chess._content_digest() != ef_peano._content_digest()


def test_kernels_mm_chess_distinct_object_file_from_peano():
    """Distinct digest → distinct object_file_name (same disambiguation channel
    used for compile_flags / b_col_maj / c_col_maj / etc.)."""
    ef_chess = kernels.mm(dim_m=64, dim_k=64, dim_n=32, use_chess=True)
    ef_peano = kernels.mm(dim_m=64, dim_k=64, dim_n=32, use_chess=False)
    assert ef_chess.object_file_name != ef_peano.object_file_name


def test_kernels_mm_chess_and_peano_are_different_kernels():
    """A kernel's identity includes use_chess: chess and peano variants of the
    same shape/dtype are distinct ExternalFunctions (not aliased)."""
    ef_chess = kernels.mm(dim_m=64, dim_k=64, dim_n=32, use_chess=True)
    ef_peano = kernels.mm(dim_m=64, dim_k=64, dim_n=32, use_chess=False)
    assert ef_chess != ef_peano


def test_kernels_mm_chess_same_params_share_one_object():
    """Two identical kernels.mm(use_chess=True) calls return equal kernels that
    share one object."""
    ef1 = kernels.mm(dim_m=64, dim_k=64, dim_n=32, use_chess=True)
    ef2 = kernels.mm(dim_m=64, dim_k=64, dim_n=32, use_chess=True)
    assert ef1 == ef2
    assert ef1.object_file is ef2.object_file


def test_kernels_mm_chess_kernel_keeps_bare_symbol():
    """A chess kernel never carries a symbol_prefix: the prefix is applied via
    llvm-objcopy, which corrupts xchesscc-produced objects.  It keeps the bare
    symbol baked into its .cc.  (Its .o filename still gets the deterministic
    digest suffix so distinct parameterizations don't share a file.)"""
    ef = kernels.mm(dim_m=64, dim_k=64, dim_n=32, use_chess=True)
    assert ef._symbol_prefix is None
    assert ef._name == ef._original_name == "matmul_i16_i16"
    assert ef.object_file_name.endswith(".o")
    assert "matmul_i16_i16" in ef.object_file_name


def test_kernels_mm_two_chess_variants_in_one_design_raise():
    """Two different parameterizations of the same chess kernel name cannot be
    disambiguated (chess .o can't be symbol-renamed), so declaring both in one
    design raises rather than silently colliding on the same exported symbol."""
    row_major, col_major = (
        kernels.mm(dim_m=64, dim_k=64, dim_n=32, c_col_maj=c_col_maj, use_chess=True)
        for c_col_maj in (False, True)
    )

    def body():
        row_major.resolve()
        col_major.resolve()

    with mlir_mod_ctx():
        with pytest.raises(ValueError, match="'matmul_i16_i16' conflicts.*link_with"):
            device(AIEDevice.npu2_1col)(body)


def test_kernels_mm_two_chess_variants_in_separate_designs_are_fine():
    """The conflict is per design: each variant alone declares cleanly."""
    for c_col_maj in (False, True):
        kernel = kernels.mm(
            dim_m=64, dim_k=64, dim_n=32, c_col_maj=c_col_maj, use_chess=True
        )
        with mlir_mod_ctx() as ctx:
            device(AIEDevice.npu2_1col)(lambda: kernel.resolve())
        assert "@matmul_i16_i16" in str(ctx.module)


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        ("mv", dict(dim_m=32, dim_k=32)),
        (
            "cascade_mm",
            dict(
                dim_m=64,
                dim_k=64,
                dim_n=32,
                input_dtype=np.int16,
                output_dtype=np.int16,
            ),
        ),
    ],
)
def test_other_matmul_factories_carry_use_chess(factory, kwargs):
    """kernels.mv and kernels.cascade_mm also forward use_chess to the EF."""
    ef = getattr(kernels, factory)(**kwargs, use_chess=True)
    assert ef._use_chess is True


def test_cascade_mm_binds_modes_explicitly():
    ef = kernels.cascade_mm(dim_m=64, dim_k=64, dim_n=32)
    for mode in ("put_only", "put_get"):
        symbol = f"matmul_scalar_cascade_{mode}_i16_i16"
        sibling = ef.object_file.bind(symbol, ef.arg_types())
        assert isinstance(sibling, Kernel)
        assert sibling.name == ef.object_file.resolve_symbol(symbol)
        assert sibling.object_file_name == ef.object_file_name
        assert sibling.object_file is ef.object_file


def test_compute_hash_distinguishes_use_chess_literal():
    """A generator that captures ``use_chess=True`` as a literal must hash
    differently from one that captures ``use_chess=False``.

    The ``_compute_hash`` function runs BEFORE the generator is invoked, so
    it can't read ``ef._use_chess`` directly — it relies on bytecode +
    co_consts for any literal constants the generator carries.  This test
    pins that behaviour: if Python ever stops emitting the bool literal into
    co_consts, the design-level cache would silently alias chess and peano
    builds at the same ``final.xclbin`` and one would overwrite the other
    on cache miss.

    Closure-captured / global-captured ``use_chess`` is a known limitation
    (same as for any other closure-captured constant); we don't pin it.
    """

    def gen_chess():
        return kernels.mm(dim_m=64, dim_k=64, dim_n=32, use_chess=True)

    def gen_peano():
        return kernels.mm(dim_m=64, dim_k=64, dim_n=32, use_chess=False)

    h_chess = _compute_hash(gen_chess, {}, [], [], [], [])
    h_peano = _compute_hash(gen_peano, {}, [], [], [], [])
    assert h_chess != h_peano


def test_mixed_chess_peano_set_is_detectable():
    """The orchestration's mixed-mode detection (in CompilableDesign.compile)
    keys on ``{getattr(f, "_use_chess", False) for f in external_kernels}``;
    a set of size > 1 means mixed and must trigger the RuntimeError.

    This pins the contract the EFs hold up (each carries ``_use_chess``) so
    that the orchestration's set comprehension actually produces > 1 element
    when peano + chess EFs are mixed.  The full RuntimeError raise path is
    hardware-verified via the chess JIT integration test, not here — it'd
    require iron device setup and would invoke aiecc.
    """
    ef_chess = kernels.mm(dim_m=64, dim_k=64, dim_n=32, use_chess=True)
    ef_peano = kernels.mm(dim_m=64, dim_k=64, dim_n=32, use_chess=False)
    chess_uses = {getattr(f, "_use_chess", False) for f in (ef_chess, ef_peano)}
    assert len(chess_uses) == 2  # mixed → orchestration would raise


def test_all_chess_set_is_unanimous():
    """Inverse of the mixed-mode test: an all-chess design has chess_uses == {True}."""
    ef1 = kernels.mm(dim_m=64, dim_k=64, dim_n=32, use_chess=True)
    ef2 = kernels.mv(dim_m=32, dim_k=32, use_chess=True)
    chess_uses = {getattr(f, "_use_chess", False) for f in (ef1, ef2)}
    assert chess_uses == {True}


# ---------------------------------------------------------------------------
# emulate_bf16_mmul_with_bfp16 toggle — adds an AIE_API macro on aie2p+bf16
# that picks the higher-throughput (8,8,8) mac dims at the cost of accuracy.
# ---------------------------------------------------------------------------


def test_kernels_mm_emulated_bf16_carries_macro(npu2_device):
    """emulate_bf16_mmul_with_bfp16=True adds the AIE_API macro on aie2p+bf16."""
    ef = kernels.mm(
        dim_m=64,
        dim_k=64,
        dim_n=32,
        input_dtype=bfloat16,
        output_dtype=bfloat16,
        emulate_bf16_mmul_with_bfp16=True,
    )
    assert "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16" in ef._compile_flags
    assert ef.mac_dims == (8, 8, 8)


def test_kernels_mm_emulated_bf16_default_off(npu2_device):
    """Default mac_dims for aie2p bf16/bf16 stays at (4, 8, 8) without the toggle."""
    ef = kernels.mm(
        dim_m=64,
        dim_k=64,
        dim_n=32,
        input_dtype=bfloat16,
        output_dtype=bfloat16,
    )
    assert "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16" not in ef._compile_flags
    assert ef.mac_dims == (4, 8, 8)


def test_kernels_mm_emulated_bf16_ignored_for_non_bf16(npu2_device):
    """The toggle is a no-op for integer dtypes (it's bf16-specific)."""
    ef = kernels.mm(
        dim_m=64,
        dim_k=64,
        dim_n=32,
        input_dtype=np.int16,
        output_dtype=np.int16,
        emulate_bf16_mmul_with_bfp16=True,
    )
    assert "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16" not in ef._compile_flags
    assert ef.mac_dims == (4, 4, 8)  # default aie2p i16/i16


def test_kernels_mm_emulated_bf16_distinct_cache_from_default(npu2_device):
    """The toggle changes the .o contents; cache must distinguish the two."""
    ef_default = kernels.mm(
        dim_m=64,
        dim_k=64,
        dim_n=32,
        input_dtype=bfloat16,
        output_dtype=bfloat16,
    )
    ef_emulated = kernels.mm(
        dim_m=64,
        dim_k=64,
        dim_n=32,
        input_dtype=bfloat16,
        output_dtype=bfloat16,
        emulate_bf16_mmul_with_bfp16=True,
    )
    assert ef_default != ef_emulated
    assert ef_default.object_file_name != ef_emulated.object_file_name
