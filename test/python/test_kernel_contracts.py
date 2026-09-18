# test_kernel_contracts.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Every kernel contract agrees with its factory, and builds a design (no NPU).

Three things a contract can only get right by matching the real kernel, each
of which is silent until a device run otherwise:

* the role list has one entry per ``arg_types()`` entry (a wrong arity is a
  compile error deep in a nightly);
* the reference takes exactly the arguments the contract hands it;
* the generic harness can lower a design for it to MLIR -- fifo types, call
  arity, DMA-padded output tiles, matmul layouts -- which executes the same
  generator the device tests and benchmarks run.
"""

import inspect
import re
from pathlib import Path

import numpy as np
import pytest
from aie.iron import In, InOut, Out, Scalar, kernels
from aie.iron.algorithms import kernel_design as kd
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.kernels import ROLES, KernelContract
from aie.utils import bfp
from aie.utils.hostruntime import set_current_device
from aie.utils.verify import Tolerance, compare
from ml_dtypes import bfloat16

# One build per factory, on the default kwargs unless a non-default variant
# is worth pinning. Matrix kernels carry the host shape they are checked at.
CASES = {
    "passthrough": ({}, dict(calls=4)),
    "passthrough/int16": (dict(dtype=np.int16), dict(calls=4)),
    "scale/int16": (dict(dtype=np.int16), dict(calls=4)),
    "scale/int32": (dict(dtype=np.int32), dict(calls=4)),
    "add": ({}, dict(calls=4)),
    "mul": ({}, dict(calls=4)),
    "relu": ({}, dict(calls=4)),
    "reduce_add": ({}, dict(calls=4)),
    "reduce_min": ({}, dict(calls=4)),
    "reduce_max": ({}, dict(calls=4)),
    "reduce_max/bf16": (dict(dtype=bfloat16), dict(calls=4)),
    "gelu": ({}, dict(calls=2)),
    "silu": ({}, dict(calls=2)),
    "bf16_exp": ({}, dict(calls=2)),
    "tanh": ({}, dict(calls=2)),
    "sigmoid": ({}, dict(calls=2)),
    "softmax": ({}, dict(calls=2)),
    "leaky_relu": ({}, dict(calls=2, scalars=(0.5,))),
    "exp2f_vec": ({}, dict(calls=2)),
    "axpy": ({}, dict(calls=2, scalars=(2.5,))),
    "convert_copy": ({}, dict(calls=2)),
    "expand": ({}, dict(calls=2)),
    "transpose/4": (dict(subtile=4), dict(calls=2)),
    "transpose/8": (dict(subtile=8), dict(calls=2)),
    "transpose/uint8": (dict(subtile=4, dtype=np.uint8), dict(calls=2)),
    "transpose/uint32": (dict(subtile=8, dtype=np.uint32), dict(calls=2)),
    "mm/bf16_f32": (
        dict(
            dim_m=64, dim_k=32, dim_n=64, input_dtype=bfloat16, output_dtype=np.float32
        ),
        dict(calls=4),
    ),
    # One tile row, and an odd number of tile rows: the C drain groups
    # two rows when it can and one otherwise.
    "mm/bf16_f32/single-tile": (
        dict(
            dim_m=64, dim_k=32, dim_n=64, input_dtype=bfloat16, output_dtype=np.float32
        ),
        dict(calls=1),
    ),
    "mm/bf16_f32/odd-rows": (
        dict(
            dim_m=64, dim_k=32, dim_n=64, input_dtype=bfloat16, output_dtype=np.float32
        ),
        dict(calls=3),
    ),
    "mm/i16_i32": (
        dict(dim_m=64, dim_k=32, dim_n=64, input_dtype=np.int16, output_dtype=np.int32),
        dict(calls=4),
    ),
    "mm/bf16_f32/b_col_maj": (
        dict(
            dim_m=64,
            dim_k=32,
            dim_n=64,
            input_dtype=bfloat16,
            output_dtype=np.float32,
            b_col_maj=True,
        ),
        dict(calls=3),
    ),
    "mm/bf16_f32/c_col_maj": (
        dict(
            dim_m=64,
            dim_k=32,
            dim_n=64,
            input_dtype=bfloat16,
            output_dtype=np.float32,
            c_col_maj=True,
        ),
        dict(calls=4),
    ),
    "mm/i16_i32/both_col_maj": (
        dict(
            dim_m=64,
            dim_k=32,
            dim_n=64,
            input_dtype=np.int16,
            output_dtype=np.int32,
            b_col_maj=True,
            c_col_maj=True,
        ),
        dict(calls=3),
    ),
    "mm/i8_i32": (
        dict(dim_m=64, dim_k=32, dim_n=64, input_dtype=np.int8, output_dtype=np.int32),
        dict(calls=4),
    ),
    "mv": (dict(dim_m=32, dim_k=32), dict(calls=4)),
    "mv/bf16": (
        dict(dim_m=32, dim_k=256, input_dtype=bfloat16, output_dtype=bfloat16),
        dict(calls=2),
    ),
    "mm_bfp": (dict(dim_m=64, dim_k=64, dim_n=64), dict(calls=4)),
    "q4nx_dequant": ({}, dict(calls=4)),
    "q4nx_dequant/group_crosses_slice": (
        dict(m_tile=48, k_tile=48, group=24, ct_k=16),
        dict(calls=2),
    ),
    "mm_bfp/mixed": (
        dict(dim_m=64, dim_k=64, dim_n=64, mixed=True),
        dict(calls=4),
    ),
    "compute_max": ({}, dict(calls=4)),
    "compute_max/bf16": (dict(dtype=bfloat16), dict(calls=4)),
    "swiglu": ({}, dict(calls=2)),
    "gray2rgba": ({}, dict(calls=2)),
    "rgba2gray": ({}, dict(calls=2)),
    "threshold": ({}, dict(calls=2, scalars=(100, 255, 0))),
    "threshold/int16": (dict(dtype=np.int16), dict(calls=2, scalars=(100, 255, 1))),
    "bitwise_or": ({}, dict(calls=2)),
    "bitwise_and": ({}, dict(calls=2)),
    "add_weighted": ({}, dict(calls=2, scalars=(8192, 8192, 0))),
    "filter2d": ({}, dict(calls=2)),
    "rgba2hue": ({}, dict(calls=2)),
    "conv2dk1": ({}, dict(calls=2, scalars=(32, 64, 64, 12))),
    "conv2dk1/uint8": (
        dict(act_dtype=np.uint8),
        dict(calls=2, scalars=(32, 64, 64, 12)),
    ),
    "conv2dk3": (
        dict(act_dtype=np.uint8),
        dict(calls=2, scalars=(32, 64, 64, 3, 3, 1, 15, 0)),
    ),
    "conv2dk3/int8": ({}, dict(calls=2, scalars=(32, 64, 64, 3, 3, 1, 15, 0))),
    "conv2dk1_i8": ({}, dict(calls=2, scalars=(32, 64, 64, 12))),
    "conv2dk1_skip/uint8": (
        dict(input_channels=128, output_channels=64, act_dtype=np.uint8),
        dict(calls=2, scalars=(32, 128, 64, 12, 1)),
    ),
    "conv2dk1_skip_init/uint8": (
        dict(input_channels=64, skip_input_channels=32, act_dtype=np.uint8),
        dict(calls=2, scalars=(32, 64, 64, 32, 12, 1, 11)),
    ),
    "conv2dk14": ({}, dict(calls=2, scalars=(224, 4, 16, 14, 17))),
    "bn_conv2dk1_relu": ({}, dict(calls=2, scalars=(32, 64, 64, 12))),
    "bn_conv2dk1_i8": ({}, dict(calls=2, scalars=(32, 64, 64, 13))),
    "bn_conv2dk1_skip": ({}, dict(calls=2, scalars=(32, 64, 64, 13, 1))),
    "bn_conv2dk1_skip/int8": (
        dict(skip_dtype=np.int8),
        dict(calls=2, scalars=(32, 64, 64, 13, 1)),
    ),
    "bn_conv2dk3_dw": ({}, dict(calls=2, scalars=(32, 64, 64, 3, 3, 1, 11, 0))),
    "bn_conv2dk3_dw/stride2": (
        dict(stride=2),
        dict(calls=2, scalars=(32, 64, 64, 3, 3, 1, 11, 0)),
    ),
    "bn_conv2dk3": ({}, dict(calls=2, scalars=(32, 64, 64, 3, 3, 1, 15, 0))),
    "bn_fc_relu_ui16_pad": (
        dict(input_channels=1280, output_channels=16),
        dict(calls=2, scalars=(1, 1280, 1280, 16, 13)),
    ),
    "mul_add": ({}, dict(calls=2, scalars=(1,))),
    "rms_norm": (dict(cols=1024), dict(calls=2)),
    "layer_norm": (dict(cols=1024), dict(calls=2)),
    "layer_norm_f32": (dict(cols=1024), dict(calls=2)),
    "layer_norm_affine_cast": (dict(cols=1024), dict(calls=2)),
    "rope": (dict(cols=1024), dict(calls=2)),
    "mm_activation_epilogue": ({}, dict(calls=2, scalars=(2,))),
    "dwconv1d": (dict(seq_len=1024, kernel_size=9), dict(calls=2, scalars=(1024,))),
}


def _factory(case_id: str):
    return getattr(kernels, case_id.split("/")[0])


@pytest.fixture(autouse=True)
def _aie2p_device():
    # Factories pick sources and mac_dims from the current device; a few
    # (exp2f_vec, convert_copy) exist only for aie2p.
    set_current_device(NPU2Col1())
    yield
    set_current_device(None)


def test_every_case_names_an_exported_factory():
    for case_id in CASES:
        assert callable(_factory(case_id)), case_id


def test_contract_coverage_is_explicit():
    """Every constructible factory carries a contract, including cascade halves."""
    without = []
    for name in kernels.__all__:
        f = getattr(kernels, name)
        if (
            not inspect.isfunction(f)
            or name.endswith("_ref")
            or name == "mm_stream_dims"
        ):
            continue
        try:
            ef = f()
        except Exception:  # a factory that needs kwargs; covered by CASES if in scope
            continue
        if getattr(ef, "contract", None) is None:
            without.append(name)
    assert not without, f"factories without contracts: {without}"


@pytest.mark.parametrize("case_id", list(CASES))
def test_roles_match_arg_types(case_id):
    fkw, _ = CASES[case_id]
    fn = _factory(case_id)(**fkw)
    c = fn.contract
    assert c is not None, f"{case_id}: no contract"
    assert len(c.roles) == len(
        fn.arg_types()
    ), f"{case_id}: {len(c.roles)} roles for {len(fn.arg_types())} arguments"
    assert set(c.roles) <= set(ROLES)


@pytest.mark.parametrize("case_id", list(CASES))
def test_reference_takes_what_the_contract_hands_it(case_id):
    fkw, _ = CASES[case_id]
    c = _factory(case_id)(**fkw).contract
    assert c.reference is not None, f"{case_id}: contract has no reference"
    params = [
        p
        for p in inspect.signature(c.reference).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert len(params) == len(c.reference_indices())


@pytest.mark.parametrize("case_id", list(CASES))
def test_tolerance_is_declared_with_its_evidence(case_id):
    fkw, _ = CASES[case_id]
    tol = _factory(case_id)(**fkw).contract.tolerance
    assert isinstance(
        tol, Tolerance
    ), f"{case_id}: tolerance is the default; declare it"
    assert (
        tol.note
    ), f"{case_id}: tolerance needs a note saying where the number comes from"


@pytest.mark.parametrize("case_id", list(CASES))
def test_harness_lowers_a_design_to_mlir(case_id):
    fkw, opts = CASES[case_id]
    factory = _factory(case_id)
    fn = factory(**fkw)
    ins = kd.sample_inputs(fn, calls=opts.get("calls", 1), shape=opts.get("shape"))
    d = kd.design(factory, params=fn.param_values(ins), **opts, **fkw)
    ref = fn.expected(ins, scalars=opts.get("scalars", ()))
    mlir = d.as_mlir()
    assert "func.call" in str(mlir) or "aie.core" in str(mlir)
    # The reference already has the output dtype the harness will compare
    # in: the kernel's, or float32 for a bfp16ebs8 output that judge decodes.
    out_dt = kd.shape_dtype(fn.arg_types()[fn.contract.out_index])[1]
    assert ref.dtype == (np.float32 if bfp.is_bfp(out_dt) else out_dt)


def test_reduction_reference_yields_one_value_per_call():
    fn = kernels.reduce_max(dtype=bfloat16)
    ins = kd.sample_inputs(fn, calls=4)
    assert fn.expected(ins).shape == (4, 1)
    # The output tile is padded to 2 bf16 for DMA alignment; only 1 is valid.
    assert kd.elems(fn.arg_types()[fn.contract.out_index]) == 2
    assert fn.contract.out_valid == 1


@pytest.mark.parametrize("factory", [kernels.mm, kernels.mv, kernels.mm_bfp])
def test_matrix_design_defaults_to_one_tile(factory):
    fn = factory()
    inputs = kd.sample_inputs(fn)
    ref = fn.expected(inputs)
    assert ref.shape == (1, fn.dims[0] * (fn.dims[2] if len(fn.dims) == 3 else 1))
    assert [a.n_elements for a in kd.host_args(fn)][-1] == kd.output_size(fn)
    assert "func.call" in str(kd.design(factory).as_mlir())


@pytest.mark.parametrize(
    "shape",
    [(0, 32, 64), (-64, 32, 64), (65, 32, 64), (64, 33, 64), (64, 32, 65)],
)
def test_matrix_design_rejects_partial_or_empty_tiles(shape):
    fn = kernels.mm(dim_m=64, dim_k=32, dim_n=64)
    for helper in (kd.sample_inputs, kd.output_size, kd.host_args):
        with pytest.raises(ValueError, match="independent tiles"):
            helper(fn, shape=shape)
    with pytest.raises(ValueError, match="independent tiles"):
        kd.design(kernels.mm, dim_m=64, dim_k=32, dim_n=64, shape=shape)


@pytest.mark.parametrize("shape", [(64, 32), (64, 32.5, 64)])
def test_matrix_design_rejects_invalid_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        kd.design(kernels.mm, shape=shape)


@pytest.mark.parametrize("calls", [0, -1, 1.5])
def test_design_rejects_invalid_call_count(calls):
    with pytest.raises(ValueError, match="positive integer"):
        kd.design(kernels.add, calls=calls)


def test_matrix_design_repeats_independent_calls():
    fn = kernels.mm()
    inputs = kd.sample_inputs(fn, calls=2)
    assert fn.expected(inputs).shape == (2, fn.dims[0] * fn.dims[2])
    assert "scf.for" in str(kd.design(kernels.mm, calls=2).as_mlir())


def test_scalar_counts_are_bound_not_inferred_from_tensor_sizes():
    assert kernels.reduce_max(tile_size=1024).contract.scalar_bindings == ((2, 1024),)
    assert kernels.rgba2hue(line_width=64).contract.scalar_bindings == ((2, 64),)
    assert kernels.gray2rgba(line_width=64).contract.scalar_bindings == ((2, 64),)
    fn = kernels.leaky_relu(tile_size=1024)
    assert fn.contract.scalar_bindings == ((2, 1024),)
    assert fn.contract.reference_indices() == [0, 3]
    mlir = str(kd.design(kernels.leaky_relu, tile_size=1024, scalars=(0.5,)).as_mlir())
    assert "1024 : i32" in mlir


def test_rounding_setup_is_merged_alwaysinline_ir():
    setter = kernels.conv_even()
    assert setter._inline
    assert setter.object_file_name.endswith(".ll")
    assert setter._symbol_prefix is None
    assert setter is kernels.conv_even()
    assert setter.name != kernels.set_rounding(kernels.RoundingMode.FLOOR).name
    mlir = str(kd.design(kernels.gelu).as_mlir())
    assert 'link_with_mode = "merge"' in mlir
    assert setter.object_file_name in mlir


@pytest.mark.parametrize("arch", ["aie2", "aie2p"])
def test_exp_factory_can_include_shared_clamp_header(arch):
    from aie.utils import config

    set_current_device(NPU1Col1() if arch == "aie2" else NPU2Col1())
    fn = kernels.bf16_exp()
    runtime_dir = Path(config.aie_runtime_lib_dir()) / arch.upper()
    assert str(runtime_dir) in fn.include_dirs


@pytest.mark.parametrize("mode", list(kernels.RoundingMode))
def test_rounding_mode_preserves_string_api(mode):
    assert str(mode) == f"{mode}" == mode.value
    assert mode == mode.value
    setter = kernels.set_rounding(mode)
    assert setter.name == f"set_rounding_{mode.value}"
    assert f"-DROUNDING_MODE={mode.value}" in setter.compile_flags
    assert setter is kernels.set_rounding(mode.value)


@pytest.mark.parametrize(
    "factory,minimum",
    [
        (kernels.conv2dk1, 2752),
        (kernels.conv2dk1_skip, 2752),
        (kernels.conv2dk3, 4736),
        (kernels.layer_norm_f32, 1152),
    ],
)
def test_stack_contract_covers_measured_core(factory, minimum):
    assert factory().contract.stack_bytes >= minimum


def test_contract_validates_argument_bindings():
    from aie.iron import Count

    with pytest.raises(ValueError, match="layouts"):
        KernelContract(roles=(In, Out), layouts=(None,))
    with pytest.raises(ValueError, match="scalar_bindings"):
        KernelContract(roles=(In, Out), scalar_bindings=((1, 64),))
    with pytest.raises(ValueError, match="distinct"):
        KernelContract(roles=(In, Out, Scalar), scalar_bindings=((2, 64), (2, 128)))
    with pytest.raises(ValueError, match="explicit scalar binding"):
        KernelContract(roles=(In, Out, Count))
    with pytest.raises(ValueError, match="initializers"):
        KernelContract(roles=(In, Out), initializers=((1, lambda fn: fn),))


def test_multi_output_contract_drives_design_and_reference():
    from aie.iron.kernel import ExternalFunction

    tile = np.ndarray[(64,), np.dtype[np.int32]]
    fn = ExternalFunction(
        "split_outputs",
        source_string="void split_outputs() {}",
        arg_types=[tile, tile, tile],
    )
    fn.contract = KernelContract(
        roles=(In, Out, Out),
        reference=lambda x: (x, -x),
    )
    assert fn.contract.out_indices == (1, 2)
    with pytest.raises(ValueError, match="multiple outputs"):
        _ = fn.contract.out_index
    inputs = kd.sample_inputs(fn, calls=3)
    refs = fn.expected(inputs)
    assert len(refs) == 2
    assert np.array_equal(refs[0], -refs[1])
    assert fn.judge(tuple(r.ravel() for r in refs), refs, calls=3)
    assert not fn.judge((refs[0].ravel(), np.ones(192, np.int32)), refs, calls=3)
    assert kd.output_size(fn, calls=3) == (192, 192)
    assert [a.direction for a in kd.host_args(fn, calls=3)] == [In, Out, Out]
    assert "split_outputs" in str(kd.design(lambda: fn, calls=3).as_mlir())


def test_split_depthwise_contract_checks_both_channel_halves():
    fn = kernels.bn_conv2dk3_dw_out_split(
        input_width=7, input_channels=16, output_split_channels=8
    )
    assert fn.contract.out_indices == (4, 5)
    inputs = kd.sample_inputs(fn, calls=3)
    refs = fn.expected(inputs, scalars=(1, 7))
    whole = kernels.bn_conv2dk3_dw_ref(*inputs, 7, 16, 16, 3, 3, 1, 7, 0)
    np.testing.assert_array_equal(np.concatenate(refs, axis=-1), whole)
    assert fn.judge(refs, refs, calls=3)
    assert not fn.judge((refs[0], refs[1] ^ 1), refs, calls=3)
    design = kd.design(
        kernels.bn_conv2dk3_dw_out_split,
        input_width=7,
        input_channels=16,
        output_split_channels=8,
        calls=3,
        scalars=(1, 7),
        params=fn.param_values(inputs),
    )
    assert "bn13_conv2dk3_ui8_out_split" in str(design.as_mlir())


@pytest.mark.parametrize(
    "factory",
    [
        kernels.bn_conv2dk1_partial_put_i8,
        kernels.bn_conv2dk1_partial_get_relu_i8,
        kernels.bn_conv2dk1_input_split_partial_put_ui8,
        kernels.bn_conv2dk1_input_split_partial_skip_get,
    ],
)
def test_cascade_factories_declare_their_unobservable_protocol(factory):
    fn = factory()
    assert len(fn.contract.roles) == len(fn.arg_types())
    assert fn.contract.cascade_partner is not None
    assert fn.contract.reference is None
    with pytest.raises(ValueError, match="cascade"):
        kd.design(factory)


@pytest.mark.parametrize("factory", [kernels.mm, kernels.mv, kernels.mm_bfp])
def test_matrix_layouts_are_reversible_per_argument(factory):
    fn = factory()
    for i, values in enumerate(kd.sample_inputs(fn, calls=3)):
        layout = fn.contract.layouts[i]
        decoded = layout.decode(layout.encode(values), calls=3)
        if bfp.is_bfp(kd.shape_dtype(fn.arg_types()[i])[1]):
            expected = (
                bfp.quantize(np.ascontiguousarray(values.swapaxes(-1, -2))).swapaxes(
                    -1, -2
                )
                if i == 1
                else bfp.quantize(values)
            )
            np.testing.assert_array_equal(decoded, expected)
        else:
            np.testing.assert_array_equal(decoded, values)


def test_param_encoding_preserves_integer_bits():
    from aie.iron import Param
    from aie.iron.kernel import ExternalFunction

    tile = np.ndarray[(4,), np.dtype[np.uint64]]
    fn = ExternalFunction(
        "param_encoding", source_string="", arg_types=[tile, tile, tile]
    )
    fn.contract = KernelContract(roles=(In, Param, Out))
    first = np.array([2**53, 2**63, 2**64 - 2, 2**64 - 1], np.uint64)
    second = first.copy()
    second[0] += 1
    encoded = kd._encode_params(fn, [first])
    assert encoded[0][2] == tuple(first.tolist())
    assert encoded != kd._encode_params(fn, [second])


def test_per_tile_matrix_references_agree_with_the_whole_problem_ones():
    """One call of the per-tile form is the whole-problem form on one tile."""
    rng = np.random.default_rng(0)
    m, k, n = 4, 8, 6
    a = rng.integers(-8, 8, (m, k)).astype(np.int16)
    b = rng.integers(-8, 8, (k, n)).astype(np.int16)
    tile = kernels.mm_tile_ref(a.ravel(), b.ravel(), dim_m=m, dim_k=k, dim_n=n)
    assert np.array_equal(tile, kernels.mm_ref(a, b).reshape(1, m * n))

    v = rng.integers(-8, 8, k).astype(np.int16)
    assert np.array_equal(
        kernels.mv_tile_ref(a.ravel(), v, dim_m=m, dim_k=k),
        kernels.mv_ref(a, v).reshape(1, m),
    )

    # bfp quantises in blocks of 8 along K, so K must be a multiple of 8.
    af = rng.standard_normal((m, k)).astype(np.float32)
    bf = rng.standard_normal((k, n)).astype(np.float32)
    for mixed, whole in ((False, kernels.mm_bfp_ref), (True, kernels.mm_bfp_mixed_ref)):
        got = kernels.mm_bfp_tile_ref(
            af.ravel(), bf.ravel(), dim_m=m, dim_k=k, dim_n=n, mixed=mixed
        )
        assert np.allclose(got, whole(af, bf).reshape(1, m * n))


def test_per_tile_matrix_references_keep_calls_independent():
    """Call i of a batch sees only tile i -- the reference never accumulates."""
    rng = np.random.default_rng(1)
    m, k, n = 4, 8, 6
    a = rng.integers(-8, 8, (3, m, k)).astype(np.int16)
    b = rng.integers(-8, 8, (3, k, n)).astype(np.int16)
    batched = kernels.mm_tile_ref(
        a.reshape(3, -1), b.reshape(3, -1), dim_m=m, dim_k=k, dim_n=n
    )
    assert batched.shape == (3, m * n)
    for i in range(3):
        one = kernels.mm_tile_ref(a[i].ravel(), b[i].ravel(), dim_m=m, dim_k=k, dim_n=n)
        assert np.array_equal(batched[i], one[0])


def test_designs_for_different_kernels_do_not_share_a_cache_key():
    h = lambda d: d.compilable.recipe_hash  # noqa: E731
    assert h(kd.design(kernels.add, calls=4)) != h(kd.design(kernels.mul, calls=4))
    assert h(kd.design(kernels.reduce_max, calls=4)) != h(
        kd.design(kernels.reduce_max, calls=4, dtype=bfloat16)
    )
    assert h(kd.design(kernels.add, calls=4)) == h(kd.design(kernels.add, calls=4))
    # A `param` is baked into the design, so its value is part of the key.
    p3 = kd.design(kernels.scale, calls=4, dtype=np.int32, params=[np.array([3])])
    p5 = kd.design(kernels.scale, calls=4, dtype=np.int32, params=[np.array([5])])
    assert h(p3) != h(p5)
    with pytest.raises(ValueError, match="need values at design time"):
        kd.design(kernels.scale, calls=4, dtype=np.int32)


def test_packed_inputs_and_baked_params_leave_the_host_side_small():
    # swiglu: three same-typed inputs -> one fifo, interleaved per call.
    fn = kernels.swiglu()
    ins = kd.sample_inputs(fn, calls=2)
    (packed,) = kd.host_layout(fn, ins)
    assert packed.shape == (2, 3, 1024)
    assert np.array_equal(packed[:, 1], ins[1])
    # filter2d: three lines packed, the kernel param dropped (it is a Buffer).
    fn = kernels.filter2d()
    ins = kd.sample_inputs(fn, calls=2)
    (lines,) = kd.host_layout(fn, ins)
    assert lines.shape == (2, 3, 1920)
    assert fn.param_values(ins)[0].size == 9
    # scale: one input fifo, the factor is a param.
    fn = kernels.scale(dtype=np.int32)
    ins = kd.sample_inputs(fn, calls=2)
    assert len(kd.host_layout(fn, ins)) == 1


def test_contract_rejects_bad_roles():
    with pytest.raises(ValueError, match="unknown"):
        KernelContract(roles=(In, "output"))
    with pytest.raises(ValueError, match="at least one Out"):
        KernelContract(roles=(In, In))
    assert KernelContract(roles=(Out, Out)).out_indices == (0, 1)


def test_harness_refuses_a_kernel_without_a_contract():
    from aie.iron.kernel import ExternalFunction

    def factory():
        return ExternalFunction("uncontracted", source_string="", arg_types=[])

    with pytest.raises(ValueError, match="declares no contract"):
        kd.design(factory)


def test_rgba2hue_reference_matches_the_kernel():
    def px(r, g, b):
        return np.array([r, g, b, 0], np.uint8)

    ref = kernels.rgba2hue_ref
    assert ref(px(0, 0, 0)).tolist() == [0]  # grey: hue 0
    assert ref(px(255, 0, 0)).tolist() == [0]  # red
    assert ref(px(0, 255, 0)).tolist() == [85]  # green: 171 * 512 >> 10
    assert ref(px(0, 0, 255)).tolist() == [170]  # blue: 341 * 512 >> 10
    # R max with G < B: negative hue wraps below 256, as the uint8 cast does.
    assert ref(px(255, 0, 255)).tolist() == [(256 - 42) & 0xFF]
    # Two pixels in one line. Yellow: G takes the tie with R, as the kernel's
    # select order does, and both channels give 43 here. Cyan comes out 127
    # rather than the exact 128 because the reciprocal truncates.
    assert ref(np.concatenate([px(255, 255, 0), px(0, 255, 255)])).tolist() == [43, 127]


def test_rgba2hue_reference_is_within_one_lsb_of_exact_hue():
    """The reciprocal is truncated, so bound the error that can introduce.

    ``rgba2hue_ref`` is bit-exact against both paths of the kernel by
    construction; this pins the other half -- that the arithmetic the two of
    them share stays within one LSB of the exact hue, over every RGB triple.
    """
    r, g, b = (
        x.ravel().astype(np.int64)
        for x in np.meshgrid(*[np.arange(256)] * 3, indexing="ij")
    )
    rgba = np.stack([r, g, b, np.zeros_like(r)], -1).astype(np.uint8).ravel()
    mx = np.maximum(np.maximum(r, g), b)
    d = mx - np.minimum(np.minimum(r, g), b)
    ds = np.where(d == 0, 1, d)
    num = np.where(
        mx == g,
        171 * ds + 85 * (b - r),
        np.where(mx == r, ds + 85 * (g - b), 341 * ds + 85 * (r - g)),
    )
    exact = np.where(d == 0, 0, num // (2 * ds)) & 0xFF
    err = ((kernels.rgba2hue_ref(rgba).astype(np.int64) - exact + 128) & 0xFF) - 128
    assert np.abs(err).max() <= 1


def test_conv_references_follow_the_kernel_layouts():
    W, IC, OC = 4, 8, 8
    # conv2dk1: one input channel lit, weights an identity in (ic8, oc8):
    # output channel c equals input channel c, requantised by >> 0.
    x = np.arange(W * IC, dtype=np.int8).reshape(IC // 8, W, 8)  # [C/8][W][8]
    ident = np.eye(8, dtype=np.int8).reshape(OC // 8, IC // 8, 8, 8)
    out = kernels.conv2dk1_ref(x.ravel(), ident.ravel(), W, IC, OC, 1)
    # scale 1: (v + 1) >> 1
    assert out.tolist() == (((x.astype(np.int64) + 1) >> 1).ravel()).tolist()
    # conv2dk3: centre tap identity, zero-padded borders; middle region
    # returns line1 requantised, top region ignores line0, bottom line2.
    w = np.zeros((OC // 8, IC // 8, 3, 3, 8, 8), np.int8)
    w[:, :, 1, 1] = np.eye(8, dtype=np.int8)  # row 1 (line1), ki 1 (x + 0)
    l0 = np.full(W * IC, 100, np.int8)
    l1 = np.arange(W * IC, dtype=np.int8)
    l2 = np.full(W * IC, -100, np.int8)
    got = kernels.conv2dk3_ref(l0, l1, l2, w.ravel(), W, IC, OC, 3, 3, 1, 1, 0)
    assert got.tolist() == (((l1.astype(np.int64) + 1) >> 1)).tolist()
    w[:, :, 0, 1] = np.eye(8, dtype=np.int8)  # add line0's centre tap
    mid = kernels.conv2dk3_ref(l0, l1, l2, w.ravel(), W, IC, OC, 3, 3, 1, 1, 0)
    top = kernels.conv2dk3_ref(l0, l1, l2, w.ravel(), W, IC, OC, 3, 3, 0, 1, 0)
    assert (
        mid.tolist() == np.clip((l1.astype(np.int64) + 100 + 1) >> 1, 0, 255).tolist()
    )
    assert top.tolist() == got.tolist()
    # Left neighbour tap on a line with a single lit pixel: shifts right by one,
    # and the left border is zero padded.
    w[:] = 0
    w[:, :, 1, 2] = np.eye(8, dtype=np.int8)  # ki 2 reads pixel x + 1
    lit = np.zeros((IC // 8, W, 8), np.int8)
    lit[0, 2, :] = 40
    got = kernels.conv2dk3_ref(l0, lit.ravel(), l2, w.ravel(), W, IC, OC, 3, 3, 1, 1, 0)
    assert got.reshape(OC // 8, W, 8)[0, :, 0].tolist() == [0, 20, 0, 0]


def test_bfp_matmul_host_layout_reference_and_judge():
    from aie.utils import bfp

    M, K, N = 64, 64, 64
    fn = kernels.mm_bfp(dim_m=64, dim_k=64, dim_n=64)
    c = fn.contract
    assert c.roles == (In, In, InOut) and fn.b_col_maj and not fn.c_col_maj
    a, b = kd.sample_inputs(fn, calls=2)
    assert a.dtype == np.float32 and a.shape == (2, M, K) and b.shape == (2, K, N)
    # Same-type operands share one FIFO, in per-call argument order.
    (packed,) = kd.host_layout(fn, [a, b])
    ha, hb = packed[:, 0], packed[:, 1]
    assert ha.dtype == np.uint8 and ha.shape == (2, M * K * 9 // 8)
    m, k, n = fn.dims
    assert np.array_equal(
        bfp.shuffle(ha[0], K, M, k, m, unshuffle=True).ravel(), bfp.encode(a[0]).ravel()
    )
    assert np.array_equal(
        bfp.shuffle(hb[0], K, N, k, n, unshuffle=True).ravel(),
        bfp.encode(np.ascontiguousarray(b[0].T)).ravel(),
    )
    ref = fn.expected([a, b])
    assert ref.dtype == np.float32 and ref.shape == (2, M * N)
    plain = (a.astype(np.float64) @ b.astype(np.float64)).reshape(2, -1)
    assert np.abs(ref - plain).max() < 0.05 * np.abs(plain).max()
    assert fn.output_dtype(ref.dtype) == np.uint8
    assert kd.output_size(fn, calls=2) == 2 * M * N * 9 // 8
    device_c = c.layouts[2].encode(ref).ravel()
    assert fn.judge(device_c, ref, calls=2)
    assert not fn.judge(bfp.encode(ref).ravel(), ref, calls=2)
    assert not fn.judge(np.full(2 * M * N * 9 // 8, 0x55, np.uint8), ref, calls=2)
    # Mixed: A and C are bf16 and stay so; B is still encoded.
    mixed = kernels.mm_bfp(dim_m=64, dim_k=64, dim_n=64, mixed=True)
    a, b = kd.sample_inputs(mixed, calls=2)
    assert a.dtype == bfloat16 and b.dtype == np.float32
    ha, hb = kd.host_layout(mixed, [a, b])
    assert ha.dtype == bfloat16 and ha.shape == (2, M * K) and hb.dtype == np.uint8
    ref = mixed.expected([a, b])
    assert ref.dtype == bfloat16 and mixed.output_dtype(ref.dtype) == bfloat16
    assert kd.output_size(mixed, calls=2) == 2 * M * N
    assert mixed.judge(mixed.contract.layouts[2].encode(ref).ravel(), ref, calls=2)


def test_bottleneck_references_round_half_even_and_saturate():
    W, IC, OC = 4, 8, 8
    ident = np.eye(8, dtype=np.int8).reshape(OC // 8, IC // 8, 8, 8)
    # Round-half-even at scale 1: 3 -> 2, 5 -> 2 (ties to even), 4 -> 2, -3 -> -2.
    x = np.zeros((IC // 8, W, 8), np.int8)
    x[0, :, 0] = [3, 5, 4, -3]
    out = kernels.bn_conv2dk1_relu_ref(x.ravel(), ident.ravel(), W, IC, OC, 1)
    assert out.reshape(OC // 8, W, 8)[0, :, 0].tolist() == [2, 2, 2, 0]  # ReLU
    out = kernels.bn_conv2dk1_i8_ref(
        x.astype(np.uint8).ravel(), ident.ravel(), W, IC, OC, 1
    )
    # uint8 view of -3 is 253: (253 + 1 - 1 + 0) >> 1 = 126
    assert out.reshape(OC // 8, W, 8)[0, :, 0].tolist() == [2, 2, 2, 126]
    # skip: conv saturates to int8 first, then the residual is added; a
    # skip_scale of 0 is no shift, the total saturates to int8.
    xs = np.full((IC // 8, W, 8), 200, np.uint8)
    skip = np.full(W * OC, 100, np.int8)
    out = kernels.bn_conv2dk1_skip_ref(xs.ravel(), ident.ravel(), skip, W, IC, OC, 0, 0)
    assert set(out.tolist()) == {127}  # 200 -> 127, + 100 -> 227 -> 127
    skip = np.full(W * OC, -100, np.int8)
    out = kernels.bn_conv2dk1_skip_ref(xs.ravel(), ident.ravel(), skip, W, IC, OC, 0, 1)
    assert set(out.tolist()) == {14}  # (127 - 100) = 27 -> (27 + 1 - 1 + 1) >> 1 = 14
    # depthwise: centre tap only, stride 1 returns line1; stride 2 every other
    # pixel; the left tap on a single lit pixel shifts right and the border
    # is zero padded.
    C = 8
    w = np.zeros((C // 8, 3, 3, 8), np.int8)
    w[0, 1, 1] = 1
    l0 = np.full(W * C, 9, np.uint8)
    l1 = np.arange(W * C, dtype=np.uint8).reshape(C // 8, W, 8)
    l1[0, :, 0] = [10, 20, 30, 40]
    l2 = np.full(W * C, 9, np.uint8)
    got = kernels.bn_conv2dk3_dw_ref(
        l0, l1.ravel(), l2, w.ravel(), W, C, C, 3, 3, 1, 0, 0
    )
    assert got.reshape(C // 8, W, 8)[0, :, 0].tolist() == [10, 20, 30, 40]
    got = kernels.bn_conv2dk3_dw_ref(
        l0, l1.ravel(), l2, w.ravel(), W, C, C, 3, 3, 1, 0, 0, stride=2
    )
    assert got.reshape(C // 8, W // 2, 8)[0, :, 0].tolist() == [10, 30]
    w[:] = 0
    w[0, 1, 0] = 1  # ki 0 reads pixel x - 1
    got = kernels.bn_conv2dk3_dw_ref(
        l0, l1.ravel(), l2, w.ravel(), W, C, C, 3, 3, 1, 0, 0
    )
    assert got.reshape(C // 8, W, 8)[0, :, 0].tolist() == [0, 10, 20, 30]
    # full 3x3 stride 2: centre tap identity halves the width; top region
    # ignores line0.
    w3 = np.zeros((OC // 8, IC // 8, 3, 3, 8, 8), np.int8)
    w3[:, :, 1, 1] = np.eye(8, dtype=np.int8)
    l1s = np.zeros((IC // 8, W, 8), np.int8)
    l1s[0, :, 0] = [10, 20, 30, 40]
    got = kernels.bn_conv2dk3_ref(
        l0, l1s.ravel(), l2, w3.ravel(), W, IC, OC, 3, 3, 1, 0, 0
    )
    assert got.reshape(OC // 8, W // 2, 8)[0, :, 0].tolist() == [10, 30]
    w3[:, :, 0, 1] = np.eye(8, dtype=np.int8)
    mid = kernels.bn_conv2dk3_ref(
        l0, l1s.ravel(), l2, w3.ravel(), W, IC, OC, 3, 3, 1, 0, 0
    )
    top = kernels.bn_conv2dk3_ref(
        l0, l1s.ravel(), l2, w3.ravel(), W, IC, OC, 3, 3, 0, 0, 0
    )
    assert mid.reshape(OC // 8, W // 2, 8)[0, :, 0].tolist() == [19, 39]
    assert top.tolist() == got.tolist()


def test_post_stage_references_follow_the_sources():
    # FC: uint16 activations, weights padded to a 16-channel stride of which
    # only the first 8 input channels are read; round-half-even, ReLU, uint8
    # range in a uint16 store.
    IC, ICp, OC = 8, 16, 8
    w = np.zeros((OC // 8, ICp // 8, 8, 8), np.int8)
    w[0, 0] = np.eye(8, dtype=np.int8)  # the padded half is never read
    w[0, 1] = 99
    x = np.zeros(IC, np.uint16)
    x[:4] = [3, 5, 600, 7]
    out = kernels.bn_fc_relu_ui16_pad_ref(x, w.ravel(), 1, IC, ICp, OC, 1)
    assert out.dtype == np.uint16
    assert out[:4].tolist() == [2, 2, 255, 4]  # 3 -> 2, 5 -> 2, 300 saturates
    # xy pool: a 7x7 map of ones through an identity 1x1 conv at scale 0 sums
    # to 49 per channel, so the pooled average is exactly 1; the padding
    # channels are zero and only the selected output tile is written.
    W, C = 7, 8
    x = np.ones((W, W * C), np.int8)
    ident = np.eye(8, dtype=np.int8).ravel()
    out = kernels.bn_conv2dk1_relu_xy_pool_padded_ref(x, ident, W, C, C, 16, 0, 0, 1, 0)
    assert out.dtype == np.uint16 and out.shape == (16,)
    assert out.tolist() == [1] * 8 + [0] * 8
    # A per-pixel value of 2 (x = 2) makes the sum 98 and the average 2.0;
    # x = 3 gives 147 / 49 = 3.0. Odd sums exercise the kernel's rounding:
    # 25 ones and 24 zeros -> 25 / 49 = 0.5102 -> (int)(5.1) % 10 == 5 ->
    # ties-to-even on the integer part -> 0.
    x = np.zeros((W, W * C), np.int8)
    x.reshape(W, W, C)[:, :, 0].flat[:25] = 1
    out = kernels.bn_conv2dk1_relu_xy_pool_padded_ref(x, ident, W, C, C, C, 0, 0, 1, 0)
    assert out[0] == 0
    x.reshape(W, W, C)[:, :, 0].flat[:] = 1
    x.reshape(W, W, C)[:, :, 0].flat[:2] = 0  # 47 / 49 = 0.959 -> 1
    out = kernels.bn_conv2dk1_relu_xy_pool_padded_ref(x, ident, W, C, C, C, 0, 0, 1, 0)
    assert out[0] == 1
    # output_split = 2, weight_index = 1: the call carries the weights of
    # its own 8-channel tile and writes only channels [8, 16).
    IC = OC = 16
    wt = np.zeros((OC // 2 // 8, IC // 8, 8, 8), np.int8)
    wt[0, 1] = np.eye(8, dtype=np.int8)  # tile 1 passes input channels 8..15
    out = kernels.bn_conv2dk1_relu_xy_pool_padded_ref(
        np.ones((W, W * IC), np.int8), wt.ravel(), W, IC, OC, OC, 0, 0, 2, 1
    )
    assert out.tolist() == [0] * 8 + [1] * 8


@pytest.mark.parametrize("act_dtype", [np.int8, np.uint8])
@pytest.mark.parametrize("input_width", [-64, -32, -1, 0, 1, 31, 33])
def test_skip_init_rejects_invalid_width(input_width, act_dtype):
    with pytest.raises(ValueError, match="positive multiple of 32"):
        kernels.conv2dk1_skip_init(input_width=input_width, act_dtype=act_dtype)


@pytest.mark.parametrize("act_dtype", [np.int8, np.uint8])
@pytest.mark.parametrize("input_width", [32, 64])
def test_skip_init_accepts_complete_width_blocks(input_width, act_dtype):
    fn = kernels.conv2dk1_skip_init(input_width=input_width, act_dtype=act_dtype)
    assert fn.arg_shape(0) == (input_width * 32,)
    assert fn.arg_shape(3) == (input_width * 64,)


def test_conv2dk14_and_skip_init_references():
    # conv2dk14: one patch of K*K RGBA pixels per output; a weight of 1 on
    # channel 0 of pixel 0 for output channel 0 reads that pixel.
    K, T, OC = 2, 8, 8
    P = K * K
    x = np.zeros((T // 8, P // 2, 8, 2, 4), np.uint8)  # [T/8][P/2][t8][p2][c]
    for t in range(T):
        x[0, 0, t, 0, 0] = 10 * (t + 1)  # pixel 0, channel 0 of patch t
    w = np.zeros((OC // 8, P // 2, 2, 4, 8), np.int8)  # [OC/8][P/2][p2][c][oc8]
    w[0, 0, 0, 0, 0] = 1
    out = kernels.conv2dk14_ref(x.ravel(), w.ravel(), T * K, 4, OC, K, 0)
    assert out.reshape(OC // 8, T, 8)[0, :, 0].tolist() == [
        10 * (t + 1) for t in range(T)
    ]
    w[0, 0, 0, 0, 0] = 2  # 2 * 80 = 160 saturates to int8
    out = kernels.conv2dk14_ref(x.ravel(), w.ravel(), T * K, 4, OC, K, 0)
    assert out.reshape(OC // 8, T, 8)[0, -1, 0] == 127
    # conv2dk1_skip_init: main conv on the two halves, projected residual.
    W, IC, OC, ICs = 4, 16, 8, 8
    w = np.zeros((OC // 8, IC // 8, 8, 8), np.int8)
    w[0, 1] = np.eye(8, dtype=np.int8)  # reads x1
    ws = np.zeros((OC // 8, ICs // 8, 8, 8), np.int8)
    ws[0, 0] = 2 * np.eye(8, dtype=np.int8)
    weights = np.concatenate([w.ravel(), ws.ravel()])
    x0 = np.full(W * 8, 50, np.uint8)
    x1 = np.full(W * 8, 30, np.uint8)
    skip = np.full(W * ICs, 20, np.uint8)
    out = kernels.conv2dk1_skip_init_ref(x0, x1, weights, skip, W, IC, OC, ICs, 0, 0, 0)
    assert set(out.tolist()) == {70}  # 30 + 2 * 20
    out = kernels.conv2dk1_skip_init_ref(x0, x1, weights, skip, W, IC, OC, ICs, 0, 1, 1)
    assert set(out.tolist()) == {25}  # proj (40 + 1) >> 1 = 20; (30 + 20 + 1) >> 1 = 25


# The kernel each amd/IRON operator builds by hand, as (factory kwargs) ->
# (symbol, source basename, compile flags). IRON declares these itself today
# (a KernelObjectArtifact plus a Kernel binding); a factory is a drop-in for
# that pair only while it produces the same symbol, source and flags, so this
# table pins them. Sources shared with IRON only -- where the two trees have
# diverged (IRON's own relu.cc / rms_norm.cc symbols, its -DROUND_CONV_EVEN
# matmul) no factory can stand in, and the library guide says so.
_IRON_KERNEL_SPECS = {
    "axpy": ({}, "saxpy", "axpy.cc", ()),
    "gelu": ({}, "gelu_bf16", "gelu.cc", ()),
    "silu": ({}, "silu_bf16", "silu.cc", ()),
    "sigmoid": ({}, "sigmoid_bf16", "sigmoid.cc", ()),
    "tanh": ({}, "tanh_bf16", "tanh.cc", ()),
    "softmax": ({}, "softmax_bf16", "softmax.cc", ()),
    "add": ({}, "eltwise_add_bf16_vector", "add.cc", ()),
    "mul": ({}, "eltwise_mul_bf16_vector", "mul.cc", ()),
    "layer_norm": (dict(cols=1024), "layer_norm", "layer_norm.cc", ()),
    "rope": (dict(cols=1024), "rope", "rope.cc", ()),
    "passthrough": (
        dict(tile_size=1024, dtype=np.int16),
        "passThroughLine",
        "passThrough.cc",
        ("-DBIT_WIDTH=16",),
    ),
    "expand": (
        dict(tile_size=1024, group_size=32),
        "expand_uint4_to_bfloat16",
        "expand.cc",
        ("-DTILE_SIZE=1024", "-DGROUP_SIZE=32"),
    ),
    "transpose": (
        dict(dim_m=32, dim_n=32, subtile=4),
        "transpose_4x4",
        "transpose.cc",
        ("-DDIM_m=32", "-DDIM_n=32"),
    ),
    "mv": (
        dict(dim_m=32, dim_k=256, input_dtype=bfloat16, output_dtype=bfloat16),
        "matvec_vectorized_bf16_bf16",
        "mv.cc",
        ("-DDIM_K=256", "-DVEC_SIZE=64"),
    ),
    "mm": (
        dict(
            dim_m=64,
            dim_k=64,
            dim_n=64,
            input_dtype=bfloat16,
            output_dtype=np.float32,
            b_col_maj=True,
        ),
        "matmul_bf16_f32",
        "mm.cc",
        ("-DDIM_M=64", "-DDIM_K=64", "-DDIM_N=64", "-Dbf16_f32_ONLY", "-DB_COL_MAJ"),
    ),
}


@pytest.mark.parametrize("name", list(_IRON_KERNEL_SPECS))
def test_factories_reproduce_the_iron_operator_kernel_specs(name):
    fkw, symbol, source, flags = _IRON_KERNEL_SPECS[name]
    fn = getattr(kernels, name)(**fkw)
    # The exported symbol may carry the memoisation digest prefix; the kernel
    # it binds is what has to match.
    assert fn.name.split("_", 1)[-1] == symbol or fn.name == symbol, fn.name
    assert Path(fn.source_file).name == source
    assert set(flags) <= set(fn.compile_flags or ()), fn.compile_flags


def test_bf16_matvec_matches_the_iron_gemv_signature():
    # (m, row_offset, A, b, c): two runtime scalars, then the tensors.
    fn = kernels.mv(
        dim_m=32, dim_k=256, input_dtype=bfloat16, output_dtype=bfloat16, vec_size=64
    )
    types = fn.arg_types()
    assert types[0] is np.int32 and types[1] is np.int32
    assert [kd.shape_dtype(t)[0] for t in types[2:]] == [(32 * 256,), (256,), (32,)]
    assert all(kd.shape_dtype(t)[1] is bfloat16 for t in types[2:])
    assert fn.contract.roles == (Scalar, Scalar, In, In, Out)
    # row_offset shifts the write into c, so one core fills several blocks.
    a = np.arange(4 * 8, dtype=np.float32).reshape(4, 8).astype(bfloat16)
    b = np.ones(8, dtype=bfloat16)
    assert kernels.mv_bf16_ref(2, 1, a, b).tolist() == [0.0, 28.0, 92.0]
    # The int16 kernel is a different source with a zero symbol and no scalars.
    i16 = kernels.mv(dim_m=32, dim_k=32)
    assert i16.contract.roles == (In, In, InOut) and hasattr(i16.also, "zero")
    with pytest.raises(ValueError, match="multiple of vec_size"):
        kernels.mv(dim_k=100, input_dtype=bfloat16, output_dtype=bfloat16)


@pytest.mark.parametrize("case_id", list(CASES))
def test_host_args_match_what_the_sampler_and_uploader_produce(case_id):
    """The declared host buffers are the ones the harness actually builds."""
    fkw, opts = CASES[case_id]
    fn = _factory(case_id)(**fkw)
    if fn.contract.unsupported:
        pytest.skip(fn.contract.unsupported)
    calls, shape = opts.get("calls", 1), opts.get("shape")
    args = kd.host_args(fn, calls=calls, shape=shape)
    ins, out = args[:-1], args[-1]
    assert [a.direction for a in args] == [In] * len(ins) + [Out]
    # Inputs: the arrays host_layout hands the device.
    staged = kd.host_layout(fn, kd.sample_inputs(fn, calls=calls, shape=shape))
    assert len(staged) == len(ins), case_id
    for got, spec in zip(staged, ins):
        assert got.shape == spec.shape, f"{case_id}: {got.shape} != {spec.shape}"
        assert got.dtype == np.dtype(spec.dtype), case_id
    # Output: the element count and dtype upload allocates.
    ref = fn.expected(
        kd.sample_inputs(fn, calls=calls, shape=shape),
        scalars=opts.get("scalars", ()),
    )
    assert out.n_elements == kd.output_size(fn, calls=calls, shape=shape), case_id
    assert np.dtype(out.dtype) == np.dtype(fn.output_dtype(ref.dtype)), case_id


def test_host_args_describe_the_layouts_a_caller_must_allocate():
    # b_col_maj puts B^T on the host; c_col_maj puts C^T there.
    fkw = dict(
        dim_m=64, dim_k=32, dim_n=64, input_dtype=bfloat16, output_dtype=np.float32
    )
    plain = kd.host_args(kernels.mm(**fkw), calls=4)
    assert [a.shape for a in plain] == [(4, 2, 2048), (4, 4096)]
    bcm = kd.host_args(kernels.mm(**fkw, b_col_maj=True), calls=4)
    assert [a.shape for a in bcm] == [a.shape for a in plain]
    ccm = kd.host_args(
        kernels.mm(
            dim_m=64,
            dim_k=32,
            dim_n=64,
            input_dtype=np.int16,
            output_dtype=np.int32,
            c_col_maj=True,
        ),
        calls=4,
    )
    assert ccm[-1].shape == (4, 4096)
    # A bfp16ebs8 operand is bytes: 9 per block of 8 along K.
    bfp_args = kd.host_args(kernels.mm_bfp(dim_m=64, dim_k=64, dim_n=64), calls=4)
    assert all(np.dtype(a.dtype) == np.uint8 for a in bfp_args)
    assert bfp_args[0].shape == (4, 2, 64 * 64 * 9 // 8)
    # Three streamed tensors share one packed fifo.
    packed = kd.host_args(kernels.swiglu(), calls=4)
    assert len(packed) == 2 and packed[0].shape == (4, 3, 1024)
    # A reduction's output keeps the DMA padding the device writes.
    red = kd.host_args(kernels.reduce_add(), calls=4)
    assert red[-1].n_elements == kd.output_size(kernels.reduce_add(), calls=4)


def test_mha_binds_its_translation_unit_as_one_object():
    """mha.cc is one compile with many symbols, bound through siblings.

    It ``#include``s softmax.cc and mm.cc, so it defines ``matmul_*`` and
    ``zero_*`` names of its own; the parameterisation prefix is what stops
    those colliding with a separate ``mm`` in the same design.
    """
    fn = kernels.mha(dim_m=64, dim_k=64, dim_n=64)
    p = fn._symbol_prefix
    assert fn.name == f"{p}_matmul_bf16_bf16_wrapper"
    expected = {
        "zero": "zero_bf16_rowmaj",
        "matmul_scalar": "matmul_bf16_bf16_wrapper_scalar",
        "matmul_rowmaj": "matmul_bf16_bf16_rowmaj",
        "partial_softmax": "partial_softmax",
        "matmul_pv": "matmul_PV",
        "rescale_o": "rescale_O",
        "init_scale_buffer": "init_scale_buffer",
    }
    for attr, symbol in expected.items():
        sib = getattr(fn.also, attr)
        assert sib.name == f"{p}_{symbol}", attr
        assert sib.object_file_name == fn.object_file_name, attr
    # mha.cc only declares passThroughLine; that copy is its own kernel.
    assert not hasattr(fn.also, "passthrough")
    # Its own matmul symbols cannot collide with a real mm in one design.
    assert kernels.mm(dim_m=64, dim_k=64, dim_n=64).name != fn.name
    # A dataflow the harness cannot drive says so rather than failing oddly.
    assert fn.contract.unsupported
    with pytest.raises(ValueError, match="cannot build"):
        kd.design(kernels.mha)
    with pytest.raises(ValueError, match="multiple of"):
        kernels.mha(dim_m=17)


def test_accumulating_kernels_are_inout_and_ship_a_zero():
    """``inout`` marks a kernel that reads its output back, and needs zeroing.

    ``mm``'s ``C += A * B`` reads C, so a design must zero the buffer before
    the first call -- which is what the ``.also.zero`` sibling is for. The
    reference still computes the whole product, so an ``inout`` output is
    excluded from ``reference_indices`` exactly like an ``out`` one.
    """
    for f, fkw in (
        (kernels.mm, {}),
        (kernels.mv, {}),
        (kernels.cascade_mm, {}),
    ):
        fn = f(**fkw)
        c = fn.contract
        assert c.accumulates, f"{f.__name__} accumulates into C"
        assert c.roles[c.out_index] is InOut
        assert hasattr(fn.also, "zero"), f"{f.__name__} needs a .also.zero to clear C"
        assert c.out_index not in c.reference_indices()
    # The autouse fixture selects aie2p, so the bfp matmul builds here too.
    bfp = kernels.mm_bfp()
    assert bfp.contract.accumulates and bfp.also.zero is not None
    # The bf16 matvec stores rather than accumulating: plain "out".
    st = kernels.mv(dim_m=32, dim_k=256, input_dtype=bfloat16, output_dtype=bfloat16)
    assert not st.contract.accumulates
    assert st.contract.roles[st.contract.out_index] is Out
    # A kernel that neither writes nor accumulates is rejected, and so is one
    # that claims both.
    with pytest.raises(ValueError, match="at least one"):
        kernels.KernelContract(roles=(In, In))
    assert kernels.KernelContract(roles=(Out, InOut)).out_indices == (0, 1)
    single = kernels.KernelContract(roles=(In, InOut))
    assert single.out_index == 1 and single.accumulates


def test_sibling_symbols_follow_the_parameterisation_prefix():
    """A kernel's siblings bind names its own object actually defines.

    Each parameterisation gets a symbol prefix so two of them can share a
    design; the whole object is prefixed, so ``.also.zero`` and the cascade
    get/put trio have to be prefixed to match.
    """
    fn = kernels.mm(dim_m=64, dim_k=64, dim_n=64)
    prefix = fn._symbol_prefix
    assert prefix and fn.name == f"{prefix}_matmul_i16_i16"
    assert fn.also.zero.name == f"{prefix}_zero_i16"
    assert fn.also.zero.object_file_name == fn.object_file_name
    # A different parameterisation gets a different prefix on every symbol.
    other = kernels.mm(dim_m=32, dim_k=32, dim_n=32)
    assert other._symbol_prefix != prefix
    assert other.also.zero.name != fn.also.zero.name
    casc = kernels.cascade_mm()
    cp = casc._symbol_prefix
    for sib in (casc.also.put_only, casc.also.put_get, casc.also.zero):
        assert (
            sib.name.startswith(f"{cp}_")
            and sib.object_file_name == casc.object_file_name
        )
    # reduce_max / compute_max pin one shared object, so they stay unprefixed.
    assert not getattr(kernels.compute_max(), "_symbol_prefix", None)


def test_unsupported_contracts_are_refused_by_the_harness():
    fn = kernels.cascade_mm()
    assert fn.contract is not None and fn.contract.unsupported
    with pytest.raises(ValueError, match="cannot build"):
        kd.design(kernels.cascade_mm)
    assert kernels.mm_bfp_shuffle(dim_n=32).contract.unsupported
    assert kernels.mm_bfp_shuffle().contract.unsupported is None


def test_bfp_shuffle_contract_uses_declared_storage_codecs():
    fn = kernels.mm_bfp_shuffle()
    inputs = kd.sample_inputs(fn, calls=3)
    (encoded,) = kd.host_layout(fn, inputs)
    ref = fn.expected(inputs)
    got = np.stack([kernels.mm_bfp_shuffle_ref(row, 64, 64, 0) for row in encoded])
    assert fn.judge(got, ref, calls=3)
    assert not fn.judge(encoded, ref, calls=3)
    assert "scalar_shuffle" in str(kd.design(kernels.mm_bfp_shuffle, calls=3).as_mlir())


def test_conv2dk1_i8_and_skip_references():
    W, IC, OC = 4, 16, 8
    ident = np.zeros((OC // 8, IC // 8, 8, 8), np.int8)
    ident[0, 0] = np.eye(8, dtype=np.int8)  # output channel c <- input channel c
    # conv2dk1_i8 keeps the sign and saturates to int8: scale 0 is no shift.
    x = np.zeros((IC // 8, W, 8), np.int8)
    x[0] = np.arange(-16, 16).reshape(W, 8)
    out = kernels.conv2dk1_i8_ref(x.ravel(), ident.ravel(), W, IC, OC, 0)
    assert out.dtype == np.int8 and out.tolist() == x[0].ravel().tolist()
    ident[0, 0] *= 2
    x[0] = 100
    out = kernels.conv2dk1_i8_ref(x.ravel(), ident.ravel(), W, IC, OC, 0)
    assert set(out.tolist()) == {127}
    x[0] = -100
    out = kernels.conv2dk1_i8_ref(x.ravel(), ident.ravel(), W, IC, OC, 0)
    assert set(out.tolist()) == {-128}
    # conv2dk1_skip: x0 holds channels 0..7, x1 channels 8..15. Weights on the
    # upper half only read x1; the conv saturates to int8 before the residual
    # is added, and the total to uint8 after its own shift.
    w = np.zeros((OC // 8, IC // 8, 8, 8), np.int8)
    w[0, 1] = np.eye(8, dtype=np.int8)
    x0 = np.full(W * 8, 50, np.uint8)
    x1 = np.arange(W * 8, dtype=np.uint8)  # 0..31
    skip = np.full(W * OC, -10, np.int8)
    out = kernels.conv2dk1_skip_ref(x0, x1, w.ravel(), skip, W, IC, OC, 0, 0)
    assert out.dtype == np.uint8
    assert out.tolist() == np.clip(x1.astype(int) - 10, 0, 255).tolist()
    w[0, 1] *= 2  # 2 * 200 = 400 saturates to 127 before the skip add
    x1[:] = 200
    out = kernels.conv2dk1_skip_ref(x0, x1, w.ravel(), skip, W, IC, OC, 0, 0)
    assert set(out.tolist()) == {117}
    # skip_scale 1 rounds half up after the add: (127 - 10 + 1) >> 1 = 59.
    out = kernels.conv2dk1_skip_ref(x0, x1, w.ravel(), skip, W, IC, OC, 0, 1)
    assert set(out.tolist()) == {59}
    # A uint8 residual is the same arithmetic on the other build.
    uskip = np.full(W * OC, 200, np.uint8)
    out = kernels.conv2dk1_skip_ref(x0, x1, w.ravel(), uskip, W, IC, OC, 0, 0)
    assert set(out.tolist()) == {255}


def test_input_limit_is_bounded_by_the_accumulator_only():
    # conv2dk1 requantises by >> 12: bounding its inputs by the uint8 output
    # would leave every random output at 0 or 1. The accumulator is the only
    # thing that bounds an input; what the kernel does when a result leaves
    # the output range is the reference's job to model.
    fn = kernels.conv2dk1()
    assert fn.input_limit(np.int8) == 127
    # Two tensors multiplied into an int32 accumulator: each is bounded by
    # the square root of the accumulator's budget, not by the int16 output.
    fn = kernels.scale(dtype=np.int16)
    budget = np.iinfo(fn.contract.acc_dtype).max // 4
    assert fn.input_limit(np.int16) == int(np.sqrt(budget))


def test_declared_layouts_pack_inputs_and_unpack_outputs():
    fkw = dict(
        dim_m=64, dim_k=32, dim_n=64, input_dtype=bfloat16, output_dtype=np.float32
    )
    a = np.arange(64 * 32, dtype=np.float32).reshape(1, 64, 32).astype(bfloat16)
    b = np.arange(32 * 64, dtype=np.float32).reshape(1, 32, 64).astype(bfloat16)
    bcm = kernels.mm(b_col_maj=True, **fkw)
    (packed,) = kd.host_layout(bcm, [a, b])
    assert np.array_equal(bcm.contract.layouts[1].decode(packed[:, 1]), b)
    ccm = kernels.mm(c_col_maj=True, **fkw)
    ref = ccm.expected([a, b])
    assert ccm.judge(ccm.contract.layouts[2].encode(ref).ravel(), ref)
    assert not ccm.judge(ref.ravel(), ref)


def test_stream_dims_follow_the_layout_flags():
    fkw = dict(dim_m=64, dim_k=32, dim_n=64)
    plain, bcm, ccm = (
        kernels.mm(**fkw),
        kernels.mm(b_col_maj=True, **fkw),
        kernels.mm(c_col_maj=True, **fkw),
    )
    assert plain.stream_dims.A == bcm.stream_dims.A == ccm.stream_dims.A
    assert plain.stream_dims.B != bcm.stream_dims.B
    assert plain.stream_dims.C != ccm.stream_dims.C
    assert (plain.b_col_maj, plain.c_col_maj) == (False, False)
    assert (bcm.b_col_maj, ccm.c_col_maj) == (True, True)


def test_softmax_tolerance_rejects_an_unwritten_tile():
    # Every softmax output of a 1024-wide tile is far below the generic LUT
    # atol of 0.05, so that floor would accept an all-zero output.
    fn = kernels.softmax()
    x = np.random.default_rng(0).standard_normal((4, 1024)).astype(bfloat16)
    ref = kernels.softmax_ref(x)
    assert float(ref.max()) < 0.05
    assert compare(ref, ref, fn.contract.tolerance)
    assert not compare(np.zeros_like(ref), ref, fn.contract.tolerance)


def test_exp2f_vec_reference_clamps_like_the_kernel():
    # The kernel clamps its input to min_x; 2**-5000 is 2**min_x on the
    # device, not 0, and the contract binds the factory's own min_x.
    x = np.array([-5000.0, -111.0, 0.0, 3.0], dtype=np.float32)
    got = kernels.exp2f_vec_ref(x)
    assert got[0] == got[1] == np.float32(2.0**-111)
    assert got[2] == 1.0 and got[3] == 8.0
    fn = kernels.exp2f_vec(min_x=-100.0)
    assert fn.contract.reference(x)[0] == np.float32(2.0**-100)


def test_vision_references_follow_the_kernel_sources():
    # threshold: OpenCV semantics, exact.
    x = np.array([0, 99, 100, 101, 255], dtype=np.uint8)
    assert kernels.threshold_ref(x, 100, 200, 0).tolist() == [0, 0, 0, 200, 200]
    assert kernels.threshold_ref(x, 100, 200, 1).tolist() == [200, 200, 200, 0, 0]
    assert kernels.threshold_ref(x, 100, 200, 2).tolist() == [0, 99, 100, 100, 100]
    assert kernels.threshold_ref(x, 100, 200, 3).tolist() == [0, 0, 0, 101, 255]
    assert kernels.threshold_ref(x, 100, 200, 4).tolist() == [0, 99, 100, 0, 0]
    # gray2rgba: three copies and an opaque alpha.
    assert kernels.gray2rgba_ref(np.array([7, 9], np.uint8)).tolist() == [
        7,
        7,
        7,
        255,
        9,
        9,
        9,
        255,
    ]
    # rgba2gray: white is white, black is black, alpha ignored.
    rgba = np.array([255, 255, 255, 0, 0, 0, 0, 255], np.uint8)
    assert kernels.rgba2gray_ref(rgba).tolist() == [255, 0]
    # add_weighted: 0.5 a + 0.5 b in Q2.14, saturating.
    a = np.array([100, 255], np.uint8)
    b = np.array([200, 255], np.uint8)
    assert kernels.add_weighted_ref(a, b, 8192, 8192, 0).tolist() == [150, 255]
    assert kernels.add_weighted_ref(a, b, 16384, 16384, 0).tolist() == [255, 255]
    # filter2d: identity kernel (Q4.12 one at the centre) copies the middle line
    # and replicates borders; a box kernel of 16/16 sums 9 pixels / 16.
    ident = np.zeros((3, 3), np.int16)
    ident[1, 1] = 4096
    l0 = np.arange(96, dtype=np.uint8)
    l1 = l0 + 10
    l2 = l0 + 20
    assert np.array_equal(kernels.filter2d_ref(l0, l1, l2, ident), l1)
    box = np.full((3, 3), 256, np.int16)  # k >> 8 == 1 per tap, sum >> 4
    flat = np.full(96, 32, np.uint8)
    assert kernels.filter2d_ref(flat, flat, flat, box).tolist() == [18] * 96


def test_declared_arg_types_survive_a_design_build():
    # The dialect's external_func rewrites a kernel's arg_types() in place with
    # MLIR types when a design resolves it, and the call-site validator relies
    # on that. The build repopulates the factory memo with the very instance it
    # resolved, so the next factory call in the same process hands back a
    # kernel whose arg_types() are MLIR types. The harness reads
    # arg_types(), which does not move.
    declared = [str(t) for t in kernels.add().arg_types()]
    kd.design(kernels.add, calls=2).as_mlir()
    fn = kernels.add()  # memoized: the instance the design resolved
    assert [str(t) for t in fn.arg_types()] == declared
    assert all(hasattr(t, "__args__") for t in fn.arg_types()[:3])
    assert kd.output_size(fn, calls=2) == 2 * 1024
    assert kd.sample_inputs(fn, calls=2)[0].shape == (2, 1024)
    kd.design(kernels.add, calls=4).as_mlir()  # a second design still builds


def test_compute_max_reference_uses_element_zero_only():
    a = np.array([[3, 100], [5, 0]], np.int32)
    b = np.array([[4, 0], [1, 100]], np.int32)
    assert kernels.compute_max_ref(a, b).tolist() == [[4], [5]]


def test_transformer_references_match_the_example_formulas():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(3, 64)).astype(bfloat16)
    x32 = x.astype(np.float32)
    # rms_norm: unit RMS rows.
    y = kernels.rms_norm_ref(x).astype(np.float32)
    assert np.allclose(np.sqrt((y * y).mean(axis=-1)), 1.0, atol=0.02)
    # layer_norm: zero mean, unit variance rows.
    y = kernels.layer_norm_ref(x).astype(np.float32)
    assert np.allclose(y.mean(axis=-1), 0.0, atol=0.02)
    assert np.allclose(y.var(axis=-1), 1.0, atol=0.03)
    # affine cast: gamma scales, beta shifts.
    gb = np.concatenate([np.full(64, 2.0), np.full(64, 0.5)]).astype(np.float32)
    ya = kernels.layer_norm_affine_cast_ref(x32, gb).astype(np.float32)
    assert np.allclose(ya.mean(axis=-1), 0.5, atol=0.05)
    # rope with a zero angle is the identity, with pi/2 a (even, odd) -> (-odd, even).
    lut = np.zeros((3, 64), np.float32)
    lut[:, 0::2] = 1.0
    assert np.array_equal(kernels.rope_ref(x, lut.astype(bfloat16)), x)
    lut[:, 0::2], lut[:, 1::2] = 0.0, 1.0
    r = kernels.rope_ref(x, lut.astype(bfloat16)).astype(np.float32)
    assert np.allclose(r[:, 0::2], -x32[:, 1::2], atol=1e-2)
    assert np.allclose(r[:, 1::2], x32[:, 0::2], atol=1e-2)
    # epilogue modes.
    xf = np.array([-2.0, 0.0, 3.0], np.float32)
    assert np.array_equal(kernels.mm_activation_epilogue_ref(xf, 0), xf)
    assert np.allclose(
        kernels.mm_activation_epilogue_ref(xf, 1), xf / (1 + np.exp(-xf))
    )
    assert np.allclose(kernels.mm_activation_epilogue_ref(xf, 2)[1], 0.0)
    assert kernels.mm_activation_epilogue_ref(xf, 3).tolist() == [0.0, 0.0, 3.0]
    # mul_add: both phases of scale_shift.
    a = np.array([1.5, -2.0], bfloat16)
    b = np.array([2.0, 4.0], bfloat16)
    assert kernels.mul_add_ref(a, b, 1).tolist() == [3.0, -8.0]
    assert kernels.mul_add_ref(a, b, 0).tolist() == [3.5, 2.0]
    # dwconv1d: taps read the padded row directly, bias is the trailing weight.
    xp = np.arange(1, 1 + 32 + kernels.DWCONV1D_TAIL, dtype=np.float32).astype(bfloat16)
    w = np.array([1, 0, 0, 5], bfloat16)  # K = 3 taps then bias
    out = kernels.dwconv1d_ref(xp, w, 32, kernel_size=3, bias=True).astype(np.float32)
    assert out.tolist() == [float(i + 5) for i in range(1, 33)]
    out = kernels.dwconv1d_ref(xp, w, 32, kernel_size=3, bias=False).astype(np.float32)
    assert out.tolist() == [float(i) for i in range(1, 33)]


def test_contract_validates_its_remaining_fields():
    with pytest.raises(ValueError, match="reduction"):
        KernelContract(roles=(In, Out), reduction=0)
    with pytest.raises(ValueError, match="stack_bytes"):
        KernelContract(roles=(In, Out), stack_bytes=0)
    with pytest.raises(ValueError, match="role"):
        KernelContract(roles=(In, "out"))
    c = KernelContract(roles=(In, Out))
    assert (c.acc_dtype, c.reduction, c.setup, c.stack_bytes) == (
        None,
        None,
        None,
        None,
    )


def test_input_limit_keeps_the_reference_inside_the_accumulator():
    # A matmul tile with int16 inputs: K products of two limits must fit in
    # what the kernel accumulates in (accauto gives int16 inputs an int64
    # accumulator), over the design's full K, not the tile's k.
    fn = kernels.mm(
        dim_m=64, dim_k=32, dim_n=64, input_dtype=np.int16, output_dtype=np.int32
    )
    if fn.contract.acc_dtype is None:
        pytest.skip("mm declares no accumulator yet")
    lim = fn.input_limit(np.int16, reduction=256)
    assert 256 * lim * lim <= np.iinfo(fn.contract.acc_dtype).max // 4
    a, b = kd.sample_inputs(fn, calls=4)
    assert int(np.abs(a).max()) <= lim and int(np.abs(b).max()) <= lim
    ref = kernels.mm_ref(a, b)
    acc = np.iinfo(fn.contract.acc_dtype)
    assert ref.min() >= acc.min and ref.max() <= acc.max
    # Float inputs and kernels without an accumulator have no limit.
    assert fn.input_limit(bfloat16) is None
    assert kernels.passthrough().input_limit(np.int32) is None


def _combo_id(v) -> str:
    """One `.dtypes` entry value as a test id: a dtype name, else its value."""
    return np.dtype(v).name if isinstance(v, type) else str(v)


def _factories_with_dtypes():
    for name in kernels.__all__:
        f = getattr(kernels, name)
        if inspect.isfunction(f) and hasattr(f, "dtypes"):
            for combo in f.dtypes:
                yield pytest.param(
                    name,
                    combo,
                    id=f"{name}/{'/'.join(_combo_id(v) for v in combo.values())}",
                )


@pytest.mark.parametrize("name,combo", list(_factories_with_dtypes()))
def test_declared_dtype_combinations_build(name, combo):
    """Every combination a factory lists as supported builds, and its arg types use it."""
    fn = getattr(kernels, name)(**combo)
    if name == "mm_bfp":  # block-floating-point operands are not numpy dtypes
        assert fn.also.zero is not None
        return
    tensor_dts = {
        np.dtype(kd.shape_dtype(t)[1]) for t in fn.arg_types() if hasattr(t, "__args__")
    }
    for v in combo.values():
        if isinstance(v, type):  # a dtype, not a shape or a flag
            assert np.dtype(v) in tensor_dts, f"{name}: {v} not among {tensor_dts}"
    if name != "mm_bfp":
        assert fn.contract is not None


@pytest.mark.parametrize("case_id", list(CASES))
def test_accumulating_kernels_declare_their_accumulator(case_id):
    """A kernel that sums more than one term says what it sums in."""
    fkw, _ = CASES[case_id]
    c = _factory(case_id)(**fkw).contract
    if c.reduction is not None and c.reduction > 1:
        assert (
            c.acc_dtype is not None
        ), f"{case_id}: reduction {c.reduction} without acc_dtype"


def test_mm_accumulators_follow_accauto():
    assert kernels.mm_acc_dtype(np.int8) is np.int32
    assert kernels.mm_acc_dtype(np.int16) is np.int64
    assert kernels.mm_acc_dtype(bfloat16) is np.float32
    fn = kernels.mm(
        dim_m=64, dim_k=32, dim_n=64, input_dtype=np.int16, output_dtype=np.int32
    )
    assert (fn.contract.acc_dtype, fn.contract.reduction) == (np.int64, 32)


def test_saturating_kernels_saturate_in_their_reference():
    # The kernel clamps to uint8, so conv2dk1_ref clamps: saturation is part
    # of the arithmetic model, not something the judge applies afterwards.
    W, IC, OC = 32, 64, 64
    x = np.full(W * IC, 127, np.int8)
    w = np.full(IC * OC, 127, np.int8)
    out = kernels.conv2dk1_ref(x, w, W, IC, OC, 0)
    assert out.dtype == np.uint8 and set(out.tolist()) == {255}


# --------------------------------------------------------------------------
# rounding mode
# --------------------------------------------------------------------------

_SET_ROUNDING_CALL = re.compile(r"^\s*(?!//)[^/\n]*\bset_rounding\s*\(", re.M)
_NARROWS = re.compile(r"to_vector<|\.srs\(|srs<|to_fixed|to_float")


@pytest.mark.parametrize("arch", ["aie2", "aie2p"])
def test_setup_is_declared_exactly_where_the_source_does_not_set_the_mode(arch):
    """A kernel that sets its own rounding mode must not also name a ``setup``.

    The two are alternatives: either the compiled source calls
    ``aie::set_rounding`` on entry, or the design sets the mode before the
    first call. Declaring both means the design fights the kernel; declaring
    neither, on a kernel that narrows an accumulator, means it runs in
    whatever mode the core booted in.
    """
    from aie.utils.compile.remarks import kernel_builds

    set_current_device(NPU1Col1() if arch == "aie2" else NPU2Col1())
    for name, ef in kernel_builds():
        c = getattr(ef, "contract", None)
        if c is None:
            continue
        src = Path(ef.source_file).read_text() if ef.source_file else ef.source_string
        sets_own = bool(_SET_ROUNDING_CALL.search(src))
        if sets_own:
            assert c.setup is None, f"{name}: source sets the mode and names a setup"
        elif c.setup is not None:
            # A setup only matters where an accumulator is narrowed: a bf16
            # or bfp16 output, or an explicit conversion in the source.
            out_dt = kd.shape_dtype(kd._arg_types(ef)[c.out_index])[1]
            narrows = (
                bfp.is_bfp(out_dt)
                or np.dtype(out_dt) == np.dtype(bfloat16)
                or _NARROWS.search(src)
            )
            assert narrows, f"{name}: names a setup but narrows nothing"


def test_a_design_runs_the_setup_a_contract_names():
    """A design for a kernel with a ``setup`` binds it; one without does not."""
    assert kernels.add().contract.setup is not None
    mlir = str(kd.design(kernels.add, calls=2).as_mlir())
    assert "set_rounding_conv_even" in mlir
    assert kernels.convert_copy().contract.setup is None
    mlir = str(kd.design(kernels.convert_copy, calls=1).as_mlir())
    assert "set_rounding" not in mlir


def test_bf16_exp_clamp_matches_the_kernel_headers():
    """``_EXP_BF16_CLAMP`` tracks ``EXP_BF16_CLAMP`` in the kernel sources.

    ``bf16_exp_ref`` describes the device as ``exp(clip(x, -C, C))``. That is
    only true while the Python constant and the C++ one agree; if they ever
    drift the reference silently stops modelling the kernel, which is the
    class of bug these contracts exist to catch.
    """
    from aie.iron.kernels.activation import _EXP_BF16_CLAMP
    from aie.utils import config

    pattern = re.compile(r"constexpr\s+float\s+EXP_BF16_CLAMP\s*=\s*([0-9.]+)f")
    checked = []
    for arch in ("AIE2", "AIE2P"):
        header = Path(config.aie_runtime_lib_dir()) / arch / "lut_based_ops.h"
        if not header.exists():
            continue
        found = pattern.search(header.read_text())
        assert found, f"{header}: no EXP_BF16_CLAMP definition"
        assert float(found.group(1)) == _EXP_BF16_CLAMP, (
            f"{header} clamps at {found.group(1)} but bf16_exp_ref uses "
            f"{_EXP_BF16_CLAMP}; the reference no longer matches the kernel"
        )
        checked.append(arch)
    assert checked, "no lut_based_ops.h found to check the clamp against"


@pytest.mark.parametrize(
    "tile_size,dtype,reason",
    [
        (64, np.int32, "iterations"),  # 256 B: 4 vector copies, below the 6 assumed
        (128, np.int16, "iterations"),  # 256 B again, via a different dtype
        (80, np.int32, "iterations"),  # 320 B: 5 copies, still short
        (100, np.int32, "vector"),  # 400 B: not a whole number of 64-B copies
    ],
)
def test_passthrough_rejects_tiles_its_loop_cannot_handle(tile_size, dtype, reason):
    """A tile below passThrough.cc's assumed trip count is refused, not run.

    The loop declares AIE_LOOP_MIN_ITERATION_COUNT(6) and steps by a whole
    64-byte vector. Violating either is undefined: a 4-iteration tile hangs
    the core on Phoenix rather than returning wrong data, so the factory has
    to reject it up front.
    """
    with pytest.raises(ValueError, match="passthrough"):
        kernels.passthrough(tile_size=tile_size, dtype=dtype)


@pytest.mark.parametrize(
    "tile_size,dtype",
    [(96, np.int32), (192, np.int16), (384, np.uint8), (4096, np.int32)],
)
def test_passthrough_accepts_its_smallest_legal_tile(tile_size, dtype):
    """384 bytes -- exactly 6 vector copies -- is the smallest tile that builds."""
    assert kernels.passthrough(tile_size=tile_size, dtype=dtype) is not None


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (kernels.reduce_add, dict(tile_size=64)),  # 4 iterations, min 8
        (kernels.reduce_min, dict(tile_size=64)),
        (kernels.reduce_max, dict(tile_size=64)),
        (kernels.reduce_max, dict(tile_size=128, dtype=bfloat16)),  # 32-lane
        (kernels.reduce_add, dict(tile_size=100)),  # not a whole vector
    ],
)
def test_reduce_rejects_tiles_below_its_declared_trip_count(factory, kwargs):
    """reduce_*.cc declares AIE_LOOP_MIN_ITERATION_COUNT(8); reduce_add hangs below it.

    reduce_max.cc says the same thing as ``assert(input_size / VECTOR_SIZE >=
    8)``, which the -DNDEBUG build drops -- so the factory is the only place
    the precondition can still be enforced.
    """
    with pytest.raises(ValueError, match="reduce_"):
        factory(**kwargs)


def test_reduce_scalar_path_has_no_trip_count_floor():
    """Only the vectorised path carries the pragma, so the scalar one stays free."""
    assert kernels.reduce_add(tile_size=64, vectorized=False) is not None


@pytest.mark.parametrize(
    "dims,mac",
    [((64, 64, 64), (4, 8, 8)), ((128, 64, 64), (8, 8, 8)), ((64, 32, 64), (4, 4, 8))],
)
def test_mm_stream_dims_match_the_blocking_the_kernel_was_compiled_for(dims, mac):
    """A and B come from taplib; C is the one layout taplib cannot express.

    The A/B transforms are a plain (r x s) blocked walk, so they ask
    TensorTiler2D for it. This pins that the answer is still the layout
    ``mm.cc`` expects, byte for byte, rather than whatever the tiler happens
    to return after a change.
    """
    (m, k, n), (r, s, t) = dims, mac
    d = kernels.mm_stream_dims(m, k, n, mac)
    assert d.A == [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    assert d.B == [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    assert d.C == [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    col = kernels.mm_stream_dims(m, k, n, mac, b_col_maj=True, c_col_maj=True)
    assert col.B == [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
    assert col.C == [(n // t, t * m), (t, r), (m // r, r * t), (r, 1)]
