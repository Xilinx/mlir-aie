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
import sys
import typing
from pathlib import Path

import numpy as np
import pytest
from aie.iron import In, InOut, Out, kernels
from aie.iron.algorithms import kernel_design as kd
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels import KernelContract, Param
from aie.iron.kernels._common import ARCH_TRAITS
from aie.utils import bfp, get_current_device
from aie.utils.hostruntime import set_current_device
from aie.utils.verify import Tolerance, compare
from ml_dtypes import bfloat16

# One table for every tier: the host checks below walk the same cases the
# device test runs (test/python/npu/kernel_cases.py), so a kernel is either
# in that table or named in NOT_JUDGED with a reason.
sys.path.insert(0, str(Path(__file__).parent / "npu"))
from kernel_cases import CASES as DEVICE_CASES  # noqa: E402


def _case_id(case) -> str:
    parts = [case.factory]
    parts += [
        f"{k}={getattr(v, '__name__', v)}" for k, v in sorted(case.kwargs.items())
    ]
    parts.append(f"calls={case.calls}")
    if case.scalars:
        parts.append("scalars=" + ",".join(str(v) for v in case.scalars))
    if case.tag:
        parts.append(case.tag)
    parts += [f"arg{i}@{offset}" for i, offset in case.arg_byte_offsets]
    return "/".join(parts)


# The fixture below binds npu2, so an npu1-only case is left to its device run.
NPU2_CASES = [case for case in DEVICE_CASES if case.supported_on("npu2")]
CASES = {
    _case_id(case): (case.kwargs, dict(calls=case.calls, scalars=case.scalars))
    for case in NPU2_CASES
}
assert len(CASES) == len(NPU2_CASES), "two device cases map to one host id"

# Exported factories the generic builder does not judge, each with its reason.
# Anything else exported must carry a contract and appear in the case table.
NOT_JUDGED = {
    "cascade_mm": "the GET half of a cascade pair; test_kernels_e2e.py builds and judges the pair",
    "cascade_mm_put": "the PUT half of that pair; its result leaves on the cascade stream",
    "set_rounding": "sets core state and has no data output; the rounding-mode tests cover it",
    **{
        name: "one half of a MobileNet bottleneck cascade pair; test_bn_cascade_pairs.py builds and judges the pair"
        for name in (
            "bn_conv2dk1_partial_put_i8",
            "bn_conv2dk1_partial_get_relu_i8",
            "bn_conv2dk1_input_split_partial_put_ui8",
            "bn_conv2dk1_input_split_partial_skip_get",
        )
    },
}


def _factory(case_id: str):
    return getattr(kernels, case_id.split("/")[0])


@pytest.fixture(autouse=True)
def _aie2p_device(npu2_device):
    # Factories pick sources and mac_dims from the current device.
    yield


def test_every_case_names_an_exported_factory():
    for case_id in CASES:
        assert callable(_factory(case_id)), case_id


def test_device_fixture_restores_previous_device(request):
    # By path: with test/python/npu collected too, "conftest" is npu's.
    conftest = request.config.pluginmanager.get_plugin(
        str(Path(__file__).with_name("conftest.py"))
    )
    previous = get_current_device(probe_runtime=False)
    binding = conftest.npu2_device.__wrapped__()
    next(binding)
    assert isinstance(get_current_device(probe_runtime=False), NPU2Col1)
    binding.close()
    assert get_current_device(probe_runtime=False) is previous


def test_factories_lists_every_exported_builder():
    """``kernels.factories()`` is what the sweeps walk; a builder it misses is never checked.

    The rule is the declared return type, so this pins the rule against the
    export list: everything exported that is neither a reference nor one of
    the two matmul query helpers must be in it.
    """
    exported = {n for n in kernels.__all__ if inspect.isfunction(getattr(kernels, n))}
    not_builders = {n for n in exported if n.endswith("_ref")}
    not_builders |= {"mm_stream_dims", "mm_acc_dtype"}
    assert set(kernels.factories()) == exported - not_builders
    assert {"mm", "mv", "cascade_mm", "cascade_mm_put"} <= set(kernels.factories())
    assert kernels.factories() == [
        n for n in kernels.__all__ if n in exported - not_builders
    ]


@pytest.mark.parametrize(
    "annotation",
    [
        ExternalFunction,
        kernels.MatrixKernel,
        "ExternalFunction",
        "kernels.MatrixKernel",
    ],
)
def test_factories_accept_subclasses_and_postponed_annotations(monkeypatch, annotation):
    def builder():
        raise AssertionError("discovery must not construct kernels")

    builder.__annotations__["return"] = annotation
    monkeypatch.setattr(kernels, "__all__", ["zero"])
    monkeypatch.setattr(kernels, "zero", builder)
    assert kernels.factories() == ["zero"]


def _first_kwargs(name: str) -> dict:
    """Keyword arguments that build ``name``: none, or its first case's."""
    signature = inspect.signature(getattr(kernels, name))
    try:
        signature.bind()
        return {}
    except TypeError:
        covered = [
            fkw for case_id, (fkw, _) in CASES.items() if case_id.split("/")[0] == name
        ]
        assert covered, f"{name}: required-argument factory needs a case"
        signature.bind(**covered[0])
        return covered[0]


def _builds():
    """Every exported factory at its defaults and at each declared dtype combination.

    A factory that refuses the current device (``NotImplementedError``) is
    skipped: it exists only for the other architecture.
    """
    for name in kernels.factories():
        f = getattr(kernels, name)
        seen = set()
        for combo in [_first_kwargs(name)] + [
            dict(c) for c in getattr(f, "dtypes", ())
        ]:
            try:
                ef = f(**combo)
            except NotImplementedError:
                continue
            if ef.object_file_name in seen:
                continue  # the default build is one of the dtypes entries
            seen.add(ef.object_file_name)
            yield name, ef


def test_contract_coverage_is_explicit():
    """Every exported factory is in the case table with a contract, or in NOT_JUDGED with a reason."""
    in_table = {case.factory for case in DEVICE_CASES}
    without, unlisted = [], []
    for name in kernels.factories():
        ef = getattr(kernels, name)(**_first_kwargs(name))
        if name in NOT_JUDGED:
            assert name not in in_table, f"{name}: in the table and in NOT_JUDGED"
            continue
        if ef.contract is None:
            without.append(name)
        if name not in in_table:
            unlisted.append(name)
    assert not without, f"factories without contracts: {without}"
    assert not unlisted, f"factories with no case: {unlisted}"
    assert set(NOT_JUDGED) <= set(kernels.factories())
    assert all(NOT_JUDGED.values())


@pytest.mark.parametrize("error", [RuntimeError, TypeError, ValueError])
def test_contract_coverage_propagates_constructor_failures(monkeypatch, error):
    def broken() -> ExternalFunction:
        raise error("factory construction failed")

    monkeypatch.setattr(kernels, "__all__", ["zero"])
    monkeypatch.setattr(kernels, "zero", broken)
    with pytest.raises(error, match="factory construction failed"):
        test_contract_coverage_is_explicit()


def test_contract_coverage_constructs_required_case_arguments(monkeypatch):
    from types import SimpleNamespace

    sizes = []

    def required(*, tile_size) -> ExternalFunction:
        sizes.append(tile_size)
        return SimpleNamespace(contract=KernelContract(roles=(Out,)))

    monkeypatch.setattr(kernels, "__all__", ["zero"])
    monkeypatch.setattr(kernels, "zero", required)
    monkeypatch.setattr(sys.modules[__name__], "NOT_JUDGED", {})
    monkeypatch.setitem(CASES, "zero", ({"tile_size": 64}, {}))
    test_contract_coverage_is_explicit()
    assert sizes == [64]


@pytest.mark.parametrize("case_id", list(CASES))
def test_roles_match_arg_types(case_id):
    fkw, _ = CASES[case_id]
    fn = _factory(case_id)(**fkw)
    c = fn.contract
    assert c is not None, f"{case_id}: no contract"
    assert len(c.roles) == len(
        fn.arg_types()
    ), f"{case_id}: {len(c.roles)} roles for {len(fn.arg_types())} arguments"
    assert set(c.roles) <= {In, Out, InOut, Param}


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
    refs = ref if isinstance(ref, tuple) else (ref,)
    for i, r in zip(fn.contract.out_indices, refs):
        out_dt = kd.shape_dtype(fn.arg_types()[i])[1]
        assert r.dtype == (np.float32 if bfp.is_bfp(out_dt) else out_dt)


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
    out_index = fn.contract.out_index
    assert ref.shape == (1, np.prod(fn.contract.layouts[out_index].shape))
    assert [a.n_elements for a in kd.host_args(fn)][-1] == kd.output_size(fn)
    assert "func.call" in str(kd.design(factory).as_mlir())


@pytest.mark.parametrize(
    "factory, kwargs",
    [(kernels.add_weighted, dict(line_width=64)), (kernels.add, {})],
    ids=["uint8", "bfloat16"],
)
def test_guard_sizes_and_strips_each_tile(factory, kwargs):
    fn = factory(**kwargs)
    out_dt = fn.output_dtype()
    n = kd.output_size(fn, calls=3)
    size = kd.output_size(fn, calls=3, guard=True)
    assert size == n + 3 * kd.GUARD_BYTES // np.dtype(out_dt).itemsize
    assert [a.n_elements for a in kd.host_args(fn, calls=3, guard=True)][-1] == size
    data = np.arange(n * np.dtype(out_dt).itemsize, dtype=np.uint8).reshape(3, -1)
    raw = np.hstack([data, np.full((3, kd.GUARD_BYTES), 0x55, np.uint8)])
    got, overrun = kd.strip_guard(fn, raw.reshape(-1).view(out_dt), calls=3)
    np.testing.assert_array_equal(got.view(np.uint8), data.reshape(-1))
    assert overrun == 0
    raw[2, data.shape[1]] = 0
    assert kd.strip_guard(fn, raw.reshape(-1).view(out_dt), calls=3)[1] == 1


def test_guarded_design_hands_the_kernel_a_view_of_its_tile():
    mlir = str(
        kd.design(
            kernels.add_weighted,
            line_width=64,
            calls=2,
            scalars=(8192, 8192, 0),
            guard=True,
        ).as_mlir()
    )
    assert "!aie.objectfifo<memref<128xi8>>" in mlir
    assert "memref<128xi8> to memref<64xui8>" in mlir
    # The tile and its guard are poisoned by a call: a loop in main would keep
    # the object FIFO lowering from unrolling the calls.
    assert "memref<128xi8> to memref<32xi32>" in mlir
    assert "func.call @kd_poison_32(" in mlir
    plain = str(
        kd.design(
            kernels.add_weighted, line_width=64, calls=2, scalars=(8192, 8192, 0)
        ).as_mlir()
    )
    assert mlir.count("scf.for") == plain.count("scf.for")


def test_guard_covers_an_initialized_output():
    mlir = str(kd.design(kernels.mm, calls=2, guard=True).as_mlir())
    assert "memref.view" in mlir


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
    assert fn.expected(inputs).shape == (
        2,
        kd.elems(fn.arg_types()[fn.contract.out_index]),
    )
    assert "scf.for" in str(kd.design(kernels.mm, calls=2).as_mlir())


def test_scalar_counts_are_bound_not_inferred_from_tensor_sizes():
    assert kernels.reduce_max(tile_size=1024).contract.parameter_bindings == (
        (2, 1024),
    )
    assert kernels.rgba2hue(line_width=64).contract.parameter_bindings == ((2, 64),)
    assert kernels.gray2rgba(line_width=64).contract.parameter_bindings == ((2, 64),)
    fn = kernels.leaky_relu(tile_size=1024)
    assert fn.contract.parameter_bindings == ((2, 1024),)
    assert fn.contract.reference_indices() == [0, 3]
    mlir = str(kd.design(kernels.leaky_relu, tile_size=1024, scalars=(0.5,)).as_mlir())
    assert "1024 : i32" in mlir


def test_rounding_setup_is_merged_alwaysinline_ir():
    setter = kernels.conv_even()
    assert setter._inline
    assert setter.object_file_name.endswith(".ll")
    assert setter._symbol_prefix is None
    assert setter == kernels.conv_even()
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
    if arch == "aie2":
        tolerance = fn.contract.tolerance
        assert tolerance is not None
        assert tolerance.atol == 2.0**-126


@pytest.mark.parametrize(
    "device,reference",
    [(NPU1Col1, kernels.swiglu_lut_ref), (NPU2Col1, kernels.swiglu_ref)],
)
def test_swiglu_default_reference_matches_architecture(device, reference):
    set_current_device(device())
    assert kernels.swiglu().contract.reference is reference


def test_factory_contract_follows_device_switch():
    set_current_device(NPU2Col1())
    npu2 = kernels.mha_softmax()
    set_current_device(NPU1Col1())
    npu1 = kernels.mha_softmax()
    assert npu1 != npu2
    assert npu1.contract.tolerance.rtol < npu2.contract.tolerance.rtol
    set_current_device(NPU2Col1())
    assert kernels.mha_softmax() == npu2


@pytest.mark.parametrize("mode", list(kernels.RoundingMode))
def test_rounding_mode_preserves_string_api(mode):
    assert str(mode) == f"{mode}" == mode.value
    assert mode == mode.value
    setter = kernels.set_rounding(mode)
    assert setter.name == f"set_rounding_{mode.value}"
    assert f"-DROUNDING_MODE={mode.value}" in setter.compile_flags
    assert setter == kernels.set_rounding(mode.value)


@pytest.mark.parametrize(
    "factory,minimum",
    [
        (kernels.conv2dk1, 1088),
        (kernels.conv2dk1_skip, 512),
        (kernels.conv2dk1_skip_init, 1216),
        (kernels.conv2dk3, 384),
    ],
)
def test_stack_contract_covers_measured_core(factory, minimum):
    assert factory().contract.stack_bytes >= minimum


@pytest.mark.parametrize(
    "device,portable,minimum",
    [
        (NPU2Col1, False, 896),
        (NPU2Col1, True, 896),
        (NPU1Col1, False, 160),
        (NPU1Col1, True, 736),
    ],
)
def test_layer_norm_f32_stack_covers_measured_core(
    monkeypatch, device, portable, minimum
):
    # aiecc's measured_stack_size, plus the 64-byte frame of __mulsf3 or
    # __divsf3 where the build calls one: compiler-rt emits no .stack_sizes.
    if portable:
        monkeypatch.setenv("AIE_KERNELS_PORTABLE", "1")
    set_current_device(device())
    mlir = str(kd.design(kernels.layer_norm_f32, cols=1024, calls=16).as_mlir())
    stack_sizes = re.findall(r"stack_size = (\d+) : i32", mlir)
    assert stack_sizes
    assert all(int(size) >= minimum for size in stack_sizes)


@pytest.mark.parametrize("portable", [False, True])
@pytest.mark.parametrize(
    "device,dim_k,dim_n,minimum",
    [
        (NPU2Col1, 56, 16, 1088),
        (NPU2Col1, 72, 32, 1024),
        (NPU2Col1, 144, 32, 2240),
        (NPU2Col1, 256, 16, 4160),
        (NPU2Col1, 256, 32, 4032),
        (NPU2Col1, 384, 16, 6208),
        (NPU1Col1, 256, 32, 512),
    ],
)
def test_mm_i8_i32_stack_covers_measured_core(
    monkeypatch, device, dim_k, dim_n, minimum, portable
):
    # aiecc's measured_stack_size, worst of the b_col_maj/c_col_maj builds.
    if portable:
        monkeypatch.setenv("AIE_KERNELS_PORTABLE", "1")
    set_current_device(device())
    fn = kernels.mm(
        dim_m=32,
        dim_k=dim_k,
        dim_n=dim_n,
        input_dtype=np.int8,
        output_dtype=np.int32,
    )
    assert kd._stack_bytes(fn) >= minimum


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        # In the fitted range: 16 * dim_k + 256.
        (dict(dim_k=56, input_dtype=np.int8, output_dtype=np.int32), 16 * 56 + 256),
        (dict(dim_k=408, input_dtype=np.int8, output_dtype=np.int32), 16 * 408 + 256),
        # Excluded at both ends: the device default (None) covers these.
        (dict(dim_k=48, input_dtype=np.int8, output_dtype=np.int32), None),
        (dict(dim_k=416, input_dtype=np.int8, output_dtype=np.int32), None),
        # Excluded by not being the tuned aie2p/vectorized/int8->int32 case.
        (
            dict(
                dim_k=200,
                input_dtype=np.int8,
                output_dtype=np.int32,
                vectorized=False,
            ),
            None,
        ),
        (dict(dim_k=200, input_dtype=np.int16, output_dtype=np.int32), None),
    ],
)
def test_mm_i8_i32_stack_formula_envelope(kwargs, expected):
    # Pins the 48 < dim_k < 416 fit boundaries themselves (kernel_cases.py and
    # test_mm_i8_i32_stack_covers_measured_core pin measured values inside
    # them), so a change to the envelope is caught even where it still
    # happens to satisfy every measured minimum above.
    set_current_device(NPU2Col1())
    fn = kernels.mm(dim_m=32, dim_n=16, **kwargs)
    assert fn.contract.stack_bytes == expected


def test_mm_stack_falls_back_off_aie2p_and_under_chess():
    set_current_device(NPU1Col1())
    assert (
        kernels.mm(
            dim_k=200, input_dtype=np.int8, output_dtype=np.int32
        ).contract.stack_bytes
        is None
    )
    set_current_device(NPU2Col1())
    assert (
        kernels.mm(
            dim_k=200, input_dtype=np.int8, output_dtype=np.int32, use_chess=True
        ).contract.stack_bytes
        == 0xD00
    )


@pytest.mark.parametrize(
    "input_width,kernel_width", [(112, 14), (336, 14), (230, 14), (240, 15)]
)
def test_conv2dk14_rejects_shapes_the_vector_paths_skip(input_width, kernel_width):
    # The vector paths step 16 patches and 2 pixels at a time.
    with pytest.raises(ValueError, match="conv2dk14"):
        kernels.conv2dk14(input_width=input_width, kernel_width=kernel_width)


@pytest.mark.parametrize(
    "device,portable,channels,minimum",
    [
        (NPU2Col1, False, 448, 1280),
        (NPU2Col1, False, 224, 1024),
        (NPU2Col1, True, 256, 1280),
        (NPU1Col1, True, 192, 416),
        (NPU1Col1, False, 416, 1056),
        (NPU1Col1, False, 1248, 9760),
    ],
)
def test_dwconv1d_channels_last_stack_covers_measured_core(
    monkeypatch, device, portable, channels, minimum
):
    # aiecc's measured_stack_size at the worst channel count of each build,
    # and at the first aie2 count that needs more than the default
    if portable:
        monkeypatch.setenv("AIE_KERNELS_PORTABLE", "1")
    set_current_device(device())
    mlir = str(
        kd.design(kernels.dwconv1d_channels_last, channels=channels, calls=1).as_mlir()
    )
    stack_sizes = re.findall(r"stack_size = (\d+) : i32", mlir)
    assert stack_sizes
    assert all(int(size) >= minimum for size in stack_sizes)


@pytest.mark.parametrize("device", [NPU1Col1, NPU2Col1])
@pytest.mark.parametrize("dim_m,dim_n", [(32, 16), (64, 32)])
@pytest.mark.parametrize("epilogue", ["none", "gelu", "silu", "sigmoid"])
def test_fused_mm_stack_covers_accumulator_and_epilogue(device, dim_m, dim_n, epilogue):
    set_current_device(device())
    kwargs = dict(
        dim_m=dim_m,
        dim_k=48,
        dim_n=dim_n,
        epilogue=epilogue,
        clamp=(-0.125, 0.75),
    )
    # The 32x16 AIE2P SiLU+clamp core measured 3648 bytes with pinned Peano.
    accumulator_bytes = np.dtype(np.float32).itemsize * dim_m * dim_n
    minimum = accumulator_bytes + max(device().default_core_stack_bytes, 1600)
    fn = kernels.fused_mm(**kwargs)
    assert fn.contract.stack_bytes >= minimum
    mlir = str(kd.design(kernels.fused_mm, calls=4, **kwargs).as_mlir())
    stack_sizes = re.findall(r"stack_size = (\d+) : i32", mlir)
    assert stack_sizes
    assert all(int(size) >= minimum for size in stack_sizes)


def test_contract_validates_argument_bindings():
    with pytest.raises(ValueError, match="layouts"):
        KernelContract(roles=(In, Out), layouts=(None,))
    with pytest.raises(ValueError, match="parameter_bindings"):
        KernelContract(roles=(In, Out), parameter_bindings=((1, 64),))
    with pytest.raises(ValueError, match="distinct"):
        KernelContract(roles=(In, Out, Param), parameter_bindings=((2, 64), (2, 128)))
    assert KernelContract(roles=(In, Out, Param)).reference_indices() == [0, 2]
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


def test_judge_scales_range_tolerance_per_call():
    fn = kernels.add()
    ref = np.zeros((2, 1024), bfloat16)
    ref[:, 0] = [1024, 4]
    got = ref.copy()
    got[:, 1] = 1
    tol = Tolerance.relative(0.0, range_frac=1 / 1024)
    assert compare(got, ref, tol).ok
    verdict = fn.judge(got.ravel(), ref, calls=2, tolerance=tol)
    assert not verdict.ok
    assert verdict.n_mismatch == 1
    assert verdict.first_bad_index == 1025
    got[1, 1] = 0
    assert fn.judge(got.ravel(), ref, calls=2, tolerance=tol).ok


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


def test_mixed_params_infer_scalar_and_tensor_abi():
    from aie.iron.kernel import ExternalFunction

    tile = np.ndarray[(4,), np.dtype[np.int32]]
    constant = np.arange(4, dtype=np.int32)
    fn = ExternalFunction(
        "mixed_params",
        source_string="void mixed_params() {}",
        arg_types=[np.int32, tile, tile, tile, np.int32, tile, tile, np.int32],
    )
    fn.contract = KernelContract(
        roles=(Param, In, Param, Param, Param, Out, Out, Param),
        parameter_bindings=((0, 4), (3, constant), (7, 7)),
        reference=lambda x, weights, factor: (
            x * weights + factor + constant,
            x - weights,
        ),
        acc_dtype=np.int32,
        stack_bytes=1024,
    )
    inputs = kd.sample_inputs(fn, calls=3)
    assert [a.shape for a in inputs] == [(3, 4), (4,)]
    assert fn.contract.reference_indices() == [1, 2, 4]
    np.testing.assert_array_equal(fn.param_values(inputs)[0], inputs[1])
    encoded = kd._encode_params(fn, fn.param_values(inputs))
    assert len(encoded) == 2 and encoded[1][2] == tuple(constant)
    refs = fn.expected(inputs, scalars=(3,))
    np.testing.assert_array_equal(refs[0], inputs[0] * inputs[1] + 3 + constant)
    np.testing.assert_array_equal(refs[1], inputs[0] - inputs[1])
    assert fn.judge(refs, refs, calls=3)
    assert [a.direction for a in kd.host_args(fn, calls=3)] == [In, Out, Out]
    assert len(kd.host_layout(fn, inputs)) == 1
    with pytest.raises(ValueError, match="expected 1 scalar"):
        kd.design(lambda: fn, params=fn.param_values(inputs)).as_mlir()
    with pytest.raises(ValueError, match="expected scalar parameter"):
        kd.design(
            lambda: fn, params=fn.param_values(inputs), scalars=(np.ones(4),)
        ).as_mlir()
    module = kd.design(
        lambda: fn, calls=3, scalars=(3,), params=fn.param_values(inputs)
    ).as_mlir()
    assert "mixed_params" in str(module)
    assert "param0" in str(module) and "param1" in str(module)


def test_param_is_kernel_only_and_host_descriptors_are_private():
    import aie.iron as iron
    from aie.utils.compile import jit
    from aie.utils.compile.jit import markers

    assert Param is kernels.Param
    assert not hasattr(iron, "Param")
    assert (iron.In, iron.Out, iron.InOut) == (jit.In, jit.Out, jit.InOut)
    for name in ("Param", "Scalar", "Count", "ROLES"):
        assert not hasattr(jit, name)
        assert not hasattr(markers, name)
    assert not hasattr(kernels, "ROLES")
    assert not hasattr(kd, "HostArg")
    assert not hasattr(iron.algorithms, "HostArg")


def test_output_only_kernel_uses_no_host_inputs():
    fn = kernels.zero(tile_size=64)
    assert kd.sample_inputs(fn) == []
    assert fn.param_values([]) == []
    assert kd.host_layout(fn, []) == []
    assert [arg.direction for arg in kd.host_args(fn)] == [Out]
    assert fn.judge(np.zeros(64, np.int32), fn.expected([]))
    assert kd.design(kernels.zero, tile_size=64) is not None


def test_parameter_only_kernel_can_have_multiple_outputs():
    from aie.iron.kernel import ExternalFunction

    tile = np.ndarray[(4,), np.dtype[np.int32]]
    fn = ExternalFunction(
        "param_outputs", source_string="", arg_types=[tile, np.int32, tile, tile]
    )
    fn.contract = KernelContract(
        roles=(Param, Param, Out, Out),
        reference=lambda weights, factor: (weights * factor, weights + factor),
        stack_bytes=1024,
    )
    inputs = kd.sample_inputs(fn)
    assert len(inputs) == 1
    assert kd.host_layout(fn, inputs) == []
    assert [arg.direction for arg in kd.host_args(fn)] == [Out, Out]
    refs = fn.expected(inputs, scalars=(2,))
    np.testing.assert_array_equal(refs[0], inputs[0] * 2)
    actuals = tuple(np.tile(ref, (3, 1)) for ref in refs)
    assert fn.judge(actuals, refs, calls=3)
    actuals[1][-1, -1] += 1
    assert not fn.judge(actuals, refs, calls=3)
    design = kd.design(
        lambda: fn, params=fn.param_values(inputs), scalars=(2,), calls=3
    )
    assert "param_outputs" in str(design.as_mlir())


def test_fifo_plan_groups_hashable_numpy_abi_types():
    from aie.iron.kernel import ExternalFunction

    i16 = np.ndarray[(4,), np.dtype[np.int16]]
    same_i16 = np.ndarray[(4,), np.dtype[np.int16]]
    i32 = np.ndarray[(4,), np.dtype[np.int32]]
    assert i16 == same_i16 and hash(i16) == hash(same_i16)
    fn = ExternalFunction(
        "typed_groups", source_string="", arg_types=[i16, i32, same_i16, i16]
    )
    fn.contract = KernelContract(roles=(In, In, In, Out))
    assert kd._fifo_plan(fn) == ([[0, 2], [1]], [0, 1, 2], [])


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

    # bfp quantizes in blocks of 8 along K, so K must be a multiple of 8.
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
    # A tensor Param is baked into the design, so its value is part of the key.
    p3 = kd.design(kernels.scale, calls=4, dtype=np.int32, params=[np.array([3])])
    p5 = kd.design(kernels.scale, calls=4, dtype=np.int32, params=[np.array([5])])
    assert h(p3) != h(p5)
    with pytest.raises(ValueError, match=r"expected 1 param value\(s\)"):
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
    assert ref(px(0, 0, 0)).tolist() == [0]  # gray: hue 0
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

    ``rgba2hue_ref`` is bit-exact against the kernel by construction; this
    pins the other half -- that the arithmetic they share stays within one LSB
    of the exact hue, over every RGB triple.
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
    # output channel c equals input channel c, requantized by >> 0.
    x = np.arange(W * IC, dtype=np.int8).reshape(IC // 8, W, 8)  # [C/8][W][8]
    ident = np.eye(8, dtype=np.int8).reshape(OC // 8, IC // 8, 8, 8)
    out = kernels.conv2dk1_ref(x.ravel(), ident.ravel(), W, IC, OC, 1)
    # scale 1: (v + 1) >> 1
    assert out.tolist() == (((x.astype(np.int64) + 1) >> 1).ravel()).tolist()
    # conv2dk3: center tap identity, zero-padded borders; middle region
    # returns line1 requantized, top region ignores line0, bottom line2.
    w = np.zeros((OC // 8, IC // 8, 3, 3, 8, 8), np.int8)
    w[:, :, 1, 1] = np.eye(8, dtype=np.int8)  # row 1 (line1), ki 1 (x + 0)
    l0 = np.full(W * IC, 100, np.int8)
    l1 = np.arange(W * IC, dtype=np.int8)
    l2 = np.full(W * IC, -100, np.int8)
    got = kernels.conv2dk3_ref(l0, l1, l2, w.ravel(), W, IC, OC, 3, 3, 1, 1, 0)
    assert got.tolist() == ((l1.astype(np.int64) + 1) >> 1).tolist()
    w[:, :, 0, 1] = np.eye(8, dtype=np.int8)  # add line0's center tap
    mid = kernels.conv2dk3_ref(l0, l1, l2, w.ravel(), W, IC, OC, 3, 3, 1, 1, 0)
    top = kernels.conv2dk3_ref(l0, l1, l2, w.ravel(), W, IC, OC, 3, 3, 0, 1, 0)
    assert (
        mid.tolist() == np.clip((l1.astype(np.int64) + 100 + 1) >> 1, 0, 255).tolist()
    )
    assert top.tolist() == got.tolist()
    # Left neighbor tap on a line with a single lit pixel: shifts right by one,
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
    assert c.roles == (In, In, InOut)
    a, b = kd.sample_inputs(fn, calls=2)
    assert a.dtype == np.float32 and a.shape == (2, M, K) and b.shape == (2, K, N)
    # Same-type operands share one FIFO, in per-call argument order.
    (packed,) = kd.host_layout(fn, [a, b])
    ha, hb = packed[:, 0], packed[:, 1]
    assert ha.dtype == np.uint8 and ha.shape == (2, M * K * 9 // 8)
    m, k, n = M, K, N
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


@pytest.mark.parametrize(
    "kwargs,label",
    [
        (dict(input_channels=8), "input_channels"),
        (dict(input_channels=24), "input_channels"),
        (dict(input_channels=0), "input_channels"),
        (dict(output_channels=12), "output_channels"),
        (dict(output_channels=0), "output_channels"),
        (dict(skip_input_channels=4), "skip_input_channels"),
        (dict(skip_input_channels=-8), "skip_input_channels"),
    ],
)
def test_skip_init_rejects_partial_channel_steps(kwargs, label):
    """The source steps channels in whole 16s (input) and 8s (output, skip)."""
    with pytest.raises(ValueError, match=f"{label} must be a positive multiple"):
        kernels.conv2dk1_skip_init(**kwargs)


def test_skip_init_accepts_whole_channel_steps():
    fn = kernels.conv2dk1_skip_init(
        input_channels=16, output_channels=8, skip_input_channels=8
    )
    assert fn.arg_shape(2) == ((16 + 8) * 8,)


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
        "mv_bf16.cc",
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
    # The exported symbol may carry the memoization digest prefix; the kernel
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
    assert fn.contract.roles == (Param, Param, In, In, Out)
    # row_offset shifts the write into c, so one core fills several blocks.
    a = np.arange(4 * 8, dtype=np.float32).reshape(4, 8).astype(bfloat16)
    b = np.ones(8, dtype=bfloat16)
    assert kernels.mv_bf16_ref(2, 1, a, b).tolist() == [0.0, 28.0, 92.0]
    # The int16 kernel accumulates and declares an independent initializer.
    i16 = kernels.mv(dim_m=32, dim_k=32)
    assert i16.contract.roles == (In, In, InOut) and i16.contract.initializers
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
    ins = [a for a in args if a.direction is In]
    outs = args[len(ins) :]
    assert [a.direction for a in args] == [In] * len(ins) + [Out] * len(outs)
    assert len(outs) == len(fn.contract.out_indices), case_id
    # Inputs: the arrays host_layout hands the device.
    staged = kd.host_layout(fn, kd.sample_inputs(fn, calls=calls, shape=shape))
    assert len(staged) == len(ins), case_id
    for got, spec in zip(staged, ins):
        assert got.shape == spec.shape, f"{case_id}: {got.shape} != {spec.shape}"
        assert got.dtype == np.dtype(spec.dtype), case_id
    # Outputs: the element counts and dtypes upload allocates.
    ref = fn.expected(
        kd.sample_inputs(fn, calls=calls, shape=shape),
        scalars=opts.get("scalars", ()),
    )
    sizes = kd.output_size(fn, calls=calls, shape=shape)
    dtypes = fn.output_dtype(None if isinstance(ref, tuple) else ref.dtype)
    if len(outs) == 1:
        sizes, dtypes = (sizes,), (dtypes,)
    for out, size, dtype in zip(outs, sizes, dtypes, strict=True):
        assert out.n_elements == size, case_id
        assert np.dtype(out.dtype) == np.dtype(dtype), case_id


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
    """mha.cc entry points bind explicitly through one artifact owner."""
    fn = kernels.mha(dim_m=64, dim_k=64, dim_n=64)
    p = fn._symbol_prefix
    assert fn.name == f"{p}_matmul_bf16_bf16_wrapper"
    tile = np.ndarray[(64 * 64,), np.dtype[bfloat16]]
    scale = np.ndarray[(64,), np.dtype[bfloat16]]
    idx = np.ndarray[(2,), np.dtype[np.int32]]
    expected = {
        "matmul_bf16_bf16_wrapper_scalar": [tile, tile, tile],
        "matmul_bf16_bf16_rowmaj": [tile, tile, tile],
        "partial_softmax": [tile, tile, scale, idx, bfloat16, *([np.int32] * 4)],
        "matmul_PV": [tile, tile, tile, scale, np.int32, np.int32, idx],
        "rescale_O": [tile, scale, np.int32, idx],
        "init_scale_buffer": [scale, np.int32],
    }
    for symbol, arg_types in expected.items():
        sib = fn.object_file.bind(symbol, arg_types)
        assert sib.name == f"{p}_{symbol}"
        assert sib.object_file is fn.object_file
    # Its own matmul symbols cannot collide with a real mm in one design.
    mm = kernels.mm(
        dim_m=64,
        dim_k=64,
        dim_n=64,
        input_dtype=bfloat16,
        output_dtype=bfloat16,
    )
    assert mm.name != fn.name
    # The wrapper is mm.cc's bf16 product with its gate bound open, so it is
    # declared, sampled and judged exactly like mm.
    assert fn.contract.unsupported is None and fn.contract.accumulates
    assert (fn.mac_dims, fn.stream_dims) == (mm.mac_dims, mm.stream_dims)
    assert dict(fn.contract.parameter_bindings)[3].tolist() == [0, 0]
    a, b = kd.sample_inputs(fn, calls=2)
    assert np.array_equal(
        fn.expected([a, b]),
        kernels.mm_tile_ref(a, b, dim_m=64, dim_k=64, dim_n=64).astype(bfloat16),
    )
    assert fn.name in str(kd.design(kernels.mha, calls=2).as_mlir())
    with pytest.raises(ValueError, match="multiple of"):
        kernels.mha(dim_m=17)


_C_ELEMENT_NAMES = {bfloat16: "bf16", np.float32: "float", np.int32: "int"}


def _extern_c_signatures(source_file: str) -> dict[str, list[tuple[str, bool]]]:
    """Parameters of every ``void`` entry point in a C++ translation unit.

    Each parameter comes back as ``(element type, is a pointer)``, qualifiers
    dropped -- the two facts one ``arg_types`` entry carries.
    """
    body = Path(source_file).read_text()
    signatures = {}
    for name, params in re.findall(r"\bvoid\s+(\w+)\(([^)]*)\)\s*\{", body):
        parsed = []
        for param in params.split(","):
            tokens = [
                token
                for token in param.replace("*", " * ").split()
                if token not in ("const", "__restrict", "restrict")
            ]
            parsed.append((tokens[0], "*" in tokens))
        signatures[name] = parsed
    return signatures


def _arg_type_facts(arg_type) -> tuple[str, bool]:
    """Return the ``(element type, is a pointer)`` pair for one arg_types entry."""
    args = typing.get_args(arg_type)
    if not args:
        return _C_ELEMENT_NAMES[arg_type], False
    (dtype,) = typing.get_args(args[1])
    return _C_ELEMENT_NAMES[dtype], True


def test_prefill_binds_its_translation_unit_as_one_object():
    """flash_attn_prefill.cc's five entry points bind through one artifact owner."""
    fn = kernels.prefill_fv(head_dim=512)
    assert fn.contract.setup is kernels.conv_even
    p = fn._symbol_prefix
    assert fn.name == f"{p}_prefill_fv_step"
    bf = lambda n: np.ndarray[(n,), np.dtype[bfloat16]]  # noqa: E731
    f32 = lambda n: np.ndarray[(n,), np.dtype[np.float32]]  # noqa: E731
    i32b = np.ndarray[(1,), np.dtype[np.int32]]
    expected = {
        "prefill_round_begin": [bf(8), bf(8), f32(8), f32(8), f32(8 * 512)],
        "prefill_qk_step": [
            bf(64), bf(8 * 512), bf(8 * 512), bf(64), bf(8),
            i32b, i32b, *([np.int32] * 5),
        ],  # fmt: skip
        "prefill_block_mid": [
            bf(64),
            bf(64),
            bf(8),
            bf(8),
            f32(8),
            f32(8),
            f32(8 * 512),
        ],
        "prefill_epilogue": [bf(64), f32(8), f32(8 * 512), np.int32],
    }
    # bind() is pure string qualification -- it consults neither the symbol
    # table nor the signature -- so the loop below alone would pass against a
    # table naming entry points that no longer exist. Read them off the C++,
    # and a renamed symbol or a changed parameter list fails here instead.
    declared = _extern_c_signatures(fn.source_file)
    assert set(declared) == set(expected) | {"prefill_fv_step"}
    assert [_arg_type_facts(t) for t in fn.arg_types()] == declared["prefill_fv_step"]
    for symbol, arg_types in expected.items():
        assert [_arg_type_facts(t) for t in arg_types] == declared[symbol]
        sib = fn.object_file.bind(symbol, arg_types)
        assert sib.name == f"{p}_{symbol}"
        assert sib.object_file is fn.object_file

    # The two geometries are two builds, not two tiles of one: differing
    # -DPREFILL_HEAD_DIM must give each its own object and its own symbol, so
    # a design can call both without a duplicate-symbol link error.
    swa = kernels.prefill_fv(head_dim=256)
    assert swa._symbol_prefix != p
    assert swa.object_file is not fn.object_file
    assert swa.name != fn.name

    # y accumulates, so it is inout and ships the zero that clears it -- at
    # argument 0 here, not a matmul's 2.
    assert fn.contract.accumulates and fn.contract.roles[0] is InOut
    assert dict(fn.contract.parameter_bindings)[3] == 0
    assert fn.zero.arg_types() == [np.ndarray[(8 * 512,), np.dtype[np.float32]]]

    # One reference serves both geometries: reorder_s makes S canonical before
    # fv_step sees it, so each is a plain (LQ, LK) x (LK, DH) product.
    for dh, lq, lk in ((512, 8, 8), (256, 16, 16)):
        k = kernels.prefill_fv(head_dim=dh)
        s, v = kd.sample_inputs(k, calls=2)
        want = np.asarray(s, np.float32).reshape(-1, lq, lk) @ np.asarray(
            v, np.float32
        ).reshape(-1, lk, dh)
        assert np.array_equal(k.expected([s, v]), want.reshape(2, lq * dh))

    # The reference accumulates in float32 because the kernel's y does. A
    # float64 one asserts a function the hardware is not defined to compute:
    # bf16 products are exact either way, but the sums part company by a whole
    # float32 ulp once the terms cancel, which mm_tile_ref would then report as
    # an error.
    assert kernels.prefill_fv_ref(s, v, dim_m=lq, dim_k=lk, dim_n=dh).dtype == (
        np.float32
    )
    assert fn.name in str(kd.design(kernels.prefill_fv, calls=2).as_mlir())
    with pytest.raises(ValueError, match="head_dim"):
        kernels.prefill_fv(head_dim=384)


def test_accumulating_kernels_are_inout_and_ship_a_zero():
    """``inout`` marks a kernel that reads its output back, and needs zeroing.

    ``mm``'s ``C += A * B`` reads C, so a design must zero the buffer before
    the first call using its independent initializer. The
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
        zero = c.initializers[0][1](fn)
        assert zero.object_file is not fn.object_file
        assert zero.contract.roles == (Out,)
        assert c.out_index not in c.reference_indices()
    # The autouse fixture selects aie2p, so the bfp matmul builds here too.
    bfp = kernels.mm_bfp()
    assert bfp.contract.accumulates and bfp.contract.initializers
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


def test_sibling_symbols_follow_the_parameterization_prefix():
    """A kernel's siblings bind names its own object actually defines.

    Each parameterization gets a symbol prefix so two of them can share a
    design; the whole object is prefixed, so the cascade
    get/put trio have to be prefixed to match.
    """
    fn = kernels.mm(dim_m=64, dim_k=64, dim_n=64)
    prefix = fn._symbol_prefix
    assert prefix and fn.name == f"{prefix}_matmul_i16_i16"
    # A different parameterization gets a different prefix on every symbol.
    other = kernels.mm(dim_m=32, dim_k=32, dim_n=32)
    assert other._symbol_prefix != prefix
    casc = kernels.cascade_mm()
    cp = casc._symbol_prefix
    for mode in ("put_only", "put_get"):
        sib = casc.object_file.bind(
            f"matmul_scalar_cascade_{mode}_i16_i16", casc.arg_types()
        )
        assert (
            sib.name.startswith(f"{cp}_")
            and sib.object_file_name == casc.object_file_name
        )
    # Only the unspecialized compute_max pins an unprefixed shared object.
    assert not getattr(kernels.compute_max(), "_symbol_prefix", None)


def test_unsupported_contracts_are_refused_by_the_harness():
    fn = kernels.mm_bfp_shuffle(dim_n=32)
    assert fn.contract is not None and fn.contract.unsupported
    with pytest.raises(ValueError, match="cannot build"):
        kd.design(kernels.mm_bfp_shuffle, dim_n=32)
    assert kernels.mm_bfp_shuffle().contract.unsupported is None
    assert kernels.mha().contract.unsupported is None


def test_bfp_shuffle_contract_uses_declared_storage_codecs():
    fn = kernels.mm_bfp_shuffle()
    inputs = kd.sample_inputs(fn, calls=3)
    (encoded,) = kd.host_layout(fn, inputs)
    ref = fn.expected(inputs)
    got = np.stack([kernels.mm_bfp_shuffle_ref(row, 64, 64, 0) for row in encoded])
    assert fn.judge(got, ref, calls=3)
    assert not fn.judge(encoded, ref, calls=3)
    assert "scalar_shuffle" in str(kd.design(kernels.mm_bfp_shuffle, calls=3).as_mlir())


def test_bfp_unshuffle_contract_reads_the_block_layout_back_to_rows():
    fn = kernels.mm_bfp_shuffle(dim_m=32, unshuffle=True)
    inputs = kd.sample_inputs(fn, calls=3)
    (shuffled,) = kd.host_layout(fn, inputs)
    ref = fn.expected(inputs)
    got = np.stack([kernels.mm_bfp_shuffle_ref(row, 64, 32, 1) for row in shuffled])
    assert fn.judge(got, ref, calls=3)
    assert not fn.judge(shuffled, ref, calls=3)
    assert fn.contract.parameter_bindings[-1] == (4, 1)


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
    # conv2dk1 requantizes by >> 12: bounding its inputs by the uint8 output
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


def test_matrix_kernels_declare_their_blocking_on_the_operand_layouts():
    """``mac_dims`` and ``stream_dims`` are views of the contract, not attributes."""
    fn = kernels.mm(
        dim_m=64, dim_k=32, dim_n=64, input_dtype=np.int16, output_dtype=np.int32
    )
    assert isinstance(fn, kernels.MatrixKernel)
    a, b, c = fn.contract.layouts
    r, s, t = fn.mac_dims
    assert (a.block, b.block, c.block) == ((r, s), (s, t), (r, t))
    assert fn.stream_dims == kernels.mm_stream_dims(64, 32, 64, (r, s, t))
    assert (a.stream, b.stream, c.stream) == tuple(fn.stream_dims)
    assert "dims" not in vars(fn) and "mac_dims" not in vars(fn)
    # The scalar kernel walks row-major operands: 1x1x1, nothing streamed.
    scalar = kernels.mm(dim_m=64, dim_k=32, dim_n=64, vectorized=False)
    assert scalar.mac_dims == (1, 1, 1)
    assert scalar.stream_dims == kernels.linalg.StreamDimsABC(None, None, None)
    assert not isinstance(kernels.mv(), kernels.MatrixKernel)
    assert kernels.mv().contract.layouts[0].stream == [(32, 2), (16, 64), (2, 1)]


def test_contract_is_given_at_construction():
    from aie.iron.kernel import ExternalFunction

    contract = KernelContract(roles=(In, Out), reference=lambda x: x)
    fn = ExternalFunction(
        "identity",
        source_string="void identity(int *a, int *b) {}",
        arg_types=[np.ndarray[(16,), np.dtype[np.int32]]] * 2,
        contract=contract,
    )
    assert fn.contract is contract
    assert ExternalFunction("bare", source_string="void bare() {}").contract is None


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
    # The host-side consequence is in the contract's layouts, where the
    # generic builder reads it: B is stored as tiles of B^T, C decoded from
    # tiles of C^T. Nothing on the function itself says "column major".
    rng = np.random.default_rng(0)
    b = rng.standard_normal((1, fkw["dim_k"], fkw["dim_n"])).astype(np.float32)
    c = rng.standard_normal((1, fkw["dim_m"], fkw["dim_n"])).astype(np.float32)
    assert not np.array_equal(
        plain.contract.layouts[1].encode(b), bcm.contract.layouts[1].encode(b)
    )
    assert np.array_equal(
        plain.contract.layouts[2].encode(c), bcm.contract.layouts[2].encode(c)
    )
    assert not np.array_equal(
        plain.contract.layouts[2].encode(c), ccm.contract.layouts[2].encode(c)
    )


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
    # filter2d: identity kernel (Q4.12 one at the center) copies the middle line
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
    # dwconv1d channels-first: taps read the padded row directly, bias is the
    # trailing weight.
    xp = np.arange(1, 1 + 32 + kernels.DWCONV1D_TAIL, dtype=np.float32).astype(bfloat16)
    w = np.array([1, 0, 0, 5], bfloat16)  # K = 3 taps then bias
    cf_ref = kernels.dwconv1d_channels_first_ref
    out = cf_ref(xp, w, 32, kernel_size=3, bias=True).astype(np.float32)
    assert out.tolist() == [float(i + 5) for i in range(1, 33)]
    out = cf_ref(xp, w, 32, kernel_size=3, bias=False).astype(np.float32)
    assert out.tolist() == [float(i) for i in range(1, 33)]
    # dwconv1d channels-last: one timestep, five independent weight planes, so
    # the result is a plain sum_t w_t * x_t across channels.
    xs = [np.full(32, t + 1, dtype=np.float32).astype(bfloat16) for t in range(5)]
    off = [np.zeros(32, np.float32).astype(bfloat16) for _ in range(4)]
    live = np.full(32, 2.0, np.float32).astype(bfloat16)
    cl_ref = kernels.dwconv1d_channels_last_ref
    assert kernels.dwconv1d_channels_last(32).contract.setup is kernels.conv_even
    # Only plane 0 is non-zero, so the result is 2 * x_0 == 2.
    out = cl_ref(live, *off, *xs, lo=-6.0, hi=6.0, clamp=False).astype(np.float32)
    assert out.tolist() == [2.0] * 32
    # Each plane pairs with its own tap: plane 3 live weights x_3 == 4.
    out = cl_ref(*off[:3], live, off[3], *xs, lo=-6.0, hi=6.0, clamp=False).astype(
        np.float32
    )
    assert out.tolist() == [8.0] * 32
    # ... and the clamp bites once the product passes the bound.
    big = np.full(32, 100.0, np.float32).astype(bfloat16)
    out = cl_ref(big, *off, *xs, lo=-6.0, hi=6.0, clamp=True).astype(np.float32)
    assert out.tolist() == [6.0] * 32


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
    if bfp.is_bfp(v):
        return "bfp16ebs8"
    return np.dtype(v).name if isinstance(v, type) else str(v)


def _factories_with_dtypes():
    for name in kernels.factories():
        f = getattr(kernels, name)
        if hasattr(f, "dtypes"):
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
    if fn.contract is None:
        assert name in NOT_JUDGED, f"{name}: no contract"
        return
    if fn.contract.accumulates and not fn.contract.unsupported:
        # The builder initializes an InOut output before every call, so a
        # kernel it can run must say how.
        assert fn.contract.initializers, f"{name}: InOut without an initializer"
    if any(bfp.is_bfp(v) for v in combo.values()):
        return  # block-floating-point operands are not numpy dtypes
    tensor_dts = {
        kd.shape_dtype(t)[1] for t in fn.arg_types() if hasattr(t, "__args__")
    }
    tensor_dts = {np.dtype(dt) for dt in tensor_dts if not bfp.is_bfp(dt)}
    for v in combo.values():
        if isinstance(v, type):  # a dtype, not a shape or a flag
            assert np.dtype(v) in tensor_dts, f"{name}: {v} not among {tensor_dts}"


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
_DIRECTIVE = re.compile(r"^\s*#\s*(\w+)\s*(.*?)\s*(?://.*)?$")


def _condition(expr: str, macros: dict) -> bool:
    """Evaluate an ``#if`` expression; an undefined name is 0, as in C."""
    expr = re.sub(
        r"defined\s*\(\s*(\w+)\s*\)|defined\s+(\w+)",
        lambda m: "1" if (m[1] or m[2]) in macros else "0",
        expr,
    )
    for _ in range(8):
        expr = re.sub(r"\b[A-Za-z_]\w*\b", lambda m: f"({macros.get(m[0], '0')})", expr)
    assert re.fullmatch(r"[\d\s()<>=!&|+*/-]*", expr), expr
    expr = expr.replace("&&", " and ").replace("||", " or ")
    expr = re.sub(r"!(?!=)", " not ", expr)
    return bool(eval(expr, {"__builtins__": {}}))


def _translation_unit(ef, arch: str) -> str:
    """Return the kernel text the compiler sees for ``arch``.

    Quoted includes are inlined and dead preprocessor branches dropped.

    ``linalg/mm_aie2.h`` guards its rounding swap on ``ROUND_CONV_EVEN``, which
    only downstream IRON defines, and a dispatcher includes one arch's body;
    reading the text alone would credit the in-tree build with calls it never
    compiles.
    """
    macros = {"__AIE_ARCH__": str(ARCH_TRAITS[arch].aie_arch)}
    for flag in ef.compile_flags or ():
        if flag.startswith("-D"):
            name, _, value = flag[2:].partition("=")
            macros[name] = value or "1"
    out = []

    def expand(src: str, base: Path | None):
        # One entry per open conditional: (enclosing live, this branch live,
        # some branch already taken).
        stack = []
        live = True
        for line in re.sub(r"\\\n", " ", src).splitlines(keepends=True):
            m = _DIRECTIVE.match(line)
            if not m:
                if live:
                    out.append(line)
                continue
            word, rest = m[1], m[2]
            if word in ("if", "ifdef", "ifndef"):
                if word == "if":
                    cond = _condition(rest, macros)
                else:
                    cond = (rest in macros) == (word == "ifdef")
                stack.append((live, live and cond, cond))
                live = live and cond
            elif word == "elif":
                outer, _, taken = stack[-1]
                cond = not taken and _condition(rest, macros)
                stack[-1] = (outer, outer and cond, taken or cond)
                live = outer and cond
            elif word == "else":
                outer, _, taken = stack[-1]
                stack[-1] = (outer, outer and not taken, True)
                live = outer and not taken
            elif word == "endif":
                live = stack.pop()[0]
            elif not live:
                continue
            elif word == "define":
                out.append(line)
                name, _, value = rest.partition(" ")
                if "(" not in name:
                    macros[name] = value.strip() or "1"
            elif word == "include" and base is not None:
                target = macros.get(rest, rest)
                if target.startswith('"'):
                    header = base / target.strip('"')
                    if header.is_file():
                        expand(header.read_text(), header.parent)

    if ef.source_file:
        expand(Path(ef.source_file).read_text(), Path(ef.source_file).parent)
    else:
        expand(ef.source_string, None)
    return "".join(out)


@pytest.mark.parametrize("arch", ["aie2", "aie2p"])
def test_setup_is_declared_exactly_where_the_source_does_not_set_the_mode(arch):
    """A kernel that sets its own rounding mode must not also name a ``setup``.

    The two are alternatives: either the compiled source calls
    ``aie::set_rounding`` on entry, or the design sets the mode before the
    first call. Declaring both means the design fights the kernel; declaring
    neither, on a kernel that narrows an accumulator, means it runs in
    whatever mode the core booted in.
    """
    set_current_device(NPU1Col1() if arch == "aie2" else NPU2Col1())
    for name, ef in _builds():
        c = ef.contract
        if c is None:
            assert name in NOT_JUDGED, f"{name}: no contract"
            continue
        src = _translation_unit(ef, arch)
        sets_own = bool(_SET_ROUNDING_CALL.search(src))
        if sets_own:
            assert c.setup is None, f"{name}: source sets the mode and names a setup"
        elif c.setup is not None:
            # A setup only matters where an accumulator is narrowed: a bf16
            # or bfp16 output, or an explicit conversion in the source.
            out_dts = [kd.shape_dtype(ef.arg_types()[i])[1] for i in c.out_indices]
            narrows = any(
                bfp.is_bfp(dt) or np.dtype(dt) == np.dtype(bfloat16) for dt in out_dts
            ) or _NARROWS.search(src)
            assert narrows, f"{name}: names a setup but narrows nothing"


def test_a_design_runs_the_setup_a_contract_names():
    """A design for a kernel with a ``setup`` binds it; one without does not."""
    assert kernels.add().contract.setup is not None
    mlir = str(kd.design(kernels.add, calls=2).as_mlir())
    assert "set_rounding_conv_even" in mlir
    assert kernels.convert_copy().contract.setup is None
    mlir = str(kd.design(kernels.convert_copy, calls=1).as_mlir())
    assert "set_rounding" not in mlir


def test_bf16_exp_reference_preserves_subnormal_tail():
    x = np.array(
        [-np.inf, -128, -88, -87.5, -87, 0, 87.5, 88, 128, np.inf],
        dtype=bfloat16,
    )
    got = kernels.bf16_exp_ref(x).astype(np.float64)
    expected = np.exp(np.clip(x.astype(np.float64), -88, 88)).astype(bfloat16)
    np.testing.assert_array_equal(got, expected.astype(np.float64))
    assert np.all(got[:4] > 0)
    assert np.all(got[:4] < np.finfo(np.float32).tiny)
    assert np.all(np.isfinite(got))
    assert got[4] >= np.finfo(np.float32).tiny
    assert got[5] == 1
    assert got[-1] > 1e38


def test_bf16_exp_clamp_matches_the_kernel_headers():
    """``_EXP_BF16_CLAMP`` tracks ``EXP_BF16_CLAMP`` in the kernel sources.

    ``bf16_exp_ref`` describes the device as ``exp(clip(x, -C, C))``. That is
    only true while the Python constant and the C++ one agree; if they ever
    drift the reference silently stops modeling the kernel, which is the
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
        assert (
            float(found.group(1)) == _EXP_BF16_CLAMP
        ), f"{header} clamps at {found.group(1)} but bf16_exp_ref uses {_EXP_BF16_CLAMP}; the reference no longer matches the kernel"
        checked.append(arch)
    assert checked, "no lut_based_ops.h found to check the clamp against"


_CONSTANT_BOUND_KERNELS = [
    ("passthrough", np.int32, 16, "tile_size", "PASSTHROUGH_ELEMS", 2),
    ("passthrough", np.int16, 32, "tile_size", "PASSTHROUGH_ELEMS", 2),
    ("passthrough", np.uint8, 64, "tile_size", "PASSTHROUGH_ELEMS", 2),
    ("reduce_add", np.int32, 16, "tile_size", "REDUCE_ADD_ELEMS", 2),
    ("reduce_min", np.int32, 16, "tile_size", "REDUCE_MIN_ELEMS", 2),
    ("reduce_max", np.int32, 16, "tile_size", "REDUCE_MAX_ELEMS", 2),
    ("reduce_max", bfloat16, 32, "tile_size", "REDUCE_MAX_ELEMS", 2),
    ("scale", np.int16, 32, "tile_size", "SCALE_ELEMS", 3),
    ("scale", np.int32, 16, "tile_size", "SCALE_ELEMS", 3),
    *[
        (name, dtype, width, "line_width", macro, count_arg)
        for name, macro, count_arg in (
            ("bitwise_and", "BITWISE_ELEMS", 3),
            ("bitwise_or", "BITWISE_ELEMS", 3),
            ("threshold", "THRESHOLD_ELEMS", 2),
        )
        for dtype, width in ((np.uint8, 64), (np.int16, 32), (np.int32, 16))
    ],
    ("add_weighted", np.uint8, 32, "line_width", "ADD_WEIGHTED_ELEMS", 3),
    ("add_weighted", np.int16, 16, "line_width", "ADD_WEIGHTED_ELEMS", 3),
]


@pytest.mark.parametrize("arch", ["aie2", "aie2p"])
@pytest.mark.parametrize("iterations", [1, 4, 5, 128])
@pytest.mark.parametrize(
    "name,dtype,width,param,macro,count_arg", _CONSTANT_BOUND_KERNELS
)
def test_kernel_bounds_are_specialized_without_iteration_promises(
    arch, iterations, name, dtype, width, param, macro, count_arg
):
    set_current_device(NPU1Col1() if arch == "aie2" else NPU2Col1())
    count = width * iterations
    fn = getattr(kernels, name)(dtype=dtype, **{param: count})
    assert f"-D{macro}={count}" in fn.compile_flags
    # Keep the runtime count operand and contract binding for existing designs.
    assert (count_arg, count) in fn.contract.parameter_bindings
    assert fn.arg_types()[count_arg] == np.int32
    src = Path(fn.source_file).read_text()
    assert f"#ifndef {macro}" in src
    assert "AIE_LOOP_MIN_ITERATION_COUNT" not in src
    assert "AIE_LOOP_RANGE" not in src
    other = getattr(kernels, name)(dtype=dtype, **{param: count + width})
    assert fn.object_file_name != other.object_file_name
    assert fn._symbol_prefix != other._symbol_prefix


@pytest.mark.parametrize("bad_size", [0, -1, 1, 65])
@pytest.mark.parametrize(
    "name,dtype,width,param,macro,count_arg", _CONSTANT_BOUND_KERNELS
)
def test_constant_bound_vector_kernels_still_require_whole_vectors(
    bad_size, name, dtype, width, param, macro, count_arg
):
    with pytest.raises(ValueError, match="positive|vector"):
        getattr(kernels, name)(dtype=dtype, **{param: bad_size})


@pytest.mark.parametrize("name", ["reduce_add", "reduce_min", "reduce_max", "scale"])
def test_scalar_bounds_are_specialized_without_vector_alignment(name):
    fn = getattr(kernels, name)(tile_size=3, vectorized=False)
    macro = f"{name.upper()}_ELEMS"
    assert f"-D{macro}=3" in fn.compile_flags


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
