# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""One harness owns host layout, embedded constants, execution and checking."""

import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from functools import cached_property
from unittest.mock import Mock

import numpy as np
import pytest
from aie.iron import kernels
from aie.iron.algorithms import kernel_design as kd
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels import KernelContract, Param, Trace
from aie.utils import get_current_device
from aie.utils.compile.jit import In, InOut, Out
from aie.utils.hostruntime import set_current_device


@pytest.fixture(autouse=True)
def _device(npu2_device):
    yield


def _tile(dtype=np.int32, size=4):
    return np.ndarray[(size,), np.dtype[dtype]]


def _kernel(*, param=False, multiple=False, trace=None):
    roles = [In, In]
    types = [_tile(), _tile()]
    if param:
        roles += [Param, Param]
        types += [_tile(), np.int32]
    roles += [Out]
    types += [_tile()]
    if multiple:
        roles += [Out]
        types += [_tile(np.float32)]
    fn = ExternalFunction("kd_test", source_string="void kd_test() {}", arg_types=types)
    fn.contract = KernelContract(
        roles=tuple(roles),
        reference=(
            (lambda x, y, w, scale: (x + y + w) * scale)
            if param
            else (lambda x, y: (x + y, x - y)) if multiple else (lambda x, y: x + y)
        ),
        trace=trace,
    )
    return fn


def test_one_configuration_drives_sampling_layout_sizes_and_descriptors():
    design = kd.KernelDesign(_kernel(param=True), calls=3, scalars=(2,), guard=True)
    inputs = design.sample_inputs(np.random.default_rng(42))
    assert [a.shape for a in inputs] == [(3, 4), (3, 4), (4,)]
    layout = design.host_layout(inputs)
    assert len(layout) == 1
    np.testing.assert_array_equal(layout[0], np.stack(inputs[:2], axis=1))
    args = design.host_args()
    assert [(a.direction, a.shape) for a in args] == [
        (In, (3, 2, 4)),
        (Out, (3, 20)),
    ]
    assert design.output_size() == args[-1].n_elements == 60
    assert args[-1].dtype is np.int32


def test_sampling_does_not_require_the_calls_to_fill_an_offset_output():
    fn = ExternalFunction(
        "offset_output",
        source_string="void offset_output() {}",
        arg_types=[_tile(), np.int32, _tile(size=8)],
    )
    fn.contract = KernelContract(
        roles=(In, Param, Out),
        parameter_bindings=((1, 0),),
        out_offset=(1, 4),
    )
    design = kd.KernelDesign(fn)
    assert design.sample_inputs()[0].shape == (1, 4)
    with pytest.raises(ValueError, match="must fill"):
        design.output_size()


def test_host_sampling_checking_and_trace_counts_do_not_bind_a_device():
    with set_current_device(None):
        fn = _kernel(trace=Trace.whole_call())
        design = kd.KernelDesign(fn, calls=2)
        inputs = design.sample_inputs()
        assert [a.shape for a in inputs] == [(2, 4), (2, 4)]
        assert design.check(fn.expected(inputs), inputs)
        assert design.traced_intervals() == 2
        assert [arg.n_elements for arg in design.host_args()] == [16, 8]
        assert len(design.host_layout(inputs)) == 1
        parameterized = kd.KernelDesign(
            _kernel(param=True), scalars=(2,), arg_byte_offsets=((2, 16),)
        )
        assert len(parameterized.sample_inputs()) == 3
        assert get_current_device(probe_runtime=False) is None


def test_declaration_and_plan_facts_are_cached(monkeypatch):
    fn = _kernel()
    arg_types = Mock(wraps=fn.arg_types)
    validate = Mock(wraps=KernelContract.validate_types)
    monkeypatch.setattr(fn, "arg_types", arg_types)
    monkeypatch.setattr(
        KernelContract, "validate_types", lambda self, types: validate(self, types)
    )
    design = kd.KernelDesign(fn, calls=2)
    facts = (
        design._contract,
        design._types,
        design._tensor_positions,
        design._out_tiles,
        design._fifo_plan,
        design._guarded,
    )
    for _ in range(3):
        design.host_layout([np.ones((2, 4)), np.ones((2, 4))])
        design.host_args()
        design.output_size()
        assert facts == (
            design._contract,
            design._types,
            design._tensor_positions,
            design._out_tiles,
            design._fifo_plan,
            design._guarded,
        )
    arg_types.assert_called_once_with()
    assert validate.call_count == 1


def test_plan_properties_are_private_and_cached():
    design = kd.KernelDesign(_kernel(), calls=2, guard=True)
    for private, public in (
        ("_contract", "contract"),
        ("_types", "arg_types"),
        ("_tensor_positions", "tensor_positions"),
        ("_out_tiles", "output_tiles"),
        ("_fifo_plan", "fifo_plan"),
        ("_guarded", "guarded_outputs"),
    ):
        assert isinstance(vars(kd.KernelDesign)[private], cached_property)
        assert getattr(design, private) is getattr(design, private)
        assert not hasattr(design, public)


@pytest.mark.parametrize(
    "name",
    [
        "fn",
        "calls",
        "guard",
        "scalars",
        "params",
        "stack_bytes",
        "aiecc_flags",
        "arg_byte_offsets",
    ],
)
def test_configuration_cannot_change_after_plan_facts_are_cached(name):
    design = kd.KernelDesign(_kernel(), calls=2, guard=True)
    design.host_args()
    before = getattr(design, name)
    with pytest.raises(AttributeError):
        setattr(design, name, None)
    assert getattr(design, name) is before


def test_constructor_validates_contract_but_design_checks_build_support():
    fn = _kernel()
    fn.contract = None
    with pytest.raises(ValueError, match="declares no contract"):
        kd.KernelDesign(fn)
    fn.contract = KernelContract(
        roles=(In, In, Out), unsupported="needs a custom schedule"
    )
    design = kd.KernelDesign(fn)
    assert len(design.sample_inputs()) == 2
    with pytest.raises(ValueError, match="needs a custom schedule"):
        design.design()


def test_specialization_reuses_encoded_values_and_keeps_stable_kernel(monkeypatch):
    specialize = Mock(side_effect=lambda **kw: object())
    monkeypatch.setattr(kd._stream, "specialize", specialize)
    fn = _kernel(param=True)
    design = kd.KernelDesign(fn, scalars=(2,), guard=True)
    weights = np.arange(4, dtype=np.int32)
    first = design.design(params=[weights])
    assert design.design(params=[weights.copy()]) is first
    assert design.design(params=[weights + 1]) is not first
    assert specialize.call_count == 2
    kwargs = specialize.call_args_list[0].kwargs
    assert kwargs["kernel"] is fn
    assert kwargs["owner"] is design
    assert kwargs["params"] == (("int32", (4,), (0, 1, 2, 3)),)
    assert kwargs["calls"] == 1 and kwargs["guard"]
    assert "factory" not in kwargs and "fn" not in kwargs
    assert design.params is None


def test_as_mlir_delegates_arguments_to_the_compilable_design(monkeypatch):
    design = kd.KernelDesign(_kernel())
    compiled = Mock()
    build = Mock(return_value=compiled)
    monkeypatch.setattr(design, "design", build)
    trace = object()
    assert design.as_mlir("host", trace_config=trace) is compiled.as_mlir.return_value
    build.assert_called_once_with()
    compiled.as_mlir.assert_called_once_with("host", trace_config=trace)


def test_contract_schedule_facts_participate_in_specialization(monkeypatch):
    specialize = Mock(return_value=object())
    monkeypatch.setattr(kd._stream, "specialize", specialize)
    fn = _kernel()
    kd.KernelDesign(fn).design()
    original = specialize.call_args.kwargs
    fn.contract = replace(fn.contract, roles=(In, Param, Out))
    kd.KernelDesign(fn, params=[np.ones(4, np.int32)]).design()
    changed = specialize.call_args.kwargs
    assert original["kernel"] is changed["kernel"]
    assert original["plan"] != changed["plan"]


@pytest.mark.parametrize(
    "change",
    ["out_offset", "initializers", "parameter_bindings", "uses_lut", "scalar_arity"],
)
def test_same_kernel_recipe_with_different_contract_build_plans_has_distinct_keys(
    change,
):
    output_type = _tile(size=8)
    contract = KernelContract(roles=(In, Param, Out), parameter_bindings=((1, 0),))
    scalars = ()
    if change == "out_offset":
        changed = replace(contract, out_offset=(1, 4))
    elif change == "initializers":
        initial_before = ExternalFunction(
            "kd_plan_init_before",
            source_string="void kd_plan_init_before() {}",
            arg_types=[output_type],
        )
        initialize = ExternalFunction(
            "kd_plan_init",
            source_string="void kd_plan_init() {}",
            arg_types=[output_type],
        )
        contract = replace(
            contract,
            roles=(In, Param, InOut),
            initializers=((2, lambda _fn: initial_before),),
        )
        changed = replace(contract, initializers=((2, lambda _fn: initialize),))
    elif change == "parameter_bindings":
        changed = replace(contract, parameter_bindings=((1, 7),))
    elif change == "uses_lut":
        changed = replace(contract, uses_lut=True)
    else:
        changed = replace(contract, parameter_bindings=())
        scalars = (0,)

    def harness(c, scalars=()):
        fn = ExternalFunction(
            "kd_same_recipe",
            source_string="void kd_same_recipe() {}",
            arg_types=[_tile(), np.int32, output_type],
            contract=c,
        )
        return kd.KernelDesign(fn, calls=2, scalars=scalars, stack_bytes=1024)

    original = harness(contract)
    different = harness(changed, scalars)
    equivalent_contract = (
        replace(changed, initializers=((2, lambda _fn: initialize),))
        if change == "initializers"
        else replace(changed)
    )
    equivalent = harness(equivalent_contract, scalars)
    assert repr(original.fn) == repr(different.fn) == repr(equivalent.fn)
    assert original.fn == different.fn == equivalent.fn
    before = original.design().compilable.recipe_hash
    after = different.design().compilable.recipe_hash
    assert before != after
    assert after == equivalent.design().compilable.recipe_hash


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("guard", [False, True])
def test_strip_guard_obeys_instance_configuration(multiple, guard):
    design = kd.KernelDesign(_kernel(multiple=multiple), calls=2, guard=guard)
    outputs = []
    for arg in design.host_args():
        if arg.direction is not Out:
            continue
        out = kd.poisoned(arg.n_elements, arg.dtype)
        out.reshape(2, -1)[:, :4] = np.arange(8).reshape(2, 4)
        outputs.append(out)
    raw = tuple(outputs) if multiple else outputs[0]
    stripped, counts = design.strip_guard(raw)
    assert counts == ((0, 0) if multiple else 0)
    if not guard:
        assert stripped is raw
    for data in stripped if multiple else (stripped,):
        np.testing.assert_array_equal(data, np.arange(8))
    if guard:
        outputs[0].view(np.uint8)[-1] = 0
        assert design.strip_guard(raw)[1] == ((1, 0) if multiple else 1)


def test_upload_derives_storage_without_building(monkeypatch):
    design = kd.KernelDesign(_kernel(multiple=True), calls=2, guard=True)
    tensor = Mock(side_effect=lambda a, **kw: a)
    monkeypatch.setattr(kd, "tensor", tensor)
    monkeypatch.setattr(design, "design", Mock(side_effect=AssertionError("built")))
    ins, outs = design.upload([np.ones((2, 4), np.int32)] * 2, poison=True)
    assert len(ins) == 1 and ins[0].shape == (16,)
    assert tuple(out.size for out in outs) == design.output_size()
    assert tuple(out.dtype for out in outs) == (np.int32, np.float32)
    assert all(np.all(out.view(np.uint8) == 0x55) for out in outs)
    assert tensor.call_count == 3


@pytest.mark.parametrize("provided_calls", [1, 3])
def test_upload_rejects_input_element_counts_that_disagree_with_calls(
    monkeypatch, provided_calls
):
    design = kd.KernelDesign(_kernel(), calls=2)
    tensor = Mock()
    monkeypatch.setattr(kd, "tensor", tensor)
    inputs = [np.ones((provided_calls, 4), np.int32)] * 2
    with pytest.raises(
        ValueError, match="input buffer 0 needs 16 elements for calls=2"
    ):
        design.upload(inputs)
    tensor.assert_not_called()


def test_upload_checks_all_input_dtypes_before_device_allocation(monkeypatch):
    fn = ExternalFunction(
        "mixed_upload",
        source_string="void mixed_upload() {}",
        arg_types=[_tile(), _tile(np.float32), _tile()],
        contract=KernelContract(roles=(In, In, Out)),
    )
    design = kd.KernelDesign(fn, calls=2)
    tensor = Mock()
    monkeypatch.setattr(kd, "tensor", tensor)
    with pytest.raises(
        ValueError, match="input buffer 1 expected dtype float32, got float64"
    ):
        design.upload([np.ones((2, 4), np.int32), np.ones((2, 4), np.float64)])
    tensor.assert_not_called()


class _Tensor:
    def __init__(self, a):
        self.array = a

    def numpy(self):
        return self.array


@pytest.mark.parametrize("custom_sample", [False, True])
def test_configured_tensor_params_roundtrip_and_reject_mismatched_inputs(
    monkeypatch, custom_sample
):
    fn = kernels.scale(tile_size=16, dtype=np.int32)
    if custom_sample:
        fn.contract = replace(
            fn.contract,
            sample=lambda rng, calls: [
                np.ones((calls, 16), np.int32),
                np.array([99], np.int32),
            ],
        )
    factor = np.array([3], np.int32)
    design = kd.KernelDesign(fn, calls=2, params=[factor])
    inputs = design.sample_inputs()
    np.testing.assert_array_equal(inputs[1], factor)
    assert not np.shares_memory(inputs[1], factor)

    def execute(input_, output):
        output.array[:] = (input_.array * np.int32(3)).ravel()

    compiled = Mock(side_effect=execute)
    specialize = Mock(return_value=compiled)
    tensor = Mock(side_effect=lambda array, **kwargs: _Tensor(array))
    monkeypatch.setattr(kd._stream, "specialize", specialize)
    monkeypatch.setattr(kd, "tensor", tensor)
    outputs = design.run(inputs)
    assert specialize.call_args.kwargs["params"] == (("int32", (1,), (3,)),)
    assert design.check(outputs, inputs)
    assert not design.check(outputs + 1, inputs)

    inputs[1][0] = 4
    before = tensor.call_count
    with pytest.raises(ValueError, match="tensor Param argument 2.*configured params"):
        design.run(inputs)
    with pytest.raises(ValueError, match="tensor Param argument 2.*configured params"):
        design.check(outputs, inputs)
    assert tensor.call_count == before
    assert compiled.call_count == 1
    np.testing.assert_array_equal(factor, [3])


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("guard", [False, True])
def test_run_returns_copied_stripped_arrays_and_check_returns_verdict(
    monkeypatch, multiple, guard
):
    fn = _kernel(multiple=multiple)
    design = kd.KernelDesign(fn, calls=2, guard=guard)
    inputs = [np.arange(8, dtype=np.int32).reshape(2, 4)] * 2
    reference = fn.expected(inputs)
    device_outputs = []

    def compiled(*args, **kwargs):
        refs = reference if multiple else (reference,)
        for out, ref in zip(args[1:], refs):
            out.array.reshape(2, -1)[:, :4] = ref
            device_outputs.append(out.array)

    monkeypatch.setattr(kd, "tensor", lambda a, **kw: _Tensor(a))
    monkeypatch.setattr(kd._stream, "specialize", Mock(return_value=compiled))
    outputs = design.run(inputs)
    assert design.check(outputs, inputs)
    arrays = outputs if multiple else (outputs,)
    assert all(a.size == 8 for a in arrays)
    assert not any(np.shares_memory(a, b) for a, b in zip(arrays, device_outputs))
    arrays[0][0] += 1
    assert not design.check(outputs, inputs)


def test_run_detects_guard_overwrite(monkeypatch):
    design = kd.KernelDesign(_kernel(), guard=True)
    monkeypatch.setattr(kd, "tensor", lambda a, **kw: _Tensor(a))
    monkeypatch.setattr(
        kd._stream,
        "specialize",
        Mock(return_value=lambda *args: args[-1].array.fill(0)),
    )
    with pytest.raises(AssertionError, match="kd_test: output guard overwritten"):
        design.run([np.ones((1, 4), np.int32)] * 2)


def test_run_uses_explicit_params_default_poison_and_trace_config(monkeypatch):
    fn = _kernel(param=True)
    weights = np.arange(4, dtype=np.int32)
    design = kd.KernelDesign(fn, scalars=(2,), params=[weights])
    inputs = [np.ones((1, 4), np.int32)] * 2 + [weights]
    device_output = _Tensor(fn.expected(inputs, scalars=(2,)).ravel())
    compiled = Mock()
    build = Mock(return_value=compiled)
    upload = Mock(return_value=(["input"], device_output))
    monkeypatch.setattr(design, "design", build)
    monkeypatch.setattr(design, "upload", upload)
    monkeypatch.setattr(
        fn, "param_values", Mock(side_effect=AssertionError("params already supplied"))
    )
    trace = object()
    assert design.check(design.run(inputs, trace_config=trace), inputs)
    build.assert_called_once_with(params=(weights,))
    upload.assert_called_once_with(inputs, poison=True)
    compiled.assert_called_once_with("input", device_output, trace_config=trace)


def test_check_forwards_scalars_inputs_and_calls_without_stripping(monkeypatch):
    fn = _kernel(param=True)
    design = kd.KernelDesign(fn, calls=3, scalars=(2,), guard=True)
    inputs = [np.ones((3, 4), np.int32)] * 2 + [np.ones(4, np.int32)]
    expected = Mock(wraps=fn.expected)
    judge = Mock(wraps=fn.judge)
    monkeypatch.setattr(fn, "expected", expected)
    monkeypatch.setattr(fn, "judge", judge)
    monkeypatch.setattr(
        design, "strip_guard", Mock(side_effect=AssertionError("stripped twice"))
    )
    outputs = np.full(12, 6, np.int32)
    assert design.check(outputs, inputs)
    expected.assert_called_once_with(inputs, scalars=(2,))
    assert judge.call_args.args[0] is outputs
    assert judge.call_args.kwargs == dict(inputs=inputs, scalars=(2,), calls=3)


def test_run_and_cycles_share_input_parameter_specialization(monkeypatch, request):
    fn = _kernel(param=True, trace=Trace.whole_call())
    design = kd.KernelDesign(fn, calls=2, scalars=(2,))
    inputs = [np.ones((2, 4), np.int32)] * 2 + [np.arange(4, dtype=np.int32)]
    configs = []

    def compiled(*args, trace_config=None):
        args[-1].array[:] = fn.expected(inputs, scalars=(2,)).ravel()
        if trace_config:
            trace_config.physical_mlir_path = "physical.mlir"
            configs.append(trace_config)

    specialize = Mock(return_value=compiled)
    monkeypatch.setattr(kd._stream, "specialize", specialize)
    monkeypatch.setattr(kd, "tensor", lambda a, **kw: _Tensor(a))
    monkeypatch.setattr(kd.TraceConfig, "trace_to_json", Mock())
    monkeypatch.setattr(kd, "get_cycles_summary", lambda _: [("core", 17, 19)])
    monkeypatch.setattr(design, "sample_inputs", Mock(return_value=inputs))
    # Keep trace artifacts in the project, not pytest's default temporary root.
    workdir = request.config.rootpath / ".kernel-design-test-trace"
    workdir.mkdir(exist_ok=True)
    request.addfinalizer(lambda: workdir.rmdir())
    assert design.check(design.run(inputs), inputs)
    assert design.cycles_per_call(workdir=workdir).kernel == (17, 19)
    assert specialize.call_count == 1
    assert configs[0].trace_size == 16384
    design.sample_inputs.assert_called_once_with()


def test_legacy_wrapper_instantiates_once_and_matches_explicit_class(monkeypatch):
    fn = _kernel(param=True)
    factory = Mock(return_value=fn)
    specialize = Mock(side_effect=lambda **kwargs: kwargs)
    monkeypatch.setattr(kd._stream, "specialize", specialize)
    options = dict(
        calls=2,
        scalars=(3,),
        params=[np.arange(4, dtype=np.int32)],
        guard=True,
        stack_bytes=2048,
        arg_byte_offsets=((2, 16),),
        aiecc_flags=["--dynamic-objFifos"],
    )
    legacy = kd.design(factory, tile_size=4, **options)
    explicit = kd.KernelDesign(fn, **options).design()
    assert repr(legacy.pop("owner")) == repr(explicit.pop("owner"))
    assert legacy == explicit
    factory.assert_called_once_with(tile_size=4)


def test_build_state_is_fresh_and_stack_facts_are_cached(monkeypatch):
    fn = _kernel()
    measure = Mock(return_value=None)
    monkeypatch.setattr(fn, "measure", measure)
    design = kd.KernelDesign(fn, calls=2)
    first, second = design._stage(()), design._stage(())
    assert first is not second
    assert first.inputs[0][0] is not second.inputs[0][0]
    assert first.outputs[0] is not second.outputs[0]
    assert first.inputs[0][0].name == second.inputs[0][0].name == "in0"
    assert first.outputs[0].name == second.outputs[0].name == "out0"
    assert design._depth == 2
    assert measure.call_count == 1


def test_repeated_mlir_builds_isolate_buffers_and_preserve_parameter_values():
    design = kd.KernelDesign(
        _kernel(param=True),
        calls=2,
        scalars=(3,),
        guard=True,
        stack_bytes=1024,
        arg_byte_offsets=((2, 16),),
    )
    weights = np.arange(4, dtype=np.int32)
    first = design.design(params=[weights]).as_mlir()
    changed = design.design(params=[weights + 1]).as_mlir()
    assert first == design.design(params=[weights.copy()]).as_mlir()
    assert first != changed
    assert "aie.objectfifo @in0" in first
    assert "aie.objectfifo @out0" in first
    assert 'sym_name = "param0"' in first
    assert 'sym_name = "param_shift"' in first
    assert "memref.view" in first
    assert "func.call @kd_poison_20" in first
    assert "stack_size = 1024" in first


def test_mlir_generation_reuses_owner_facts_but_not_stage_state(monkeypatch):
    counts = Counter()
    facts = (
        "_contract",
        "_types",
        "_tensor_positions",
        "_fifo_plan",
        "_out_tiles",
        "_guarded",
        "_kernel_frame_bytes",
        "_depth",
    )

    class CountedDesign(kd.KernelDesign):
        pass

    def counted(name):
        original = vars(kd.KernelDesign)[name].func

        @cached_property
        def fact(self):
            counts[name] += 1
            return original(self)

        fact.__set_name__(CountedDesign, name)
        return fact

    for name in facts:
        setattr(CountedDesign, name, counted(name))

    measure = Mock(return_value=None)
    monkeypatch.setattr(ExternalFunction, "measure", measure)
    owner = CountedDesign(_kernel(), calls=2, guard=True)
    owner.sample_inputs()
    original_init = kd._KernelStage.__init__
    stages = []

    def initialize(stage, design, params):
        assert design is owner
        original_init(stage, design, params)
        stages.append(stage)

    monkeypatch.setattr(kd._KernelStage, "__init__", initialize)
    signature = repr(owner)
    first = owner.as_mlir()
    assert first == owner.as_mlir()
    assert first != owner.as_mlir(
        trace_config=kd.TraceConfig(trace_size=1024, trace_file="trace.txt")
    )
    compiled = owner.design().compilable
    assert first == str(compiled.mlir_generator(**compiled.compile_kwargs))
    assert repr(owner) == signature
    assert counts == Counter({name: 1 for name in facts})
    assert measure.call_count == 2  # The measured kernel and the guard fill.
    assert len(stages) == 3
    assert len({id(stage.fifos_in[0]) for stage in stages}) == 3
    assert len({id(stage.fifos_out[0]) for stage in stages}) == 3


def test_owner_repr_and_recipe_hash_are_stable_across_processes():
    script = """
import numpy as np
from aie.iron.algorithms.kernel_design import KernelDesign
from aie.iron.device import NPU2Col1
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels import KernelContract, Param
from aie.utils.compile.jit import In, InOut
from aie.utils.hostruntime import set_current_device

set_current_device(NPU2Col1())
tile = np.ndarray[(4,), np.dtype[np.int32]]
initialize = ExternalFunction(
    "stable_init", source_string="void stable_init() {}", arg_types=[tile]
)
fn = ExternalFunction(
    "stable_owner",
    source_string="void stable_owner() {}",
    arg_types=[tile, np.int32, tile],
    contract=KernelContract(
        roles=(In, Param, InOut),
        parameter_bindings=((1, 3),),
        initializers=((2, lambda _fn: initialize),),
    ),
)
owner = KernelDesign(fn, calls=2, guard=True, stack_bytes=1024)
print(repr(owner))
print(owner.design().compilable.recipe_hash)
"""
    results = [
        subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        for seed in ("1", "2")
    ]
    assert results[0] == results[1]
    assert "object at 0x" not in results[0]
