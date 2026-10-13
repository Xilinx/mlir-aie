# test_kernel_built_for_arch.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""A library kernel compiles only for the architecture its factory chose,
and is judged by the build that produced its output.

A factory reads the bound device when it is called and picks its source and
flags for that architecture (aie2 when none is bound). A script that calls
it before binding, then runs a design on an NPU2, used to hand Peano the
aie2 LUT headers for aie2p ("call to 'mul_elem_16_2' is ambiguous").

Which reference and tolerance judge an output is chosen when it is judged,
from the device it ran on and the kernel's portable flag, so a contract read
from a kernel built before binding judges the kernel the design built.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from aie.iron import In, Out, kernels
from aie.iron.algorithms import kernel_design as kd
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels import BUILDS, Build, KernelContract
from aie.iron.kernels.activation import (
    _GELU_VTANH_TOLERANCE,
    _LUT_MODEL_TOLERANCE,
    _vtanh_family_tolerance,
)
from aie.iron.kernels.fused import fused_mm
from aie.iron.kernels.norm import (
    _LAYER_NORM_BF16_AIE2,
    _LAYER_NORM_BF16_AIE2P,
    _NORM_BF16,
    _RMS_NORM_BF16_AIE2,
    _RMS_NORM_BF16_AIE2P,
)
from aie.iron.kernels.transformer import (
    _AFFINE_TOLERANCE,
    _EPILOGUE_LUT_TOLERANCE,
    _EPILOGUE_TOLERANCE,
    _NORM_F32,
    _NORM_F32_AIE2,
    _NORM_F32_AIE2P,
)
from aie.utils import config, get_current_device
from aie.utils.compile.utils import compile_external_kernels
from aie.utils.hostruntime import set_current_device
from aie.utils.verify import Tolerance, compare
from ml_dtypes import bfloat16

# Every finite bf16, as 64 calls of 1024 (the last 256 lanes repeat zero).
_FINITE_BF16 = np.zeros(64 * 1024, np.uint16)
_FINITE_BF16[:65280] = [
    b for b in range(1 << 16) if np.isfinite(np.uint16(b).view(bfloat16))
]
_FINITE_BF16 = _FINITE_BF16.view(bfloat16).reshape(64, 1024)


def _peano_available() -> bool:
    try:
        return os.path.isfile(config.peano_cxx_path())
    except RuntimeError:
        return False


@pytest.fixture
def no_device():
    previous = get_current_device(probe_runtime=False)
    set_current_device(None)
    try:
        yield
    finally:
        set_current_device(previous)


@pytest.fixture
def npu1_device():
    previous = get_current_device(probe_runtime=False)
    set_current_device(NPU1Col1())
    try:
        yield
    finally:
        set_current_device(previous)


def test_a_kernel_built_unbound_refuses_an_aie2p_compile(no_device, tmp_path):
    exp = kernels.bf16_exp(tile_size=1024)
    assert exp.built_for_arch == "aie2"
    with pytest.raises(ValueError, match="built for aie2 but .* for aie2p.*or none"):
        compile_external_kernels([exp], str(tmp_path), "aie2p")
    assert not list(tmp_path.iterdir())


def test_a_sibling_binding_carries_the_factory_arch(no_device, tmp_path):
    """JIT rediscovers ``object_file.bind(...)`` bindings; they keep the arch."""
    exp = kernels.bf16_exp(tile_size=1024)
    sibling = exp.object_file.bind("sibling", exp.arg_types())
    saved = set(ExternalFunction._instances)
    ExternalFunction._instances.clear()
    try:
        ExternalFunction._register_object(sibling)
        (binding,) = ExternalFunction._instances
        assert binding is not exp and binding.built_for_arch == "aie2"
        with pytest.raises(ValueError, match="built for aie2 but .* for aie2p"):
            compile_external_kernels([binding], str(tmp_path), "aie2p")
    finally:
        ExternalFunction._instances.clear()
        ExternalFunction._instances.update(saved)
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
def test_a_kernel_built_bound_compiles_for_its_arch(npu2_device, tmp_path):
    if not (Path(config.aie_runtime_lib_dir()) / "AIE2P").is_dir():
        pytest.skip("this build has no aie_runtime_lib/AIE2P")
    exp = kernels.bf16_exp(tile_size=1024)
    assert exp.built_for_arch == "aie2p"
    compile_external_kernels([exp], str(tmp_path), "aie2p")
    assert (tmp_path / exp.object_file_name).is_file()


@pytest.mark.parametrize("factory", [kernels.gelu, kernels.gelu_sized])
def test_a_lut_gelu_built_unbound_is_judged_as_the_npu2_build(no_device, factory):
    fn = factory(tile_size=1024, use_lut=True)
    set_current_device(NPU2Col1())
    c = fn.contract_for()
    assert c.reference is kernels.gelu_lut_ref
    assert c.tolerance is _LUT_MODEL_TOLERANCE
    expected = fn.expected([_FINITE_BF16])
    npu2 = kernels.gelu_lut_ref(_FINITE_BF16)
    verdict = fn.judge(npu2.ravel(), expected, calls=64)
    assert verdict, verdict.detail
    # One ulp of the npu2 model, not of whichever model would pass.
    npu1 = kernels.gelu_aie2_lut_ref(_FINITE_BF16)
    assert not fn.judge(npu1.ravel(), expected, calls=64)


def test_a_gelu_built_unbound_takes_reference_and_tolerance_from_one_row(no_device):
    fn = kernels.gelu_sized(tile_size=2048)
    set_current_device(NPU2Col1())
    c = fn.contract_for()
    assert (c.reference, c.tolerance) == (kernels.gelu_ref, _GELU_VTANH_TOLERANCE)
    set_current_device(NPU1Col1())
    c = fn.contract_for()
    assert (c.reference, c.tolerance) == (
        kernels.gelu_aie2_lut_ref,
        _LUT_MODEL_TOLERANCE,
    )


def test_an_unbound_swiglu_is_judged_as_the_build_that_ran(no_device):
    fn = kernels.swiglu()
    set_current_device(NPU1Col1())
    assert fn.contract_for().reference is kernels.swiglu_table_ref
    set_current_device(NPU2Col1())
    assert fn.contract_for().reference is kernels.swiglu_ref


def _nudged(ref, ulps):
    """``ref`` with its largest element moved ``ulps`` ulps away from zero."""
    got = np.array(ref, copy=True)
    flat = got.reshape(-1)
    i = int(np.argmax(np.abs(flat.astype(np.float32))))
    bits = flat.view(np.uint16 if flat.dtype == bfloat16 else np.uint32)
    bits[i] += ulps
    return got


@pytest.mark.parametrize(
    "factory, kwargs, scalars, npu1, npu2",
    [
        (kernels.rms_norm, {}, (), _RMS_NORM_BF16_AIE2, _RMS_NORM_BF16_AIE2P),
        (
            kernels.rms_norm_eps,
            {},
            (1e-5,),
            _RMS_NORM_BF16_AIE2,
            _RMS_NORM_BF16_AIE2P,
        ),
        (kernels.layer_norm, {}, (), _LAYER_NORM_BF16_AIE2, _LAYER_NORM_BF16_AIE2P),
        (
            kernels.layer_norm_affine_cast,
            dict(cols=1024),
            (),
            _AFFINE_TOLERANCE["aie2"],
            _AFFINE_TOLERANCE["aie2p"],
        ),
    ],
    ids=["rms_norm", "rms_norm_eps", "layer_norm", "layer_norm_affine_cast"],
)
def test_a_norm_built_unbound_keeps_the_bound_measured_where_it_ran(
    no_device, factory, kwargs, scalars, npu1, npu2
):
    fn = factory(**kwargs)
    inputs = kd.sample_inputs(fn, calls=2)
    for device, measured in ((NPU1Col1(), npu1), (NPU2Col1(), npu2)):
        set_current_device(device)
        assert fn.contract_for().tolerance is measured
        ref = fn.expected(inputs, scalars=scalars)
        assert fn.judge(ref.ravel(), ref, calls=2)
        # _NORM_BF16, the generic bound, admits this; one ulp does not.
        off = _nudged(ref, 2)
        assert _NORM_BF16.rtol > 0.1
        assert not fn.judge(off.ravel(), ref, calls=2)


def test_an_f32_layer_norm_built_unbound_keeps_its_measured_bound(no_device):
    fn = kernels.layer_norm_f32(cols=1024)
    inputs = kd.sample_inputs(fn, calls=2)
    for device, measured in (
        (NPU1Col1(), _NORM_F32_AIE2),
        (NPU2Col1(), _NORM_F32_AIE2P),
    ):
        set_current_device(device)
        assert fn.contract_for().tolerance is measured
        ref = fn.expected(inputs)
        off = ref.copy()
        off[0, 0] += np.float32(1e-4)
        assert fn.judge(ref.ravel(), ref, calls=2)
        assert not fn.judge(off.ravel(), ref, calls=2)


def test_a_per_build_contract_names_no_build_it_was_not_given(no_device):
    fn = kernels.rms_norm()
    assert fn.contract.reference is None and fn.contract.tolerance is None
    with pytest.raises(ValueError, match="bind the device it ran on, or pass arch="):
        fn.contract_for()
    x = kd.sample_inputs(fn)
    with pytest.raises(ValueError, match="bind the device it ran on"):
        fn.expected(x)
    ref = fn.expected(x, arch="aie2p")
    assert fn.judge(ref.ravel(), ref, arch="aie2p")
    # An explicit tolerance needs no build.
    assert fn.judge(ref.ravel(), ref, tolerance=_NORM_BF16)
    # A contract the same for every build needs none either.
    add = kernels.add()
    ref = add.expected(kd.sample_inputs(add))
    assert add.judge(ref.ravel(), ref)


def test_a_per_build_contract_has_one_row_per_build():
    rows = tuple((b, kernels.gelu_ref, _NORM_BF16) for b in BUILDS)
    KernelContract(roles=(In, Out), per_build=rows)
    with pytest.raises(ValueError, match="one row for each"):
        KernelContract(roles=(In, Out), per_build=rows[1:])
    with pytest.raises(ValueError, match="one row for each"):
        KernelContract(roles=(In, Out), per_build=rows + rows[:1])
    with pytest.raises(ValueError, match="leaves reference and tolerance"):
        KernelContract(roles=(In, Out), per_build=rows, tolerance=_NORM_BF16)


@pytest.fixture
def portable(monkeypatch):
    monkeypatch.setenv("AIE_KERNELS_PORTABLE", "1")


@pytest.mark.parametrize("device", [NPU1Col1, NPU2Col1], ids=["npu1", "npu2"])
def test_a_portable_build_is_judged_by_its_untuned_branch(no_device, portable, device):
    set_current_device(device())
    for fn in (
        kernels.rms_norm(),
        kernels.layer_norm(),
        kernels.layer_norm_affine_cast(cols=1024),
    ):
        assert "-DAIE_KERNELS_PORTABLE" in fn.compile_flags
        assert fn.contract_for().tolerance is _NORM_BF16
    assert kernels.layer_norm_f32(cols=1024).contract_for().tolerance is _NORM_F32
    c = kernels.mm_activation_epilogue().contract_for()
    assert c.reference is kernels.mm_activation_epilogue_ref
    assert c.tolerance is _EPILOGUE_TOLERANCE
    assert kernels.bf16_exp().contract_for().reference is kernels.bf16_exp_ref
    # gelu.cc compiles gelu_aie2p.h unless AIE2's tuned branch is on.
    gelu = kernels.gelu(use_lut=True).contract_for()
    assert gelu.reference is kernels.gelu_lut_ref


def test_the_portable_flag_comes_from_the_kernel_not_the_environment(
    npu1_device, monkeypatch
):
    tuned = kernels.layer_norm_f32(cols=1024)
    epilogue = kernels.mm_activation_epilogue()
    monkeypatch.setenv("AIE_KERNELS_PORTABLE", "1")
    assert tuned.contract_for().tolerance is _NORM_F32_AIE2
    assert epilogue.contract_for().tolerance is _EPILOGUE_LUT_TOLERANCE
    assert kernels.layer_norm_f32(cols=1024).build_for() == Build("aie2", True)


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
def test_a_portable_compile_pairs_with_the_portable_row(
    npu1_device, monkeypatch, tmp_path
):
    tuned = kernels.layer_norm_f32(cols=1024)
    monkeypatch.setenv("AIE_KERNELS_PORTABLE", "1")
    portable = kernels.layer_norm_f32(cols=1024)
    assert portable.object_file_name != tuned.object_file_name
    compile_external_kernels([tuned, portable], str(tmp_path), "aie2")
    tuned_o = (tmp_path / tuned.object_file_name).read_bytes()
    portable_o = (tmp_path / portable.object_file_name).read_bytes()
    assert tuned_o != portable_o
    assert tuned.contract_for().tolerance is _NORM_F32_AIE2
    assert portable.contract_for().tolerance is _NORM_F32


def test_attention_and_fused_matmul_built_unbound_are_judged_where_they_ran(
    no_device,
):
    softmax = kernels.mha_softmax()
    fv = kernels.prefill_fv()
    fused = fused_mm(epilogue="gelu")
    set_current_device(NPU1Col1())
    npu1_softmax = softmax.contract_for().tolerance
    assert "exp2_bf16.h cubic" in npu1_softmax.note
    assert fv.contract_for().tolerance.kind == "bound"
    assert fused.contract_for().tolerance.kind == "relative"
    set_current_device(NPU2Col1())
    npu2_softmax = softmax.contract_for().tolerance
    assert "aie::exp2 interpolant" in npu2_softmax.note
    assert npu1_softmax.rtol < npu2_softmax.rtol
    assert fv.contract_for().tolerance.kind == "relative"
    assert fused.contract_for().tolerance.kind == "bound"


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
def test_a_lut_silu_sized_is_judged_by_its_table_model(npu2_device, tmp_path):
    vtanh = kernels.silu_sized(tile_size=1024)
    lut = kernels.silu_sized(tile_size=1024, use_lut=True)
    assert "-DACTIVATIONS_TANH_LUT=1" in lut.compile_flags
    assert vtanh.contract_for().reference is kernels.silu_ref
    c = lut.contract_for()
    assert (c.reference, c.tolerance) == (kernels.silu_table_ref, _LUT_MODEL_TOLERANCE)
    compile_external_kernels([vtanh, lut], str(tmp_path), "aie2p")
    assert (tmp_path / lut.object_file_name).read_bytes() != (
        tmp_path / vtanh.object_file_name
    ).read_bytes()
    # The table build against the true function, every finite bf16, is held
    # to an eighth of the relative error measured for vtanh's on npu2.
    x = _FINITE_BF16.astype(np.float64)
    with np.errstate(over="ignore"):
        silu = np.nan_to_num(x / (1 + np.exp(-x)))
    table = kernels.silu_table_ref(_FINITE_BF16).astype(np.float32)
    lut_bound = Tolerance.relative(0.01, 0.02)
    verdict = compare(table, silu.astype(np.float32), lut_bound)
    assert verdict, verdict.detail
    assert _vtanh_family_tolerance("silu").rtol >= 8 * lut_bound.rtol
