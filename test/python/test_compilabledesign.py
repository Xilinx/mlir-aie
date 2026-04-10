# test_compilabledesign.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Unit tests for CompilableDesign pure-logic surfaces — no NPU required.

Tests that exercise compile() or end-to-end kernel execution live in
test/python/npu-xrt/test_iron_jit_e2e.py (requires xrt_python_bindings).
"""

import json
import time
from pathlib import Path

import pytest

from aie.iron.compile.compilabledesign import CompilableDesign, _compute_hash
from aie.iron.compile.context import get_compile_arg
from aie.iron.compile.markers import Compile, In, InOut, Out

# ---------------------------------------------------------------------------
# Shared generator factories
# ---------------------------------------------------------------------------


def _gemm_gen():
    def gemm(
        a: In, b: In, c: Out, *, M: Compile[int], K: Compile[int], N: Compile[int]
    ):
        pass

    return gemm


def _scalar_gen():
    def f(a: In, c: Out, alpha: float, *, N: Compile[int]):
        pass

    return f


def _inout_gen():
    def f(x: InOut, *, M: Compile[int]):
        pass

    return f


# ---------------------------------------------------------------------------
# Construction defaults
# ---------------------------------------------------------------------------


def test_default_use_cache_is_true():
    d = CompilableDesign(_gemm_gen())
    assert d.use_cache is True


def test_default_compile_kwargs_is_empty():
    d = CompilableDesign(_gemm_gen())
    assert d.compile_kwargs == {}


def test_default_compile_flags_is_empty_list():
    d = CompilableDesign(_gemm_gen())
    assert d.compile_flags == []


def test_default_aiecc_flags_is_empty_list():
    d = CompilableDesign(_gemm_gen())
    assert d.aiecc_flags == []


def test_default_source_files_is_empty_list():
    d = CompilableDesign(_gemm_gen())
    assert d.source_files == []


def test_default_include_paths_is_empty_list():
    d = CompilableDesign(_gemm_gen())
    assert d.include_paths == []


def test_default_object_files_is_empty_list():
    d = CompilableDesign(_gemm_gen())
    assert d.object_files == []


def test_compile_kwargs_none_becomes_empty_dict():
    d = CompilableDesign(_gemm_gen(), compile_kwargs=None)
    assert d.compile_kwargs == {}


# ---------------------------------------------------------------------------
# Construction: param categorisation stored on the object
# ---------------------------------------------------------------------------


def test_compile_params_classified():
    d = CompilableDesign(_gemm_gen())
    assert d._compile_params == ["M", "K", "N"]


def test_tensor_params_classified():
    d = CompilableDesign(_gemm_gen())
    assert d._tensor_params == ["a", "b", "c"]


def test_scalar_params_classified():
    d = CompilableDesign(_scalar_gen())
    assert d._scalar_params == ["alpha"]


def test_inout_classified_as_tensor():
    d = CompilableDesign(_inout_gen())
    assert d._tensor_params == ["x"]


def test_path_generator_has_empty_param_lists():
    d = CompilableDesign(Path("/nonexistent/design.mlir"))
    assert d._compile_params == []
    assert d._tensor_params == []
    assert d._scalar_params == []


# ---------------------------------------------------------------------------
# Construction: paths normalised to Path objects
# ---------------------------------------------------------------------------


def test_source_files_strings_converted_to_paths():
    d = CompilableDesign(_gemm_gen(), source_files=["kernel.cc", "helper.cc"])
    assert all(isinstance(sf, Path) for sf in d.source_files)
    assert d.source_files[0].name == "kernel.cc"


def test_include_paths_strings_converted_to_paths():
    d = CompilableDesign(
        _gemm_gen(), include_paths=["/usr/include", "/opt/aie/include"]
    )
    assert all(isinstance(p, Path) for p in d.include_paths)


def test_object_files_strings_converted_to_paths():
    d = CompilableDesign(_gemm_gen(), object_files=["add.o", "mul.o"])
    assert all(isinstance(of, Path) for of in d.object_files)


def test_mixed_path_and_str_in_source_files():
    d = CompilableDesign(_gemm_gen(), source_files=[Path("a.cc"), "b.cc"])
    assert d.source_files[0] == Path("a.cc")
    assert d.source_files[1] == Path("b.cc")


# ---------------------------------------------------------------------------
# _generator_name
# ---------------------------------------------------------------------------


def test_generator_name_callable():
    gen = _gemm_gen()
    d = CompilableDesign(gen)
    assert d._generator_name() == gen.__name__


def test_generator_name_path():
    p = Path("/some/dir/design.mlir")
    d = CompilableDesign(p)
    assert d._generator_name() == str(p)


def test_generator_name_lambda():
    fn = lambda: None  # noqa: E731
    d = CompilableDesign(fn)
    assert "<lambda>" in d._generator_name()


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_contains_generator_name():
    gen = _gemm_gen()
    d = CompilableDesign(gen, compile_kwargs={"M": 512})
    r = repr(d)
    assert gen.__name__ in r
    assert "512" in r


def test_repr_contains_compile_kwargs():
    gen = _gemm_gen()
    d = CompilableDesign(gen, compile_kwargs={"M": 1024, "K": 256})
    r = repr(d)
    assert "1024" in r
    assert "256" in r


# ---------------------------------------------------------------------------
# _compute_hash / __hash__: stability and uniqueness
# ---------------------------------------------------------------------------


def test_hash_is_stable_across_two_constructions():
    gen = _gemm_gen()
    d1 = CompilableDesign(gen, compile_kwargs={"M": 512, "K": 256, "N": 128})
    d2 = CompilableDesign(gen, compile_kwargs={"M": 512, "K": 256, "N": 128})
    assert hash(d1) == hash(d2)


def test_hash_differs_for_different_kwargs_value():
    gen = _gemm_gen()
    d1 = CompilableDesign(gen, compile_kwargs={"M": 512})
    d2 = CompilableDesign(gen, compile_kwargs={"M": 1024})
    assert hash(d1) != hash(d2)


def test_hash_differs_for_different_kwargs_key():
    gen = _gemm_gen()
    d1 = CompilableDesign(gen, compile_kwargs={"M": 512})
    d2 = CompilableDesign(gen, compile_kwargs={"K": 512})
    assert hash(d1) != hash(d2)


def test_hash_stable_regardless_of_kwargs_dict_insertion_order():
    """JSON dump is sorted, so insertion order must not matter."""
    gen = _gemm_gen()
    d1 = CompilableDesign(gen, compile_kwargs={"M": 512, "K": 256})
    d2 = CompilableDesign(gen, compile_kwargs={"K": 256, "M": 512})
    assert hash(d1) == hash(d2)


def test_hash_differs_for_different_aiecc_flags():
    gen = _gemm_gen()
    d1 = CompilableDesign(gen, aiecc_flags=[])
    d2 = CompilableDesign(gen, aiecc_flags=["--verbose"])
    assert hash(d1) != hash(d2)


def test_hash_differs_for_different_compile_flags():
    gen = _gemm_gen()
    d1 = CompilableDesign(gen, compile_flags=[])
    d2 = CompilableDesign(gen, compile_flags=["-O3"])
    assert hash(d1) != hash(d2)


def test_hash_differs_for_different_generators():
    # Use meaningfully different bodies so that co_code differs.
    def gen_a(*, M: Compile[int]):
        x = M + 1  # noqa: F841
        return x

    def gen_b(*, M: Compile[int]):
        x = M * 2  # noqa: F841
        return x

    d1 = CompilableDesign(gen_a, compile_kwargs={"M": 512})
    d2 = CompilableDesign(gen_b, compile_kwargs={"M": 512})
    assert hash(d1) != hash(d2)


def test_hash_for_path_generator_uses_path_string():
    d1 = CompilableDesign(Path("/a/design.mlir"))
    d2 = CompilableDesign(Path("/b/design.mlir"))
    assert hash(d1) != hash(d2)


def test_hash_for_path_generator_stable_when_file_absent():
    d = CompilableDesign(Path("/nonexistent/design.mlir"))
    assert hash(d) == hash(d)


def test_hash_for_existing_source_file_includes_mtime(tmp_path):
    """Changing a source file (hence mtime) must change the hash."""
    src = tmp_path / "kernel.cc"
    src.write_text("// v1")
    d1 = CompilableDesign(_gemm_gen(), source_files=[src])
    h1 = hash(d1)

    time.sleep(0.01)
    src.write_text("// v2")

    d2 = CompilableDesign(_gemm_gen(), source_files=[src])
    assert h1 != hash(d2)


def test_hash_is_24_hex_chars():
    d = CompilableDesign(_gemm_gen())
    hex_str = d._compute_cache_hash()
    assert len(hex_str) == 24
    assert all(c in "0123456789abcdef" for c in hex_str)


def test_hash_is_valid_python_hash():
    """__hash__ must return a valid Python hash (fits in a signed int, != -1)."""
    d = CompilableDesign(_gemm_gen())
    h = hash(d)
    assert isinstance(h, int)
    assert h != -1
    # Must be usable as a dict/set key.
    mapping = {d: "ok"}
    assert mapping[d] == "ok"


# ---------------------------------------------------------------------------
# get_artifacts before compile
# ---------------------------------------------------------------------------


def test_get_artifacts_returns_none_before_compile():
    d = CompilableDesign(_gemm_gen())
    assert d.get_artifacts() is None


# ---------------------------------------------------------------------------
# split_runtime_args
# ---------------------------------------------------------------------------


def test_split_all_positional_tensors():
    def f(a: In, b: Out, *, N: Compile[int]):
        pass

    d = CompilableDesign(f, compile_kwargs={"N": 256})
    x, y = object(), object()
    tensors, scalars = d.split_runtime_args((x, y), {})
    assert tensors == [x, y]
    assert scalars == {}


def test_split_tensor_and_scalar_kwarg():
    gen = _scalar_gen()
    d = CompilableDesign(gen, compile_kwargs={"N": 512})
    a, c = object(), object()
    tensors, scalars = d.split_runtime_args((a, c), {"alpha": 0.5})
    assert tensors == [a, c]
    assert scalars == {"alpha": 0.5}


def test_split_inout_classified_as_tensor():
    def f(x: InOut, *, M: Compile[int]):
        pass

    d = CompilableDesign(f, compile_kwargs={"M": 128})
    obj = object()
    tensors, scalars = d.split_runtime_args((obj,), {})
    assert tensors == [obj]
    assert scalars == {}


def test_split_all_kwargs_tensors():
    def f(a: In, b: Out, *, N: Compile[int]):
        pass

    d = CompilableDesign(f, compile_kwargs={"N": 256})
    x, y = object(), object()
    tensors, scalars = d.split_runtime_args((), {"a": x, "b": y})
    assert tensors == [x, y]
    assert scalars == {}


def test_split_compile_params_excluded_from_walk():
    """compile_kwargs params must not consume runtime positional args."""

    def f(a: In, *, M: Compile[int]):
        pass

    d = CompilableDesign(f, compile_kwargs={"M": 512})
    obj = object()
    tensors, scalars = d.split_runtime_args((obj,), {})
    assert tensors == [obj]


def test_split_empty_args_and_kwargs():
    def f(a: In, *, N: Compile[int]):
        pass

    d = CompilableDesign(f, compile_kwargs={"N": 256})
    tensors, scalars = d.split_runtime_args((), {})
    assert tensors == []
    assert scalars == {}


def test_split_scalar_positional_arg():
    def f(a: In, alpha: float, *, N: Compile[int]):
        pass

    d = CompilableDesign(f, compile_kwargs={"N": 256})
    obj = object()
    tensors, scalars = d.split_runtime_args((obj, 0.5), {})
    assert tensors == [obj]
    assert scalars.get("alpha") == 0.5


def test_split_path_generator_passes_everything_as_tensors():
    d = CompilableDesign(Path("/nonexistent/design.mlir"))
    a, b = object(), object()
    tensors, scalars = d.split_runtime_args((a, b), {"extra": 1})
    assert tensors == [a, b]
    assert scalars == {"extra": 1}


# ---------------------------------------------------------------------------
# validate_tensor_args (currently no-op; must not raise)
# ---------------------------------------------------------------------------


def test_validate_tensor_args_is_no_op():
    d = CompilableDesign(_gemm_gen())
    d.validate_tensor_args([object(), object(), object()])
    d.validate_tensor_args([])
    d.validate_tensor_args([None])


# ---------------------------------------------------------------------------
# to_json / from_json round-trip
# ---------------------------------------------------------------------------


def test_to_json_is_valid_json():
    gen = _gemm_gen()
    d = CompilableDesign(gen, compile_kwargs={"M": 512})
    data = json.loads(d.to_json())
    assert isinstance(data, dict)


def test_to_json_contains_all_fields():
    gen = _gemm_gen()
    d = CompilableDesign(
        gen,
        use_cache=False,
        compile_kwargs={"M": 512, "K": 256, "N": 128},
        aiecc_flags=["--verbose"],
        compile_flags=["-O3"],
        source_files=["kernel.cc"],
        include_paths=["/opt/inc"],
        object_files=["add.o"],
    )
    data = json.loads(d.to_json())
    assert data["use_cache"] is False
    assert data["compile_kwargs"] == {
        "M": {"__type__": "int", "__value__": 512},
        "K": {"__type__": "int", "__value__": 256},
        "N": {"__type__": "int", "__value__": 128},
    }
    assert data["aiecc_flags"] == ["--verbose"]
    assert data["compile_flags"] == ["-O3"]
    assert "kernel.cc" in data["source_files"][0]
    assert "/opt/inc" in data["include_paths"][0]
    assert "add.o" in data["object_files"][0]
    assert "generator_name" in data
    assert "cache_hash" in data


def test_to_json_compile_kwargs_typed_encoding():
    import numpy as np

    gen = _gemm_gen()
    d = CompilableDesign(gen, compile_kwargs={"M": 512, "dtype": np.float32})
    data = json.loads(d.to_json())
    # int values are encoded with type tag
    assert data["compile_kwargs"]["M"] == {"__type__": "int", "__value__": 512}
    # unknown types fall back to string with unknown marker
    assert data["compile_kwargs"]["dtype"]["__type__"] == "unknown"
    assert isinstance(data["compile_kwargs"]["dtype"]["__value__"], str)


def test_from_json_requires_generator():
    gen = _gemm_gen()
    d = CompilableDesign(gen, compile_kwargs={"M": 512})
    with pytest.raises(ValueError, match="generator must be supplied"):
        CompilableDesign.from_json(d.to_json(), generator=None)


def test_from_json_restores_use_cache():
    gen = _gemm_gen()
    d = CompilableDesign(gen, use_cache=False)
    d2 = CompilableDesign.from_json(d.to_json(), generator=gen)
    assert d2.use_cache is False


def test_from_json_restores_flags():
    gen = _gemm_gen()
    d = CompilableDesign(gen, aiecc_flags=["--verbose"], compile_flags=["-O3"])
    d2 = CompilableDesign.from_json(d.to_json(), generator=gen)
    assert d2.aiecc_flags == ["--verbose"]
    assert d2.compile_flags == ["-O3"]


def test_from_json_restores_source_and_include_paths():
    gen = _gemm_gen()
    d = CompilableDesign(gen, source_files=["k.cc"], include_paths=["/opt"])
    d2 = CompilableDesign.from_json(d.to_json(), generator=gen)
    assert any("k.cc" in str(sf) for sf in d2.source_files)
    assert any("/opt" in str(p) for p in d2.include_paths)


def test_from_json_with_object_files():
    gen = _gemm_gen()
    d = CompilableDesign(gen, object_files=["add.o"])
    d2 = CompilableDesign.from_json(d.to_json(), generator=gen)
    assert any("add.o" in str(of) for of in d2.object_files)


def test_from_json_compile_kwargs_round_trip_typed():
    gen = _gemm_gen()
    d = CompilableDesign(gen, compile_kwargs={"M": 512})
    d2 = CompilableDesign.from_json(d.to_json(), generator=gen)
    # int values are round-tripped exactly (not as strings)
    assert d2.compile_kwargs["M"] == 512
    assert isinstance(d2.compile_kwargs["M"], int)


# ---------------------------------------------------------------------------
# _generate_mlir: compile param validation (no MLIR generation needed)
# ---------------------------------------------------------------------------


def test_generate_mlir_raises_type_error_for_missing_compile_param():
    """TypeError when a required Compile[T] param is absent from compile_kwargs."""
    from aie.iron.kernel import ExternalFunction

    def gen(*, M: Compile[int], K: Compile[int]):
        pass

    d = CompilableDesign(gen, compile_kwargs={"M": 512})  # K missing

    with pytest.raises(TypeError, match="compile_kwargs do not match"):
        d._generate_mlir(ExternalFunction)


def test_generate_mlir_type_error_message_includes_generator_name():
    from aie.iron.kernel import ExternalFunction

    def my_special_gen(*, M: Compile[int]):
        pass

    d = CompilableDesign(my_special_gen, compile_kwargs={})  # M missing

    with pytest.raises(TypeError, match="my_special_gen"):
        d._generate_mlir(ExternalFunction)


def test_generate_mlir_injects_compile_context():
    """CompileContext values must be visible via get_compile_arg() inside the generator."""
    from aie.iron.kernel import ExternalFunction

    observed = {}

    def gen(*, M: Compile[int], K: Compile[int]):
        observed["M"] = get_compile_arg("M")
        observed["K"] = get_compile_arg("K")
        # Return a real (empty) MLIR module via the unplaced path.
        from aie.extras.context import mlir_mod_ctx

        with mlir_mod_ctx() as ctx:
            pass
        return ctx.module

    d = CompilableDesign(gen, compile_kwargs={"M": 256, "K": 64})
    d._generate_mlir(ExternalFunction)

    assert observed["M"] == 256
    assert observed["K"] == 64


def test_generate_mlir_clears_external_function_instances_before_call():
    """Stale ExternalFunction instances must not leak into a new generation."""
    from aie.iron.kernel import ExternalFunction

    stale = object()
    ExternalFunction._instances.add(stale)

    def gen(*, M: Compile[int]):
        # Verify the stale instance was cleared before we ran.
        assert stale not in ExternalFunction._instances
        from aie.extras.context import mlir_mod_ctx

        with mlir_mod_ctx() as ctx:
            pass
        return ctx.module

    d = CompilableDesign(gen, compile_kwargs={"M": 1})
    d._generate_mlir(ExternalFunction)


def test_generate_mlir_unplaced_style_uses_return_value():
    """When generator returns a module object, _generate_mlir must return it."""
    from aie.iron.kernel import ExternalFunction
    from aie.extras.context import mlir_mod_ctx

    with mlir_mod_ctx() as ctx:
        pass
    real_module = ctx.module

    def gen(*, M: Compile[int]):
        return real_module  # unplaced style

    d = CompilableDesign(gen, compile_kwargs={"M": 1})
    result = d._generate_mlir(ExternalFunction)
    assert result is real_module


# ---------------------------------------------------------------------------
# _generate_mlir: Guard 2-A and 2-B validation
# ---------------------------------------------------------------------------


def test_generate_mlir_guard_2a_tensor_name_in_compile_kwargs():
    """compile_kwargs must not contain names annotated as In/Out/InOut."""
    from aie.iron.kernel import ExternalFunction

    def gen(a: In, *, M: Compile[int]):
        pass

    d = CompilableDesign(gen, compile_kwargs={"a": object(), "M": 1})
    with pytest.raises(TypeError, match="runtime tensors"):
        d._generate_mlir(ExternalFunction)


def test_generate_mlir_guard_2b_unknown_key_in_compile_kwargs():
    """compile_kwargs must not contain keys absent from the generator signature."""
    from aie.iron.kernel import ExternalFunction

    def gen(a: In, *, M: Compile[int]):
        pass

    d = CompilableDesign(gen, compile_kwargs={"M": 1, "NOSUCHPARAM": 99})
    with pytest.raises(TypeError, match="not in the generator signature"):
        d._generate_mlir(ExternalFunction)


def test_generate_mlir_raises_on_verification_failure():
    """RuntimeError must be raised when the generated MLIR module fails verify()."""
    from aie.iron.kernel import ExternalFunction
    from unittest.mock import MagicMock

    bad_module = MagicMock()
    bad_module.operation.verify.return_value = False

    def gen(*, M: Compile[int]):
        return bad_module  # unplaced style — returns a module directly

    d = CompilableDesign(gen, compile_kwargs={"M": 1})
    with pytest.raises(RuntimeError, match="MLIR verification failed"):
        d._generate_mlir(ExternalFunction)


def test_split_runtime_args_path_generator_filters_kernel_objects():
    """Kernel/ExternalFunction instances must be stripped even for Path generators."""
    from aie.iron.kernel import Kernel

    d = CompilableDesign(Path("/nonexistent/design.mlir"))
    k = Kernel("my_func", "my_func.o")
    a, b = object(), object()
    tensors, scalars = d.split_runtime_args((a, k, b), {})
    assert k not in tensors
    assert a in tensors
    assert b in tensors


# ---------------------------------------------------------------------------
# transform_typed
# ---------------------------------------------------------------------------


def test_parse_expected_tensor_sizes_matches_real_mlir_format(tmp_path):
    """Regex must extract DMA element counts from lowered aie.runtime_sequence MLIR."""
    gen = _gemm_gen()
    d = CompilableDesign(gen)
    sample_mlir = """\
module {
  aie.device(npu2) {
    aie.runtime_sequence(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
      aie.dma_configure_task_for @of_in {
        aie.dma_bd(%arg0 : memref<1024xi32>, 0, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
      }
      aie.dma_configure_task_for @of_out {
        aie.dma_bd(%arg1 : memref<1024xi32>, 0, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
      }
    }
  }
}
"""
    mlir_path = tmp_path / "input_with_addresses.mlir"
    mlir_path.write_text(sample_mlir)
    sizes = d._parse_expected_tensor_sizes(tmp_path)
    assert sizes == [1024, 1024], f"Expected [1024, 1024], got {sizes}"


def test_transform_typed_returns_module():
    import numpy as np
    from aie.iron.algorithms import transform_typed
    from aie.iron.device import NPU1Col1
    from aie.utils.hostruntime import set_current_device

    set_current_device(NPU1Col1())
    try:
        tensor_ty = np.ndarray[(1024,), np.dtype[np.int32]]
        # This should not raise and should return an MLIR module
        module = transform_typed(lambda x: x + 1, tensor_ty, tile_size=16)
        assert module is not None
        assert hasattr(module, "operation")
    finally:
        set_current_device(None)
