# test_compilabledesign.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Unit tests for CompilableDesign pure-logic surfaces — no NPU required.

Tests that exercise compile() or end-to-end kernel execution live in
test/python/npu/test_iron_jit_e2e.py (requires a host runtime backend).
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import CodeType

import pytest

from aie.extras.context import mlir_mod_ctx
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.kernel import ExternalFunction, Kernel
from aie.utils.compile.jit._dma_size_parser import parse_dma_sizes
from aie.utils.compile.jit.compilabledesign import CompilableDesign, _compute_hash
from aie.utils.compile.jit.context import get_compile_arg
from aie.utils.compile.jit.markers import CompileTime, DispatchTime, In, InOut, Out
from aie.utils.hostruntime import set_current_device

# ---------------------------------------------------------------------------
# Shared generator factories
# ---------------------------------------------------------------------------


def _gemm_gen():
    def gemm(
        a: In,
        b: In,
        c: Out,
        *,
        M: CompileTime[int],
        K: CompileTime[int],
        N: CompileTime[int],
    ):
        pass

    return gemm


def _scalar_gen():
    def f(a: In, c: Out, alpha: float, *, N: CompileTime[int]):
        pass

    return f


def _inout_gen():
    def f(x: InOut, *, M: CompileTime[int]):
        pass

    return f


def _dispatch_gen():
    def f(a: In, c: Out, scale: DispatchTime[int], *, N: CompileTime[int]):
        pass

    return f


# ---------------------------------------------------------------------------
# Construction defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attr,expected",
    [
        ("use_cache", True),
        ("compile_kwargs", {}),
        ("compile_flags", ()),
        ("aiecc_flags", ()),
        ("source_files", ()),
        ("include_paths", ()),
        ("object_files", ()),
    ],
)
def test_construction_default(attr, expected):
    d = CompilableDesign(_gemm_gen())
    assert getattr(d, attr) == expected


def test_compile_kwargs_none_becomes_empty_dict():
    d = CompilableDesign(_gemm_gen(), compile_kwargs=None)
    assert d.compile_kwargs == {}


# ---------------------------------------------------------------------------
# Construction: param categorisation stored on the object
# ---------------------------------------------------------------------------


def test_compile_params_classified():
    d = CompilableDesign(_gemm_gen())
    assert d.compile_params == ["M", "K", "N"]


def test_tensor_params_classified():
    d = CompilableDesign(_gemm_gen())
    assert d.tensor_params == ["a", "b", "c"]


def test_scalar_params_classified():
    d = CompilableDesign(_scalar_gen())
    assert d.scalar_params == ["alpha"]


def test_inout_classified_as_tensor():
    d = CompilableDesign(_inout_gen())
    assert d.tensor_params == ["x"]


def test_dispatch_params_classified():
    d = CompilableDesign(_dispatch_gen())
    assert d.dispatch_params == ["scale"]
    # DispatchTime[T] params must not also land in scalar_params or compile_params.
    assert d.scalar_params == []
    assert d.compile_params == ["N"]


def test_path_generator_has_empty_param_lists():
    d = CompilableDesign(Path("/nonexistent/design.mlir"))
    assert d.compile_params == []
    assert d.tensor_params == []
    assert d.dispatch_params == []
    assert d.scalar_params == []


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
    assert d.generator_name == gen.__name__


def test_generator_name_path():
    p = Path("/some/dir/design.mlir")
    d = CompilableDesign(p)
    assert d.generator_name == str(p)


def test_generator_name_lambda():
    fn = lambda: None  # noqa: E731
    d = CompilableDesign(fn)
    assert "<lambda>" in d.generator_name


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


def test_hash_differs_for_different_include_paths():
    """-I directories change which headers the C++ kernel compiles against."""
    gen = _gemm_gen()
    d1 = CompilableDesign(gen, include_paths=["/a/include"])
    d2 = CompilableDesign(gen, include_paths=["/b/include"])
    assert hash(d1) != hash(d2)


def test_hash_differs_for_include_path_order():
    """-I search order decides which header wins, so it is not order-free."""
    gen = _gemm_gen()
    d1 = CompilableDesign(gen, include_paths=["/a/include", "/b/include"])
    d2 = CompilableDesign(gen, include_paths=["/b/include", "/a/include"])
    assert hash(d1) != hash(d2)


def test_hash_differs_for_different_generators():
    # Use meaningfully different bodies so that co_code differs.
    def gen_a(*, M: CompileTime[int]):
        x = M + 1  # noqa: F841
        return x

    def gen_b(*, M: CompileTime[int]):
        x = M * 2  # noqa: F841
        return x

    d1 = CompilableDesign(gen_a, compile_kwargs={"M": 512})
    d2 = CompilableDesign(gen_b, compile_kwargs={"M": 512})
    assert hash(d1) != hash(d2)


def test_hash_works_when_peano_install_dir_is_invalid(monkeypatch):
    """``hash(design)`` must not require Peano to be installed.

    The CI "Build and Test (Release, ...)" matrix doesn't install Peano,
    yet pytest still imports CompilableDesign and hashes designs as part
    of its membership / equality checks.  ``_compute_hash`` must fall back
    to the "absent" sentinel rather than letting the ``RuntimeError`` raised
    by ``peano_cxx_path`` / ``peano_install_dir`` escape.
    """
    import aie.utils.configure as _aiecc_configure

    monkeypatch.setattr(_aiecc_configure, "peano_install_dir", "peano_not_found")

    gen = _gemm_gen()
    d1 = CompilableDesign(gen, compile_kwargs={"M": 512})
    d2 = CompilableDesign(gen, compile_kwargs={"M": 512})
    d3 = CompilableDesign(gen, compile_kwargs={"M": 1024})

    # Neither call should raise; equal-keyed designs must still match,
    # different-keyed designs must still differ.
    assert hash(d1) == hash(d2)
    assert hash(d1) != hash(d3)


def test_hash_stable_for_generator_with_dispatch_param():
    """A DispatchTime[T] param in the signature must not affect hash(design).

    DispatchTime[T] values are never part of compile_kwargs (Guard 0-A), so
    two designs built from the same generator + compile_kwargs must hash
    identically -- there is no way to make hash(design) depend on a runtime
    scalar value.
    """
    gen = _dispatch_gen()
    d1 = CompilableDesign(gen, compile_kwargs={"N": 512})
    d2 = CompilableDesign(gen, compile_kwargs={"N": 512})
    assert hash(d1) == hash(d2)


def test_hash_differs_for_compile_time_change_with_dispatch_param_present():
    """Contrast case.

    CompileTime[T] changes still rehash even when the same generator also
    declares a DispatchTime[T] param.
    """
    gen = _dispatch_gen()
    d1 = CompilableDesign(gen, compile_kwargs={"N": 512})
    d2 = CompilableDesign(gen, compile_kwargs={"N": 1024})
    assert hash(d1) != hash(d2)


def test_dispatch_time_value_rejected_before_it_can_reach_hash():
    """A DispatchTime[T] name in compile_kwargs is rejected at construction.

    Before hash(design) could ever be computed from it (Guard 0-A). Waiting
    until generation (the old Guard 2-A2 location) would be too late: a
    cache-hit compile() never calls _generate_mlir, so __hash__ would already
    have read the polluted compile_kwargs.
    """
    gen = _dispatch_gen()
    with pytest.raises(TypeError, match="runtime scalars"):
        CompilableDesign(gen, compile_kwargs={"scale": 4, "N": 512})


def test_hash_works_when_dispatch_toolchain_is_missing(monkeypatch):
    """``hash(design)`` must not require the dispatch toolchain to be installed.

    aie-opt/aie-translate/a host C++ compiler are needed to *compile* a
    DispatchTime[T] design, but hashing it must still work without them.
    """
    import aie.utils.config as _config

    def _raise(*_a, **_kw):
        raise RuntimeError("not found")

    monkeypatch.setattr(_config, "aie_opt_path", _raise)
    monkeypatch.setattr(_config, "aie_translate_path", _raise)
    monkeypatch.setattr(_config, "host_cxx_path", _raise)

    gen = _dispatch_gen()
    d1 = CompilableDesign(gen, compile_kwargs={"N": 512})
    d2 = CompilableDesign(gen, compile_kwargs={"N": 512})
    assert hash(d1) == hash(d2)


def test_hash_for_path_generator_uses_path_string():
    d1 = CompilableDesign(Path("/a/design.mlir"))
    d2 = CompilableDesign(Path("/b/design.mlir"))
    assert hash(d1) != hash(d2)


def test_hash_for_existing_source_file_tracks_content(tmp_path):
    """Changing a source file's content must change the hash.

    The key digests bytes, so an edit is observable without waiting out
    filesystem mtime granularity.
    """
    src = tmp_path / "kernel.cc"
    src.write_text("// v1")
    d1 = CompilableDesign(_gemm_gen(), source_files=[src])
    h1 = hash(d1)

    src.write_text("// v2")

    d2 = CompilableDesign(_gemm_gen(), source_files=[src])
    assert h1 != hash(d2)


def test_hash_survives_touch_of_an_unchanged_source(tmp_path):
    """Restamping a source file must NOT change the hash.

    This is the property that makes the cache reusable at all across a fresh
    checkout, a reinstall or a ``cp``: those restamp every file without
    changing a byte, and under an mtime-keyed hash every entry was invalidated
    for reasons that had nothing to do with the build.
    """
    src = tmp_path / "kernel.cc"
    src.write_text("// v1")
    d1 = CompilableDesign(_gemm_gen(), source_files=[src])
    h1 = hash(d1)

    st = src.stat()
    os.utime(src, (st.st_atime + 10_000, st.st_mtime + 10_000))

    d2 = CompilableDesign(_gemm_gen(), source_files=[src])
    assert hash(d2) == h1


def test_hash_survives_touch_of_an_unchanged_object_file(tmp_path):
    """Same property for prebuilt objects, which are keyed the same way."""
    obj = tmp_path / "kernel.o"
    obj.write_bytes(b"\x7fELF-not-really")
    d1 = CompilableDesign(_gemm_gen(), object_files=[obj])
    h1 = hash(d1)

    st = obj.stat()
    os.utime(obj, (st.st_atime + 10_000, st.st_mtime + 10_000))

    d2 = CompilableDesign(_gemm_gen(), object_files=[obj])
    assert hash(d2) == h1


def test_hash_for_path_generator_survives_touch_but_tracks_content(tmp_path):
    """A static .mlir design is keyed the same way as a C++ source."""
    design = tmp_path / "design.mlir"
    design.write_text("module {}\n")
    h1 = hash(CompilableDesign(design))

    st = design.stat()
    os.utime(design, (st.st_atime + 10_000, st.st_mtime + 10_000))
    assert hash(CompilableDesign(design)) == h1

    design.write_text("module { // edited\n}\n")
    assert hash(CompilableDesign(design)) != h1


def test_hash_changes_when_content_changes_under_a_preserved_mtime(tmp_path):
    """Two different kernels at one path with one mtime must not share a key.

    This is the correctness half rather than the hit-rate half.  An mtime-keyed
    hash cannot tell these apart, so the second build is served the first
    build's artifact -- a stale-artifact ride, where a changed kernel runs as
    the old binary.  Restoring the mtime is not exotic: ``git checkout`` of
    another revision, a restore from an archive that carries timestamps, and
    any build step that copies with ``-p`` all reproduce it.
    """
    src = tmp_path / "kernel.cc"
    src.write_text("void k() { /* v1 */ }")
    st = src.stat()
    h1 = hash(CompilableDesign(_gemm_gen(), source_files=[src]))

    src.write_text("void k() { /* v2 -- different code */ }")
    os.utime(src, (st.st_atime, st.st_mtime))  # same timestamp, new bytes
    assert src.stat().st_mtime == st.st_mtime

    assert hash(CompilableDesign(_gemm_gen(), source_files=[src])) != h1


def test_hash_keys_on_the_aiecc_the_compile_uses(monkeypatch):
    """aiecc must be resolved as compilation resolves it, not via PATH."""
    import aie.utils.config as config

    seen = []

    def fake_aiecc_path():
        seen.append(True)
        return config.__file__  # any real file; only its mtime is consumed

    monkeypatch.setattr(config, "aiecc_path", fake_aiecc_path)
    CompilableDesign(_gemm_gen())._compute_cache_hash()
    assert seen, "artifact hash did not consult config.aiecc_path()"


def test_content_digest_streams_a_large_file(tmp_path):
    """Digesting must not materialise the whole input.

    In-tree kernels already carry multi-megabyte generated LUT headers, so the
    read is chunked; this pins that the chunked path agrees with hashlib.
    """
    import hashlib

    from aie.utils.compile.jit import _hash as _hash_mod

    blob = tmp_path / "big.h"
    payload = (b"0123456789abcdef" * 64) * 1024 + b"tail"  # 1 MiB + 4, two reads
    blob.write_bytes(payload)

    assert _hash_mod._content_digest(blob) == hashlib.sha256(payload).hexdigest()


def test_unreadable_input_does_not_alias_onto_a_readable_one(tmp_path):
    """An input we cannot read must not share a key with one we can.

    Skipping it would collapse "missing" and "empty" onto the same digest, so a
    design whose kernel disappeared would hit the entry built when it was there.
    """
    from aie.utils.compile.jit import _hash as _hash_mod

    missing = tmp_path / "gone.cc"
    empty = tmp_path / "empty.cc"
    empty.write_bytes(b"")

    absent = _hash_mod._content_digest(missing)
    assert absent.startswith("<unreadable:")
    assert absent != _hash_mod._content_digest(empty)


def _design(body, name="design"):
    """A generator built from source, so a pair can differ in exactly one way."""
    ns = {"__name__": "designs.probe"}
    exec(body, ns)  # noqa: S102 -- controlling the source is the point
    return ns[name]


def test_hash_is_stable_for_a_generator_with_a_nested_function(tmp_path):
    """repr() of a nested code object embeds its address; the key must not."""
    script = tmp_path / "probe.py"
    script.write_text(
        "from aie.utils.compile.jit._hash import _compute_recipe_hash\n"
        "def make():\n"
        "    def design(a, b):\n"
        "        def core(x):\n"
        "            return x + 1\n"
        "        return core\n"
        "    return design\n"
        "print(_compute_recipe_hash(make(), {}, (), ()))\n"
    )
    seen = {
        subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(3)
    }
    assert len(seen) == 1, f"recipe hash differed between processes: {seen}"


def test_hash_distinguishes_designs_differing_only_in_a_nested_body():
    """Only recursion into the nested code object can tell these apart.

    Both are named `design` in one module, so qualname and module cannot
    discriminate, and the outer bytecode is asserted byte-identical.
    """

    def build(leaf):
        return _design(
            "def design(a, b):\n"
            "    def core(x):\n"
            f"        return x + {leaf}\n"
            "    return core\n"
        )

    a, b = build(1), build(2)
    assert a.__qualname__ == b.__qualname__
    assert (
        a.__code__.co_code == b.__code__.co_code
    ), "outer bytecode differs; this test would not exercise the nested walk"
    assert _compute_hash(a, {}, [], [], [], []) != _compute_hash(b, {}, [], [], [], [])


def test_hash_distinguishes_designs_calling_different_symbols():
    """A symbol is an index into co_names, so the bytecode is identical."""

    def build(kernel):
        return _design(
            "def design(a_ty, b_ty, c_ty):\n"
            "    def core_body(a, b, c):\n"
            f"        {kernel}(a, b, c)\n"
            "    return core_body\n"
        )

    def nested(fn):
        return next(c for c in fn.__code__.co_consts if isinstance(c, CodeType))

    a, b = build("matmul_bf16"), build("matmul_i8")
    assert (
        nested(a).co_code == nested(b).co_code
    ), "nested bytecode differs; this test would not exercise co_names"
    assert _compute_hash(a, {}, [], [], [], []) != _compute_hash(b, {}, [], [], [], [])


def test_hash_distinguishes_designs_differing_only_in_a_compile_time_default():
    """A CompileTime[T] left at its default never reaches compile_kwargs."""

    def build(tile):
        return _design(
            f"def design(a, b, *, TILE=({tile})):\n"
            "    def core(x):\n"
            "        return x * TILE\n"
            "    return core\n"
        )

    a, b = build(512), build(1024)
    assert (
        a.__code__.co_code == b.__code__.co_code
    ), "bytecode differs; the default would not be the only difference"
    assert _compute_hash(a, {}, [], [], [], []) != _compute_hash(b, {}, [], [], [], [])


def test_hash_of_a_callable_compile_time_value_is_stable_and_not_blind():
    """The callable branch of _kwarg_repr needs the same treatment."""

    def build(factor):
        return _design(
            "def act(x):\n"
            "    def inner(y):\n"
            f"        return y * {factor}\n"
            "    return inner(x)\n",
            name="act",
        )

    gen = _design("def design(a, b):\n    return a\n")

    def key(act):
        return _compute_hash(gen, {"act": act}, [], [], [], [])

    assert key(build(2)) == key(build(2))
    assert key(build(2)) != key(build(3))


def test_hash_survives_a_move_of_the_design_file():
    """co_filename and line info are not part of the design."""
    src = "def design(a):\n    def core(x):\n        return x + 1\n    return core\n"

    def build(filename, line_offset=0):
        ns = {"__name__": "designs.probe"}
        exec(  # noqa: S102 -- the location is what this test varies
            compile("\n" * line_offset + src, filename, "exec"), ns
        )
        return ns["design"]

    def key(fn):
        return _compute_hash(fn, {}, [], [], [], [])

    assert key(build("/checkout-a/design.py")) == key(build("/checkout-b/design.py"))
    assert key(build("/checkout-a/design.py")) == key(
        build("/checkout-a/design.py", line_offset=8)
    )
    # Reflowing a body leaves co_code alone and rewrites co_linetable, which a
    # uniform line shift does not.
    flat = _design("def design(a):\n    return (a + 1)\n")
    reflowed = _design("def design(a):\n    return (\n        a\n        + 1\n    )\n")
    assert flat.__code__.co_code == reflowed.__code__.co_code
    assert key(flat) == key(reflowed)


def _nest(leaf, depth=40):
    src = f"def f0():\n    {leaf}\n"
    for i in range(1, depth):
        body = "\n".join("    " + line for line in src.splitlines())
        src = f"def f{i}():\n{body}\n    return f{i - 1}\n"
    return _design(src, name=f"f{depth - 1}")


def test_hash_distinguishes_bodies_nested_past_any_fixed_depth():
    """A bound on the walk would silently collide designs differing past it."""
    assert _compute_hash(_nest("return 1"), {}, [], [], [], []) != _compute_hash(
        _nest("return 2"), {}, [], [], [], []
    )


def test_hash_handles_deeply_nested_functions():
    """The walk must terminate on pathological nesting."""
    assert len(_compute_hash(_nest("pass"), {}, [], [], [], [])) == 24


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
    def f(a: In, b: Out, *, N: CompileTime[int]):
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
    def f(x: InOut, *, M: CompileTime[int]):
        pass

    d = CompilableDesign(f, compile_kwargs={"M": 128})
    obj = object()
    tensors, scalars = d.split_runtime_args((obj,), {})
    assert tensors == [obj]
    assert scalars == {}


def test_split_all_kwargs_tensors():
    def f(a: In, b: Out, *, N: CompileTime[int]):
        pass

    d = CompilableDesign(f, compile_kwargs={"N": 256})
    x, y = object(), object()
    tensors, scalars = d.split_runtime_args((), {"a": x, "b": y})
    assert tensors == [x, y]
    assert scalars == {}


def test_split_compile_params_excluded_from_walk():
    """compile_kwargs params must not consume runtime positional args."""

    def f(a: In, *, M: CompileTime[int]):
        pass

    d = CompilableDesign(f, compile_kwargs={"M": 512})
    obj = object()
    tensors, scalars = d.split_runtime_args((obj,), {})
    assert tensors == [obj]


def test_split_empty_args_and_kwargs():
    def f(a: In, *, N: CompileTime[int]):
        pass

    d = CompilableDesign(f, compile_kwargs={"N": 256})
    tensors, scalars = d.split_runtime_args((), {})
    assert tensors == []
    assert scalars == {}


def test_split_scalar_positional_arg():
    def f(a: In, alpha: float, *, N: CompileTime[int]):
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
        "M": ["int", 512],
        "K": ["int", 256],
        "N": ["int", 128],
    }
    assert data["aiecc_flags"] == ["--verbose"]
    assert data["compile_flags"] == ["-O3"]
    assert "kernel.cc" in data["source_files"][0]
    assert "opt/inc" in data["include_paths"][0].replace("\\", "/")
    assert "add.o" in data["object_files"][0]
    assert "generator_name" in data
    assert "cache_hash" in data


def test_to_json_compile_kwargs_typed_encoding():
    import numpy as np

    gen = _gemm_gen()
    d = CompilableDesign(gen, compile_kwargs={"M": 512, "dtype": np.float32})
    data = json.loads(d.to_json())
    # int values are encoded with type tag
    assert data["compile_kwargs"]["M"] == ["int", 512]
    # unknown types fall back to ["str", repr-string]
    assert data["compile_kwargs"]["dtype"][0] == "str"
    assert isinstance(data["compile_kwargs"]["dtype"][1], str)


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
    assert d2.aiecc_flags == ("--verbose",)
    assert d2.compile_flags == ("-O3",)


def test_from_json_restores_source_and_include_paths():
    gen = _gemm_gen()
    d = CompilableDesign(gen, source_files=["k.cc"], include_paths=["/opt"])
    d2 = CompilableDesign.from_json(d.to_json(), generator=gen)
    assert any("k.cc" in str(sf) for sf in d2.source_files)
    assert any("opt" in str(p).replace("\\", "/") for p in d2.include_paths)


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
    """TypeError when a required CompileTime[T] param is absent from compile_kwargs."""

    def gen(*, M: CompileTime[int], K: CompileTime[int]):
        pass

    d = CompilableDesign(gen, compile_kwargs={"M": 512})  # K missing

    with pytest.raises(TypeError, match="compile_kwargs do not match"):
        d._generate_mlir(ExternalFunction)


def test_generate_mlir_type_error_message_includes_generator_name():

    def my_special_gen(*, M: CompileTime[int]):
        pass

    d = CompilableDesign(my_special_gen, compile_kwargs={})  # M missing

    with pytest.raises(TypeError, match="my_special_gen"):
        d._generate_mlir(ExternalFunction)


def test_generate_mlir_injects_compile_context():
    """CompileContext values must be visible via get_compile_arg() inside the generator."""

    observed = {}

    def gen(*, M: CompileTime[int], K: CompileTime[int]):
        observed["M"] = get_compile_arg("M")
        observed["K"] = get_compile_arg("K")
        # Return a real (empty) MLIR module via the unplaced path.

        with mlir_mod_ctx() as ctx:
            pass
        return ctx.module

    d = CompilableDesign(gen, compile_kwargs={"M": 256, "K": 64})
    d._generate_mlir(ExternalFunction)

    assert observed["M"] == 256
    assert observed["K"] == 64


def test_generate_mlir_clears_external_function_instances_before_call():
    """Stale ExternalFunction instances must not leak into a new generation."""

    stale = object()
    ExternalFunction._instances.add(stale)

    def gen(*, M: CompileTime[int]):
        # Verify the stale instance was cleared before we ran.
        assert stale not in ExternalFunction._instances

        with mlir_mod_ctx() as ctx:
            pass
        return ctx.module

    d = CompilableDesign(gen, compile_kwargs={"M": 1})
    d._generate_mlir(ExternalFunction)


def test_generate_mlir_unplaced_style_uses_return_value():
    """When generator returns a module object, _generate_mlir must use it (not ctx.module).

    Option B memoization re-parses the cached MLIR text into a fresh Module
    per call, so identity is not preserved — content equivalence is the
    contract.
    """

    with mlir_mod_ctx() as ctx:
        pass
    real_module = ctx.module
    expected_text = str(real_module)

    def gen(*, M: CompileTime[int]):
        return real_module  # unplaced style

    d = CompilableDesign(gen, compile_kwargs={"M": 1})
    result = d._generate_mlir(ExternalFunction)
    assert str(result) == expected_text


# ---------------------------------------------------------------------------
# _generate_mlir: Guard 2-A and 2-B validation
# ---------------------------------------------------------------------------


def test_generate_mlir_guard_2a_tensor_name_in_compile_kwargs():
    """compile_kwargs must not contain names annotated as In/Out/InOut."""

    def gen(a: In, *, M: CompileTime[int]):
        pass

    d = CompilableDesign(gen, compile_kwargs={"a": object(), "M": 1})
    with pytest.raises(TypeError, match="runtime tensors"):
        d._generate_mlir(ExternalFunction)


def test_generate_mlir_guard_2b_unknown_key_in_compile_kwargs():
    """compile_kwargs must not contain keys absent from the generator signature."""

    def gen(a: In, *, M: CompileTime[int]):
        pass

    d = CompilableDesign(gen, compile_kwargs={"M": 1, "NOSUCHPARAM": 99})
    with pytest.raises(TypeError, match="not in the generator signature"):
        d._generate_mlir(ExternalFunction)


def test_generate_mlir_dispatch_param_receives_wrapped_type():
    """DispatchTime[T] params generate from the wrapped type T, not a value."""
    import numpy as np

    observed = {}

    def gen(scale: DispatchTime[np.int32], *, M: CompileTime[int]):
        observed["scale"] = scale
        with mlir_mod_ctx() as ctx:
            pass
        return ctx.module

    d = CompilableDesign(gen, compile_kwargs={"M": 1})
    d._generate_mlir(ExternalFunction)

    assert observed["scale"] is np.int32


def test_generate_mlir_raises_on_verification_failure():
    """RuntimeError must be raised when the generated MLIR module fails verify()."""
    from unittest.mock import MagicMock

    bad_module = MagicMock()
    bad_module.operation.verify.return_value = False

    def gen(*, M: CompileTime[int]):
        return bad_module  # unplaced style — returns a module directly

    d = CompilableDesign(gen, compile_kwargs={"M": 1})
    with pytest.raises(RuntimeError, match="MLIR verification failed"):
        d._generate_mlir(ExternalFunction)


def test_split_runtime_args_path_generator_filters_kernel_objects():
    """Kernel/ExternalFunction instances must be stripped even for Path generators."""

    d = CompilableDesign(Path("/nonexistent/design.mlir"))
    k = Kernel("my_func", "my_func.o")
    a, b = object(), object()
    tensors, scalars = d.split_runtime_args((a, k, b), {})
    assert k not in tensors
    assert a in tensors
    assert b in tensors


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------


def test_parse_dma_sizes_matches_real_mlir_format(tmp_path):
    """Reads element counts directly from the runtime_sequence's typed args
    on a sample matching what aiecc emits.
    """

    sample_mlir = """\
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
      %c0_i32 = arith.constant 0 : i32
      %c1024_i32 = arith.constant 1024 : i32
      %0 = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = %c0_i32 len = %c1024_i32 sizes = [1, 1, 1, 1024] strides = [0, 0, 0, 1]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @of_out {
        aie.dma_bd(%arg1 : memref<1024xi32> offset = %c0_i32 len = %c1024_i32 sizes = [1, 1, 1, 1024] strides = [0, 0, 0, 1]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%0)
      aiex.dma_await_task(%1)
    }
  }
}
"""
    mlir_path = tmp_path / "input_with_addresses.mlir"
    mlir_path.write_text(sample_mlir)
    sizes = parse_dma_sizes(tmp_path)
    assert sizes == [1024, 1024], f"Expected [1024, 1024], got {sizes}"


def test_parse_dma_sizes_handles_repeated_transfer(tmp_path):
    """matmul-style: the same host arg can carry several dma_bd ops (one per
    tile_row reload, or both an MM2S fill and S2MM drain on an InOut buffer).
    Reading the runtime_sequence arg type — instead of summing dma_bd lens —
    means this case Just Works without any per-arg accumulator.
    """

    sample_mlir = """\
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<1024xi32>) {
      %c0_i32 = arith.constant 0 : i32
      %c1024_i32 = arith.constant 1024 : i32
      %0 = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = %c0_i32 len = %c1024_i32 sizes = [1, 1, 1, 1024] strides = [0, 0, 0, 1]) {burst_length = 0 : i32}
        aie.end
      }
      %1 = aiex.dma_configure_task_for @of_in_again {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = %c0_i32 len = %c1024_i32 sizes = [1, 1, 1, 1024] strides = [0, 0, 0, 1]) {burst_length = 0 : i32}
        aie.end
      }
    }
  }
}
"""
    (tmp_path / "input_with_addresses.mlir").write_text(sample_mlir)
    sizes = parse_dma_sizes(tmp_path)
    assert sizes == [1024], f"Expected [1024] (signature-based), got {sizes}"


def test_parse_dma_sizes_handles_disjoint_fan_out(tmp_path):
    """Multi-column fan-out: one host arg fans into N disjoint per-column
    DMAs.  The runtime_sequence arg type still reports the full buffer size,
    no DMA accounting needed.
    """

    sample_mlir = """\
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<1024xi32>) {
      %c0_i32 = arith.constant 0 : i32
      %c512_i32 = arith.constant 512 : i32
      %0 = aiex.dma_configure_task_for @of_in_a {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = %c0_i32 len = %c512_i32 sizes = [1, 1, 1, 512] strides = [0, 0, 0, 1]) {burst_length = 0 : i32}
        aie.end
      }
      %1 = aiex.dma_configure_task_for @of_in_b {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = %c512_i32 len = %c512_i32 sizes = [1, 1, 1, 512] strides = [0, 0, 0, 1]) {burst_length = 0 : i32}
        aie.end
      }
    }
  }
}
"""
    (tmp_path / "input_with_addresses.mlir").write_text(sample_mlir)
    sizes = parse_dma_sizes(tmp_path)
    assert sizes == [1024], f"Expected [1024] (union), got {sizes}"


def test_parse_dma_sizes_picks_uncalled_root_when_helper_present(tmp_path):
    """If a module declares a main runtime_sequence + a helper invoked via
    aiex.run, the parser must pick the main (call-graph root) regardless of
    declaration order — the helper's args must NOT be reported.
    """

    # Layout matches test/aiecc/cpp_expand_load_pdis.mlir: @main's sequence
    # @sequence configures @helper-device and runs its sequence @helper_seq.
    # The helper device is declared FIRST so any "pick first sequence in
    # source order" heuristic would pick the wrong one; the call-graph-root
    # algorithm must pick @sequence.
    sample_mlir = """\
module {
  aie.device(npu2) @helper {
    aie.runtime_sequence @helper_seq(%h0: memref<2048xi32>) {
    }
  }
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%a: memref<1024xi32>, %b: memref<1024xi32>) {
      aiex.configure @helper {
        aiex.run @helper_seq(%a) : (memref<1024xi32>)
      }
    }
  }
}
"""
    (tmp_path / "input_with_addresses.mlir").write_text(sample_mlir)
    sizes = parse_dma_sizes(tmp_path)
    assert sizes == [1024, 1024], f"Expected main's args [1024, 1024], got {sizes}"


def test_parse_dma_sizes_returns_none_when_multi_device_has_multiple_roots(tmp_path):
    """Multi-device modules with more than one top-level runtime_sequence
    can't be unambiguously matched to a flat host tensor list — bail rather
    than validate against the wrong signature.
    """

    sample_mlir = """\
module {
  aie.device(npu1) {
    aie.runtime_sequence @dev_a_main(%x: memref<1024xi32>) {
    }
  }
  aie.device(npu2) {
    aie.runtime_sequence @dev_b_main(%y: memref<2048xi32>) {
    }
  }
}
"""
    (tmp_path / "input_with_addresses.mlir").write_text(sample_mlir)
    assert parse_dma_sizes(tmp_path) is None


def test_parse_dma_sizes_returns_none_for_dynamic_shape_arg(tmp_path):
    """A dynamic-dim memref arg means the kernel's host contract isn't a
    fixed element count — skip validation rather than guess."""

    sample_mlir = """\
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<?xi32>) {
    }
  }
}
"""
    (tmp_path / "input_with_addresses.mlir").write_text(sample_mlir)
    assert parse_dma_sizes(tmp_path) is None


def test_parse_dma_sizes_returns_none_for_unparseable_text(tmp_path):
    """Garbage in the file must come back as None, not raise."""

    (tmp_path / "input_with_addresses.mlir").write_text("not actually MLIR\n")
    assert parse_dma_sizes(tmp_path) is None


def test_parse_dma_sizes_returns_none_when_file_missing(tmp_path):
    """Absent input_with_addresses.mlir must return None, not raise."""

    assert parse_dma_sizes(tmp_path) is None


def test_compute_hash_changes_when_active_device_changes_arch():
    """Cross-compile correctness: switching the iron-active device to a
    different-arch NPU must change the design's cache-key hash, so a
    Strix-cross-compile of the same generator doesn't silently collide
    with a Phoenix compile in the per-design cache directory.

    Regression for the case where _compute_hash used DefaultNPURuntime
    (XRT-detected hardware, fixed) instead of iron.get_current_device()
    (override-aware) — designs with no per-arch source files (e.g. an
    inline passthrough) hit cache collision until the hash started
    tracking the iron-active device.
    """
    import aie.iron as iron

    gen = _gemm_gen()
    cd = CompilableDesign(gen, compile_kwargs={"M": 64, "K": 64, "N": 64})

    set_current_device(NPU1Col1())
    h_phx = cd._compute_cache_hash()

    set_current_device(NPU2Col1())
    h_strix = cd._compute_cache_hash()

    # Restore Phoenix as the active device for any later test.
    set_current_device(NPU1Col1())

    assert (
        h_phx != h_strix
    ), f"Phoenix and Strix cache hashes must differ; both were {h_phx}"


def test_kernels_mm_mac_dims_per_arch():
    """kernels.mm exposes per-arch MMUL geometry via .mac_dims so designs
    can drive their DMA layout transforms from the kernel itself instead
    of hardcoding for one NPU generation.

    Regression for the matmul example that used to hardcode (4, 4, 4)
    and silently produce garbage on AIE2P, which uses (4, 4, 8) for the
    same i16/i16 dtype combo.
    """
    import numpy as np
    import aie.iron.kernels as kernels

    set_current_device(NPU1Col1())
    mm_aie2 = kernels.mm(
        dim_m=64, dim_k=64, dim_n=64, input_dtype=np.int16, output_dtype=np.int16
    )

    set_current_device(NPU2Col1())
    mm_aie2p = kernels.mm(
        dim_m=64, dim_k=64, dim_n=64, input_dtype=np.int16, output_dtype=np.int16
    )

    set_current_device(NPU1Col1())

    assert mm_aie2.mac_dims == (
        4,
        4,
        4,
    ), f"AIE2 i16/i16 mac_dims expected (4, 4, 4), got {mm_aie2.mac_dims}"
    assert mm_aie2p.mac_dims == (
        4,
        4,
        8,
    ), f"AIE2P i16/i16 mac_dims expected (4, 4, 8), got {mm_aie2p.mac_dims}"


def test_compile_mixed_explicit_paths_raises():
    """Passing only one of (xclbin_path, inst_path) is rejected up front."""

    def gen():
        pass

    cd = CompilableDesign(gen)
    with pytest.raises(ValueError, match="must be set together"):
        cd.compile(xclbin_path="/tmp/foo.xclbin", inst_path=None)
    with pytest.raises(ValueError, match="must be set together"):
        cd.compile(xclbin_path=None, inst_path="/tmp/foo.bin")


# ---------------------------------------------------------------------------
# specialize(): config overrides + CompileTime[T] kwargs
# ---------------------------------------------------------------------------


def test_config_param_names_matches_construction():
    """config_param_names() covers every ctor param except generator+kwargs."""
    from aie.utils.compile.jit.compilabledesign import config_param_names

    assert config_param_names(CompilableDesign) == {
        "use_cache",
        "compile_flags",
        "source_files",
        "include_paths",
        "aiecc_flags",
        "object_files",
        "full_elf",
    }


def test_specialize_binds_compile_kwargs():
    """Non-config overrides become CompileTime[T] kwargs."""
    d = CompilableDesign(_gemm_gen())
    s = d.specialize(M=512, K=256, N=128)
    assert s.compile_kwargs == {"M": 512, "K": 256, "N": 128}


def test_specialize_overrides_config():
    """A config-named override replaces that config on the new design."""
    d = CompilableDesign(_gemm_gen())
    assert d.full_elf is False
    s = d.specialize(full_elf=True)
    assert s.full_elf is True
    # original is unchanged (specialize returns a new design)
    assert d.full_elf is False


def test_specialize_preserves_other_config():
    """Config not mentioned in the override is carried over from self."""
    d = CompilableDesign(
        _gemm_gen(), aiecc_flags=["--dynamic-objFifos"], use_cache=False
    )
    s = d.specialize(full_elf=True, M=512)
    assert s.full_elf is True
    assert s.aiecc_flags == ("--dynamic-objFifos",)
    assert s.use_cache is False
    assert s.compile_kwargs == {"M": 512}


def test_specialize_mixes_config_and_compile_kwargs():
    """A single call can override config and bind CompileTime[T] together."""
    d = CompilableDesign(_gemm_gen(), compile_kwargs={"M": 1})
    s = d.specialize(full_elf=True, M=512, K=256)
    assert s.full_elf is True
    # call-time compile kwargs win over pre-bound
    assert s.compile_kwargs == {"M": 512, "K": 256}


def test_specialize_config_lists_are_independent_copies():
    """Mutating the child's list config must not affect the parent."""
    d = CompilableDesign(_gemm_gen(), source_files=["a.cc"])
    s = d.specialize(full_elf=True)
    assert [p.name for p in s.source_files] == ["a.cc"]
    # tuples are immutable; the point is the child got its own copy, not a
    # shared alias -- equal by value, distinct config surface.
    assert s.source_files == d.source_files


def test_compile_pdi_path_requires_explicit_paths():
    """pdi_path without explicit xclbin_path + inst_path is rejected up front."""

    def gen():
        pass

    cd = CompilableDesign(gen)
    with pytest.raises(ValueError, match="pdi_path requires explicit"):
        cd.compile(pdi_path="/tmp/foo.pdi")


def test_compile_elf_path_requires_explicit_paths():
    """elf_path without explicit xclbin_path + inst_path is rejected up front."""

    def gen():
        pass

    cd = CompilableDesign(gen)
    with pytest.raises(ValueError, match="elf_path requires explicit"):
        cd.compile(elf_path="/tmp/foo.elf")


def test_get_pdi_path_none_before_compile():
    """get_pdi_path() returns None when no compile has happened yet."""

    def gen():
        pass

    cd = CompilableDesign(gen)
    assert cd.get_pdi_path() is None


def test_get_pdi_paths_empty_before_compile():
    """get_pdi_paths() returns [] when no compile has happened yet."""

    def gen():
        pass

    cd = CompilableDesign(gen)
    assert cd.get_pdi_paths() == []


# ---------------------------------------------------------------------------
# compile(): DispatchTime[T] guards (raise before any aiecc/dispatch-bridge
# subprocess runs, so no compiler toolchain is needed to exercise these)
# ---------------------------------------------------------------------------


def test_compile_dispatch_time_rejects_full_elf():
    """DispatchTime[T] + full_elf=True raises before any compilation is attempted."""
    d = CompilableDesign(_dispatch_gen(), compile_kwargs={"N": 512}, full_elf=True)
    with pytest.raises(NotImplementedError, match="full_elf"):
        d.compile()


def test_compile_dispatch_time_rejects_full_elf_path_kwarg():
    """Same guard via the full_elf_path= call-time kwarg, not just the config."""
    d = CompilableDesign(_dispatch_gen(), compile_kwargs={"N": 512})
    with pytest.raises(NotImplementedError, match="full_elf"):
        d.compile(full_elf_path="/tmp/foo.elf")


def test_compile_dispatch_time_rejects_explicit_paths():
    """DispatchTime[T] + explicit xclbin_path/inst_path raises up front."""
    d = CompilableDesign(_dispatch_gen(), compile_kwargs={"N": 512})
    with pytest.raises(NotImplementedError, match="explicit-path"):
        d.compile(xclbin_path="/tmp/foo.xclbin", inst_path="/tmp/foo.bin")


def test_get_dispatch_lib_path_none_before_compile():
    """get_dispatch_lib_path() returns None when no compile has happened yet."""
    d = CompilableDesign(_dispatch_gen(), compile_kwargs={"N": 512})
    assert d.get_dispatch_lib_path() is None


def test_get_dispatch_lib_path_none_for_non_dispatch_design():
    """get_dispatch_lib_path() returns None for a design with no DispatchTime[T] params."""
    d = CompilableDesign(_gemm_gen())
    assert d.get_dispatch_lib_path() is None
