# test_jit_utils.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings

# Unit tests for compile_external_kernel and cache key utilities.

import os
import tempfile
import pytest
import numpy as np

import aie.iron as iron
from aie.iron import ExternalFunction, ObjectFifo, Worker, Runtime, Program
from aie.iron import CompileTime, In, Out
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, NPU2Col1
from aie.utils.compile.utils import compile_external_kernel
from aie.utils.compile.cache.utils import _create_function_cache_key

# ---------------------------------------------------------------------------
# Session-scoped helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def npu_target_arch():
    """Return the target architecture string for the current NPU."""
    device = iron.get_current_device()
    if isinstance(device, (NPU2, NPU2Col1)):
        return "aie2p"
    return "aie2"


@pytest.fixture(autouse=True)
def _clear_external_function_instances():
    """Prevent ExternalFunction instances from leaking between tests."""
    ExternalFunction._instances.clear()
    yield
    ExternalFunction._instances.clear()


# ---------------------------------------------------------------------------
# compile_external_kernel
#
# Regression: the original implementation silently returned without compiling
# or raising when source_file did not exist, leaving the caller to encounter
# a confusing downstream linker error.
# ---------------------------------------------------------------------------


def test_compile_external_kernel_missing_source_file_raises(npu_target_arch):
    """FileNotFoundError must be raised when source_file does not exist.

    ExternalFunction construction succeeds (source_file is stored but not read
    until hash time). The FileNotFoundError is raised inside
    compile_external_kernel when it checks that the file exists before copying.
    """
    with tempfile.TemporaryDirectory() as kernel_dir:
        with pytest.raises(FileNotFoundError):
            func = ExternalFunction("my_kernel", source_file="/nonexistent/kernel.cc")
            compile_external_kernel(func, kernel_dir, target_arch=npu_target_arch)


def test_compile_external_kernel_source_string(npu_target_arch):
    """source_string must be compiled to an object file."""
    func = ExternalFunction(
        "add_one",
        source_string="""extern "C" {
            void add_one(int* a, int* b, int n) {
                for (int i = 0; i < n; i++) b[i] = a[i] + 1;
            }
        }""",
    )
    with tempfile.TemporaryDirectory() as kernel_dir:
        compile_external_kernel(func, kernel_dir, target_arch=npu_target_arch)
        obj = os.path.join(kernel_dir, "add_one.o")
        assert os.path.exists(obj)
        assert os.path.getsize(obj) > 0


def test_compile_external_kernel_source_file(npu_target_arch):
    """source_file must be copied into kernel_dir and compiled to an object file."""
    with (
        tempfile.TemporaryDirectory() as src_dir,
        tempfile.TemporaryDirectory() as kernel_dir,
    ):
        src = os.path.join(src_dir, "my_kernel.cc")
        with open(src, "w") as f:
            f.write("""extern "C" {
                void my_kernel(int* a, int* b, int n) {
                    for (int i = 0; i < n; i++) b[i] = a[i] + 1;
                }
            }""")

        func = ExternalFunction("my_kernel", source_file=src)
        compile_external_kernel(func, kernel_dir, target_arch=npu_target_arch)

        assert os.path.exists(os.path.join(kernel_dir, "my_kernel.cc"))
        obj = os.path.join(kernel_dir, "my_kernel.o")
        assert os.path.exists(obj)
        assert os.path.getsize(obj) > 0


def test_compile_external_kernel_source_file_uses_true_basename(npu_target_arch):
    """The copied/embedded source name must be the real file's basename, not
    the entry point name, so it stays stable regardless of which entry point
    of a multi-entry-point source file happens to compile first."""
    with (
        tempfile.TemporaryDirectory() as src_dir,
        tempfile.TemporaryDirectory() as kernel_dir,
    ):
        src = os.path.join(src_dir, "conv.cc")
        with open(src, "w") as f:
            f.write("""extern "C" {
                void conv_step(int* a, int* b, int n) {
                    for (int i = 0; i < n; i++) b[i] = a[i] + 1;
                }
            }""")

        func = ExternalFunction("conv_step", source_file=src)
        compile_external_kernel(func, kernel_dir, target_arch=npu_target_arch)

        assert os.path.exists(os.path.join(kernel_dir, "conv.cc"))
        assert not os.path.exists(os.path.join(kernel_dir, "conv_step.cc"))


def test_compile_external_kernel_shared_object_deterministic(npu_target_arch):
    """Several ExternalFunctions with distinct entry points but a shared
    ``object_file_name`` must produce a byte-identical object regardless of
    which entry point is compiled first.

    Regression: only the first-visited entry point actually compiles (the
    rest hit the output-file-exists cache check), so its (previously
    entry-point-derived) source naming used to leak into the shared object.
    Since ``ExternalFunction._instances`` is a content-hashed set, the
    "first-visited" entry point is perturbed by edits to unrelated kernels,
    making the embedded name (and thus the object's bytes) nondeterministic.
    """
    with tempfile.TemporaryDirectory() as src_dir:
        src = os.path.join(src_dir, "conv.cc")
        with open(src, "w") as f:
            f.write("""extern "C" {
                void conv_begin(int* a, int* b, int n) {}
                void conv_step(int* a, int* b, int n) {}
            }""")

        def build(order):
            ExternalFunction._instances.clear()
            funcs = {
                "conv_begin": ExternalFunction(
                    "conv_begin", object_file_name="conv.o", source_file=src
                ),
                "conv_step": ExternalFunction(
                    "conv_step", object_file_name="conv.o", source_file=src
                ),
            }
            with tempfile.TemporaryDirectory() as kernel_dir:
                for name in order:
                    compile_external_kernel(
                        funcs[name], kernel_dir, target_arch=npu_target_arch
                    )
                with open(os.path.join(kernel_dir, "conv.o"), "rb") as f:
                    return f.read()

        first = build(["conv_begin", "conv_step"])
        second = build(["conv_step", "conv_begin"])
        assert first == second


def test_compile_external_kernel_same_source_multiple_entries_no_collision(
    npu_target_arch,
):
    """Two ExternalFunctions built from the SAME source file (different entry
    points, different object_file_name) must compile cleanly -- this is the
    normal "one .cc, several kernels" pattern (e.g. kernels.mm's matmul +
    zero symbols)."""
    with (
        tempfile.TemporaryDirectory() as src_dir,
        tempfile.TemporaryDirectory() as kernel_dir,
    ):
        src = os.path.join(src_dir, "mm.cc")
        with open(src, "w") as f:
            f.write("""extern "C" {
                void zero_kernel(int* a) {}
                void matmul_kernel(int* a) {}
            }""")

        zero = ExternalFunction("zero_kernel", source_file=src)
        matmul = ExternalFunction("matmul_kernel", source_file=src)
        compile_external_kernel(zero, kernel_dir, target_arch=npu_target_arch)
        compile_external_kernel(matmul, kernel_dir, target_arch=npu_target_arch)

        assert os.path.exists(os.path.join(kernel_dir, "mm.cc"))
        assert os.path.exists(os.path.join(kernel_dir, "zero_kernel.o"))
        assert os.path.exists(os.path.join(kernel_dir, "matmul_kernel.o"))


def test_compile_external_kernel_source_file_already_in_kernel_dir(npu_target_arch):
    """kernel_dir == the source file's own directory must not crash.

    Naming the copy after the true basename means source_file and the
    computed destination can be the literal same path (e.g. a caller that
    compiles a design in place). shutil.copy2 raises SameFileError when src
    and dst are identical, so compile_external_kernel must skip the copy
    rather than attempt it unconditionally.
    """
    with tempfile.TemporaryDirectory() as kernel_dir:
        src = os.path.join(kernel_dir, "in_place.cc")
        with open(src, "w") as f:
            f.write('extern "C" void in_place(int* a) {}')

        func = ExternalFunction("in_place", source_file=src)
        compile_external_kernel(func, kernel_dir, target_arch=npu_target_arch)

        obj = os.path.join(kernel_dir, "in_place.o")
        assert os.path.exists(obj)
        assert os.path.getsize(obj) > 0


def test_compile_external_kernel_marks_compiled(npu_target_arch):
    """compile_external_kernel must set func._compiled = True on success."""
    func = ExternalFunction(
        "add_one",
        source_string='extern "C" void add_one(int* a, int* b, int n) {}',
    )
    with tempfile.TemporaryDirectory() as kernel_dir:
        assert not func._compiled
        compile_external_kernel(func, kernel_dir, target_arch=npu_target_arch)
        assert func._compiled


def test_compile_external_kernel_skip_if_already_compiled(npu_target_arch):
    """compile_external_kernel must be a no-op when func._compiled is already True."""
    func = ExternalFunction(
        "add_one",
        source_string='extern "C" void add_one() {}',
    )
    func._compiled = True
    with tempfile.TemporaryDirectory() as kernel_dir:
        compile_external_kernel(func, kernel_dir, target_arch=npu_target_arch)
        assert not os.path.exists(os.path.join(kernel_dir, "add_one.o"))


def test_compile_external_kernel_skip_if_object_file_exists(npu_target_arch):
    """compile_external_kernel must be a no-op when the output object file already exists."""
    func = ExternalFunction(
        "add_one",
        source_string='extern "C" void add_one() {}',
    )
    with tempfile.TemporaryDirectory() as kernel_dir:
        obj = os.path.join(kernel_dir, func.object_file_name)
        with open(obj, "wb") as f:
            f.write(b"placeholder")
        compile_external_kernel(func, kernel_dir, target_arch=npu_target_arch)
        with open(obj, "rb") as f:
            assert f.read() == b"placeholder"


# ---------------------------------------------------------------------------
# _create_function_cache_key: closure key collision fix
#
# Regression: closures differing only in captured value previously produced
# identical keys because co_code/co_consts/co_names do not change when a
# free variable changes.
# ---------------------------------------------------------------------------


def test_closure_cache_key_distinguishes_captured_values():
    """_create_function_cache_key must produce different keys for closures
    that capture different values."""

    def make(v):
        return lambda a: a + v

    f1, f2 = make(1), make(2)
    dummy_fn = lambda: None
    key1 = _create_function_cache_key(dummy_fn, [f1], {})
    key2 = _create_function_cache_key(dummy_fn, [f2], {})
    assert key1 != key2


def test_closure_cache_key_mutable_object_no_repr():
    """Cache key must change when a mutable object's attributes change,
    even when it has no __eq__, __hash__, or __repr__ override."""

    class Config:
        def __init__(self, val):
            self.val = val

        # deliberately no __repr__, __eq__, or __hash__

    def make_fn(c):
        # cfg must be captured as a closure cell, not a global
        return lambda a: a + c.val

    cfg = Config(1)
    fn = make_fn(cfg)
    dummy_fn = lambda: None
    key1 = _create_function_cache_key(dummy_fn, [fn], {})
    cfg.val = 2  # mutate in-place — only deep state has changed
    key2 = _create_function_cache_key(dummy_fn, [fn], {})
    assert key1 != key2


def test_closure_cache_key_list_mutation():
    """Cache key must change when a list captured by a closure is mutated."""

    def make_fn(items):
        # items must be captured as a closure cell, not a global
        return lambda a: a + items[0]

    items = [1, 2, 3]
    fn = make_fn(items)
    dummy_fn = lambda: None
    key1 = _create_function_cache_key(dummy_fn, [fn], {})
    items[0] = 99
    key2 = _create_function_cache_key(dummy_fn, [fn], {})
    assert key1 != key2


def test_closure_cache_key_is_stable_without_mutation():
    """Cache key must be identical across repeated calls when nothing changes."""

    class Config:
        def __init__(self, val):
            self.val = val

    def make_fn(c):
        return lambda a: a + c.val

    cfg = Config(42)
    fn = make_fn(cfg)
    dummy_fn = lambda: None
    key1 = _create_function_cache_key(dummy_fn, [fn], {})
    key2 = _create_function_cache_key(dummy_fn, [fn], {})
    assert key1 == key2


def test_closure_cache_key_no_closure():
    """A callable with no closure must produce a stable key."""
    fn = lambda a: a + 1  # no captured variables
    dummy_fn = lambda: None
    key1 = _create_function_cache_key(dummy_fn, [fn], {})
    key2 = _create_function_cache_key(dummy_fn, [fn], {})
    assert key1 == key2


# ---------------------------------------------------------------------------
# End-to-end JIT closure test
# ---------------------------------------------------------------------------

_NUM_ELEMS = 1024
_TILE_SIZE = 16
_tile_ty = np.ndarray[(_TILE_SIZE,), np.dtype[np.int32]]
_tensor_ty = np.ndarray[(_NUM_ELEMS,), np.dtype[np.int32]]


@iron.jit
def _transform(input_tensor: In, output_tensor: Out, *, kernel_fn: CompileTime[object]):
    """JIT-compiled element-wise transform using a caller-supplied lambda.

    ``kernel_fn`` is a compile-time callable — changing it produces a new cache
    entry and recompiles. Types are constructed from compile-time constants
    rather than from the runtime tensors (which are not available at generation
    time). This is why iron.algorithms.transform cannot be used directly here;
    that function requires real tensors to infer shape/dtype.
    """
    of_in = ObjectFifo(_tile_ty, name="in")
    of_out = ObjectFifo(_tile_ty, name="out")

    def core_body(of_in, of_out):
        for _ in range_(_NUM_ELEMS // _TILE_SIZE):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(_TILE_SIZE):
                elem_out[i] = kernel_fn(elem_in[i])
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod()])

    def sequence(A, B, in_h, out_h):
        in_h.fill(A)
        out_h.drain(B, wait=True)

    rt = Runtime(
        sequence,
        [_tensor_ty, _tensor_ty, of_in.prod(), of_out.cons()],
    )
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


@pytest.mark.parametrize("add_value", [1, 2, 3])
def test_jit_closure_parametrize(add_value):
    """@jit must produce correct output for each distinct closure value.

    Before the fix, all three parametrize cases shared the same in-memory
    cache key (captured value was ignored), so only the first value ever
    executed correctly.
    """
    input_tensor = iron.arange(_NUM_ELEMS, dtype=np.int32)
    output_tensor = iron.zeros(_NUM_ELEMS, dtype=np.int32, device="npu")
    _transform(input_tensor, output_tensor, kernel_fn=lambda x: x + add_value)
    np.testing.assert_array_equal(
        output_tensor.numpy(), input_tensor.numpy() + add_value
    )
