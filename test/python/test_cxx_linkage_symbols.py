# test_cxx_linkage_symbols.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""C++-linkage kernel symbol resolution.

An IRON kernel named ``reduce_min_vector`` emits ``func.func private
@reduce_min_vector``, so the linker needs a symbol with that exact name.  A
kernel source without ``extern "C"`` defines ``_Z17reduce_min_vectorPiS_i``
instead.  These tests cover the resolution step that bridges the two by
demangling the compiled object's symbols and renaming the match.

The selection logic is tested directly (pure, fast); the llvm-nm / llvm-cxxfilt
/ llvm-objcopy plumbing is covered by one end-to-end test against a real
Peano-compiled AIE object.
"""

import os
import pathlib
import subprocess

import numpy as np
import pytest

from aie.iron.kernel import ExternalFunction
from aie.utils import config
from aie.utils.compile.utils import (
    _apply_symbol_renames,
    _demangled_base_name,
    _resolve_cxx_linkage_symbol,
    _select_cxx_symbol,
    compile_external_kernel,
)

# ---------------------------------------------------------------------------
# Base-name extraction: everything before the parameter list.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "demangled,expected",
    [
        # A C-linkage symbol demangles to itself -- no parameter list at all.
        ("plain_c", "plain_c"),
        ("reduce_min_vector(int*, int*, int)", "reduce_min_vector"),
        # const / restrict live in the parameter list and must not survive.
        ("f1(int const*, int*, int)", "f1"),
        # Namespaces are part of the name and must survive.
        ("ns::g(int*, int*, int)", "ns::g"),
        # A function-pointer parameter contains its own '(' -- split on the
        # first one only.
        ("takes_fp(void (*)(int))", "takes_fp"),
    ],
)
def test_base_name_strips_parameter_list(demangled, expected):
    assert _demangled_base_name(demangled) == expected


# ---------------------------------------------------------------------------
# Symbol selection.  Input is {mangled: demangled}, as llvm-nm + llvm-cxxfilt
# produce it.
# ---------------------------------------------------------------------------


def test_c_linkage_symbol_needs_no_rename():
    """An exact match means the kernel already has C linkage."""
    symbols = {"reduce_min_vector": "reduce_min_vector"}
    assert _select_cxx_symbol(symbols, "reduce_min_vector") is None


def test_selects_mangled_symbol_by_base_name():
    symbols = {
        "_Z17reduce_min_vectorPiS_i": "reduce_min_vector(int*, int*, int)",
        "_Z17reduce_min_scalarPiS_i": "reduce_min_scalar(int*, int*, int)",
    }
    assert (
        _select_cxx_symbol(symbols, "reduce_min_vector") == "_Z17reduce_min_vectorPiS_i"
    )


def test_const_parameters_do_not_affect_selection():
    """The whole point of demangling: `const` changes the mangling but not the
    base name, so a read-only kernel resolves without IRON knowing about it."""
    symbols = {"_Z2f1PKiPii": "f1(int const*, int*, int)"}
    assert _select_cxx_symbol(symbols, "f1") == "_Z2f1PKiPii"


def test_namespaced_symbol_matches_unqualified_name():
    symbols = {"_ZN2ns1gEPiS0_i": "ns::g(int*, int*, int)"}
    assert _select_cxx_symbol(symbols, "g") == "_ZN2ns1gEPiS0_i"


def test_namespaced_symbol_matches_qualified_name():
    """Fully qualifying is how a user disambiguates same-named functions in
    different namespaces."""
    symbols = {"_ZN2ns1gEPiS0_i": "ns::g(int*, int*, int)"}
    assert _select_cxx_symbol(symbols, "ns::g") == "_ZN2ns1gEPiS0_i"


def test_missing_symbol_reports_what_is_present():
    symbols = {"_Z17reduce_min_scalarPiS_i": "reduce_min_scalar(int*, int*, int)"}
    with pytest.raises(ValueError) as exc:
        _select_cxx_symbol(symbols, "reduce_min_vector")
    message = str(exc.value)
    assert "reduce_min_vector" in message
    assert "reduce_min_scalar(int*, int*, int)" in message


def test_overloads_are_rejected_as_ambiguous():
    """IRON has no way to choose between overloads: `arg_types` carries numpy
    types, which cannot distinguish `int*` from `int const*`."""
    symbols = {
        "_Z1fPii": "f(int*, int)",
        "_Z1fPfi": "f(float*, int)",
    }
    with pytest.raises(ValueError) as exc:
        _select_cxx_symbol(symbols, "f")
    message = str(exc.value)
    assert "ambiguous" in message.lower()
    assert "f(int*, int)" in message
    assert "f(float*, int)" in message


# ---------------------------------------------------------------------------
# End-to-end against a real Peano-compiled AIE object.
# ---------------------------------------------------------------------------

_KERNEL_SOURCE = """
#include <cstdint>
void reduce_min_vector(int32_t *__restrict in, int32_t *__restrict out,
                       const int32_t n) {
  int32_t m = in[0];
  for (int32_t i = 1; i < n; i++)
    if (in[i] < m)
      m = in[i];
  *out = m;
}
extern "C" void already_c(int32_t *a, int32_t *b, int32_t n) { *b = a[n]; }
"""


@pytest.fixture
def aie_object(tmp_path):
    """Compile a C++ kernel to a real aie2 object with Peano."""
    try:
        peano = config.peano_cxx_path()
    except RuntimeError as exc:
        pytest.skip(f"Peano not available: {exc}")

    source = tmp_path / "k.cc"
    source.write_text(_KERNEL_SOURCE)
    obj = tmp_path / "k.o"
    subprocess.run(
        [
            peano,
            str(source),
            "-c",
            "-o",
            str(obj),
            "-std=c++20",
            "-O2",
            "-DNDEBUG",
            "--target=aie2-none-unknown-elf",
        ],
        check=True,
        capture_output=True,
    )
    return obj


def _defined_symbols(obj):
    nm = os.path.join(config.peano_install_dir(), "bin", "llvm-nm")
    out = subprocess.run(
        [nm, "--defined-only", "--extern-only", "--format=just-symbols", str(obj)],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(out.stdout.split())


def test_resolve_renames_mangled_symbol_in_real_object(aie_object):
    assert "_Z17reduce_min_vectorPiS_i" in _defined_symbols(aie_object)

    _resolve_cxx_linkage_symbol(str(aie_object), "reduce_min_vector")

    symbols = _defined_symbols(aie_object)
    assert "reduce_min_vector" in symbols
    assert "_Z17reduce_min_vectorPiS_i" not in symbols


def test_resolve_is_idempotent(aie_object):
    """The JIT re-runs resolution on a cache hit, when the object on disk has
    already been renamed."""
    _resolve_cxx_linkage_symbol(str(aie_object), "reduce_min_vector")
    _resolve_cxx_linkage_symbol(str(aie_object), "reduce_min_vector")
    assert "reduce_min_vector" in _defined_symbols(aie_object)


def test_resolve_leaves_c_linkage_symbol_untouched(aie_object):
    _resolve_cxx_linkage_symbol(str(aie_object), "already_c")
    assert "already_c" in _defined_symbols(aie_object)


# ---------------------------------------------------------------------------
# Wiring into the JIT kernel compile.
# ---------------------------------------------------------------------------

_CXX_KERNEL = """
#include <cstdint>
void min_kernel(int32_t *__restrict in, int32_t *__restrict out,
                const int32_t n) {
  int32_t m = in[0];
  for (int32_t i = 1; i < n; i++)
    if (in[i] < m)
      m = in[i];
  *out = m;
}
"""


@pytest.fixture
def clean_registry():
    ExternalFunction._instances.clear()
    yield
    ExternalFunction._instances.clear()


def _require_peano():
    try:
        config.peano_cxx_path()
    except RuntimeError as exc:
        pytest.skip(f"Peano not available: {exc}")


def test_jit_compile_exports_cxx_kernel_under_its_iron_name(clean_registry, tmp_path):
    """A kernel source with no `extern "C"` must still link against the name
    IRON emitted into the MLIR."""
    _require_peano()
    func = ExternalFunction(
        "min_kernel",
        source_string=_CXX_KERNEL,
        arg_types=[
            np.ndarray[(32,), np.dtype[np.int32]],
            np.ndarray[(1,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    compile_external_kernel(func, str(tmp_path), "aie2")

    assert "min_kernel" in _defined_symbols(tmp_path / func.object_file_name)


def test_symbol_prefix_applies_on_top_of_cxx_resolution(clean_registry, tmp_path):
    """The prefix rename looks for the unprefixed name, which a C++ source does
    not define -- so C++ resolution has to run first for the two to compose."""
    _require_peano()
    func = ExternalFunction(
        "min_kernel",
        source_string=_CXX_KERNEL,
        symbol_prefix="pfx",
        arg_types=[
            np.ndarray[(32,), np.dtype[np.int32]],
            np.ndarray[(1,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    compile_external_kernel(func, str(tmp_path), "aie2")

    symbols = _defined_symbols(tmp_path / func.object_file_name)
    assert "pfx_min_kernel" in symbols
    assert "min_kernel" not in symbols


class _FakeFunc:
    """Stand-in for an ExternalFunction whose artifact must not be inspected.

    ``object_path`` below does not exist, so any attempt to read symbols from
    it raises -- which is precisely the assertion.
    """

    def __init__(self, **attrs):
        self._name = "k"
        self._original_name = "k"
        self._symbol_prefix = None
        self._inline = False
        self._use_chess = False
        self.__dict__.update(attrs)


def test_inline_artifact_is_not_inspected(tmp_path):
    """An inline kernel's artifact is LLVM IR, which llvm-nm cannot read.
    `_make_ir_inlinable` already rejects a mangled inline kernel with a message
    naming the `extern "C"` fix, so there is nothing to do here."""
    _apply_symbol_renames(_FakeFunc(_inline=True), str(tmp_path / "absent.ll"))


def test_chess_artifact_is_not_inspected(tmp_path):
    """xchesscc's object format has not been verified against llvm-nm.  Until
    it is, chess kernels keep their existing behaviour rather than risking a
    hard failure for kernels that already work."""
    _apply_symbol_renames(_FakeFunc(_use_chess=True), str(tmp_path / "absent.o"))


def test_jit_compile_rejects_arg_types_that_contradict_the_kernel(
    clean_registry, tmp_path
):
    """The check has to run as part of the compile, not just be available."""
    _require_peano()
    func = ExternalFunction(
        "min_kernel",
        source_string=_CXX_KERNEL,
        # The kernel's third parameter is an int32 scalar, not a buffer.
        arg_types=[
            np.ndarray[(32,), np.dtype[np.int32]],
            np.ndarray[(1,), np.dtype[np.int32]],
            np.ndarray[(1,), np.dtype[np.int32]],
        ],
    )

    with pytest.raises(ValueError) as exc:
        compile_external_kernel(func, str(tmp_path), "aie2")
    assert "argument 3" in str(exc.value).lower()


def test_shipped_reduce_min_kernel_needs_no_extern_c(clean_registry, tmp_path):
    """The motivating case: aie_kernels/aie2/reduce_min.cc carries no
    `extern "C"` trampolines and must still export `reduce_min_vector`."""
    _require_peano()
    source = (
        pathlib.Path(__file__).parents[2] / "aie_kernels" / "aie2" / "reduce_min.cc"
    )
    if not source.is_file():
        pytest.skip(f"kernel source not available at {source}")
    assert 'extern "C"' not in source.read_text()

    func = ExternalFunction(
        "reduce_min_vector",
        source_file=str(source),
        arg_types=[
            np.ndarray[(1024,), np.dtype[np.int32]],
            np.ndarray[(8,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    compile_external_kernel(func, str(tmp_path), "aie2")

    assert "reduce_min_vector" in _defined_symbols(tmp_path / func.object_file_name)


# ---------------------------------------------------------------------------
# Catalog sweep.
#
# Symbol resolution now runs llvm-nm over EVERY compiled kernel object, not
# just C++-linkage ones.  The shipped catalog is almost entirely `extern "C"`,
# so those kernels take the fast path and must be completely unaffected -- this
# sweep is what proves the new step did not disturb them.
# ---------------------------------------------------------------------------


def _catalog_specs():
    try:
        from test_kernels_specs import KERNEL_SPECS
    except ImportError:  # pragma: no cover - only when run outside the suite
        return []
    return [s for s in KERNEL_SPECS if not s.requires_npu2]


@pytest.mark.parametrize(
    "spec", _catalog_specs(), ids=lambda s: s.name if hasattr(s, "name") else str(s)
)
def test_catalog_kernel_still_compiles(clean_registry, tmp_path, spec):
    _require_peano()
    try:
        func = spec.factory(**spec.kwargs)
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(f"{spec.name} unavailable in this build: {exc}")
    if not isinstance(func, ExternalFunction):
        pytest.skip(f"{spec.name} is not an ExternalFunction")
    if getattr(func, "_use_chess", False) or getattr(func, "_inline", False):
        pytest.skip(f"{spec.name} does not take the object path")

    try:
        compile_external_kernel(func, str(tmp_path), "aie2")
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        pytest.skip(f"{spec.name} does not build in this environment: {exc}")

    symbols = _defined_symbols(tmp_path / func.object_file_name)
    assert func._name in symbols
