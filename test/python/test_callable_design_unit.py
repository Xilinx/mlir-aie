# test_callable_design_unit.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# RUN: %pytest %s
"""Unit tests for CallableDesign and @jit pure-logic surfaces — no NPU required.

Tests that exercise compile() or actual NPU kernel execution live in
test/python/npu-xrt/test_iron_jit_e2e.py (requires xrt_python_bindings).
"""

from pathlib import Path

import pytest

from unittest.mock import MagicMock, patch

from aie.utils.compile.jit.compilabledesign import CompilableDesign
from aie.utils.compile.jit.markers import Compile, In, InOut, Out
from aie.utils.callabledesign import CallableDesign
from aie.utils.jit import _JIT_CONFIG_KEYS, jit
from aie.iron.kernel import ExternalFunction, Kernel

# ---------------------------------------------------------------------------
# CallableDesign construction
# ---------------------------------------------------------------------------
#
# Forwarding tests (CallableDesign correctly delegates to its inner
# CompilableDesign) are covered by the @jit decorator block below, which
# exercises the same paths via a more realistic surface.  Construction-
# defaults and split_runtime_args semantics are pinned in
# test_compilabledesign.py.


def test_repr_contains_callable_design():
    def gen(a: In, *, M: Compile[int]):
        pass

    cd = CallableDesign(gen, compile_kwargs={"M": 1})
    assert "CallableDesign" in repr(cd)


# ---------------------------------------------------------------------------
# @jit decorator — construction-time behaviour only
# ---------------------------------------------------------------------------


class TestJitDecorator:

    def test_bare_decorator_returns_callable_design(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            @jit
            def gen(a: In, *, M: Compile[int]):
                pass

        assert isinstance(gen, CallableDesign)

    def test_bare_decorator_empty_compile_kwargs(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            @jit
            def gen(a: In, *, M: Compile[int]):
                pass

        assert gen.compilable.compile_kwargs == {}

    def test_bare_decorator_default_use_cache(self):
        @jit
        def gen(a: In):
            pass

        assert gen.compilable.use_cache is True

    def test_with_compile_params_only(self):
        @jit(M=512, N=512)
        def gen(a: In, b: In, c: Out, *, M: Compile[int], N: Compile[int]):
            pass

        assert isinstance(gen, CallableDesign)
        assert gen.compilable.compile_kwargs == {"M": 512, "N": 512}

    def test_with_config_only(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            @jit(use_cache=False)
            def gen(a: In, *, M: Compile[int]):
                pass

        assert isinstance(gen, CallableDesign)
        assert gen.compilable.compile_kwargs == {}
        assert gen.compilable.use_cache is False

    def test_with_mixed_config_and_compile_kwargs(self):
        @jit(M=256, use_cache=False, aiecc_flags=["--verbose"])
        def gen(a: In, *, M: Compile[int]):
            pass

        assert gen.compilable.compile_kwargs == {"M": 256}
        assert gen.compilable.use_cache is False
        assert "--verbose" in gen.compilable.aiecc_flags

    def test_source_files_forwarded(self):
        @jit(source_files=["kernel.cc"])
        def gen(a: In):
            pass

        assert gen.compilable.source_files[0].name == "kernel.cc"

    def test_compile_flags_forwarded(self):
        @jit(compile_flags=["-O3"])
        def gen(a: In):
            pass

        assert "-O3" in gen.compilable.compile_flags

    def test_include_paths_forwarded(self):
        @jit(include_paths=["/opt/inc"])
        def gen(a: In):
            pass

        assert any("/opt/inc" in str(p) for p in gen.compilable.include_paths)

    def test_object_files_forwarded(self):
        @jit(object_files=["add.o"])
        def gen(a: In):
            pass

        assert gen.compilable.object_files[0].name == "add.o"

    def test_partial_decorator_applied_later(self):
        partial = jit(M=512)
        assert callable(partial)

        def gen(a: In, *, M: Compile[int]):
            pass

        result = partial(gen)
        assert isinstance(result, CallableDesign)
        assert result.compilable.compile_kwargs == {"M": 512}

    def test_empty_compile_kwargs_stored_as_empty_dict(self):
        @jit(use_cache=True)
        def gen(a: In):
            pass

        assert gen.compilable.compile_kwargs == {}

    def test_jit_config_keys_covers_all_compilable_design_params(self):
        expected = {
            "use_cache",
            "source_files",
            "aiecc_flags",
            "compile_flags",
            "include_paths",
            "object_files",
            "trace_config",
        }
        assert _JIT_CONFIG_KEYS == expected

    def test_unknown_key_becomes_compile_kwarg(self):
        @jit(my_custom_param=42)
        def gen(a: In, *, my_custom_param: Compile[int]):
            pass

        assert gen.compilable.compile_kwargs == {"my_custom_param": 42}

    def test_multiple_compile_params_all_captured(self):
        @jit(M=512, K=256, N=128)
        def gen(a: In, *, M: Compile[int], K: Compile[int], N: Compile[int]):
            pass

        assert gen.compilable.compile_kwargs == {"M": 512, "K": 256, "N": 128}

    def test_guard_1a_unknown_kwarg_to_jit_raises(self):
        """@iron.jit must raise TypeError when a kwarg matches neither a config key nor a Compile[T] param.

        Fails fast at decoration time so a typo like @jit(TYPO=512) doesn't
        silently run a kernel with no value bound.
        """
        import pytest

        with pytest.raises(TypeError, match="TYPO"):

            @jit(TYPO=512)
            def gen(a: In, *, M: Compile[int]):
                pass

    def test_guard_1b_unbound_required_compile_params_logs_at_debug(self, caplog):
        """Bare @iron.jit with required Compile[T] params and no pre-binding
        must log at DEBUG (not warn) — TypeError is the actual safety net at
        compile time, so decoration-time noise was demoted to debug.
        """
        import logging

        with caplog.at_level(logging.DEBUG, logger="aie.utils.callabledesign"):

            @jit  # bare — no pre-bound compile params
            def gen(a: In, *, M: Compile[int], N: Compile[int]):
                pass

        relevant = [
            r
            for r in caplog.records
            if r.levelno == logging.DEBUG
            and "no defaults and no pre-bound values" in r.getMessage()
        ]
        assert (
            relevant
        ), "Expected a DEBUG-level log for unbound required Compile[T] params"
        msg = relevant[-1].getMessage()
        assert (
            "M" in msg or "N" in msg
        ), f"DEBUG message should mention the unbound params; got: {msg}"

    def test_guard_1b_unbound_required_compile_params_silent_at_default(self):
        """Same scenario, default log level (WARNING): nothing should be
        emitted to warnings.warn or to the default-level logger."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @jit
            def gen(a: In, *, M: Compile[int], N: Compile[int]):
                pass

        unbound_warnings = [
            x
            for x in w
            if issubclass(x.category, UserWarning)
            and "no defaults and no pre-bound values" in str(x.message)
        ]
        assert not unbound_warnings, (
            "@jit decoration must not emit a UserWarning for unbound required "
            "Compile[T] params at default log level (it should be DEBUG-only). "
            f"Got: {[str(x.message) for x in unbound_warnings]}"
        )

    def test_guard_1c_unannotated_scalar_with_default_raises(self):
        """An unannotated non-tensor param with a default value would silently
        bake the default into the compiled MLIR and ignore per-call overrides.
        @iron.jit must reject this at decoration time and point users at
        Compile[T] = default (or, eventually, Rtp[T])."""
        with pytest.raises(TypeError, match="silently ignored"):

            @jit
            def gen(x: In, y: Out, *, factor: int = 7, N: Compile[int]):
                pass

    def test_guard_1c_unannotated_scalar_no_default_is_allowed(self):
        """No-default unannotated params already error naturally at compile
        time (TypeError missing kwarg) — they're not silent.  Guard 1-C only
        rejects the silent-default case, so this should decorate fine."""

        # Should NOT raise at decoration time.
        @jit
        def gen(x: In, y: Out, *, factor: int, N: Compile[int]):
            pass

        assert isinstance(gen, CallableDesign)

    def test_guard_1c_compile_t_with_default_is_allowed(self):
        """Compile[T] with default IS the supported way to express 'I want a
        default value, recompile on per-call change'.  Must not be rejected
        by Guard 1-C."""

        @jit
        def gen(x: In, y: Out, *, N: Compile[int] = 1024):
            pass

        assert isinstance(gen, CallableDesign)

    def test_jit_creates_distinct_designs_per_decoration(self):
        @jit(M=256)
        def gen_a(a: In, *, M: Compile[int]):
            pass

        @jit(M=512)
        def gen_b(a: In, *, M: Compile[int]):
            pass

        assert gen_a.compilable.compile_kwargs != gen_b.compilable.compile_kwargs
        assert hash(gen_a.compilable) != hash(gen_b.compilable)


# ---------------------------------------------------------------------------
# Fix 1: ExternalFunction/Kernel filtering in split_runtime_args
# ---------------------------------------------------------------------------


def test_external_function_positional_not_in_tensor_args():
    """ExternalFunction passed positionally must not appear in tensor_args."""
    # We cannot instantiate ExternalFunction (requires source_file/source_string),
    # so use Kernel which is the base class checked by isinstance in the fix.
    kernel_obj = Kernel("my_func", "my_func.o")

    def f(a: In, b: Out):
        pass

    cd = CallableDesign(f)
    a, b = object(), object()
    # Pass the Kernel instance between the two tensor args positionally.
    tensors, scalars = cd.compilable.split_runtime_args((a, kernel_obj, b), {})
    assert (
        kernel_obj not in tensors
    ), "Kernel instance must be filtered from tensor_args"
    assert a in tensors
    assert b in tensors
    assert (
        kernel_obj not in scalars.values()
    ), "Kernel instance must not appear in scalar_kwargs"


# NOTE: trace_config end-to-end behaviour (forwarded to NPUKernel.__init__,
# not to kernel.__call__) is covered by a real NPU run in
# test/python/npu-xrt/test_iron_jit_e2e.py::test_trace_config_forwarded_to_kernel.
# A pure-unit version was removed because it required mocking compile() in a
# way that diverged from prod behaviour (left _kernel_dir unset, masking a
# real defect).


# ---------------------------------------------------------------------------
# Guard 3-A: tensor param as runtime kwarg raises TypeError
# ---------------------------------------------------------------------------


def test_guard_3a_tensor_param_as_runtime_kwarg_raises():
    """Tensor-annotated params passed as keyword args must raise TypeError."""

    def gen(a: In, b: Out, *, M: Compile[int]):
        pass

    cd = CallableDesign(gen, compile_kwargs={"M": 1})
    a_obj = object()
    b_obj = object()
    with pytest.raises(TypeError, match="tensor param"):
        cd(a_obj, b=b_obj)  # 'b' is Out — must be positional


# ---------------------------------------------------------------------------
# Guard 3-C: too many positional args raises TypeError
# ---------------------------------------------------------------------------


def test_guard_3c_too_many_positional_raises():
    """More positional args than tensor+scalar slots must raise TypeError."""

    def gen(a: In, *, M: Compile[int]):
        pass  # only 1 tensor slot, 0 scalar slots

    cd = CallableDesign(gen, compile_kwargs={"M": 1})
    with pytest.raises(TypeError, match="positional argument"):
        cd(object(), object(), object())  # 3 positional, only 1 expected


def test_as_mlir_call_time_kwarg_overrides_prebound():
    """as_mlir() must let call-time Compile[T] kwargs override pre-bound values.

    Call-time-wins semantics are shared with __call__ so the MLIR you
    inspect matches the MLIR that would be compiled for the same kwargs.
    """

    def gen(a: In, b: Out, *, N: Compile[int] = 1024):
        pass

    cd = CallableDesign(gen, compile_kwargs={"N": 1024})

    # Capture the CompilableDesign that as_mlir() ends up calling
    # generate_mlir on so we can assert its effective compile_kwargs reflect
    # the override.
    captured_self = []

    def fake_generate(self):
        captured_self.append(self)
        return "<mlir>"

    with patch.object(
        CompilableDesign, "generate_mlir", autospec=True, side_effect=fake_generate
    ):
        result = cd.as_mlir(N=512)

    assert result == "<mlir>"
    assert len(captured_self) == 1
    bound = captured_self[0]
    assert bound.compile_kwargs["N"] == 512, (
        f"as_mlir() must override pre-bound N=1024 with call-time N=512; "
        f"CompilableDesign got compile_kwargs={bound.compile_kwargs}"
    )
    # The original CallableDesign must remain unchanged for future calls.
    assert cd.compilable.compile_kwargs["N"] == 1024


def test_as_mlir_no_warning_when_no_conflict():
    """as_mlir() must not warn when call-time kwargs match pre-bound values."""
    import warnings as _warnings

    def gen(a: In, b: Out, *, N: Compile[int] = 1024):
        pass

    cd = CallableDesign(gen, compile_kwargs={"N": 1024})

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        with patch.object(CompilableDesign, "generate_mlir", return_value="<mlir>"):
            cd.as_mlir(N=1024)  # same value — no conflict

    conflict_warnings = [
        w
        for w in caught
        if "ignored" in str(w.message).lower() or "pre-bound" in str(w.message).lower()
    ]
    assert (
        not conflict_warnings
    ), "as_mlir() must not warn when call-time and pre-bound values match"
