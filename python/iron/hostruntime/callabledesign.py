# callabledesign.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""CallableDesign: JIT-compiles on first call and runs on the NPU.

``CallableDesign`` wraps a ``CompilableDesign`` (or creates one implicitly)
and provides a ``__call__`` interface that:

1. Compiles the MLIR generator on first invocation (or returns a cached kernel
   on subsequent calls with the same compile configuration).
2. Supports two usage patterns for ``Compile[T]`` parameters:

   * **Pre-bound** (``@iron.jit(M=512)``): compile params fixed at decoration
     time; no extra kwargs needed at call time.
   * **Call-time** (bare ``@iron.jit``): ``Compile[T]``-annotated params
     passed as kwargs at each call site; different values produce independently
     cached kernels.

3. Splits runtime arguments into tensor args (``In``/``Out``/``InOut``
   annotated) and scalar kwargs (everything else) using the generator's type
   annotations — no heuristic type-checking needed.
4. Validates tensor shapes/dtypes against the compiled kernel specification.
5. Invokes the ``NPUKernel`` with the tensor args and scalar kwargs.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Callable

from aie.utils.compile.cache.utils import _create_function_cache_key
from aie.utils.npukernel import NPUKernel

from aie.iron.compile.compilabledesign import CompilableDesign

logger = logging.getLogger(__name__)


class CallableDesign:
    """JIT-compiling, callable wrapper around a ``CompilableDesign``.

    Supports two ``Compile[T]`` binding patterns:

    * **Pre-bound** — pass compile params at decoration time (Triton style)::

          @iron.jit(M=512, K=512, N=512)
          def gemm(a: In, b: In, c: Out,
                   M: Compile[int], K: Compile[int], N: Compile[int]):
              ...

          gemm(a, b, c)  # compiles once, cached thereafter

    * **Call-time** — pass compile params as kwargs at each call site::

          @iron.jit
          def gemm(a: In, b: In, c: Out,
                   M: Compile[int], K: Compile[int], N: Compile[int]):
              ...

          gemm(a, b, c, M=512, K=512, N=512)  # compiled for this shape
          gemm(a2, b2, c2, M=1024, K=1024, N=1024)  # separate cached kernel

    Args:
        mlir_generator: A callable, ``Path`` to a ``.mlir`` file, or an
            existing ``CompilableDesign`` instance.
        compile_kwargs: Values for ``Compile[T]``-annotated parameters.
            Ignored when *mlir_generator* is already a ``CompilableDesign``.
        use_cache: Enable filesystem caching. Forwarded to ``CompilableDesign``.
        source_files: C++ kernel source files. Forwarded to ``CompilableDesign``.
        aiecc_flags: Extra ``aiecc`` flags. Forwarded to ``CompilableDesign``.
        compile_flags: Extra Peano compiler flags. Forwarded to ``CompilableDesign``.
        include_paths: Extra ``-I`` paths. Forwarded to ``CompilableDesign``.
        object_files: Pre-compiled ``.o`` files. Forwarded to ``CompilableDesign``.
        trace_config: Optional ``TraceConfig`` for hardware trace collection.
            When set, ``trace_config.trace_size`` is injected as a
            ``trace_size`` compile kwarg so generators can use
            ``trace_size: Compile[int] = 0`` instead of receiving the full
            ``TraceConfig`` object.
    """

    def __init__(
        self,
        mlir_generator: Callable | Path | CompilableDesign,
        *,
        compile_kwargs: dict[str, Any] | None = None,
        use_cache: bool = True,
        source_files: list[str | Path] | None = None,
        aiecc_flags: list[str] | None = None,
        compile_flags: list[str] | None = None,
        include_paths: list[str | Path] | None = None,
        object_files: list[str | Path] | None = None,
        trace_config=None,
    ):
        if isinstance(mlir_generator, CompilableDesign):
            self.compilable = mlir_generator
        else:
            self.compilable = CompilableDesign(
                mlir_generator,
                compile_kwargs=compile_kwargs,
                use_cache=use_cache,
                source_files=source_files,
                aiecc_flags=aiecc_flags,
                compile_flags=compile_flags,
                include_paths=include_paths,
                object_files=object_files,
            )

        self.trace_config = trace_config

        # Pre-build the named wrapper object used as the cache-key identity for
        # Path-based generators.  Creating it once here avoids allocating a new
        # anonymous class and instance on every __call__ invocation.
        if isinstance(self.compilable.mlir_generator, Path):
            self._path_cache_fn = type(
                "_PathKernel", (), {"__name__": str(self.compilable.mlir_generator)}
            )()
        else:
            self._path_cache_fn = None

        # Per-instance in-process kernel cache: cache_key → NPUKernel.
        # Using a plain dict (no size cap) because there is no cross-function
        # interference risk; the number of distinct (shape, dtype, compile_kwargs)
        # combinations per function is naturally bounded in practice.
        self._kernel_cache: dict = {}

        # Warn if any required Compile[T] params are unbound at decoration time.
        # These must be supplied as kwargs at every call site.
        if (
            callable(self.compilable.mlir_generator)
            and not self.compilable.compile_kwargs
        ):
            import inspect as _inspect

            sig = _inspect.signature(self.compilable.mlir_generator)
            unbound_required = [
                name
                for name in self.compilable._compile_params
                if sig.parameters[name].default is _inspect.Parameter.empty
            ]
            if unbound_required:
                warnings.warn(
                    f"{self.compilable._generator_name()!r} has Compile[T] "
                    f"parameters with no defaults and no pre-bound values: "
                    f"{unbound_required}.\n"
                    f"  You must pass these as keyword arguments at every call:\n"
                    f"    kernel(..., {', '.join(f'{n}=...' for n in unbound_required)})\n"
                    f"  Omitting them will raise TypeError at compile time.",
                    stacklevel=3,
                )

    def _extract_compile_kwargs(
        self, runtime_kwargs: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Split runtime_kwargs into call-time compile params and scalar kwargs.

        Returns:
            (call_compile_kwargs, scalar_runtime_kwargs, effective_compile_kwargs)
        """
        call_compile_kwargs: dict[str, Any] = {}
        scalar_runtime_kwargs: dict[str, Any] = {}

        compile_param_names = set(self.compilable._compile_params)
        for name, val in runtime_kwargs.items():
            if name in compile_param_names:
                call_compile_kwargs[name] = val
            else:
                scalar_runtime_kwargs[name] = val

        # Pre-bound compile_kwargs win: placed last in the merge so they
        # overwrite any same-named call-time values.
        effective_compile_kwargs = {
            **call_compile_kwargs,
            **self.compilable.compile_kwargs,
        }

        return call_compile_kwargs, scalar_runtime_kwargs, effective_compile_kwargs

    def _build_compilable(
        self,
        call_compile_kwargs: dict[str, Any],
        effective_compile_kwargs: dict[str, Any],
    ) -> CompilableDesign:
        """Return a compilable for this call's effective compile kwargs.

        If call-time compile params were supplied a transient ``CompilableDesign``
        is created so the original ``self.compilable`` remains unchanged for
        future calls.  Otherwise ``self.compilable`` is returned directly.
        """
        if call_compile_kwargs:
            return CompilableDesign(
                self.compilable.mlir_generator,
                compile_kwargs=effective_compile_kwargs,
                use_cache=self.compilable.use_cache,
                compile_flags=self.compilable.compile_flags,
                source_files=self.compilable.source_files,
                include_paths=self.compilable.include_paths,
                aiecc_flags=self.compilable.aiecc_flags,
                object_files=self.compilable.object_files,
            )
        return self.compilable

    def __call__(self, *runtime_args, **runtime_kwargs):
        """Compile (if needed), then run the kernel.

        ``Compile[T]``-annotated kwargs in *runtime_kwargs* are extracted and
        merged with any pre-bound ``compile_kwargs``; remaining kwargs are
        forwarded to the NPU kernel as scalar arguments.

        Positional args fill tensor params (``In``/``Out``/``InOut``) in the
        order they appear in the generator signature.

        Args:
            *runtime_args: Runtime tensor and/or scalar positional arguments.
            **runtime_kwargs: Mix of call-time ``Compile[T]`` params and
                runtime scalar kernel arguments.

        Returns:
            The result of ``NPUKernel.__call__``.
        """
        # --- Split call-time Compile[T] params from runtime scalar kwargs ---
        # trace_config is handled specially: if annotated as Compile[object] on
        # the generator, it flows through the normal Compile[T] classification so
        # the generator receives it and can conditionally enable tracing in the
        # generated MLIR.  We extract it from effective_compile_kwargs after the
        # merge (below) rather than popping it here.
        call_compile_kwargs, scalar_runtime_kwargs, effective_compile_kwargs = (
            self._extract_compile_kwargs(runtime_kwargs)
        )

        # Guard 3-A: tensor params must not appear as runtime kwargs.
        tensor_names = set(self.compilable._tensor_params)
        confused_tensor_kwargs = set(scalar_runtime_kwargs.keys()) & tensor_names
        if confused_tensor_kwargs:
            raise TypeError(
                f"{self.compilable._generator_name()!r} received tensor "
                f"param(s) as keyword arguments: {confused_tensor_kwargs}.\n"
                f"  Params annotated In/Out/InOut must be passed positionally.\n"
                f"  Compile[T] params (passed as kwargs): "
                f"{self.compilable._compile_params}."
            )

        # Guard 3-C: too many positional args.
        if callable(self.compilable.mlir_generator):
            max_positional = len(self.compilable._tensor_params) + len(
                self.compilable._scalar_params
            )
            if len(runtime_args) > max_positional:
                raise TypeError(
                    f"{self.compilable._generator_name()!r} takes at most "
                    f"{max_positional} positional argument(s) "
                    f"(tensor: {len(self.compilable._tensor_params)}, "
                    f"scalar: {len(self.compilable._scalar_params)}) "
                    f"but {len(runtime_args)} were given.\n"
                    f"  Compile[T] parameters {self.compilable._compile_params} "
                    f"must be keyword arguments, not positional."
                )

        # --- Resolve trace_config ---
        # Two patterns are supported:
        #   1. JIT config: trace_config set on CallableDesign.__init__ (or via
        #      @iron.jit(trace_config=...)).  trace_config.trace_size is
        #      injected as a "trace_size" compile kwarg so generators can use
        #      the simpler ``trace_size: Compile[int] = 0`` signature.
        #   2. Compile kwarg (legacy): trace_config passed as a Compile[T]
        #      param on the generator (``trace_config: Compile[... | None]``).
        trace_config = self.trace_config
        if trace_config is not None:
            # Inject trace_size as a compile kwarg for the generator.
            if "trace_size" not in effective_compile_kwargs:
                effective_compile_kwargs["trace_size"] = trace_config.trace_size
                call_compile_kwargs["trace_size"] = trace_config.trace_size
        else:
            # Legacy path: extract trace_config from compile kwargs.
            trace_config = effective_compile_kwargs.get("trace_config", None)

        # Build a separate dict for the cache key that excludes trace_config:
        # trace_config is a per-call object whose identity should not drive cache
        # misses.
        cache_compile_kwargs = {
            k: v for k, v in effective_compile_kwargs.items() if k != "trace_config"
        }

        # Guard 3-B: raise if call-time value differs from a pre-bound value.
        # Identical values are silently accepted.
        prebound = set(self.compilable.compile_kwargs.keys())
        overridden = {
            k: (call_compile_kwargs[k], self.compilable.compile_kwargs[k])
            for k in set(call_compile_kwargs.keys()) & prebound
            if call_compile_kwargs[k] != self.compilable.compile_kwargs[k]
        }
        if overridden:
            detail = ", ".join(
                f"{k}={call!r} ignored, using pre-bound {pre!r}"
                for k, (call, pre) in overridden.items()
            )
            raise TypeError(
                f"{self.compilable._generator_name()!r} has pre-bound "
                f"Compile[T] value(s) that override call-site value(s): "
                f"{detail}.\n"
                f"  Pre-bound values always win. Use bare @iron.jit to "
                f"allow per-call compile parameters.",
            )

        compilable = self._build_compilable(
            call_compile_kwargs, effective_compile_kwargs
        )

        # --- In-process kernel cache lookup ---
        # Use the generator (or its string path) as the cache key identity.
        # For Path generators: wrap in an object with __name__ so that
        # _create_function_cache_key does not crash (it accesses .__name__).
        generator = compilable.mlir_generator
        if callable(generator):
            cache_fn = generator
        else:
            # Use the pre-built named wrapper created once in __init__.
            cache_fn = self._path_cache_fn

        cache_key = _create_function_cache_key(
            cache_fn,
            runtime_args,
            cache_compile_kwargs,
        )

        if compilable.use_cache and cache_key in self._kernel_cache:
            kernel = self._kernel_cache[cache_key]
        else:
            # Compile on demand.
            xclbin_path, inst_path = compilable.compile()

            # Set physical MLIR path for trace parsing (contains lowered
            # npu_write32 ops).  Mirrors utils/jit.py lines 175-178.
            if trace_config is not None:
                kernel_dir = xclbin_path.parent
                physical_mlir = kernel_dir / "input_with_addresses.mlir"
                if physical_mlir.exists():
                    trace_config.physical_mlir_path = str(physical_mlir)

            kernel = NPUKernel(
                xclbin_path,
                inst_path,
                kernel_name="MLIR_AIE",
                trace_config=trace_config,
            )
            if compilable.use_cache:
                self._kernel_cache[cache_key] = kernel

        tensor_args, remaining_scalars = compilable.split_runtime_args(
            runtime_args, scalar_runtime_kwargs
        )
        compilable.validate_tensor_args(tensor_args)
        return kernel(*tensor_args, **remaining_scalars)

    def lower(self, *runtime_args, **runtime_kwargs) -> str:
        """Generate and return the MLIR text for this kernel without compiling.

        Accepts the same arguments as ``__call__``.  Tensor args may be real
        tensors (shape and dtype are read from them) or ``None`` (in which case
        the generator body must use ``Compile[T]`` params for all shape/dtype
        info).

        Returns:
            The MLIR module as a string (suitable for inspection or debugging).

        Note:
            Unlike ``__call__``, this method does not raise ``TypeError`` when a
            call-time ``Compile[T]`` value conflicts with a pre-bound value.
            ``lower()`` is an inspection tool; pre-bound values silently win,
            consistent with the merge semantics of ``__call__``.
        """
        from aie.iron.kernel import ExternalFunction

        call_compile_kwargs, _scalar_runtime_kwargs, effective_compile_kwargs = (
            self._extract_compile_kwargs(runtime_kwargs)
        )

        compilable = self._build_compilable(
            call_compile_kwargs, effective_compile_kwargs
        )

        module = compilable._generate_mlir(ExternalFunction)
        return str(module)

    def __repr__(self) -> str:
        return f"CallableDesign({self.compilable!r})"
