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
2. Supports two usage patterns for ``CompileTime[T]`` parameters:

   * **Pre-bound** (``@iron.jit(M=512)``): compile params fixed at decoration
     time; no extra kwargs needed at call time.
   * **Call-time** (bare ``@iron.jit``): ``CompileTime[T]``-annotated params
     passed as kwargs at each call site; different values produce independently
     cached kernels.

   When both are supplied for the same name, the call-time value wins.

3. Splits runtime arguments into tensor args (``In``/``Out``/``InOut``
   annotated) and scalar kwargs (everything else) using the generator's type
   annotations — no heuristic type-checking needed.
4. Validates tensor shapes/dtypes against the compiled kernel specification.
5. Invokes the ``NPUKernel`` with the tensor args and scalar kwargs.
"""

from __future__ import annotations

import inspect as _inspect
import logging
from pathlib import Path
from typing import Any, Callable

from aie.utils.compile.cache.utils import _create_function_cache_key
from aie.utils.compile.jit.compilabledesign import CompilableDesign

# NPUKernel and DefaultNPURuntime pull in the XRT runtime stack on import.
# Defer to first call so importing CallableDesign on machines without an NPU
# (e.g. CI nodes that only need as_mlir / generate_mlir for inspection) does
# not fail at module load.

logger = logging.getLogger(__name__)


def _evict_xrt_context(xclbin_path: Path) -> None:
    """Evict a stale XRT hw_context after IOCTL EINVAL so the retry gets a fresh one."""
    from aie.utils import DefaultNPURuntime

    if DefaultNPURuntime is None or not hasattr(DefaultNPURuntime, "_context_cache"):
        return
    try:
        resolved = str(xclbin_path.resolve())
        mtime = xclbin_path.stat().st_mtime
        entry = DefaultNPURuntime._context_cache.pop((resolved, mtime), None)
        if entry is not None:
            DefaultNPURuntime._cleanup_entry(entry)
    except Exception:
        # Recovery path: must not raise, but log loudly — silent failure would
        # keep recycling a broken _context_cache into every retry.
        logger.warning(
            "_evict_xrt_context: failed to evict %s; retry may reuse a stale "
            "hardware context",
            xclbin_path,
            exc_info=True,
        )


class CallableDesign:
    """JIT-compiling, callable wrapper around a ``CompilableDesign``.

    Supports two ``CompileTime[T]`` binding patterns:

    * **Pre-bound** — pass compile params at decoration time (Triton style)::

          @iron.jit(M=512, K=512, N=512)
          def gemm(a: In, b: In, c: Out,
                   M: CompileTime[int], K: CompileTime[int], N: CompileTime[int]):
              ...

          gemm(a, b, c)  # compiles once, cached thereafter

    * **Call-time** — pass compile params as kwargs at each call site::

          @iron.jit
          def gemm(a: In, b: In, c: Out,
                   M: CompileTime[int], K: CompileTime[int], N: CompileTime[int]):
              ...

          gemm(a, b, c, M=512, K=512, N=512)  # compiled for this shape
          gemm(a2, b2, c2, M=1024, K=1024, N=1024)  # separate cached kernel

    Args:
        mlir_generator: A callable, ``Path`` to a ``.mlir`` file, or an
            existing ``CompilableDesign`` instance.
        compile_kwargs: Values for ``CompileTime[T]``-annotated parameters.
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
            ``trace_size: CompileTime[int] = 0`` instead of receiving the full
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

        self._kernel_cache: dict = {}

        if (
            logger.isEnabledFor(logging.DEBUG)
            and callable(self.compilable.mlir_generator)
            and not self.compilable.compile_kwargs
        ):
            sig = _inspect.signature(self.compilable.mlir_generator)
            unbound_required = [
                name
                for name in self.compilable.compile_params
                if sig.parameters[name].default is _inspect.Parameter.empty
            ]
            if unbound_required:
                logger.debug(
                    "%r has CompileTime[T] parameters with no defaults and no "
                    "pre-bound values: %s. Pass these as keyword arguments at "
                    "every call site: kernel(..., %s).",
                    self.compilable.generator_name,
                    unbound_required,
                    ", ".join(f"{n}=..." for n in unbound_required),
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

        compile_param_names = set(self.compilable.compile_params)
        for name, val in runtime_kwargs.items():
            if name in compile_param_names:
                call_compile_kwargs[name] = val
            else:
                scalar_runtime_kwargs[name] = val

        # Call-time compile_kwargs win, matching Triton (and CompilableDesign
        # .specialize() — pre-bound is the starting set, the call-site kwarg
        # overrides). __call__ and as_mlir share these semantics.
        effective_compile_kwargs = {
            **self.compilable.compile_kwargs,
            **call_compile_kwargs,
        }

        return call_compile_kwargs, scalar_runtime_kwargs, effective_compile_kwargs

    def _build_compilable(
        self,
        call_compile_kwargs: dict[str, Any],
    ) -> CompilableDesign:
        """Return a compilable for this call's effective compile kwargs.

        If call-time compile params were supplied a transient ``CompilableDesign``
        is created so the original ``self.compilable`` remains unchanged for
        future calls.  Otherwise ``self.compilable`` is returned directly.
        """
        if call_compile_kwargs:
            return self.compilable.specialize(**call_compile_kwargs)
        return self.compilable

    def _compile_and_build_kernel(
        self,
        compilable: CompilableDesign,
        cache_key,
        trace_config,
    ) -> NPUKernel:
        from aie.utils.npukernel import NPUKernel

        xclbin_path, inst_path = compilable.compile()
        if trace_config is not None and compilable._kernel_dir is not None:
            physical_mlir = compilable._kernel_dir / "input_with_addresses.mlir"
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
        return kernel

    def __call__(self, *runtime_args, **runtime_kwargs):
        """Compile (if needed), then run the kernel.

        ``CompileTime[T]``-annotated kwargs in *runtime_kwargs* are extracted and
        merged with any pre-bound ``compile_kwargs``; remaining kwargs are
        forwarded to the NPU kernel as scalar arguments.

        Positional args fill tensor params (``In``/``Out``/``InOut``) in the
        order they appear in the generator signature.

        Args:
            *runtime_args: Runtime tensor and/or scalar positional arguments.
            **runtime_kwargs: Mix of call-time ``CompileTime[T]`` params and
                runtime scalar kernel arguments.

        Returns:
            The result of ``NPUKernel.__call__``.
        """
        # --- Split call-time CompileTime[T] params from runtime scalar kwargs ---
        # trace_config is handled specially: if annotated as CompileTime[object] on
        # the generator, it flows through the normal CompileTime[T] classification so
        # the generator receives it and can conditionally enable tracing in the
        # generated MLIR.  We extract it from effective_compile_kwargs after the
        # merge (below) rather than popping it here.
        call_compile_kwargs, scalar_runtime_kwargs, effective_compile_kwargs = (
            self._extract_compile_kwargs(runtime_kwargs)
        )

        # Guard 3-A: tensor params must not appear as runtime kwargs.
        tensor_names = set(self.compilable.tensor_params)
        confused_tensor_kwargs = set(scalar_runtime_kwargs.keys()) & tensor_names
        if confused_tensor_kwargs:
            raise TypeError(
                f"{self.compilable.generator_name!r} received tensor "
                f"param(s) as keyword arguments: {confused_tensor_kwargs}.\n"
                f"  Params annotated In/Out/InOut must be passed positionally.\n"
                f"  CompileTime[T] params (passed as kwargs): "
                f"{self.compilable.compile_params}."
            )

        # Guard 3-C: too many positional args.
        if callable(self.compilable.mlir_generator):
            max_positional = len(self.compilable.tensor_params) + len(
                self.compilable.scalar_params
            )
            if len(runtime_args) > max_positional:
                raise TypeError(
                    f"{self.compilable.generator_name!r} takes at most "
                    f"{max_positional} positional argument(s) "
                    f"(tensor: {len(self.compilable.tensor_params)}, "
                    f"scalar: {len(self.compilable.scalar_params)}) "
                    f"but {len(runtime_args)} were given.\n"
                    f"  CompileTime[T] parameters {self.compilable.compile_params} "
                    f"must be keyword arguments, not positional."
                )

        trace_config = self.trace_config
        if trace_config is not None:
            if "trace_size" not in effective_compile_kwargs:
                effective_compile_kwargs["trace_size"] = trace_config.trace_size
                call_compile_kwargs["trace_size"] = trace_config.trace_size
        else:
            trace_config = effective_compile_kwargs.get("trace_config", None)

        # Build a separate dict for the cache key that excludes trace_config:
        # trace_config is a per-call object whose identity should not drive cache
        # misses.
        cache_compile_kwargs = {
            k: v for k, v in effective_compile_kwargs.items() if k != "trace_config"
        }

        compilable = self._build_compilable(call_compile_kwargs)

        # In-process key includes runtime_args (tensor shapes); on-disk key in
        # _compute_cache_hash does not. Divergence is intentional: if a generator
        # omits CompileTime[T] for shape, the disk artifact reuses but the in-process
        # slot changes, so validate_tensor_args() surfaces the mismatch.
        generator = compilable.mlir_generator
        if callable(generator):
            cache_fn = generator
        else:
            cache_fn = self._path_cache_fn

        cache_key = _create_function_cache_key(
            cache_fn,
            runtime_args,
            cache_compile_kwargs,
        )

        if compilable.use_cache and cache_key in self._kernel_cache:
            kernel = self._kernel_cache[cache_key]
        else:
            kernel = self._compile_and_build_kernel(compilable, cache_key, trace_config)

        tensor_args, remaining_scalars = compilable.split_runtime_args(
            runtime_args, scalar_runtime_kwargs
        )
        compilable.validate_tensor_args(tensor_args)

        try:
            return kernel(*tensor_args, **remaining_scalars)
        except RuntimeError as exc:
            # IOCTL EINVAL → stale XRT hw_context. Match libxrt's "err=-22"
            # or xrt::error's "XRT...Invalid argument"; bare "Invalid argument"
            # is too generic and is not treated as EINVAL.
            msg = str(exc)
            is_xrt_einval = "err=-22" in msg or (
                "Invalid argument" in msg and ("XRT" in msg or "xrt" in msg)
            )
            if not is_xrt_einval:
                raise
            logger.warning(
                "XRT IOCTL EINVAL detected for %r; evicting hw_context and "
                "retrying once. Original error: %s",
                self.compilable.generator_name,
                msg,
            )

            self._kernel_cache.pop(cache_key, None)
            xclbin_path, _ = compilable.compile()
            _evict_xrt_context(xclbin_path)
            kernel = self._compile_and_build_kernel(compilable, cache_key, trace_config)
            return kernel(*tensor_args, **remaining_scalars)

    def specialize(self, **compile_kwargs) -> "CallableDesign":
        """Return a new ``CallableDesign`` with additional ``CompileTime[T]`` kwargs bound.

        The given kwargs are merged onto any pre-bound ``compile_kwargs`` with
        call-time values winning — matching ``__call__`` / ``as_mlir`` semantics.
        Config (``source_files``, ``aiecc_flags``, etc.) is preserved.

        Use together with :meth:`compile` to perform ahead-of-time compilation
        of a JIT-decorated design at known shapes::

            @iron.jit
            def matmul(...): ...

            matmul.specialize(M=256, K=256, N=256, element_type=np.int16).compile()
        """
        return CallableDesign(
            self.compilable.specialize(**compile_kwargs),
            trace_config=self.trace_config,
        )

    def compile(
        self,
        xclbin_path: Path | str | None = None,
        inst_path: Path | str | None = None,
        elf_path: Path | str | None = None,
    ) -> tuple[Path, Path]:
        """Eagerly compile this design and return ``(xclbin_path, inst_path)``.

        With no arguments, pre-warms the on-disk cache so subsequent calls with
        matching ``compile_kwargs`` hit the cache instead of paying ``aiecc``
        time on first invocation.

        With both ``xclbin_path`` and ``inst_path`` set, writes artifacts
        directly to those paths and bypasses the cache — useful for build
        systems (e.g. Makefiles) that manage their own dependency tracking.
        Mixed (only one of ``xclbin_path`` / ``inst_path`` given) raises
        ``ValueError``.

        ``elf_path`` is optional: when set, aiecc also wraps the NPU
        instructions into an ELF (via ``aiebu-asm``) at that path.  Needed by
        C++ testbenches that load instructions through ``xrt::elf`` +
        ``xrt::module``; requires explicit ``xclbin_path`` + ``inst_path``.
        """
        return self.compilable.compile(
            xclbin_path=xclbin_path, inst_path=inst_path, elf_path=elf_path
        )

    def as_mlir(self, *runtime_args, **runtime_kwargs) -> str:
        """Return the resolved MLIR text for this kernel without compiling.

        Accepts the same arguments as ``__call__``.  Tensor args may be real
        tensors (shape and dtype are read from them) or ``None`` (in which case
        the generator body must use ``CompileTime[T]`` params for all shape/dtype
        info).

        Returns:
            The MLIR module as a string (suitable for inspection, debugging,
            or feeding to a separate aiecc invocation).
        """
        call_compile_kwargs, _scalar_runtime_kwargs, _ = self._extract_compile_kwargs(
            runtime_kwargs
        )
        compilable = self._build_compilable(call_compile_kwargs)
        return str(compilable.generate_mlir())

    def __repr__(self) -> str:
        return f"CallableDesign({self.compilable!r})"
