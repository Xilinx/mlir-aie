# compilabledesign.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""CompilableDesign: bundles an MLIR generator with its compile-time configuration.

Pairs an MLIR generator function (or ``.mlir`` file path) with explicit
compile-time parameters and produces an ``xclbin`` + ``insts.bin`` artifact
pair via ``compile()``.

Hashing is split into two halves so callers can distinguish "recipe changed"
from "rebuild needed":

* ``recipe_hash``   — generator bytecode + compile_kwargs + aiecc/compile flags
* ``artifact_hash`` — source / object mtimes + tool mtimes + target arch

``hash(design)`` composes both into a 24-hex cache key; no MLIR generation
needed for a cache lookup.

Generation is memoized at the instance level via :attr:`CompilableDesign._generated`
(a ``functools.cached_property`` returning the MLIR text + collected
``ExternalFunction`` instances). Caching the *text* — not the Module — is what
makes this safe across MLIR Contexts: every consumer call re-parses into a
fresh ``mlir_mod_ctx()`` and gets a Module bound to its own Context. Safe
because all constructor inputs are frozen, so the generated MLIR is a pure
function of the instance.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from aie.utils.compile import (
    NPU_CACHE_HOME,
    compile_external_kernel,
    compile_mlir_module,
)
from aie.utils.compile.cache.utils import file_lock
from aie.utils.compile.utils import _cleanup_failed_compilation
from aie.extras.context import mlir_mod_ctx
from aie.ir import Module as _Module

from ._dma_size_parser import parse_dma_sizes
from ._hash import _compute_artifact_hash, _compute_hash, _compute_recipe_hash
from ._introspect import (
    _introspect_generator,
    _is_compile_param,
    _is_tensor_param,
    split_params,
)
from ._serialization import _TensorPlaceholder, _decode_kwarg, _encode_kwarg
from .context import compile_context
from .markers import CompileTime, In, InOut, Out

logger = logging.getLogger(__name__)


class CompilableDesign:
    """Bundles an MLIR generator with compile-time parameters.

    Args:
        mlir_generator: A callable that accepts ``CompileTime[T]`` kwargs and
            either returns an MLIR module (e.g., built inside an
            ``mlir_mod_ctx()`` block) or returns ``None`` after building the
            module into the active MLIR context (e.g., via
            ``Program(...).resolve_program()``),
            OR a ``pathlib.Path`` to a pre-written ``.mlir`` file.
        use_cache: When ``True`` (default), a file-system cache keyed by the
            bytecode+kwargs hash is consulted before recompiling.
        compile_kwargs: Values for the ``CompileTime[T]``-annotated parameters.
            Validated against the generator signature via ``inspect.Signature.bind``.
        compile_flags: Extra flags forwarded to the Peano C++ compiler.
        source_files: Paths to C++ kernel source files.  Their mtimes are
            included in the cache key so that edits correctly invalidate the cache.
        include_paths: Extra ``-I`` paths forwarded to the C++ compiler.
        aiecc_flags: Extra flags forwarded to ``aiecc``.
        object_files: Pre-compiled ``.o`` files to link with.
    """

    def __init__(
        self,
        mlir_generator: Callable | Path,
        *,
        use_cache: bool = True,
        compile_kwargs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
        source_files: list[str | Path] | None = None,
        include_paths: list[str | Path] | None = None,
        aiecc_flags: list[str] | None = None,
        object_files: list[str | Path] | None = None,
    ):
        self.mlir_generator = mlir_generator
        self.use_cache = use_cache
        # Freeze all inputs so callers can't mutate config after construction
        # (which would silently invalidate the cache hash). MappingProxyType +
        # tuples are read-only views; equality with plain dict/list still works.
        self.compile_kwargs: Mapping[str, Any] = MappingProxyType(
            dict(compile_kwargs or {})
        )
        self.compile_flags: tuple[str, ...] = tuple(compile_flags or ())
        self.source_files: tuple[Path, ...] = tuple(
            Path(sf) for sf in (source_files or ())
        )
        self.include_paths: tuple[Path, ...] = tuple(
            Path(p) for p in (include_paths or ())
        )
        self.aiecc_flags: tuple[str, ...] = tuple(aiecc_flags or ())
        self.object_files: tuple[Path, ...] = tuple(
            Path(of) for of in (object_files or ())
        )

        # Cached artifact paths (set after compile()).
        self._xclbin_path: Path | None = None
        self._inst_path: Path | None = None
        self._kernel_dir: Path | None = None
        self._expected_tensor_sizes: list[int] | None = None

        # Introspect generator signature to split param categories.  Cache
        # hints+sig on self so split_runtime_args / _generate_mlir reuse the
        # same memoised intro instead of re-running typing.get_type_hints
        # and inspect.signature on every call.
        if callable(mlir_generator):
            self._hints, self._sig, (cp, tp, sp) = _introspect_generator(mlir_generator)
            self.compile_params = list(cp)
            self.tensor_params = list(tp)
            self.scalar_params = list(sp)
        else:
            self._hints = {}
            self._sig = None
            self.compile_params = []
            self.tensor_params = []
            self.scalar_params = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def specialize(self, **compile_kwargs) -> "CompilableDesign":
        """Return a new ``CompilableDesign`` with additional ``CompileTime[T]`` kwargs bound.

        The given kwargs are merged onto ``self.compile_kwargs`` with call-time
        values winning.  All other config (``source_files``, ``aiecc_flags``,
        ``include_paths``, etc.) is preserved.
        """
        return CompilableDesign(
            self.mlir_generator,
            compile_kwargs={**self.compile_kwargs, **compile_kwargs},
            use_cache=self.use_cache,
            compile_flags=list(self.compile_flags),
            source_files=list(self.source_files),
            include_paths=list(self.include_paths),
            aiecc_flags=list(self.aiecc_flags),
            object_files=list(self.object_files),
        )

    def compile(
        self,
        xclbin_path: Path | str | None = None,
        inst_path: Path | str | None = None,
        elf_path: Path | str | None = None,
    ) -> tuple[Path, Path]:
        """Compile the generator to ``(xclbin_path, inst_path)``.

        When both ``xclbin_path`` and ``inst_path`` are given, artifacts are
        written directly to those paths; the parent directory is used as
        ``work_dir`` for intermediate files (``.o``, lowered ``.mlir``).  The
        on-disk cache is bypassed in this mode — the caller is presumed to
        manage their own dependency tracking (e.g. via a Makefile).

        When both are ``None`` (the default), behavior is unchanged: artifacts
        land in ``~/.npu/cache/<hash>/`` and the cache is consulted first.

        Mixed (only one of ``xclbin_path`` / ``inst_path`` given) raises
        ``ValueError``.

        ``elf_path`` is optional and orthogonal: when set, aiecc also wraps
        the NPU instructions into an ELF (via ``aiebu-asm``) at that path,
        suitable for C++ testbenches that load through ``xrt::elf`` +
        ``xrt::module``.  Requires ``xclbin_path`` / ``inst_path`` to be set
        too — the cache path doesn't track ELF artifacts.
        """
        from aie.iron.kernel import ExternalFunction
        from aie.utils import DefaultNPURuntime

        if (xclbin_path is None) != (inst_path is None):
            raise ValueError(
                "compile(): xclbin_path and inst_path must be set together "
                "(both paths to write artifacts directly, or both None to use "
                f"the JIT cache).  Got xclbin_path={xclbin_path!r}, "
                f"inst_path={inst_path!r}."
            )
        explicit_paths = xclbin_path is not None
        cache_hash = None

        if elf_path is not None and not explicit_paths:
            raise ValueError(
                "compile(): elf_path requires explicit xclbin_path + inst_path "
                "(the JIT cache does not track ELF artifacts)."
            )

        if explicit_paths:
            assert xclbin_path is not None and inst_path is not None
            # Absolutize so compile_external_kernel's `cwd=kernel_dir` doesn't
            # turn relative paths into "build/build/foo.cc" etc.
            xclbin_path = Path(xclbin_path).resolve()
            inst_path = Path(inst_path).resolve()
            if elf_path is not None:
                elf_path = Path(elf_path).resolve()
            # Per-xclbin scratch dir (mirrors aiecc's default <input>.prj
            # naming) so two siblings sharing one build/ don't clobber each
            # other's input_with_addresses.mlir / .o files.
            kernel_dir = xclbin_path.parent / f"{xclbin_path.stem}.prj"
            lock_file_path = kernel_dir / ".lock"
        else:
            cache_hash = self._compute_cache_hash()
            kernel_dir = NPU_CACHE_HOME / cache_hash
            lock_file_path = kernel_dir / ".lock"
            xclbin_path = kernel_dir / "final.xclbin"
            inst_path = kernel_dir / "insts.bin"

        with file_lock(lock_file_path):
            os.makedirs(kernel_dir, exist_ok=True)

            xclbin_exists = xclbin_path.exists()
            inst_exists = inst_path.exists()

            if not explicit_paths and self.use_cache and xclbin_exists and inst_exists:
                logger.debug(
                    "Cache hit for '%s' (hash=%s)", self.generator_name, cache_hash
                )
                self._xclbin_path = xclbin_path
                self._inst_path = inst_path
                self._kernel_dir = kernel_dir
                # Populate on cache hit too, else validate_tensor_args no-ops
                # and callers see kernel garbage instead of a shape error.
                if self._expected_tensor_sizes is None:
                    self._expected_tensor_sizes = parse_dma_sizes(kernel_dir)
                return xclbin_path, inst_path

            if explicit_paths:
                logger.debug(
                    "Compiling '%s' to %s (explicit paths, cache bypassed)",
                    self.generator_name,
                    xclbin_path,
                )
            else:
                logger.debug(
                    "Cache miss for '%s' (hash=%s); compiling...",
                    self.generator_name,
                    cache_hash,
                )

            try:
                # _EXTERN_CACHE ops bind to a prior MLIR Context; reuse across
                # contexts segfaults in func.function_type. Clear per compile.
                from aie.iron.kernels._common import _EXTERN_CACHE

                _EXTERN_CACHE.clear()
                mlir_module = self._generate_mlir(ExternalFunction)

                from aie.utils.compile import resolve_target_arch
                import aie.iron as _iron

                # Prefer iron's current device (matches `iron.get_current_device()`
                # inside the generator); fall back to XRT-detected only when unset.
                # Without this, a Strix-targeted design (iron set to NPU2) running
                # in an env without pyxrt silently builds .o files for aie2 instead
                # of aie2p — link succeeds, runtime times out (ERT_CMD_STATE_TIMEOUT).
                try:
                    device = _iron.get_current_device()
                except (RuntimeError, AttributeError):
                    device = (
                        DefaultNPURuntime.device()
                        if DefaultNPURuntime is not None
                        else None
                    )
                target_arch = resolve_target_arch(device)

                external_kernels = list(ExternalFunction._instances)
                ExternalFunction._instances.clear()

                # aiecc invokes one toolchain front-end per compile; all EFs
                # must agree or the resulting xclbin is silently broken.
                chess_uses = {getattr(f, "_use_chess", False) for f in external_kernels}
                if len(chess_uses) > 1:
                    chess_funcs = [f._name for f in external_kernels if f._use_chess]
                    peano_funcs = [
                        f._name for f in external_kernels if not f._use_chess
                    ]
                    raise RuntimeError(
                        "Mixed peano + chess ExternalFunctions in one "
                        f"@iron.jit design ({self.generator_name!r}): "
                        f"chess={chess_funcs}, peano={peano_funcs}.  aiecc "
                        "can only invoke one front-end per compile; pick one "
                        "toolchain consistently across all kernels.* helper "
                        "calls in this design."
                    )
                use_chess = chess_uses == {True}

                for func in external_kernels:
                    if not func._compiled:
                        compile_external_kernel(func, kernel_dir, target_arch)

                compile_mlir_module(
                    mlir_module=mlir_module,
                    insts_path=inst_path,
                    xclbin_path=xclbin_path,
                    elf_path=elf_path,
                    work_dir=kernel_dir,
                    use_chess=use_chess,
                    options=list(self.aiecc_flags) if self.aiecc_flags else None,
                )

                # aiecc may exit 0 even when xclbin generation fails silently
                # (missing xclbinutil/bootgen); verify outputs exist.
                expected_outputs = [xclbin_path, inst_path]
                if elf_path is not None:
                    expected_outputs.append(Path(elf_path))
                missing = [p for p in expected_outputs if not p.exists()]
                if missing:
                    raise RuntimeError(
                        "[aiecc] Compilation appeared to succeed (exit code 0) "
                        "but expected output file(s) were not created: "
                        + ", ".join(str(p) for p in missing)
                    )
            except Exception:
                _cleanup_failed_compilation(kernel_dir)
                raise

        self._xclbin_path = xclbin_path
        self._inst_path = inst_path
        self._kernel_dir = kernel_dir
        # Parse expected tensor sizes for runtime validation.
        self._expected_tensor_sizes = parse_dma_sizes(kernel_dir)
        return xclbin_path, inst_path

    def get_artifacts(self) -> tuple[Path, Path] | None:
        """Return cached artifact paths without recompiling, or ``None``."""
        if self._xclbin_path is None or self._inst_path is None:
            return None
        return self._xclbin_path, self._inst_path

    def split_runtime_args(
        self, runtime_args: tuple, runtime_kwargs: dict[str, Any]
    ) -> tuple[list, dict[str, Any]]:
        """Split ``runtime_args``/``runtime_kwargs`` into tensor list and scalar dict.

        Uses the ``In``/``Out``/``InOut`` annotation order from the generator
        signature.  Positional ``runtime_args`` are consumed left-to-right to
        fill tensor params (in signature order), then scalar params.

        ``Kernel`` and ``ExternalFunction`` instances are compile-time-only
        objects resolved at link time; they are silently filtered out and
        never forwarded to the NPU kernel as runtime arguments.

        Returns:
            ``(tensor_args, scalar_kwargs)``
        """
        from aie.iron.kernel import ExternalFunction, Kernel

        if not callable(self.mlir_generator):
            # Static .mlir file: pass everything through as tensors,
            # but still filter compile-time-only kernel objects.
            filtered = [a for a in runtime_args if not isinstance(a, Kernel)]
            return filtered, runtime_kwargs

        tensor_args = []
        scalar_kwargs = dict(runtime_kwargs)

        # Reuse the cached intro from __init__ — same generator, same hints/sig.
        hints = self._hints
        sig = self._sig
        assert sig is not None
        params = [
            (name, p)
            for name, p in sig.parameters.items()
            if name not in self.compile_kwargs
        ]

        # Walk the non-compile parameters in order, consuming positional args.
        # Kernel/ExternalFunction instances are compile-time only; skip them
        # in the positional stream so they never land in tensor_args or
        # scalar_kwargs.
        def _next_non_kernel(it):
            while True:
                val = next(it)
                if not isinstance(val, Kernel):
                    return val

        pos_iter = iter(runtime_args)
        for name, param in params:
            ann = hints.get(name, param.annotation)
            if _is_tensor_param(ann):
                # Try positional first, then kwargs.
                if name in scalar_kwargs:
                    tensor_args.append(scalar_kwargs.pop(name))
                else:
                    try:
                        tensor_args.append(_next_non_kernel(pos_iter))
                    except StopIteration:
                        pass
            else:
                # Scalar param: leave in scalar_kwargs (already there from kwargs)
                # or consume from positional.
                if name not in scalar_kwargs:
                    try:
                        val = _next_non_kernel(pos_iter)
                        scalar_kwargs[name] = val
                    except StopIteration:
                        pass

        return tensor_args, scalar_kwargs

    def generate_mlir(self):
        """Generate and return the MLIR module without compiling to xclbin.

        Useful for inspecting generated MLIR, debugging, or offline analysis.
        Does not require an NPU or XRT to be present.

        Returns:
            The generated ``mlir.ir.Module``.
        """
        from aie.iron.kernel import ExternalFunction

        return self._generate_mlir(ExternalFunction)

    def validate_tensor_args(self, tensor_args: list) -> None:
        """Validate that *tensor_args* element counts match the compiled kernel.

        Compares each tensor's element count against the per-host-arg
        addressable footprint extracted from the compiled
        ``aiex.runtime_sequence``.  ``parse_dma_sizes`` returns
        ``max(offset + len)`` so multi-column fan-outs, repeated transfers
        (matmul B reloaded each tile_row), and InOut buffers (for_each
        fill+drain on the same arg) all give the host-tensor size directly.

        Args with no associated DMA (entry == 0) are skipped — those are
        runtime params not directly transferred by the design.

        No-op when expected sizes are unavailable (e.g. offline compilation
        or when ``input_with_addresses.mlir`` was not produced).
        """
        if not self._expected_tensor_sizes:
            return
        import numpy as np

        for i, (tensor, expected) in enumerate(
            zip(tensor_args, self._expected_tensor_sizes)
        ):
            if expected == 0:
                continue
            try:
                actual = int(np.size(tensor))
            except (TypeError, ValueError, AttributeError):
                # Non-array-like tensor argument (e.g. a scalar passed by mistake);
                # skip rather than raise so the kernel call surfaces the real
                # type error.
                continue
            if actual != expected:
                param_name = (
                    self.tensor_params[i]
                    if i < len(self.tensor_params)
                    else f"arg[{i}]"
                )
                raise RuntimeError(
                    f"Tensor argument {param_name!r} has {actual} elements but "
                    f"the kernel was compiled for {expected} elements.\n"
                    f"CompileTime[T] parameters used at compile time: "
                    f"{self.compile_kwargs!r}"
                )

    def to_json(self) -> str:
        """Serialise the non-callable parts of this design to JSON.

        The generator callable itself cannot be serialised; callers must
        supply it back to ``from_json``.

        Note:
            The on-the-wire format is internal to ``CompilableDesign``;
            ``compile_kwargs`` are encoded as ``[type, value]`` pairs (not a
            dict).  Do not treat the output as a stable public schema.
        """
        data = {
            "generator_name": self.generator_name,
            "use_cache": self.use_cache,
            "compile_kwargs": {
                k: _encode_kwarg(v) for k, v in self.compile_kwargs.items()
            },
            "compile_flags": self.compile_flags,
            "source_files": [str(sf) for sf in self.source_files],
            "include_paths": [str(p) for p in self.include_paths],
            "aiecc_flags": self.aiecc_flags,
            "object_files": [str(of) for of in self.object_files],
            "cache_hash": self._compute_cache_hash(),
        }
        return json.dumps(data)

    @classmethod
    def from_json(
        cls, json_str: str, generator: Callable | None = None
    ) -> CompilableDesign:
        """Deserialise a ``CompilableDesign`` from JSON.

        Args:
            json_str: JSON string produced by ``to_json()``.
            generator: The original callable (required unless ``mlir_generator``
                is a ``.mlir`` file path encoded in the JSON).

        Note:
            ``compile_kwargs`` values for the types ``int``, ``float``,
            ``str``, and ``bool`` are round-tripped exactly.  Values of other
            types are stored as strings and decoded as strings.
        """
        data = json.loads(json_str)
        if generator is None:
            raise ValueError(
                "generator must be supplied to CompilableDesign.from_json() "
                "because callables cannot be serialised."
            )
        return cls(
            mlir_generator=generator,
            use_cache=data.get("use_cache", True),
            compile_kwargs={
                k: _decode_kwarg(v) for k, v in data.get("compile_kwargs", {}).items()
            },
            compile_flags=data.get("compile_flags", []),
            source_files=data.get("source_files", []),
            include_paths=data.get("include_paths", []),
            aiecc_flags=data.get("aiecc_flags", []),
            object_files=data.get("object_files", []),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def generator_name(self) -> str:
        """Human-readable name for the generator (function name or .mlir path)."""
        if isinstance(self.mlir_generator, Path):
            return str(self.mlir_generator)
        return getattr(self.mlir_generator, "__name__", repr(self.mlir_generator))

    @property
    def recipe_hash(self) -> str:
        """Hash of the design recipe: generator + CompileTime[T] kwargs + flags.

        Stable across rebuilds; identifies the *what* of compilation. Two
        designs with equal ``recipe_hash`` produce identical MLIR.
        """
        return _compute_recipe_hash(
            self.mlir_generator,
            self.compile_kwargs,
            self.aiecc_flags,
            self.compile_flags,
        )

    @property
    def artifact_hash(self) -> str:
        """Hash of the build environment: source/object mtimes + tool mtimes + arch.

        Changes whenever a kernel ``.cc``, an ``.o``, Peano, aiecc, or the
        target arch changes; identifies the *with what* of compilation.
        """
        return _compute_artifact_hash(
            self.mlir_generator,
            self.source_files,
            self.object_files,
        )

    def _compute_cache_hash(self) -> str:
        return _compute_hash(
            self.mlir_generator,
            self.compile_kwargs,
            self.source_files,
            self.object_files,
            self.aiecc_flags,
            self.compile_flags,
        )

    @functools.cached_property
    def _generated(self) -> tuple[str, list]:
        """Run the generator once and cache ``(mlir_text, external_kernels)``.

        Generation (executing the user's MLIR builder) is the dominant
        in-process cost on a cold-disk cache path. Caching the *text* — not
        the Module — is safe across MLIR Contexts: the Module is bound to the
        Context that built it, but the text is not, so each ``_generate_mlir``
        call can re-parse into the live Context and return a fresh Module.

        ExternalFunction instances are Python objects with no Context
        affinity; cached directly. Their ``_compiled`` flag survives reuse,
        so a second ``compile()`` correctly skips already-built kernels.

        Safe to memoize because :class:`CompilableDesign` inputs are frozen
        at ``__init__`` (``compile_kwargs`` is a ``MappingProxyType``; the
        list fields are tuples), so the generated MLIR is a pure function
        of ``self``.
        """
        from aie.iron.kernel import ExternalFunction

        if isinstance(self.mlir_generator, Path):
            # Static .mlir file: text already on disk; no kernels to collect.
            return self.mlir_generator.read_text(), []

        hints = self._hints

        # Guard 2-A: compile_kwargs must not contain tensor param names.
        tensor_names = set(self.tensor_params)
        confused_tensor_keys = set(self.compile_kwargs.keys()) & tensor_names
        if confused_tensor_keys:
            raise TypeError(
                f"CompilableDesign for {self.generator_name!r}: "
                f"compile_kwargs contains name(s) annotated as runtime tensors "
                f"(In/Out/InOut), not CompileTime[T] parameters: {confused_tensor_keys}.\n"
                f"  Tensor params must be supplied at call time, not compile time.\n"
                f"  CompileTime[T] params are: {self.compile_params}."
            )

        # Guard 2-B: compile_kwargs must not contain entirely unknown keys.
        known_params = (
            set(self.compile_params) | set(self.tensor_params) | set(self.scalar_params)
        )
        unknown_keys = set(self.compile_kwargs.keys()) - known_params
        if unknown_keys:
            raise TypeError(
                f"CompilableDesign for {self.generator_name!r}: "
                f"compile_kwargs contains key(s) not in the generator signature: "
                f"{unknown_keys}.\n"
                f"  Valid CompileTime[T] params are: {self.compile_params}."
            )

        sig = self._sig
        assert sig is not None
        compile_only_params = {
            name: p
            for name, p in sig.parameters.items()
            if _is_compile_param(hints.get(name, p.annotation))
        }
        compile_only_sig = inspect.Signature(
            parameters=list(compile_only_params.values())
        )
        try:
            compile_only_sig.bind(**self.compile_kwargs)
        except TypeError as exc:
            raise TypeError(
                f"CompilableDesign for '{self.generator_name}': "
                f"compile_kwargs do not match CompileTime[T] parameters — {exc}"
            ) from exc

        ExternalFunction._instances.clear()

        _tensor_placeholders = {
            name: _TensorPlaceholder(name) for name in self.tensor_params
        }
        _gen_call_kwargs = {**_tensor_placeholders, **self.compile_kwargs}

        # Re-register any ExternalFunction instances passed as CompileTime[T] params
        # so the generator's kernel-call paths see them already-registered.
        for _v in _gen_call_kwargs.values():
            if isinstance(_v, ExternalFunction):
                ExternalFunction._instances.add(_v)

        with compile_context(**self.compile_kwargs):
            with mlir_mod_ctx() as ctx:  # pyright: ignore[reportGeneralTypeIssues]
                result = self.mlir_generator(**_gen_call_kwargs)
                module = ctx.module if result is None else result
                if not module.operation.verify():
                    raise RuntimeError(
                        f"MLIR verification failed for '{self.generator_name}'"
                    )
                mlir_text = str(module)

        external_kernels = list(ExternalFunction._instances)
        ExternalFunction._instances.clear()
        return mlir_text, external_kernels

    def _generate_mlir(self, ExternalFunction):
        """Return an MLIR ``Module`` bound to a fresh Context.

        Thin wrapper over :attr:`_generated`: parse the cached MLIR text into
        a new ``mlir_mod_ctx()`` and re-register the cached ``ExternalFunction``
        instances so ``compile()`` can collect them.
        """
        mlir_text, external_kernels = self._generated
        ExternalFunction._instances.update(external_kernels)
        with mlir_mod_ctx():  # pyright: ignore[reportGeneralTypeIssues]
            return _Module.parse(mlir_text)

    def __hash__(self) -> int:
        # Fold the 96-bit cache hash down to a signed 64-bit Python hash.
        h = int(self._compute_cache_hash(), 16)
        # Wrap to the range [-(2^63), 2^63-1] by taking modulo and adjusting.
        bits = sys.hash_info.width  # typically 64
        mask = (1 << bits) - 1
        h = h & mask
        if h >= (1 << (bits - 1)):
            h -= 1 << bits
        return h if h != -1 else -2  # -1 is reserved by CPython

    def __repr__(self) -> str:
        return (
            f"CompilableDesign(generator={self.generator_name!r}, "
            f"compile_kwargs={self.compile_kwargs!r})"
        )
