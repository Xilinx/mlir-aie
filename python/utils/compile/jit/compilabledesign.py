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
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import os
import sys
import typing
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, get_args, get_origin

from aie.utils.compile import (
    NPU_CACHE_HOME,
    compile_external_kernel,
    compile_mlir_module,
)
from aie.utils.compile.cache.utils import file_lock
from aie.utils.compile.utils import _cleanup_failed_compilation
from aie.extras.context import mlir_mod_ctx

from ._dma_size_parser import parse_dma_sizes
from .context import compile_context
from .markers import Compile, In, InOut, Out

logger = logging.getLogger(__name__)

_PRIMITIVE_TYPES = (int, float, str, bool, bytes)

_KWARG_TYPE_MAP = {"bool": bool, "int": int, "float": float, "str": str}


def _encode_kwarg(value: Any) -> Any:
    """Encode a compile_kwarg value as [typename, value] for JSON storage."""
    if isinstance(value, bool):  # must check bool before int
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, float):
        return ["float", value]
    if isinstance(value, str):
        return ["str", value]
    return ["str", str(value)]


def _decode_kwarg(encoded: Any) -> Any:
    """Decode a compile_kwarg value from JSON storage."""
    if not isinstance(encoded, list) or len(encoded) != 2:
        return encoded  # legacy plain value or unknown format
    t, v = encoded
    converter = _KWARG_TYPE_MAP.get(t, str)
    return converter(v)


class _TensorPlaceholder:
    """Sentinel for In/Out/InOut params during MLIR generation; any attribute
    access raises so generator bodies can't read tensor shape/dtype at compile
    time (use ``Compile[T]`` for that instead)."""

    def __init__(self, param_name: str) -> None:
        object.__setattr__(self, "_param_name", param_name)

    def _raise(self, op: str = "") -> None:
        name = object.__getattribute__(self, "_param_name")
        suffix = f": {op}" if op else ""
        raise RuntimeError(
            f"Generator parameter {name!r} is a runtime tensor (In/Out/InOut) "
            f"and is not available at compile time{suffix}. "
            f"Use Compile[T] parameters for shape/dtype information instead."
        )

    def __getattr__(self, name: str):
        self._raise(f".{name}")

    def __setattr__(self, name: str, value) -> None:
        self._raise(f".{name} = ...")

    def __getitem__(self, key):
        self._raise(f"[{key!r}]")

    def __repr__(self) -> str:
        name = object.__getattribute__(self, "_param_name")
        return f"<_TensorPlaceholder for {name!r}>"


# Sentinel: annotation origins that represent runtime tensor directions.
_TENSOR_ANNOTATIONS = (In, Out, InOut)


def _is_compile_param(annotation) -> bool:
    """Return True for ``Compile[T]`` or ``Optional[Compile[T]]``."""
    if annotation is Compile:
        return True
    origin = get_origin(annotation)
    if origin is Compile:
        return True
    # get_type_hints rewrites `Compile[T] = None` defaults to Optional[...].
    if origin is typing.Union:
        return any(_is_compile_param(arg) for arg in get_args(annotation))
    return False


def _is_tensor_param(annotation) -> bool:
    """Return True if *annotation* is ``In``, ``Out``, or ``InOut``."""
    return annotation in _TENSOR_ANNOTATIONS


@functools.lru_cache(maxsize=None)
def _introspect_generator(generator: Callable):
    """Memoise ``(hints, signature, (compile, tensor, scalar))`` for a generator.

    All three derived values are pure functions of the generator's source —
    the same answer for every call of the same function object — but
    ``typing.get_type_hints`` and ``inspect.signature`` together cost
    ~40us per invocation.  Memoising shaves both ``split_params`` and
    ``CompilableDesign.split_runtime_args`` to a dict lookup on the JIT
    hot path.

    The returned param tuples are immutable; callers that need lists copy.
    """
    try:
        hints = typing.get_type_hints(generator)
    except Exception as exc:
        logger.debug("get_type_hints failed for %r: %s", generator, exc)
        hints = {}

    sig = inspect.signature(generator)
    compile_params: list[str] = []
    tensor_params: list[str] = []
    scalar_params: list[str] = []
    for name, param in sig.parameters.items():
        ann = hints.get(name, param.annotation)
        if ann is inspect.Parameter.empty:
            scalar_params.append(name)
        elif _is_compile_param(ann):
            compile_params.append(name)
        elif _is_tensor_param(ann):
            tensor_params.append(name)
        else:
            scalar_params.append(name)

    return (
        hints,
        sig,
        (
            tuple(compile_params),
            tuple(tensor_params),
            tuple(scalar_params),
        ),
    )


def split_params(generator: Callable) -> tuple[list[str], list[str], list[str]]:
    """Inspect *generator* and return (compile_params, tensor_params, scalar_params).

    * ``compile_params``  — names with ``Compile[T]`` annotation
    * ``tensor_params``   — names with ``In``/``Out``/``InOut`` annotation (in order)
    * ``scalar_params``   — names with any other annotation (runtime scalars)

    Uses ``typing.get_type_hints()`` so that stringified annotations (produced
    by ``from __future__ import annotations`` or PEP 563 mode) are evaluated
    correctly.  Falls back to ``inspect.signature`` annotations on any error
    (e.g. when the generator's globals are not resolvable at call time).
    """
    _, _, (cp, tp, sp) = _introspect_generator(generator)
    return list(cp), list(tp), list(sp)


def _compute_recipe_hash(
    generator: Callable | Path,
    compile_kwargs: Mapping[str, Any],
    aiecc_flags: list[str] | tuple[str, ...],
    compile_flags: list[str] | tuple[str, ...],
) -> str:
    """Hash of the "recipe": generator bytecode + Compile[T] kwargs + flags.

    Pure function of the design specification; does not touch the filesystem
    or environment. Two CompilableDesigns with the same recipe_hash will
    produce identical MLIR (modulo nondeterminism in the generator body).
    """
    h = hashlib.sha256()

    if isinstance(generator, Path):
        h.update(str(generator).encode())
        try:
            h.update(str(generator.stat().st_mtime).encode())
        except FileNotFoundError:
            pass
    else:
        code = generator.__code__
        h.update(code.co_code)
        h.update(repr(code.co_consts).encode())
        h.update(getattr(generator, "__qualname__", "").encode())
        h.update(getattr(generator, "__module__", "").encode())

    def _kwarg_repr(v):
        if callable(v) and hasattr(v, "__code__"):
            code = v.__code__
            closure = (
                tuple(c.cell_contents for c in v.__closure__) if v.__closure__ else None
            )
            try:
                closure_repr = repr(closure)
            except Exception:
                closure_repr = "<unhashable closure>"
            return (
                "fn:",
                bytes(code.co_code).hex(),
                repr(code.co_consts),
                repr(getattr(v, "__defaults__", None)),
                closure_repr,
            )
        return str(v)

    try:
        kwargs_json = json.dumps(
            {k: _kwarg_repr(v) for k, v in sorted(compile_kwargs.items())}
        ).encode()
    except (TypeError, ValueError):
        kwargs_json = repr(sorted(compile_kwargs.items())).encode()
    h.update(kwargs_json)

    h.update(repr(sorted(aiecc_flags)).encode())
    h.update(repr(sorted(compile_flags)).encode())

    return h.hexdigest()


def _compute_artifact_hash(
    generator: Callable | Path,
    source_files: list[Path] | tuple[Path, ...],
    object_files: list[Path] | tuple[Path, ...],
) -> str:
    """Hash of the "artifacts": source/object mtimes + tool mtimes + target arch.

    Captures everything that can change the *output* of compilation without
    changing the *recipe*: edited C++ kernels, swapped object files, upgraded
    Peano / aiecc, retargeted device.
    """
    h = hashlib.sha256()

    for sf in sorted(source_files, key=str):
        h.update(str(sf).encode())
        try:
            h.update(str(Path(sf).stat().st_mtime).encode())
        except (FileNotFoundError, OSError):
            pass

    for of in sorted(object_files, key=str):
        h.update(str(of).encode())
        try:
            h.update(str(Path(of).stat().st_mtime).encode())
        except (FileNotFoundError, OSError):
            pass

    # Static .mlir is arch-agnostic; compiled kernels need a target identifier.
    # Missing components collapse to a constant + WARNING log so cross-arch cache
    # collisions surface instead of silently aliasing.
    if not isinstance(generator, Path):
        try:
            import aie.iron as _iron
            from aie.utils import DefaultNPURuntime
            from aie.utils.compile.utils import resolve_target_arch

            try:
                device = _iron.get_current_device()
            except (RuntimeError, AttributeError):
                device = (
                    DefaultNPURuntime.device()
                    if DefaultNPURuntime is not None
                    else None
                )
            target_arch = resolve_target_arch(device)
        except (ImportError, AttributeError, RuntimeError, ValueError) as exc:
            logger.warning(
                "_compute_artifact_hash: target_arch unresolved (%s); using 'unknown'",
                exc,
            )
            target_arch = "unknown"

        try:
            from aie.utils import config as _config

            peano_cxx = _config.peano_cxx_path()
            peano_mtime = str(Path(peano_cxx).stat().st_mtime)
        except (
            ImportError,
            AttributeError,
            FileNotFoundError,
            OSError,
            RuntimeError,
        ) as exc:
            try:
                from aie.utils import config as _config

                peano_mtime = f"path:{_config.peano_install_dir()}"
                logger.warning(
                    "_compute_artifact_hash: peano cxx unavailable (%s); "
                    "keying on install dir path only",
                    exc,
                )
            except (ImportError, AttributeError, RuntimeError) as exc2:
                logger.warning("_compute_artifact_hash: peano absent (%s)", exc2)
                peano_mtime = "absent"

        try:
            import shutil as _shutil

            _aiecc_path = _shutil.which("aiecc")
            aiecc_mtime = (
                str(Path(_aiecc_path).stat().st_mtime) if _aiecc_path else "absent"
            )
        except (FileNotFoundError, OSError) as exc:
            logger.warning("_compute_artifact_hash: aiecc absent (%s)", exc)
            aiecc_mtime = "absent"

        h.update(
            f"target_arch={target_arch}|peano_mtime={peano_mtime}|aiecc_mtime={aiecc_mtime}".encode()
        )

    return h.hexdigest()


def _compute_hash(
    generator: Callable | Path,
    compile_kwargs: Mapping[str, Any],
    source_files: list[Path] | tuple[Path, ...],
    object_files: list[Path] | tuple[Path, ...],
    aiecc_flags: list[str] | tuple[str, ...],
    compile_flags: list[str] | tuple[str, ...],
) -> str:
    """Stable 24-hex SHA-256 cache key combining recipe + artifact hashes."""
    recipe = _compute_recipe_hash(generator, compile_kwargs, aiecc_flags, compile_flags)
    artifact = _compute_artifact_hash(generator, source_files, object_files)
    return hashlib.sha256(f"{recipe}|{artifact}".encode()).hexdigest()[:24]


class CompilableDesign:
    """Bundles an MLIR generator with compile-time parameters.

    Args:
        mlir_generator: A callable that accepts ``Compile[T]`` kwargs and
            either returns an MLIR module (e.g., built inside an
            ``mlir_mod_ctx()`` block) or returns ``None`` after building the
            module into the active MLIR context (e.g., via
            ``Program(...).resolve_program()``),
            OR a ``pathlib.Path`` to a pre-written ``.mlir`` file.
        use_cache: When ``True`` (default), a file-system cache keyed by the
            bytecode+kwargs hash is consulted before recompiling.
        compile_kwargs: Values for the ``Compile[T]``-annotated parameters.
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
        """Return a new ``CompilableDesign`` with additional ``Compile[T]`` kwargs bound.

        The given kwargs are merged onto ``self.compile_kwargs`` with call-time
        values winning.  All other config (``source_files``, ``aiecc_flags``,
        ``include_paths``, etc.) is preserved.
        """
        return CompilableDesign(
            self.mlir_generator,
            compile_kwargs={**self.compile_kwargs, **compile_kwargs},
            use_cache=self.use_cache,
            compile_flags=self.compile_flags,
            source_files=self.source_files,
            include_paths=self.include_paths,
            aiecc_flags=self.aiecc_flags,
            object_files=self.object_files,
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

        if elf_path is not None and not explicit_paths:
            raise ValueError(
                "compile(): elf_path requires explicit xclbin_path + inst_path "
                "(the JIT cache does not track ELF artifacts)."
            )

        if explicit_paths:
            # Absolutize so compile_external_kernel's `cwd=kernel_dir` doesn't
            # turn relative paths into "build/build/foo.cc" etc.
            xclbin_path = Path(xclbin_path).resolve()
            inst_path = Path(inst_path).resolve()
            if elf_path is not None:
                elf_path = Path(elf_path).resolve()
            kernel_dir = xclbin_path.parent
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
                        "toolchain consistently across all kernels.X helper "
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
                    expected_outputs.append(elf_path)
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
            runtime_args = [a for a in runtime_args if not isinstance(a, Kernel)]
            return runtime_args, runtime_kwargs

        tensor_args = []
        scalar_kwargs = dict(runtime_kwargs)

        # Reuse the cached intro from __init__ — same generator, same hints/sig.
        hints = self._hints
        sig = self._sig
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
        (matmul B reloaded each tile_row), and InOut buffers (for_each_typed
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
                    f"Compile[T] parameters used at compile time: "
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
        """Hash of the design recipe: generator + Compile[T] kwargs + flags.

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

    def _generate_mlir(self, ExternalFunction):
        """Call the generator (or read the .mlir file) and return the MLIR module."""
        if isinstance(self.mlir_generator, Path):
            # Static MLIR file.
            mlir_path = self.mlir_generator
            with mlir_mod_ctx() as ctx:
                ctx.module.parse(mlir_path.read_text())
            return ctx.module

        # Validate that all Compile[T] params are supplied.
        # Reuse the cached intro from __init__ — same generator, same hints.
        hints = self._hints

        # Guard 2-A: compile_kwargs must not contain tensor param names.
        tensor_names = set(self.tensor_params)
        confused_tensor_keys = set(self.compile_kwargs.keys()) & tensor_names
        if confused_tensor_keys:
            raise TypeError(
                f"CompilableDesign for {self.generator_name!r}: "
                f"compile_kwargs contains name(s) annotated as runtime tensors "
                f"(In/Out/InOut), not Compile[T] parameters: {confused_tensor_keys}.\n"
                f"  Tensor params must be supplied at call time, not compile time.\n"
                f"  Compile[T] params are: {self.compile_params}."
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
                f"  Valid Compile[T] params are: {self.compile_params}."
            )

        sig = self._sig
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
                f"compile_kwargs do not match Compile[T] parameters — {exc}"
            ) from exc

        # Clear stale ExternalFunction instances before generation.
        ExternalFunction._instances.clear()

        # Build the call kwargs: Compile[T] params from compile_kwargs,
        # plus None placeholders for In/Out/InOut params (which are not
        # available at compile time — the generator must not read them).
        _tensor_placeholders = {
            name: _TensorPlaceholder(name) for name in self.tensor_params
        }
        _gen_call_kwargs = {**_tensor_placeholders, **self.compile_kwargs}

        # Re-register any ExternalFunction instances passed as Compile[T] params
        # so that compile() collects them for compilation after generation returns.
        for _v in _gen_call_kwargs.values():
            if isinstance(_v, ExternalFunction):
                ExternalFunction._instances.add(_v)

        with compile_context(**self.compile_kwargs):
            with mlir_mod_ctx() as ctx:
                result = self.mlir_generator(**_gen_call_kwargs)

        module = ctx.module if result is None else result
        if not module.operation.verify():
            raise RuntimeError(f"MLIR verification failed for '{self.generator_name}'")
        return module

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
