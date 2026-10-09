# _object_cache.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Compiled kernel objects, shared across JIT cache entries.

A design's cache entry is keyed on the whole design, so a kernel used by two
designs -- or by one design at two sizes -- was compiled once per design
directory. What determines an object's bytes is much narrower: the
``KernelObject`` recipe, the target, the design's include paths, whether IR is
retained, and the compiler and object tools. Objects are keyed on those inputs,
built once under ``<root>/<key>/``, and copied into each design directory that
links them.

An entry records the inputs its compile read (see ``_manifest``) and is rebuilt
when one changes, so a header edit reaches every object that included it. The
copied depfile lets the design's own manifest record the same inputs.

**Peano only.** xchesscc reports no inputs, so a Chess object's headers cannot
be checked; one shared across designs would stay stale for all of them. Chess
kernels keep compiling into the design directory.

A kernel library object is compiled against ``aie_api/aie.hpp`` precompiled
(see ``PrecompiledAieApi``), built once per target and flag set under
``<root>/pch/<key>/``: parsing that header is nearly all of such a compile.
"""

from __future__ import annotations

import glob
import hashlib
import logging
import os
import re
import subprocess
import threading
from pathlib import Path

from aie.utils import config
from aie.utils.compile.cache.utils import file_lock
from aie.utils.compile.utils import (
    _cleanup_failed_compilation,
    _compile_external_kernel,
    _copy_source,
    cxx_core_compile_command,
)

from . import _manifest
from ._hash import _tool_identity

logger = logging.getLogger(__name__)

_PREFIX = "aie_api.hpp"
_PCH = "aie_api.pch"
_IDENTIFIER = re.compile(rb"[A-Za-z_]\w*")
_DIRECTIVE = re.compile(rb"^[ \t]*#[ \t]*(\w+)(.*)$", re.M)
_DEFINE = re.compile(rb"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)", re.M)
_INCLUDED = re.compile(
    rb"(?:^[ \t]*#[ \t]*(?:include_next|include|import)[ \t]*"
    rb"|__has_include(?:_next)?[ \t]*\([ \t]*)"
    rb"([<\"][^>\"\n]*|[A-Za-z_])",
    re.M,
)
_MACRO = re.compile(rb"^[ \t]*#[ \t]*define[ \t]+(\w+)(?:\(([^)]*)\))?(.*)$", re.M)
# A header included by a macro's expansion could be named anything.
_ANY = b"*"
_CONDITIONALS = {b"if", b"ifdef", b"ifndef", b"elif", b"elifdef", b"elifndef"}
# Statement pragmas: they cannot sit before a header's declarations and reach them.
_LOCAL_PRAGMAS = {(b"once",), (b"clang", b"loop"), (b"unroll",), (b"nounroll",)}

# Each entry is loaded once a process: clang refuses a PCH whose inputs changed
# since, which the cache takes as any failed compile against it.
_loaded: dict[Path, PrecompiledAieApi | None] = {}
_loading: dict[Path, threading.Lock] = {}
_loading_lock = threading.Lock()


def _invocations(text: bytes, names: set[bytes]) -> list[bytes]:
    """The argument text of each call in ``text`` of a macro in ``names``."""
    pattern = re.compile(rb"\b(?:" + b"|".join(map(re.escape, names)) + rb")\s*\(")
    found = []
    for m in pattern.finditer(text):
        depth, end = 1, m.end()
        while end < len(text) and depth:
            depth += {ord("("): 1, ord(")"): -1}.get(text[end], 0)
            end += 1
        found.append(text[m.end() : end - 1])
    return found


def _pragma_kind(arg: bytes, params: list[bytes]) -> str | None:
    """Classify a ``_Pragma`` argument, looking through one stringizing call.

    Returns:
        "local" for a statement's pragma, "forwarded" for one of ``params``
        handed on, else None.
    """
    words = _IDENTIFIER.findall(arg)
    for w in (words, words[1:]):
        if tuple(w[:1]) in _LOCAL_PRAGMAS or tuple(w[:2]) in _LOCAL_PRAGMAS:
            return "local"
        if len(w) == 1 and w[0] in params:
            return "forwarded"
    return None


def _file_id(path) -> tuple[int, int]:
    st = os.stat(path)
    return st.st_dev, st.st_ino


def _key(func, target_arch, include_dirs, embed_bitcode) -> str:
    identity = (
        func.object_file_name,
        func.object_file._source,
        target_arch,
        tuple(str(d) for d in include_dirs or ()),
        embed_bitcode,
        str(config.cxx_header_path()),
        _tool_identity("peano", config.peano_cxx_path),
        _tool_identity("nm", config.nm_path),
        _tool_identity("objcopy", config.objcopy_path),
    )
    return hashlib.sha256(repr(identity).encode()).hexdigest()[:24]


class PrecompiledAieApi:
    """``aie_api/aie.hpp`` precompiled for one target and flag set.

    A kernel compiled with ``-include-pch`` reads the header before its own
    first line rather than where it includes it. That builds the same object
    only while neither side reaches the other, which the cache checks: the
    header is built with every command-line macro it names, no include dir of
    the kernel holds a name the header includes or probes (``shadowed_by``),
    and no file of the kernel's own defines a name the header reads, tests a
    macro it defines or sets a pragma beyond a statement's (``serves``).
    """

    def __init__(self, entry: Path):
        self.entry = entry
        self.path = entry / _PCH
        self.files = {
            _file_id(entry / p) for p in _manifest._parse_depfile(entry / f"{_PCH}.d")
        }
        self.identifiers = set((entry / "identifiers").read_bytes().split())
        self.defines = set((entry / "defines").read_bytes().split())
        self.included = (entry / "included").read_bytes().split()

    @staticmethod
    def scan(entry: Path) -> None:
        """Write the identifiers of every file the header read, and the macros it leaves changed.

        The macros are those defined, undefined or redefined between a
        translation unit's first line (``predefined.h``) and the end of the
        header (``defined.h``), both as clang's ``-dM`` lists them.
        """
        identifiers = set()
        included = set()
        for p in _manifest._parse_depfile(entry / f"{_PCH}.d"):
            text = (entry / p).read_bytes().replace(b"\\\n", b" ")
            identifiers.update(_IDENTIFIER.findall(text))
            for path in _INCLUDED.findall(text):
                included.add(path[1:] if path[:1] in b'<"' else _ANY)
        changed = set((entry / "defined.h").read_bytes().splitlines()) ^ set(
            (entry / "predefined.h").read_bytes().splitlines()
        )
        defines = set(_DEFINE.findall(b"\n".join(changed)))
        (entry / "identifiers").write_bytes(b"\n".join(sorted(identifiers)))
        (entry / "defines").write_bytes(b"\n".join(sorted(defines)))
        (entry / "included").write_bytes(b"\n".join(sorted(included)))

    def shadowed_by(self, include_dirs: list[Path]) -> bool:
        """Whether one of ``include_dirs`` could change what the header reads.

        That is, whether one holds a path the header includes or probes with
        ``__has_include``.
        """
        header_dir = _file_id(config.cxx_header_path())
        for d in include_dirs:
            try:
                # Every compile searches the header's dir first; clang drops a repeat of it.
                if _file_id(d) == header_dir:
                    continue
                names = set(os.listdir(os.fsencode(d)))
            except OSError:
                continue
            if names and _ANY in self.included:
                return True
            for path in self.included:
                first = path.split(b"/")[0]
                if (first in names or first == b"..") and os.path.lexists(
                    os.path.join(os.fsencode(d), path)
                ):
                    return True
        return False

    def serves(self, kernel_dir: Path, depfile: Path) -> bool:
        """Whether the compile that wrote ``depfile`` built what it would have without the header.

        Its own files are what it read beyond the header's. One of them must
        include the header, so the kernel would have read it anyway.
        """
        own = [
            kernel_dir / p
            for p in _manifest._parse_depfile(depfile)
            if _file_id(kernel_dir / p) not in self.files
        ]
        texts = {path: path.read_bytes().replace(b"\\\n", b" ") for path in own}
        macros = [
            (name, params.replace(b",", b" ").split(), body)
            for text in texts.values()
            for name, params, body in _MACRO.findall(text)
        ]
        # A macro handing its argument to _Pragma: its uses name the pragma.
        pragma = {b"_Pragma"}
        while more := {
            name
            for name, params, body in macros
            if name not in pragma
            and any(
                _pragma_kind(arg, params) == "forwarded"
                for arg in _invocations(body, pragma)
            )
        }:
            pragma |= more
        uses = [
            (arg, params)
            for _, params, body in macros
            for arg in _invocations(body, pragma)
        ] + [
            (arg, [])
            for text in texts.values()
            for arg in _invocations(_MACRO.sub(b"", text), pragma)
        ]
        for arg, params in uses:
            if _pragma_kind(arg, params) is None:
                logger.debug("%s sets _Pragma(%r)", depfile, arg)
                return False
        includes_header = False
        for path, text in texts.items():
            for kind, rest in _DIRECTIVE.findall(text):
                words = rest.split()
                if kind == b"include":
                    includes_header |= rest.strip()[1:-1] == b"aie_api/aie.hpp"
                elif kind in (b"define", b"undef"):
                    name = _IDENTIFIER.match(rest.strip())
                    if name is not None and name.group() in self.identifiers:
                        logger.debug("%s redefines %s", path, name.group())
                        return False
                elif kind == b"pragma":
                    if (
                        tuple(words[:1]) not in _LOCAL_PRAGMAS
                        and tuple(words[:2]) not in _LOCAL_PRAGMAS
                    ):
                        logger.debug("%s sets #pragma %s", path, rest.strip())
                        return False
                elif kind in _CONDITIONALS:
                    tested = set(_IDENTIFIER.findall(rest)) & self.defines
                    if tested:
                        logger.debug("%s tests %s", path, sorted(tested))
                        return False
        return includes_header


class KernelObjectCache:
    """Content-addressed store of compiled kernel objects under ``root``."""

    def __init__(self, root: Path, lock_timeout_seconds: int):
        self.root = Path(root).absolute()
        self.lock_timeout_seconds = lock_timeout_seconds

    @staticmethod
    def accepts(func) -> bool:
        """Whether the cache holds ``func``: not a Chess kernel, nor one with no source recipe."""
        recipe = getattr(getattr(func, "object_file", None), "_source", None)
        return recipe is not None and not recipe.use_chess

    def fetch(self, func, kernel_dir, target_arch, include_dirs, embed_bitcode) -> bool:
        """Copy ``func``'s object into ``kernel_dir``, compiling it on a miss.

        Returns False, touching nothing, for a kernel the cache does not hold
        (see ``accepts``).
        """
        if not self.accepts(func):
            return False
        entry = self.root / _key(func, target_arch, include_dirs, embed_bitcode)
        obj = entry / func.object_file_name
        with file_lock(entry / ".lock", timeout_seconds=self.lock_timeout_seconds):
            if obj.is_file() and _manifest.is_valid(entry):
                logger.debug("Kernel object cache hit for %s (%s)", obj.name, entry)
            else:
                logger.debug("Kernel object cache miss for %s (%s)", obj.name, entry)
                _cleanup_failed_compilation(entry)
                try:
                    pch = self.precompiled_header(
                        func, entry, target_arch, include_dirs, embed_bitcode
                    )
                    built = False
                    if pch is not None:
                        try:
                            _compile_external_kernel(
                                func,
                                str(entry),
                                target_arch,
                                include_dirs,
                                embed_bitcode,
                                precompiled_header=str(pch.path),
                            )
                            built = pch.serves(entry, entry / f"{obj.name}.d")
                        except RuntimeError as e:
                            logger.debug("%s against %s: %s", obj.name, pch.path, e)
                        if not built:
                            _cleanup_failed_compilation(entry)
                    if not built:
                        _compile_external_kernel(
                            func, str(entry), target_arch, include_dirs, embed_bitcode
                        )
                    _manifest.record(entry, [func], [])
                except BaseException:
                    _cleanup_failed_compilation(entry)
                    raise
            stamps = entry.glob(f"{glob.escape(obj.name)}.prefix_state.*.json")
            for artifact in (obj, entry / f"{obj.name}.d", *stamps):
                if artifact.is_file():
                    _copy_source(os.path.join(kernel_dir, artifact.name), str(artifact))
        return True

    def precompiled_header(
        self, func, entry, target_arch, include_dirs, embed_bitcode
    ) -> PrecompiledAieApi | None:
        """The ``aie_api`` PCH ``func``'s object can be compiled against, if any.

        Only for a kernel library source compiled to a plain object: the
        library is written to read the header before anything of its own,
        which ``PrecompiledAieApi.serves`` then checks.
        """
        source = func.source_file
        if source is None or embed_bitcode or func._inline:
            return None
        library = os.path.realpath(config.aie_kernels_dir())
        if os.path.commonpath([library, os.path.realpath(source)]) != library:
            return None
        flags = func.compile_flags
        if any(
            f in ("-D", "-U", "-I") or f.startswith(("-include", "-imacros"))
            for f in flags
        ):
            return None
        macros = [f for f in flags if f.startswith(("-D", "-U"))]
        flag_dirs = [f[2:] for f in flags if f.startswith("-I")]
        others = [f for f in flags if not f.startswith(("-D", "-U", "-I"))]
        read: list[str] = []
        for _ in range(len(macros) + 1):
            pch = self._aie_api(target_arch, [*others, *read])
            if pch is None:
                return None
            named = [
                f for f in macros if f[2:].split("=", 1)[0].encode() in pch.identifiers
            ]
            if named == read:
                break
            read = named
        else:
            return None
        # The kernel's source is copied into the entry, its quoted includes' first dir.
        dirs = [
            ".",
            *func.include_dirs,
            *flag_dirs,
            os.path.dirname(os.path.abspath(source)),
        ]
        if pch.shadowed_by([entry / d for d in (*dirs, *(include_dirs or ()))]):
            logger.debug("an include dir of %s shadows aie_api", func.object_file_name)
            return None
        return pch

    def _aie_api(self, target_arch, compile_args) -> PrecompiledAieApi | None:
        cmd = cxx_core_compile_command(
            _PREFIX, target_arch, _PCH, compile_args=compile_args
        )
        cmd[1:1] = ["-x", "c++-header"]
        identity = (cmd, _tool_identity("peano", config.peano_cxx_path))
        entry = (
            self.root / "pch" / hashlib.sha256(repr(identity).encode()).hexdigest()[:24]
        )
        if entry in _loaded:
            return _loaded[entry]
        with _loading_lock:
            loading = _loading.setdefault(entry, threading.Lock())
        # Waiting on a cold header beats compiling plainly: a kernel's parse of
        # aie_api is most of its compile, and the header takes about one parse.
        with loading:
            if entry not in _loaded:
                try:
                    _loaded[entry] = self._load(entry, cmd, target_arch, compile_args)
                except TimeoutError:
                    return None
            return _loaded[entry]

    def _load(self, entry, cmd, target_arch, compile_args) -> PrecompiledAieApi | None:
        with file_lock(entry / ".lock", timeout_seconds=self.lock_timeout_seconds):
            if not ((entry / _PCH).is_file() and _manifest.is_valid(entry)):
                _cleanup_failed_compilation(entry)
                (entry / _PREFIX).write_text("#include <aie_api/aie.hpp>\n")
                (entry / "empty.hpp").write_text("")
                steps = [cmd]
                for source, macros in (
                    (_PREFIX, "defined.h"),
                    ("empty.hpp", "predefined.h"),
                ):
                    step = cxx_core_compile_command(
                        source, target_arch, macros, compile_args=compile_args
                    )
                    step[1:1] = ["-x", "c++", "-E", "-dM"]
                    step.append("-Wno-unused-command-line-argument")
                    steps.append(step)
                for step in steps:
                    ret = subprocess.run(step, cwd=entry, capture_output=True)
                    if ret.returncode:
                        logger.debug("aie_api PCH in %s: %s", entry, ret.stderr)
                        _cleanup_failed_compilation(entry)
                        return None
                PrecompiledAieApi.scan(entry)
                _manifest.record(entry, [], [])
        return PrecompiledAieApi(entry)
