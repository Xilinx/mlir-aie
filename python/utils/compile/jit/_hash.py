# _hash.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Content-addressed hashing for the JIT cache.

Two halves so callers can distinguish "recipe changed" from "rebuild needed":

* `_compute_recipe_hash`   — generator identity + the Python it reaches +
  compile_kwargs + aiecc/compile flags. Target-independent design identity.
* `_compute_artifact_hash` — source / object content + tool mtimes +
  target device.  Captures things that change the *output* of compilation
  without changing the *recipe*.

Design inputs (sources, objects, a `Path` generator) are identified by their
**content**.  mtime is not a property of a file, it is a property of how the
file arrived: a fresh clone, a `pip install`, a `cp` or a `touch` all restamp it
without changing a byte, and restoring one hides a change that did happen.  Tool
identity stays on mtime, which is cheap and moves whenever the toolchain is
rebuilt or reinstalled.

The Python a generator reaches is identified by what is running, which is
not always what is on disk, and cheaply, since the key is taken on every
call. A module imported from a file is named by its source's content when
the key first reaches that import, read once per import; a reload looks
again. The script
being run counts as that kind of module, under the name it is imported as.
Code with no file behind it (a notebook cell, the REPL) is identified by its
code and the plain constants it reads -- through the mutable containers
holding them as well, which are digested member by member, each object's
content once per walk and recorded at every position it is read at -- since
those can change without any file changing.

A generator that reconfigures the state it reads (a full-ELF design appending
each configuration it learns to a module-level list, say) would otherwise
never find its own entry: every run moves the key.  Its mutations are told
apart from the user's by when they happen -- ``_mask_generator_mutations``
walks the same roots again after the generator runs and pins what moved, so
later keys read the pinned identities, while a user's edit between compiles
to anything the run did not touch still moves the key and rebuilds.

`_compute_hash` composes both into the 24-hex cache-key
``CompilableDesign`` uses to address ``$NPU_CACHE_HOME``.

Carved out of ``compilabledesign.py`` to keep the main file focused on the
``CompilableDesign`` class itself.
"""

from __future__ import annotations

import dis
import hashlib
import json
import logging
import marshal
import os
import site
import sys
import sysconfig
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import cache, partial
from pathlib import Path
from types import CodeType, FunctionType, MethodType, ModuleType
from typing import Any, Callable, Mapping

from ._introspect import declare

logger = logging.getLogger(__name__)

# Read granularity for content digests.  Bounded so a large input is streamed
# rather than materialised, which keeps MemoryError off this path.
_DIGEST_CHUNK = 1 << 20


def _content_digest(path: Path | str) -> str:
    """Digest a file by content, streamed.

    Returns a marker instead of raising: an input we cannot read is still an
    input, and it must not silently collapse onto the same key as an input we
    can.  The marker embeds the error class so "missing" and "unreadable" stay
    distinguishable, which keeps a later fail-closed change a pure policy edit
    rather than a re-plumbing.
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while chunk := fh.read(_DIGEST_CHUNK):
                h.update(chunk)
    except (OSError, ValueError) as exc:
        return f"<unreadable:{type(exc).__name__}>"
    return h.hexdigest()


def _device_identity_key(device) -> tuple[str, str, str, str]:
    """Return the cache-relevant identity of an IRON device."""
    if device is None:
        return ("none", "", "", "")
    return (
        f"{type(device).__module__}.{type(device).__qualname__}",
        str(getattr(device, "arch", "")),
        str(getattr(device, "cols", "")),
        str(getattr(device, "rows", "")),
    )


def _without_location(const):
    """Return a constant with file and line info dropped, nested code objects included.

    Applied to every constant, not just code objects, so one cannot keep its
    location by sitting inside a tuple.
    """
    if isinstance(const, tuple):
        return tuple(_without_location(c) for c in const)
    if isinstance(const, slice):
        return (Ellipsis, "slice", const.start, const.stop, const.step)
    if not isinstance(const, CodeType):
        return const
    return const.replace(
        co_consts=tuple(_without_location(c) for c in const.co_consts),
        co_filename="",
        co_firstlineno=1,
        co_linetable=b"",
    )


def _code_identity(code: CodeType) -> bytes:
    """Stable bytes for a code object: what marshal writes into a ``.pyc``.

    ``repr()`` of a code object embeds its address, so a key built from it moves
    between processes. marshal covers bytecode, names, varnames, flags and
    constants, recursing into nested code, with no addresses and a hash-seed
    independent frozenset order. ``co_names`` matters: ``matmul_bf16(a, b, c)``
    and ``matmul_i8(a, b, c)`` compile to identical bytecode.

    Location is stripped first: it is not part of the design, and keying on it
    would split the cache per checkout. Format 3 and up write an object a
    second time as a back-reference only when its refcount is above one, so
    before 3.12 made interned names immortal the bytes moved with whatever
    else held a name. Format 2 has no references; it cannot write the
    constant slices 3.14 folds into ``co_consts``, so those go in as tuples.
    """
    return marshal.dumps(_without_location(code), 2)


_PLAIN = (int, float, complex, str, bytes, bool, type(None))

# Mutable containers a generator can read compile inputs through (a
# module-level list of kernel flags, say).
_CONTAINER = (list, tuple, dict, set, frozenset)

# Deepest nesting of containers and instances the key will walk before it
# refuses (see ``_too_deep``).  Bounded so key time cannot grow with an
# arbitrary object graph, and so state that mutates on every call (a memo
# cache hanging off a reachable instance) cannot silently cost a key
# computation each call.  32 is far past any real design input -- flags,
# shapes and configs sit within a handful of levels -- and well inside
# Python's own recursion limit, so an over-deep graph fails with this
# readable error instead of a RecursionError from deep inside the walk.
_MAX_KEY_DEPTH = 32


def _too_deep(where: str) -> ValueError:
    """Return the error for a key that would walk deeper than ``_MAX_KEY_DEPTH``.

    Failing here is the graceful failure: the alternative to a clear error
    is either hashing an unbounded object graph on every call (the cache
    performance problem) or summarising it (a weaker key that can silently
    reuse a stale entry -- the very bug this walk exists to close).  numba
    makes the same call for oversized constants: it refuses to cache rather
    than key what it cannot freeze; JAX rejects non-hashable static inputs
    outright.  None of them publishes a nesting number, so the bound is
    pragmatic, not theoretical.
    """
    return ValueError(
        f"cache-key traversal exceeded the nesting bound of {_MAX_KEY_DEPTH} "
        f"at {where}: flatten the state the generator reads (a plain "
        f"constant, or a digest of it) instead of nesting it deeper."
    )


def _plain(value) -> bool:
    if isinstance(value, tuple):
        return all(_plain(v) for v in value)
    return isinstance(value, _PLAIN)


def _names(code: CodeType) -> set[str]:
    """Return the global and attribute names ``code`` and its nested code use."""
    names, todo = set(), [code]
    while todo:
        c = todo.pop()
        names.update(c.co_names)
        todo.extend(k for k in c.co_consts if isinstance(k, CodeType))
    return names


_STORE_GLOBAL = bytes([dis.opmap["STORE_GLOBAL"]])


def _rebound_globals(module: ModuleType) -> set[str]:
    """Return the globals of ``module`` its own functions assign (``global x``).

    Such a global is state bound after import, as aie.utils binds its default
    NPU runtime on first use; one bound by the module's body, or imported
    into it, is not.
    """
    todo = []
    for value in vars(module).values():
        own = isinstance(value, type) and value.__module__ == module.__name__
        members = vars(value).values() if own else (value,)
        for f in members:
            if isinstance(f, (staticmethod, classmethod)):
                f = f.__func__
            if isinstance(f, FunctionType) and f.__module__ == module.__name__:
                todo.append(f.__code__)
    names = set()
    while todo:
        code = todo.pop()
        todo.extend(k for k in code.co_consts if isinstance(k, CodeType))
        # Opcodes sit at even offsets; a match there is worth disassembling.
        if _STORE_GLOBAL in code.co_code[::2]:
            names.update(
                i.argval
                for i in dis.get_instructions(code)
                if i.opname == "STORE_GLOBAL"
            )
    return names


@cache
def _installed_dirs() -> tuple[Path, ...]:
    paths = sysconfig.get_paths()
    dirs = {paths[k] for k in ("stdlib", "platstdlib", "purelib", "platlib")}
    dirs.add(site.getusersitepackages())
    return tuple(Path(d).resolve() for d in dirs)


@cache
def _design_source(package: str, file: str | None) -> Path | None:
    """Return the source a module is identified by, or None if the key does not follow it.

    That is any Python source but the interpreter's and installed packages',
    whose versions do not move under a design, except mlir-aie's own, which
    decides what the design generates.
    """
    if not file or not file.endswith(".py"):
        return None
    path = Path(file).resolve()
    if package != "aie" and any(path.is_relative_to(d) for d in _installed_dirs()):
        return None
    return path


def _import_name(name: str) -> str:
    """Return the name module ``name`` is imported as.

    A script run by path or with ``-m`` is ``__main__``; it keys as the module
    its spec names, or its file's stem, so running a design's file and
    importing it share a key. A ``__main__`` with no file behind it (the
    REPL, ``-c``, a notebook) keeps its name.
    """
    module = sys.modules.get(name)
    if name != "__main__" or module is None:
        return name
    if module.__spec__ is not None:
        return module.__spec__.name
    file = vars(module).get("__file__")
    return Path(file).stem if file else name


def _stamped_module(value) -> ModuleType | None:
    """Return the module ``value`` is, or comes from, if the key identifies it by its file."""
    if not isinstance(value, ModuleType):
        owner = value if isinstance(value, (type, FunctionType)) else type(value)
        value = sys.modules.get(owner.__module__ or "")
    if value is None:
        return None
    package = value.__name__.partition(".")[0]
    if _design_source(package, vars(value).get("__file__")) is None:
        return None
    return value


# Per import of a file-backed module: the spec it was imported under, its
# source's content digest when the key first reached it, and the modules its
# globals reach. A reload brings a new spec, so the module is looked at again.
_IMPORTS: dict[str, tuple[Any, str, tuple[str, ...]]] = {}


def _import_identity(module: ModuleType) -> tuple[str, tuple[str, ...]]:
    entry = _IMPORTS.get(module.__name__)
    if entry is None or entry[0] is not module.__spec__:
        stamp = _content_digest(module.__file__ or "")
        # Importing a submodule binds it on its package, so a package's
        # namespace grows with whatever else the process imports. Its
        # submodules are reached through what they define, or by the code that
        # names them (see _python_identity). Likewise a global the module's
        # functions rebind holds whatever the process has done so far.
        rebound = _rebound_globals(module)
        reached = {
            _stamped_module(v)
            for k, v in vars(module).items()
            if k not in rebound
            and (
                not isinstance(v, ModuleType) or v.__name__ != f"{module.__name__}.{k}"
            )
        }
        names = tuple(sorted(m.__name__ for m in reached if m is not None))
        entry = _IMPORTS[module.__name__] = (module.__spec__, stamp, names)
    return entry[1], entry[2]


def _python_identity(
    roots,
    frozen: Mapping[int, str] | None = None,
    snapshot: dict[int, str] | None = None,
    obj_snapshot: dict[int, object] | None = None,
) -> bytes:
    """Identify the Python ``roots`` reach: their code, helpers and modules.

    A generator's bytecode names a helper but does not contain it, so the key
    follows globals, closure cells and attributes of modules read by the
    *walked* code (the generator and fileless helpers); a file-backed helper's
    own reads ride its file's stamp.  Functions and
    classes with no file behind them are identified by their code and the
    plain constants they read, directly or through a mutable container (a
    list of kernel compile flags on the module, say), which is digested
    member by member — its kind and length, the pairing of a mapping's
    keys to its values, and which value sits at which position included;
    a module imported from a file, or the script being
    run, by its source's stamp (see ``_import_identity``), and every
    module its globals reach in turn. Installed third-party code is
    not followed. No design is generated to find any of it.

    Each mutable object's content is digested once per walk and memoized
    by id, and every position it is read at records that digest: a shared
    sub-object is an input at each of its positions but is walked once, so
    the work follows the object graph rather than the number of paths
    through it. Digests are address-free and order-free — mapping entries
    and set members are folded in a canonical (sorted) order — and a
    mapping's keys are named the way values are (a plain key by its repr,
    code or a module by its binding, a container by its digest), so the
    same running design hashes identically in every process.

    An ``ExternalFunction`` it reaches is recorded by its content-based repr,
    as a ``CompileTime`` one is: the cache manifest never sees a kernel's
    compile flags.

    ``frozen`` maps an object's ``id`` to the descriptor a previous walk of
    this domain produced for it (see ``_mask_generator_mutations``): the walk
    reads the pinned descriptor instead of the object, so a later key
    reproduces the earlier one over the objects that pinning covers.  With
    ``snapshot`` given, the walk fills it with the ``id`` and descriptor of
    every container and file-backed instance it digested, pinned ones
    included -- the two walks a mutation mask is diffed from.
    """
    # aie.iron imports this module.
    from aie.iron.kernel import ExternalFunction

    frozen = dict(frozen) if frozen is not None else {}

    records: set[bytes] = set()
    seen: set[int] = set()
    # Content digest per mutable object for THIS walk, keyed by id.  id() is
    # an address, but it only decides where the one-time computation happens;
    # the digest itself is content, so no key can move with it.  Per walk,
    # not per process: any of this state may be mutated between keys.
    digests: dict[int, str] = {}
    # The descent currently being digested: a reference back up it is a
    # cycle and contributes a marker instead of recursing.
    active: set[int] = set()
    todo: list = []
    modules: list[ModuleType] = []

    def describe(value, where: str, depth: int) -> str:
        """Return the position-free identity of ``value``: what it holds.

        ``where`` threads the read position through for error messages and
        key qualification only; it never enters a digest, so which path
        happens to compute a digest first cannot move the key. Values the
        walk does not follow — installed code, containers outside
        ``_CONTAINER`` — all return one uniform marker, as interchangeable
        as they were before.
        """
        if isinstance(value, (staticmethod, classmethod)):
            value = value.__func__
        if isinstance(value, property):
            legs = ",".join(
                "()" if leg is None else describe(leg, where, depth)
                for leg in (value.fget, value.fset, value.fdel)
            )
            return f"<property:{legs}>"
        if _plain(value) or isinstance(value, ExternalFunction):
            return repr(value)
        if isinstance(value, _CONTAINER):
            if id(value) in frozen:
                # Pinned by a run of the generator (see
                # _mask_generator_mutations): the identity the key already
                # holds for this object, not what it holds now.  Pinned
                # subtrees are not walked, so they cost neither time nor a
                # turn against the nesting bound.
                if snapshot is not None:
                    snapshot[id(value)] = frozen[id(value)]
                    if obj_snapshot is not None:
                        obj_snapshot[id(value)] = value
                return frozen[id(value)]
            d = digests.get(id(value))
            if d is None:
                if id(value) in active:
                    return "<cycle>"
                if depth >= _MAX_KEY_DEPTH:
                    raise _too_deep(where)
                digests[id(value)] = d = _digest(value, where, depth)
            if snapshot is not None:
                snapshot[id(value)] = d
                if obj_snapshot is not None:
                    obj_snapshot[id(value)] = value
            return d
        if (module := _stamped_module(value)) is not None:
            modules.append(module)
            if isinstance(value, ModuleType):
                return f"module:{_import_name(value.__name__)}"
            # The position binds what sits in it: two same-file helpers, or
            # two modules, are not interchangeable at one read even though
            # the file's stamp covers both, so the binding travels with the
            # content. The file says what an instance's attributes mean,
            # not what they hold, so the state read through it is an input
            # like a container's members are.
            if isinstance(value, (FunctionType, type)):
                # _import_name keeps a design's file run as a script on the
                # key its import gets.  A function also binds its code: two
                # products of one factory share module and qualname while
                # their constants differ.
                if isinstance(value, FunctionType):
                    ident = _callable_identity(value, where, depth)
                    return (
                        f"{_import_name(value.__module__)}.{value.__qualname__}"
                        f"#{ident}"
                    )
                # A class's file covers its code but not its mutable
                # class-level state (Config.FLAGS); that state is an input
                # like an instance's attributes are.
                binding = f"{_import_name(value.__module__)}.{value.__qualname__}"
                if id(value) in frozen:
                    if snapshot is not None:
                        snapshot[id(value)] = frozen[id(value)]
                        if obj_snapshot is not None:
                            obj_snapshot[id(value)] = value
                    return frozen[id(value)]
                d = digests.get(id(value))
                if d is None:
                    if id(value) in active:
                        return f"{binding}=<cycle>"
                    if depth >= _MAX_KEY_DEPTH:
                        raise _too_deep(where)
                    digests[id(value)] = d = _type_digest(value, binding, where, depth)
                if snapshot is not None:
                    snapshot[id(value)] = f"{binding}={d}"
                    if obj_snapshot is not None:
                        obj_snapshot[id(value)] = value
                return f"{binding}={d}"
            binding = (
                f"{_import_name(type(value).__module__)}.{type(value).__qualname__}"
            )
            state = getattr(value, "__dict__", None)
            if not state:
                return binding
            if id(value) in frozen:
                if snapshot is not None:
                    snapshot[id(value)] = frozen[id(value)]
                    if obj_snapshot is not None:
                        obj_snapshot[id(value)] = value
                return frozen[id(value)]
            d = digests.get(id(value))
            if d is None:
                if id(value) in active:
                    return f"{binding}=<cycle>"
                if depth >= _MAX_KEY_DEPTH:
                    raise _too_deep(where)
                digests[id(value)] = d = _instance_digest(value, binding, where, depth)
            if snapshot is not None:
                snapshot[id(value)] = f"{binding}={d}"
                if obj_snapshot is not None:
                    obj_snapshot[id(value)] = value
            return f"{binding}={d}"
        if isinstance(value, (FunctionType, type)):
            home = sys.modules.get(value.__module__)
            # Followed by code only where no file stands for it; an
            # installed package's is versioned with the install.  The body
            # is walked once (``seen``); the binding is what every position
            # records, however many positions reach it.
            if (
                home is None
                or home.__name__ == "__main__"
                or not vars(home).get("__file__")
            ):
                if id(value) not in seen:
                    seen.add(id(value))
                    todo.append(value)
                # Two same-named fileless helpers (lambdas from one factory,
                # a swapped OPS list) share a qualname; the position must
                # bind the code, or swapping them leaves the key unchanged.
                # Classes have no __code__; their binding is the name.
                if isinstance(value, FunctionType):
                    ident = _callable_identity(value, where, depth)
                    return f"{value.__module__}.{value.__qualname__}#{ident}"
                return f"{value.__module__}.{value.__qualname__}"
        return "<not-followed>"

    def _callable_identity(fn, where: str, depth: int) -> str:
        """Code identity plus the closure cells a factory lambda/closure captures.

        Two products of one factory share code and qualname while their cells
        differ (``def make(flag): return lambda: [flag]``), so the position
        must bind the cells or swapping the products leaves the key unchanged.

        Cells are described like any other read, with the function itself in
        ``active`` so a self-referential closure (``def f(): return f``)
        terminates: the re-entrant frame contributes code identity only.
        """
        parts = [_code_identity(fn.__code__).hex()]
        if id(fn) in active:
            return parts[0]
        active.add(id(fn))
        try:
            cells = fn.__closure__ or ()
            for n, cell in zip(fn.__code__.co_freevars, cells):
                if depth + 1 >= _MAX_KEY_DEPTH:
                    raise _too_deep(where)
                try:
                    content = cell.cell_contents
                except ValueError:  # an empty cell
                    parts.append(f"<{n}:empty>")
                    continue
                parts.append(
                    f"<{n}={describe(content, f'{where}.<cell>.{n}', depth + 1)}>"
                )
            return "|".join(parts)
        finally:
            active.discard(id(fn))

    def _digest(value, where: str, depth: int) -> str:
        """Digest a container's content: its kind, length and member identities."""
        active.add(id(value))
        try:
            h = hashlib.sha256()
            h.update(f"<{type(value).__name__}:{len(value)}>".encode())
            if isinstance(value, dict):
                # Insertion order is deliberately not an input: equal
                # mappings built in different orders (say, folded from a
                # set) must agree, at the price that a generator reading
                # ``list(d.values())`` order after a pop-and-reinsert sees
                # new order at the old key -- such a generator must rebuild
                # the mapping to rekey.  A mapping entry's key qualifies its
                # value, so a value reassigned between keys cannot keep the
                # old entry's digest, and the entries fold in the keys'
                # order so equal mappings built in different orders agree.
                # The key is
                # named the way a value is: ``repr(key)`` would embed a
                # function's address and move the key in every process.
                entries = []
                for k, v in value.items():
                    kd = describe(k, f"{where}[key].key", depth + 1)
                    entries.append((kd, describe(v, f"{where}[{kd}]", depth + 1)))
                for kd, vd in sorted(entries):
                    h.update(f"{kd}={vd}\0".encode())
            elif isinstance(value, (set, frozenset)):
                # A set is read without order; fold its members sorted.
                for d in sorted(
                    describe(member, f"{where}.item", depth + 1) for member in value
                ):
                    h.update(f"{d}\0".encode())
            else:
                for index, member in enumerate(value):
                    d = describe(member, f"{where}[{index}]", depth + 1)
                    h.update(f"{d}\0".encode())
            return h.hexdigest()
        finally:
            active.discard(id(value))

    def _instance_digest(value, binding: str, where: str, depth: int) -> str:
        """Digest the state a file-backed instance holds, attribute by attribute."""
        active.add(id(value))
        try:
            h = hashlib.sha256()
            h.update(binding.encode())
            state = value.__dict__
            for attr in sorted(state):
                member = describe(state[attr], f"{where}.{attr}", depth + 1)
                h.update(f".{attr}={member}\0".encode())
            return h.hexdigest()
        finally:
            active.discard(id(value))

    def _type_digest(value, binding: str, where: str, depth: int) -> str:
        """Digest a file-backed class's mutable class-level state."""
        active.add(id(value))
        try:
            h = hashlib.sha256()
            h.update(binding.encode())
            for attr, member in sorted(vars(value).items()):
                # Private *containers* are registries (ExternalFunction._instances)
                # whose churn must not move the key; private scalars and
                # callables are configuration (Config._STEP) and stay keyed.
                # Skipping a private scalar would be a stale hit, which is
                # worse than the churn a registry costs.
                if attr.startswith("_") and isinstance(member, _CONTAINER):
                    continue
                d = describe(member, f"{where}.{attr}", depth + 1)
                h.update(f".{attr}={d}\0".encode())
            return h.hexdigest()
        finally:
            active.discard(id(value))

    def visit(value, where: str):
        """Record the identity of ``value`` as read at ``where``."""
        records.add(f"{where}={describe(value, where, 0)}".encode())

    root_ids = {id(r) for r in roots if isinstance(r, FunctionType)}
    for root in roots:
        if isinstance(root, FunctionType) and id(root) not in seen:
            seen.add(id(root))
            todo.append(root)
        else:
            visit(root, "<root>")
    roots = root_ids
    while todo:
        obj = todo.pop()
        home = _import_name(obj.__module__)
        name = f"{home}.{obj.__qualname__}"
        if isinstance(obj, type):
            for attr, value in vars(obj).items():
                if not attr.startswith("__") or isinstance(value, FunctionType):
                    visit(value, f"{name}.{attr}")
            continue
        if id(obj) not in roots:
            records.add(name.encode() + _code_identity(obj.__code__))
            # A root's defaults are keyed with its filtered signature already
            # (a CompileTime default moves the key, a DispatchTime or
            # overridden one does not); visiting them here would double-count
            # them and let a dispatch default move the key.  A reached
            # helper's defaults are content and are visited.
            visit(obj.__defaults__, f"{name}.__defaults__")
            visit(tuple(sorted((obj.__kwdefaults__ or {}).items())), f"{name}.kw")
        names = _names(obj.__code__)
        scope = dict(obj.__globals__)
        where = {n: f"{home}.{n}" for n in scope}
        # A closure reaches its helpers and constants through cells rather
        # than globals; those shadow a global of the same name.
        for n, cell in zip(obj.__code__.co_freevars, obj.__closure__ or ()):
            try:
                scope[n] = cell.cell_contents
            except ValueError:  # an empty cell
                continue
            where[n] = f"{name}.<closure>.{n}"
            names.add(n)
        if (module := _stamped_module(obj)) is not None:
            modules.append(module)
        for n in names & scope.keys():
            visit(scope[n], where[n])
            owners = [scope[n]]
            while owners:
                owner = owners.pop()
                if not isinstance(owner, ModuleType):
                    continue
                members = vars(owner)
                for m in names & members.keys():
                    member = members[m]
                    visit(member, f"{owner.__name__}.{m}")
                    if (
                        isinstance(member, ModuleType)
                        and member.__name__ == f"{owner.__name__}.{m}"
                    ):
                        owners.append(member)

    stamps: dict[str, str] = {}
    pending = [m.__name__ for m in modules]
    while pending:
        name = pending.pop()
        module = sys.modules.get(name)
        if name in stamps or module is None:
            continue
        stamps[name], reached = _import_identity(module)
        pending += reached
    records.update(f"{_import_name(n)}@{s}".encode() for n, s in stamps.items())
    return b"\0".join(sorted(records))


# Only the outermost watch masks.  A generator nested in another generator's
# run mutates on the outer one's behalf, and a mask taken mid-run would pin
# identities the nested key never read.  Per thread: two concurrent compiles
# each mask their own generator.
_MASKING = threading.local()


@contextmanager
def _mask_generator_mutations(
    generator: Callable, extra_roots: Sequence = ()
) -> Iterator[None]:
    """Pin what ``generator`` mutates about its own inputs while it runs.

    A self-configuring generator -- one that records what a run learns in
    the module state the key walks, like appending every configuration it
    composes to a module-level list -- would otherwise miss its own cache
    entry: the key moves with every run, so the entry a cold compile builds
    is already stale for the next lookup.  The generator's mutation and the
    user's are structurally the same edit, so they are told apart by when
    they happen: this wraps the generator call, walks the same roots the
    key walks before and after it, and pins each container or file-backed
    instance whose identity moved to the descriptor the key read before the
    run.  The pins land on the generator as ``__iron_key_mask__`` (a mapping
    of ``id`` to descriptor), which ``_compute_recipe_hash`` threads back
    into the walk -- so the entry a run builds is the one the next lookup
    finds.  A change made outside the run to an object the run did not pin
    still moves the key and rebuilds; a pinned object is read at its pinned
    identity until the next run re-derives the mask, so a user edit to a
    *pinned* object between compiles is the one change that does not move
    the key (it is indistinguishable from the generator's own rewrites).

    The mask lives on the function rather than in a cache entry: it is a
    same-process statement about which objects this generator rewrites, and
    a fresh process re-derives it on its first cold compile.  An ``id`` is
    only unique while its object lives, so the mask keeps the pinned objects
    referenced (``__iron_key_mask_keepalive__``) -- a dead pin could
    otherwise be inherited by a new object that reuses the id, masking the
    wrong object.  This never raises on the compile's
    behalf: a walk that cannot be taken, or a generator that cannot hold
    the attribute, simply leaves the key as sensitive as it was.

    Args:
        generator: The generator function about to run.
        extra_roots: The remaining walk roots the key uses -- the
            generator's compile-time kwargs.
    """
    if getattr(_MASKING, "on", False):
        yield
        return
    roots = [generator, *extra_roots]
    frozen = dict(getattr(generator, "__iron_key_mask__", None) or {})
    pre: dict[int, str] = {}
    pre_objs: dict[int, object] = {}
    try:
        _python_identity(roots, frozen=frozen, snapshot=pre, obj_snapshot=pre_objs)
    except Exception:  # the key's helper must not fail the compile
        logger.debug(
            "identity snapshot before generation failed; not masking", exc_info=True
        )
        pre = {}
    _MASKING.on = True
    try:
        yield
    finally:
        # Every early exit below must be an `if`, never a `return`: a return
        # inside a finally discards an exception propagating out of the
        # generator, and the caller then proceeds as if the generator had
        # finished -- reading the result it never assigned.
        _MASKING.on = False
        if pre:
            post: dict[int, str] = {}
            try:
                _python_identity(roots, frozen=frozen, snapshot=post)
            except Exception:  # as above: leave the key as sensitive as it was
                logger.debug(
                    "identity snapshot after generation failed; not masking",
                    exc_info=True,
                )
            else:
                moved = {i: d for i, d in pre.items() if i in post and post[i] != d}
                if moved:
                    mask = dict(getattr(generator, "__iron_key_mask__", None) or {})
                    mask.update(moved)
                    try:
                        setattr(generator, "__iron_key_mask__", mask)
                        # An id stays unique only while its object lives:
                        # hold the pinned objects so no new object can
                        # inherit their pins by reusing an id.
                        keep = [pre_objs[i] for i in moved if i in pre_objs]
                        prev = (
                            getattr(generator, "__iron_key_mask_keepalive__", None)
                            or []
                        )
                        # The mask accumulates ids across runs; the keepalive
                        # must cover every pinned id, not just this run's.
                        setattr(
                            generator,
                            "__iron_key_mask_keepalive__",
                            [*prev, *keep],
                        )
                    except (AttributeError, TypeError):
                        logger.debug(
                            "generator %r cannot hold a key mask; its own "
                            "mutations keep moving the key",
                            getattr(generator, "__name__", generator),
                        )


def _compute_recipe_hash(
    generator: Callable | Path,
    compile_kwargs: Mapping[str, Any],
    aiecc_flags: list[str] | tuple[str, ...],
    compile_flags: list[str] | tuple[str, ...],
    full_elf: bool = False,
    include_paths: list[Path] | tuple[Path, ...] = (),
    insts_only: bool = False,
) -> str:
    """Hash of the "recipe": generator bytecode + reached Python + CompileTime[T] kwargs + flags.

    Captures the target-independent generator and compile configuration. It
    omits device identity, so equal recipe hashes can produce different
    target-specialized MLIR.

    ``full_elf`` is part of the recipe: full-ELF and xclbin+insts builds emit
    different MLIR (the former injects ``npu.load_pdi``) and different
    artifacts, so they must not share a cache entry.

    ``include_paths`` likewise: they are ``-I`` directories forwarded to the
    C++ compiler, so two otherwise identical designs pointed at different
    header trees compile to different objects. Hashed in ORDER, not sorted
    like the flag lists above, because ``-I`` search order decides which
    header wins when two directories provide the same name.

    A generator carrying ``__iron_key_mask__`` (see
    ``_mask_generator_mutations``) has the pins read by its identity walk:
    the mutations the generator itself made do not move the key, everything
    else the walk reaches still does.
    """
    h = hashlib.sha256()

    if isinstance(generator, Path):
        h.update(str(generator).encode())
        h.update(_content_digest(generator).encode())
    else:
        h.update(_code_identity(generator.__code__))
        h.update(getattr(generator, "__qualname__", "").encode())
        h.update(_import_name(generator.__module__ or "").encode())
        h.update(
            _python_identity(
                [generator, *compile_kwargs.values()],
                frozen=getattr(generator, "__iron_key_mask__", None),
            )
        )
        d = declare(generator)
        assert d.signature is not None, "a generator's declaration has a signature"
        dispatch_params = {m.name for m in d.per_calls}
        # Dispatch defaults are call-time values; explicitly bound defaults are
        # unused. Neither changes the compiled program.
        h.update(
            repr(
                [
                    param.replace(
                        default=(
                            param.empty
                            if name in dispatch_params or name in compile_kwargs
                            else param.default
                        ),
                    )
                    for name, param in d.signature.parameters.items()
                ]
            ).encode()
        )

    def _kwarg_repr(v):
        if isinstance(v, MethodType):
            v = v.__func__
        if isinstance(v, FunctionType):
            closure = (
                tuple(c.cell_contents for c in v.__closure__) if v.__closure__ else None
            )
            try:
                closure_repr = repr(closure)
            except Exception:
                closure_repr = "<unhashable closure>"
            return (
                "fn:",
                _code_identity(v.__code__).hex(),
                repr(v.__defaults__),
                repr(v.__kwdefaults__),
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
    h.update(f"full_elf={full_elf}".encode())
    h.update(repr([str(p) for p in include_paths]).encode())
    if insts_only:
        # An instruction stream alone is a different artifact from the
        # xclbin + insts pair the same generator would otherwise produce.
        h.update(b"insts_only=True")

    return h.hexdigest()


def _tool_identity(
    name: str, resolve: Callable[[], str | Path], *, expected: bool = True
) -> str:
    """Identify a resolved compiler component without probing an executable.

    A tool that is not ``expected`` may legitimately be missing, so its absence
    is hashed without a warning.
    """
    try:
        path = Path(resolve()).resolve()
        stat = path.stat()
        return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"
    except (ImportError, AttributeError, OSError, RuntimeError) as exc:
        if expected:
            logger.warning("_compute_artifact_hash: %s absent (%s)", name, exc)
        return "absent"


def _aiecc_option(flags: list[str] | tuple[str, ...], name: str) -> str | None:
    """Return the value of a string-valued aiecc option."""
    options = (f"--{name}", f"-{name}")
    for index, flag in enumerate(flags):
        for option in options:
            if flag.startswith(f"{option}="):
                return flag.split("=", 1)[1]
            if flag == option and index + 1 < len(flags):
                return flags[index + 1]
    return None


def _compute_artifact_hash(
    generator: Callable | Path,
    source_files: list[Path] | tuple[Path, ...],
    object_files: list[Path] | tuple[Path, ...],
    fold_ddr_addr_offset: bool,
    has_dispatch_params: bool = False,
    full_elf: bool = False,
    insts_only: bool = False,
    aiecc_flags: list[str] | tuple[str, ...] = (),
    emit_elf: bool = False,
    work_dir: Path | None = None,
) -> str:
    """Hash of the "artifacts": source/object content + tool mtimes + device.

    Captures everything that can change the *output* of compilation without
    changing the *recipe*: edited C++ kernels, swapped object files, upgraded
    Peano / aiecc, retargeted device.

    ``fold_ddr_addr_offset`` is the active backend's DDR-patch ABI: XRT/CPU emit
    a folded ``insts.bin`` and HRX an unfolded one, so the two must never share a
    cache entry. It is resolved once by the caller and passed in explicitly (no
    silent default) so the cache key and the compilation can never disagree.

    ``has_dispatch_params`` additionally hashes the host C++ compiler used to
    build the dispatch library. Its generated source is covered by aiecc's
    identity above; Python does not run a separate translation pipeline.

    Every tool that packages a requested image is hashed too: ``aiebu-asm`` for
    an ELF, ``xclbinutil`` for an xclbin, and nothing for an instruction stream
    alone. Both images embed a PDI, so they also hash the ``bootgen`` aiecc
    would run. aiecc may link bootgen and aiebu in instead, which Python cannot
    tell, so a missing ``bootgen`` or ``aiebu-asm`` is not an error.
    """
    from aie.utils import config as _config

    h = hashlib.sha256()
    tools = {
        "peano": _config.peano_cxx_path,
        "aiecc": _config.aiecc_path,
        "nm": _config.nm_path,
        "objcopy": _config.objcopy_path,
    }

    for sf in sorted(source_files, key=str):
        h.update(str(sf).encode())
        h.update(_content_digest(sf).encode())

    for of in sorted(object_files, key=str):
        h.update(str(of).encode())
        h.update(_content_digest(of).encode())

    h.update(f"fold_ddr_addr_offset={fold_ddr_addr_offset}".encode())
    if (
        insts_only
        and not full_elf
        and not fold_ddr_addr_offset
        and not any(
            flag.split("=", 1)[0] in ("--fold-ddr-addr-offset", "-fold-ddr-addr-offset")
            for flag in aiecc_flags
        )
    ):
        # Backend-selected unfolded streams cannot reuse folded-default entries.
        h.update(b"insts_only_unfolded_abi=1")
    # Static .mlir is target-agnostic; compiled kernels need a device identifier.
    # Missing components collapse to a constant + WARNING log so cross-target
    # cache collisions surface instead of silently aliasing.
    if not isinstance(generator, Path):
        try:
            from aie.utils import get_current_device
            from aie.utils.compile.utils import resolve_target_arch

            device = get_current_device(probe_runtime=False)
            target_arch = resolve_target_arch(device)
            target_device = _device_identity_key(device)
        except (ImportError, AttributeError, RuntimeError, ValueError) as exc:
            logger.warning(
                "_compute_artifact_hash: target_arch unresolved (%s); using 'unknown'",
                exc,
            )
            target_arch = "unknown"
            target_device = ("unknown", "", "", "")

        h.update(f"target_arch={target_arch}|target_device={target_device!r}".encode())
        if has_dispatch_params:
            tools["host_cxx"] = _config.host_cxx_path
        # Library factories pick their sources from this tree when the
        # generator runs, after the key is taken, so the key names the tree
        # and its contents, which a candidate edited in place changes.
        if kernel_tree := os.environ.get("MLIR_AIE_KERNEL_SOURCES"):
            from aie.utils.benchmark import kernel_tree_digest

            h.update(f"kernel_sources={kernel_tree}|{kernel_tree_digest()}".encode())
    optional = {"aiebu-asm"}
    if full_elf or not insts_only:
        tools["bootgen"] = partial(_config.aiecc_tool_path, "bootgen")
        optional.add("bootgen")
    if full_elf:
        tools["aiebu-asm"] = partial(_config.aiecc_tool_path, "aiebu-asm")
    elif not insts_only:
        xclbinutil_override = _aiecc_option(aiecc_flags, "xclbinutil-path")
        if not xclbinutil_override:
            xclbinutil_override = os.environ.get("AIE_XCLBINUTIL")
        if (
            xclbinutil_override
            and not Path(xclbinutil_override).is_absolute()
            and any(sep and sep in xclbinutil_override for sep in (os.sep, os.altsep))
            and work_dir is None
        ):
            raise ValueError(
                "A relative xclbinutil path requires an explicit output path so "
                "it can be resolved from aie's work directory; use an absolute "
                "path with the JIT cache."
            )
        tools["xclbinutil"] = partial(
            _config.aiecc_tool_path,
            "xclbinutil",
            override=xclbinutil_override,
            cwd=work_dir,
        )
        if emit_elf:
            tools["aiebu-asm"] = partial(_config.aiecc_tool_path, "aiebu-asm")
    for name, resolve in tools.items():
        identity = _tool_identity(name, resolve, expected=name not in optional)
        h.update(f"{name}={identity}".encode())

    return h.hexdigest()


def _compute_hash(
    generator: Callable | Path,
    compile_kwargs: Mapping[str, Any],
    source_files: list[Path] | tuple[Path, ...],
    object_files: list[Path] | tuple[Path, ...],
    aiecc_flags: list[str] | tuple[str, ...],
    compile_flags: list[str] | tuple[str, ...],
    full_elf: bool = False,
    fold_ddr_addr_offset: bool = True,
    has_dispatch_params: bool = False,
    include_paths: list[Path] | tuple[Path, ...] = (),
    insts_only: bool = False,
    emit_elf: bool = False,
    work_dir: Path | None = None,
) -> str:
    """Stable 24-hex SHA-256 cache key combining recipe + artifact hashes."""
    recipe = _compute_recipe_hash(
        generator,
        compile_kwargs,
        aiecc_flags,
        compile_flags,
        full_elf,
        include_paths,
        insts_only,
    )
    artifact = _compute_artifact_hash(
        generator,
        source_files,
        object_files,
        fold_ddr_addr_offset,
        has_dispatch_params,
        full_elf,
        insts_only,
        aiecc_flags,
        emit_elf,
        work_dir,
    )
    return hashlib.sha256(f"{recipe}|{artifact}".encode()).hexdigest()[:24]
