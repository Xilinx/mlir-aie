# _manifest.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Recorded dependency manifest for the JIT cache.

The cache key cannot know a design's kernel sources: they are registered while
MLIR is generated, which happens after the key is computed and not at all on a
hit. So instead of predicting inputs, record the ones a build actually consumed
and check them on the next lookup.

**Peano only.** Inputs are recorded when every kernel in the design was compiled
by Peano, because Peano is asked for a depfile and so reports exactly which files
it opened. The Chess front-end is driven through ``xchesscc_wrapper`` and emits
nothing equivalent, so for a Chess design the consumed set is unknowable from
here. Recording a set that is silently short would read as verified and is worse
than recording nothing.

Such a build still writes a manifest, marked ``complete: false``. That is not the
same as a manifest with no inputs -- a design that genuinely consumes nothing
records ``complete: true`` and an empty list, and is checked like any other. The
distinction matters because the two must behave differently: an unverifiable
design has to keep the cache behaviour it had before this file existed, whereas
writing nothing at all would make every lookup discard the entry and rebuild,
which is strictly worse than the status quo it is meant to preserve.

Everything here fails closed: an absent, unparsable, misshapen or partial
manifest is a miss. A manifest is checked for shape rather than trusted, since a
lookup that raises on a corrupt one fails neither open nor closed.

Where a build path cannot report its inputs, the manifest says so -- an
unreadable depfile and an undigestable input both record ``complete: false``,
never nothing at all. Unverifiable is not uncacheable.

Recorded paths are absolute: an entry is written by one process and checked by
another that need not share a working directory. Depfile tokens anchor against
``kernel_dir`` (the cwd Peano runs in, so what their relative paths mean),
declared sources against the caller's cwd.

**Known limits.** Two paths fail open rather than closed -- they can report a
valid entry that is actually stale -- so they are worth knowing before relying
on this. Both are shared with ccache and neither is fixable by recording inputs
more carefully, because in both cases the recorded inputs are genuinely
unchanged.

1. *Include resolution.* Only files the compiler opened are recorded. A file
   that later appears earlier in the ``-I`` search order and shadows a recorded
   header changes what the next compile would resolve, while every recorded
   input still matches.

2. *Size and mtime agreement.* ``is_valid`` skips the digest when both size and
   mtime match a record. An edit that preserves byte count, after something has
   reset mtimes -- a fresh clone, a checkout, an extracted archive -- therefore
   reads as unchanged. The digest is what catches a same-size edit; this
   shortcut is what makes the common case cheap.

   No clock reset is needed for this to bite: two same-size writes inside one
   timestamp tick are indistinguishable, and the tick belongs to the
   filesystem. On NFS it can be a whole second, and the client attribute cache
   may serve a stale size and mtime for ``acregmax`` on top. ``ccache`` gates
   the same shortcut behind ``sloppiness``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_NAME = "deps.json"
_VERSION = 1


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _entry(path: Path) -> dict:
    st = path.stat()
    return {
        "path": str(path),
        "size": st.st_size,
        "mtime": st.st_mtime,
        "sha256": _digest(path),
    }


def _absolute(base: Path, path: Path) -> Path:
    """Anchor a possibly-relative path against `base`, keeping symlinks intact.

    Lexical, not ``resolve()``: keeping the path the compiler opened means
    repointing a symlink is caught, since stat and the digest follow the link.
    """
    return Path(os.path.normpath(base / path))


def _parse_depfile(depfile: Path) -> list[Path]:
    r"""Read a make-style depfile: 'target: a b \\\n  c d'.

    Make quoting is undone rather than split through: a space arrives as ``\\ ``
    and a dollar as ``$$``, so splitting on bare whitespace would turn one real
    dependency into two tokens naming no file -- and those are dropped, leaving
    a manifest silently short while still calling itself complete.
    """
    text = depfile.read_text()
    tokens: list[str] = []
    current: list[str] = []
    past_target = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt == "\n":  # line continuation: a separator
                i += 2
                if current:
                    tokens.append("".join(current))
                    current = []
                continue
            if nxt in " #":  # escaped: the character itself
                current.append(nxt)
                i += 2
                continue
            current.append(c)  # lone backslash: a path separator, not quoting
            i += 1
            continue
        if c == "$" and text[i : i + 2] == "$$":
            current.append("$")
            i += 2
            continue
        if not past_target and c == ":":
            past_target = True
            current = []
            i += 1
            continue
        if c.isspace():
            if current:
                tokens.append("".join(current))
                current = []
            i += 1
            continue
        current.append(c)
        i += 1
    if current:
        tokens.append("".join(current))
    return [Path(tok) for tok in tokens] if past_target else []


def record(kernel_dir, external_kernels, source_files, used_chess=False) -> None:
    """Record the inputs this build consumed, if they can be known exactly.

    Every kernel compiled through Peano leaves a depfile naming what the
    compiler actually opened.  A kernel compiled any other way -- today, the
    Chess path, which takes no -MD -- leaves none, and its includes are
    therefore unknown.  Rather than record a set that is silently short, mark
    the manifest ``complete: false``: the entry stays usable, so such a design
    keeps exactly the behaviour it has now, but nothing claims it was checked.
    """
    kernel_dir = Path(kernel_dir)
    compiled = [
        f
        for f in external_kernels
        if getattr(f, "_source_file", None) or getattr(f, "_source_string", None)
    ]
    depfiles = list(kernel_dir.glob("*.d"))
    if used_chess or (compiled and not depfiles):
        logger.debug(
            "cache manifest: %d kernel(s) compiled without a depfile; recording "
            "an incomplete manifest, inputs will not be checked",
            len(compiled),
        )
        _write(kernel_dir, [], complete=False)
        return

    found: set[Path] = set()
    for dep in depfiles:
        try:
            # Peano runs with cwd=kernel_dir, so that is what a relative token means.
            found.update(_absolute(kernel_dir, p) for p in _parse_depfile(dep))
        except OSError:
            # Unknowable, as for Chess. Writing nothing would instead read as a
            # miss, and the caller answers a miss by discarding the entry.
            logger.debug(
                "cache manifest: %s could not be read; recording an incomplete "
                "manifest, inputs will not be checked",
                dep,
            )
            _write(kernel_dir, [], complete=False)
            return

    # Declared paths are the caller's, read where the caller stands -- the same
    # reading compile_external_kernel gives them.
    cwd = Path.cwd()
    for f in compiled:
        if getattr(f, "_source_file", None):
            found.add(_absolute(cwd, Path(f._source_file)))
    found.update(_absolute(cwd, Path(sf)) for sf in source_files)
    _write(kernel_dir, sorted({p for p in found if p.is_file()}, key=str))


def _write(kernel_dir: Path, inputs: list[Path], complete: bool = True) -> None:
    entries: list[dict] = []
    for path in inputs:
        try:
            entries.append(_entry(path))
        except OSError:
            # Unverifiable, but not uncacheable: mark it incomplete rather than
            # writing nothing and costing the entry on every later lookup.
            logger.warning(
                "cache manifest: %s unreadable; recording an incomplete manifest",
                path,
            )
            entries, complete = [], False
            break
    payload = {"version": _VERSION, "complete": complete, "inputs": entries}
    tmp = Path(kernel_dir) / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, Path(kernel_dir) / MANIFEST_NAME)


def write_for_test(kernel_dir, inputs) -> None:
    """Write a manifest directly. For tests that fabricate a cache entry."""
    _write(Path(kernel_dir), [Path(i) for i in inputs])


def is_valid(kernel_dir: Path) -> bool:
    """Return True only if every recorded input is still exactly as recorded.

    Fails closed: a missing, unparsable or partial manifest is a miss, never a
    hit.  Anything this cannot verify, it does not trust.

    The one exception is a manifest that declares itself incomplete, which is
    how a build path that cannot report its inputs (today, Chess) records
    itself.  There is nothing to check, so this reports the entry usable and
    leaves it exactly as cacheable as it was before manifests existed.  Treating
    it as a miss would be worse than the status quo, not safer: the caller
    discards the directory on a miss, so every lookup would wipe a valid entry
    and rebuild it.
    """
    path = Path(kernel_dir) / MANIFEST_NAME
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    # Shape is an input to check, not an invariant to assume: raising on a
    # malformed manifest fails neither open nor closed.
    if not isinstance(payload, dict) or payload.get("version") != _VERSION:
        return False
    if not payload.get("complete", True):
        logger.debug(
            "cache manifest: %s records no verifiable inputs; not checking", path
        )
        return True
    inputs = payload.get("inputs", ())
    if not isinstance(inputs, list):
        return False
    for rec in inputs:
        if not isinstance(rec, dict):
            return False
        recorded_path = rec.get("path")
        size = rec.get("size")
        mtime = rec.get("mtime")
        sha256 = rec.get("sha256")
        if (
            not isinstance(recorded_path, str)
            or not isinstance(size, int)
            or not isinstance(mtime, (int, float))
            or not isinstance(sha256, str)
        ):
            return False
        f = Path(recorded_path)
        try:
            st = f.stat()
        except OSError:
            return False
        # tsbuildinfo pattern: size+mtime agree -> trust it, skip the hash.
        if st.st_size == size and st.st_mtime == mtime:
            continue
        try:
            if _digest(f) != sha256:
                return False
        except OSError:
            return False
    return True
