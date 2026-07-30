# test_cache_manifest.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Unit tests for the JIT cache's recorded dependency manifest -- no NPU required."""

import json
import time

import pytest

from aie.utils.compile.jit import _manifest


class _Kernel:
    """Minimal stand-in for an ExternalFunction, which needs an MLIR context."""

    def __init__(self, source_file=None, source_string=None):
        self._source_file = source_file
        self._source_string = source_string


def _depfile(kernel_dir, obj, deps):
    (kernel_dir / f"{obj}.d").write_text(
        f"{kernel_dir / obj}: \\\n  " + " \\\n  ".join(str(d) for d in deps) + "\n"
    )


# ---------------------------------------------------------------------------
# Validation fails closed
# ---------------------------------------------------------------------------


def test_absent_manifest_is_not_valid(tmp_path):
    """An entry with no recorded inputs must never read as verified."""
    assert not _manifest.is_valid(tmp_path)


def test_unparsable_manifest_is_not_valid(tmp_path):
    (tmp_path / _manifest.MANIFEST_NAME).write_text("{ not json")
    assert not _manifest.is_valid(tmp_path)


def test_manifest_from_a_future_version_is_not_valid(tmp_path):
    (tmp_path / _manifest.MANIFEST_NAME).write_text(
        json.dumps({"version": 999, "inputs": []})
    )
    assert not _manifest.is_valid(tmp_path)


@pytest.mark.parametrize(
    "build_payload",
    [
        pytest.param(lambda src: [1, 2, 3], id="payload-is-not-an-object"),
        pytest.param(
            lambda src: {"version": 1, "inputs": str(src)}, id="inputs-is-not-a-list"
        ),
        pytest.param(
            lambda src: {"version": 1, "inputs": [str(src)]},
            id="input-is-not-an-object",
        ),
        pytest.param(
            lambda src: {"version": 1, "inputs": [{"path": str(src)}]},
            id="input-is-partial",
        ),
        pytest.param(
            lambda src: {
                "version": 1,
                "inputs": [{"path": str(src), "size": "4", "mtime": 0.0}],
            },
            id="input-field-is-the-wrong-type",
        ),
    ],
)
def test_malformed_manifest_is_not_valid(tmp_path, build_payload):
    """Well-formed JSON that is not a manifest must miss, not raise.

    Each payload names a file that exists, so validation reaches the recorded
    fields instead of stopping at a failed ``stat``.
    """
    src = tmp_path / "k.cc"
    src.write_text("// k")
    (tmp_path / _manifest.MANIFEST_NAME).write_text(json.dumps(build_payload(src)))
    assert not _manifest.is_valid(tmp_path)


def test_deleted_input_is_not_valid(tmp_path):
    src = tmp_path / "k.cc"
    src.write_text("// v1")
    _manifest.write_for_test(tmp_path, [src])
    assert _manifest.is_valid(tmp_path)
    src.unlink()
    assert not _manifest.is_valid(tmp_path)


# ---------------------------------------------------------------------------
# Validation tracks content, not just timestamps
# ---------------------------------------------------------------------------


def test_edited_input_invalidates(tmp_path):
    src = tmp_path / "k.cc"
    src.write_text("// v1")
    _manifest.write_for_test(tmp_path, [src])

    time.sleep(0.01)
    src.write_text("// v2")
    assert not _manifest.is_valid(tmp_path)


def test_touched_but_unchanged_input_still_valid(tmp_path):
    """A new mtime over identical bytes must not force a rebuild."""
    src = tmp_path / "k.cc"
    src.write_text("// v1")
    _manifest.write_for_test(tmp_path, [src])

    time.sleep(0.01)
    src.write_text("// v1")  # same content, new mtime
    assert _manifest.is_valid(tmp_path)


def test_same_size_different_content_invalidates(tmp_path):
    """Size alone must not be trusted; the digest is the decider."""
    src = tmp_path / "k.cc"
    src.write_text("aaaa")
    _manifest.write_for_test(tmp_path, [src])

    time.sleep(0.01)
    src.write_text("bbbb")
    assert not _manifest.is_valid(tmp_path)


# ---------------------------------------------------------------------------
# Recording: what the compiler reported, and when to record nothing
# ---------------------------------------------------------------------------


def test_record_captures_depfile_contents(tmp_path):
    """Inputs the compiler reported must be recorded, not just the named source."""
    body = tmp_path / "body.cc"
    body.write_text("// body")
    shim = tmp_path / "shim.cc"
    shim.write_text('#include "body.cc"\n')
    _depfile(tmp_path, "k.o", [shim, body])

    _manifest.record(tmp_path, [_Kernel(source_file=str(shim))], ())
    recorded = {
        i["path"]
        for i in json.loads((tmp_path / _manifest.MANIFEST_NAME).read_text())["inputs"]
    }
    assert str(body) in recorded, "the included body was not recorded"
    assert str(shim) in recorded

    time.sleep(0.01)
    body.write_text("// body v2")
    assert not _manifest.is_valid(tmp_path)


def test_chess_build_records_an_incomplete_manifest(tmp_path):
    """Chess reports no inputs, so the entry says so instead of claiming a check.

    It stays usable: a Chess design cached fine before manifests existed, and
    writing nothing would make every later lookup discard the directory and
    rebuild -- worse than the behaviour this is meant to preserve.
    """
    shim = tmp_path / "shim.cc"
    shim.write_text("// k")
    _manifest.record(tmp_path, [_Kernel(source_file=str(shim))], (), used_chess=True)

    payload = json.loads((tmp_path / _manifest.MANIFEST_NAME).read_text())
    assert payload["complete"] is False
    assert payload["inputs"] == []
    assert _manifest.is_valid(tmp_path)

    # And it keeps saying so: an edited kernel is genuinely undetectable here,
    # which is exactly why the manifest does not claim otherwise.
    time.sleep(0.01)
    shim.write_text("// k v2")
    assert _manifest.is_valid(tmp_path)


def test_kernel_without_a_depfile_records_an_incomplete_manifest(tmp_path):
    """A kernel compiled by some other route leaves the set unknowable."""
    shim = tmp_path / "shim.cc"
    shim.write_text("// k")
    _manifest.record(tmp_path, [_Kernel(source_file=str(shim))], ())

    payload = json.loads((tmp_path / _manifest.MANIFEST_NAME).read_text())
    assert payload["complete"] is False
    assert _manifest.is_valid(tmp_path)


def test_incomplete_is_not_the_same_as_no_inputs(tmp_path):
    """An empty input list still gets checked; an incomplete manifest does not.

    Both record zero inputs, so only the flag separates "this design consumes
    nothing" from "this build could not tell what it consumed".
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    _manifest.record(empty, [], ())
    assert json.loads((empty / _manifest.MANIFEST_NAME).read_text())["complete"] is True

    unknowable = tmp_path / "unknowable"
    unknowable.mkdir()
    shim = unknowable / "shim.cc"
    shim.write_text("// k")
    _manifest.record(unknowable, [_Kernel(source_file=str(shim))], (), used_chess=True)
    payload = json.loads((unknowable / _manifest.MANIFEST_NAME).read_text())
    assert payload["complete"] is False
    assert payload["inputs"] == []


def test_design_without_kernels_records_its_declared_sources(tmp_path):
    """No compiled kernels means nothing to discover, but sources still count."""
    src = tmp_path / "extra.cc"
    src.write_text("// x")
    _manifest.record(tmp_path, [], (src,))
    assert _manifest.is_valid(tmp_path)

    time.sleep(0.01)
    src.write_text("// y")
    assert not _manifest.is_valid(tmp_path)


def test_unreadable_depfile_records_an_incomplete_manifest(tmp_path):
    """A depfile that cannot be read leaves the set unknowable, like Chess.

    Writing nothing would read as a miss, and the caller answers a miss by
    discarding the directory -- wiping a good entry on every later lookup.
    """
    shim = tmp_path / "shim.cc"
    shim.write_text("// k")
    _depfile(tmp_path, "k.o", [shim])
    # A directory where the depfile should be fails the read for every user.
    # chmod(0o000) would not: the Ryzen AI Software CI job runs as root, which
    # bypasses the permission bits and made this assertion vacuous there.
    depfile = tmp_path / "k.o.d"
    depfile.unlink()
    depfile.mkdir()

    _manifest.record(tmp_path, [_Kernel(source_file=str(shim))], ())

    payload = json.loads((tmp_path / _manifest.MANIFEST_NAME).read_text())
    assert payload["complete"] is False
    assert _manifest.is_valid(tmp_path)


def test_unreadable_input_records_an_incomplete_manifest(tmp_path, monkeypatch):
    """An input that cannot be digested is unverifiable, not uncacheable.

    _write only digests paths that already passed is_file(), so the branch under
    test is the narrow one where the read fails anyway: the file goes away, or
    the I/O errors, between the two calls. Raising it directly is also what
    survives running as root, where chmod(0o000) is not a read barrier.
    """
    shim = tmp_path / "shim.cc"
    shim.write_text("// k")
    secret = tmp_path / "secret.h"
    secret.write_text("// h")
    _depfile(tmp_path, "k.o", [shim, secret])

    entry = _manifest._entry

    def unreadable_secret(path):
        if path.name == "secret.h":
            raise OSError("input vanished under us")
        return entry(path)

    monkeypatch.setattr(_manifest, "_entry", unreadable_secret)
    _manifest.record(tmp_path, [_Kernel(source_file=str(shim))], ())

    payload = json.loads((tmp_path / _manifest.MANIFEST_NAME).read_text())
    assert payload["complete"] is False
    assert payload["inputs"] == []
    assert _manifest.is_valid(tmp_path)


# ---------------------------------------------------------------------------
# Recorded paths must not depend on where the caller happens to stand
# ---------------------------------------------------------------------------


def test_relative_declared_source_is_recorded_absolutely(tmp_path, monkeypatch):
    """``source_files`` is user-supplied and routinely relative.

    Stored verbatim it is checked from whatever directory a later process runs
    in, so it either misses forever or stats a same-named file elsewhere.
    """
    kernel_dir = tmp_path / "cache"
    kernel_dir.mkdir()
    project = tmp_path / "project"
    (project / "kernels").mkdir(parents=True)
    src = project / "kernels" / "vector_add.cc"
    src.write_text("// v1")

    monkeypatch.chdir(project)
    _manifest.record(kernel_dir, [], ("kernels/vector_add.cc",))

    recorded = json.loads((kernel_dir / _manifest.MANIFEST_NAME).read_text())["inputs"]
    assert [i["path"] for i in recorded] == [str(src)]

    # The next lookup runs from somewhere else, as a second process would.
    monkeypatch.chdir(tmp_path)
    assert _manifest.is_valid(kernel_dir)
    time.sleep(0.01)
    src.write_text("// v2")
    assert not _manifest.is_valid(kernel_dir)


def test_relative_kernel_source_is_recorded_absolutely(tmp_path, monkeypatch):
    """Same for ``ExternalFunction(source_file=...)``, stored exactly as given."""
    kernel_dir = tmp_path / "cache"
    kernel_dir.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    src = project / "k.cc"
    src.write_text("// v1")

    monkeypatch.chdir(project)
    _manifest.record(kernel_dir, [_Kernel(source_file="k.cc")], (), used_chess=True)
    _manifest.record(kernel_dir, [], ("k.cc",))

    monkeypatch.chdir(tmp_path)
    assert _manifest.is_valid(kernel_dir)
    time.sleep(0.01)
    src.write_text("// v2")
    assert not _manifest.is_valid(kernel_dir)


def test_relative_depfile_entry_resolves_against_the_compile_directory(tmp_path):
    """Depfile paths are relative to the compiler's cwd, and Peano runs in
    kernel_dir -- not in whatever directory writes the manifest."""
    header = tmp_path / "hdr.h"
    header.write_text("// h")
    (tmp_path / "k.o.d").write_text(f"{tmp_path / 'k.o'}: hdr.h\n")

    _manifest.record(tmp_path, [_Kernel(source_string="// k")], ())

    payload = json.loads((tmp_path / _manifest.MANIFEST_NAME).read_text())
    assert [i["path"] for i in payload["inputs"]] == [str(header)]

    time.sleep(0.01)
    header.write_text("// h v2")
    assert not _manifest.is_valid(tmp_path)


def test_depfile_entry_with_an_escaped_space_is_recorded(tmp_path):
    """clang escapes a space in a dependency path as ``\\ ``.

    Splitting on bare whitespace shreds it into two tokens naming no file, both
    dropped -- leaving a real input unrecorded under ``complete: true``.
    """
    spaced = tmp_path / "my dir"
    spaced.mkdir()
    header = spaced / "hdr.h"
    header.write_text("// h")
    (tmp_path / "k.o.d").write_text(
        f"{tmp_path / 'k.o'}: {str(header).replace(' ', chr(92) + ' ')}\n"
    )

    _manifest.record(tmp_path, [_Kernel(source_string="// k")], ())

    payload = json.loads((tmp_path / _manifest.MANIFEST_NAME).read_text())
    assert [i["path"] for i in payload["inputs"]] == [str(header)]

    time.sleep(0.01)
    header.write_text("// h v2")
    assert not _manifest.is_valid(tmp_path)
