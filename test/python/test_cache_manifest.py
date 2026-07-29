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
