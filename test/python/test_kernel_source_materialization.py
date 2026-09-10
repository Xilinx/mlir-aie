# test_kernel_source_materialization.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Unit tests for concurrent kernel source materialization — no NPU required."""

import os
import types

import pytest

import aie.utils.compile.utils as compile_utils
from aie.utils.compile.utils import _materialize_source, compile_external_kernels

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="test/python/ has no Windows pytest job to run under"
)

SOURCE = "// kernel\nvoid k() {}\n" * 64


def _stub_func(name, source_file):
    """A stand-in for ExternalFunction carrying only what the compile path reads."""
    return types.SimpleNamespace(
        _name=name,
        _original_name=name,
        _source_file=str(source_file),
        _source_string=None,
        _include_dirs=[],
        _compile_flags=[f"-D{name.upper()}"],
        _inline=False,
        _symbol_prefix=None,
        _use_chess=False,
        _compiled=False,
        object_file_name=f"{name}.o",
    )


@pytest.fixture
def stub_compiler(monkeypatch):
    """Replace the Peano invocation with a no-op that just produces the object."""

    def fake_compile(source_path, output_path, **kwargs):
        with open(output_path, "w"):
            pass

    monkeypatch.setattr(compile_utils, "compile_cxx_core_function", fake_compile)


def test_rewrite_leaves_an_open_reader_on_its_own_file(tmp_path):
    """A compile that already opened the source is unaffected by a sibling's write.

    Reading back the same bytes would prove nothing -- every writer stages
    identical content, so an in-place truncate is invisible once it finishes.
    What matters is that the reader is never looking at the file being written,
    which is what spares it the SIGBUS when the size drops under its mapping.
    """
    dest = tmp_path / "kernel.cc"
    _materialize_source(str(dest), text=SOURCE)

    with open(dest) as reader:
        _materialize_source(str(dest), text="// replaced\n")
        assert reader.read() == SOURCE
        assert os.fstat(reader.fileno()).st_ino != os.stat(dest).st_ino

    assert dest.read_text() == "// replaced\n"


def test_shared_source_kernels_do_not_clobber_each_other(tmp_path, stub_compiler):
    """Kernels sharing one .cc write one destination, so those writes must be atomic."""
    upstream = tmp_path / "shared.cc"
    upstream.write_text(SOURCE)
    kernel_dir = tmp_path / "work"
    kernel_dir.mkdir()
    dest = kernel_dir / "shared.cc"

    # The copy is named after the .cc but the grouping key is the kernel name,
    # so these two are not ordered against each other despite sharing a path.
    put, get = (_stub_func(n, upstream) for n in ("kernel_put", "kernel_get"))

    compile_external_kernels([put], str(kernel_dir), "aie2p")
    first = os.stat(dest).st_ino
    compile_external_kernels([get], str(kernel_dir), "aie2p")

    assert (
        os.stat(dest).st_ino != first
    ), "second kernel rewrote the shared source in place"
    assert dest.read_text() == SOURCE


def test_source_that_is_already_in_place_is_not_rewritten(tmp_path, stub_compiler):
    """A kernel_dir reaching its own source through a symlink must not overwrite it."""
    kernel_dir = tmp_path / "work"
    kernel_dir.mkdir()
    upstream = kernel_dir / "shared.cc"
    upstream.write_text(SOURCE)
    link = tmp_path / "link"
    link.symlink_to(kernel_dir)

    before = os.stat(upstream).st_ino
    compile_external_kernels(
        [_stub_func("kernel_put", link / "shared.cc")], str(kernel_dir), "aie2p"
    )

    assert os.stat(upstream).st_ino == before
    assert upstream.read_text() == SOURCE


def test_materialized_source_keeps_umask_permissions(tmp_path):
    """Staging through mkstemp must not narrow the source to 0600."""
    written = tmp_path / "from_string.cc"
    _materialize_source(str(written), text=SOURCE)

    copied = tmp_path / "from_file.cc"
    _materialize_source(str(copied), copy_from=str(written))

    expected = 0o666 & ~compile_utils._UMASK
    assert os.stat(written).st_mode & 0o777 == expected
    assert os.stat(copied).st_mode & 0o777 == expected


def test_failed_write_leaves_no_temp_files(tmp_path):
    """A raising write must not litter the kernel directory with .tmp sources."""
    dest = tmp_path / "kernel.cc"
    with pytest.raises(FileNotFoundError):
        _materialize_source(str(dest), copy_from=str(tmp_path / "missing.cc"))

    assert list(tmp_path.iterdir()) == []
