# test_kernel_source_materialization.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Unit tests for concurrent kernel source materialization — no NPU required."""

import os
import types

import aie.utils.compile.utils as compile_utils
import pytest
from aie.utils.compile.utils import (
    _copy_source,
    _write_source,
    compile_external_kernels,
)

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="test/python/ has no Windows pytest job to run under"
)

SOURCE = "// kernel\nvoid k() {}\n" * 64


def _stub_func(name, source_file):
    """Build a stand-in for ExternalFunction carrying what the compile path reads."""
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
    calls = []

    def fake_compile(source_path, output_path, **kwargs):
        calls.append(kwargs)
        with open(output_path, "w"):
            pass

    monkeypatch.setattr(compile_utils, "compile_cxx_core_function", fake_compile)
    return calls


def test_rewrite_leaves_an_open_reader_on_its_own_file(tmp_path):
    """A compile that already opened the source is unaffected by a sibling's write.

    Reading back the same bytes would prove nothing -- every writer stages
    identical content, so an in-place truncate is invisible once it finishes.
    What matters is that the reader is never looking at the file being written,
    which is what spares it the SIGBUS when the size drops under its mapping.
    """
    dest = tmp_path / "kernel.cc"
    _write_source(str(dest), SOURCE)

    with open(dest) as reader:
        _write_source(str(dest), "// replaced\n")
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


def test_design_include_dirs_follow_kernel_and_source_dirs(tmp_path, stub_compiler):
    upstream = tmp_path / "source" / "kernel.cc"
    upstream.parent.mkdir()
    upstream.write_text(SOURCE)
    kernel_dir = tmp_path / "work"
    kernel_dir.mkdir()
    func = _stub_func("kernel", upstream)
    func._include_dirs = ["kernel/include"]

    compile_external_kernels(
        [func], kernel_dir, "aie2p", include_dirs=[tmp_path / "design/include"]
    )

    assert stub_compiler[0]["include_dirs"] == [
        "kernel/include",
        str(upstream.parent),
        tmp_path / "design/include",
    ]


def test_source_string_design_include_dirs_follow_kernel_dirs(tmp_path, stub_compiler):
    func = _stub_func("kernel", None)
    func._source_file = None
    func._source_string = SOURCE
    func._include_dirs = ["kernel/include"]

    compile_external_kernels(
        [func], tmp_path, "aie2p", include_dirs=[tmp_path / "design/include"]
    )

    assert stub_compiler[0]["include_dirs"] == [
        "kernel/include",
        tmp_path / "design/include",
    ]
    assert func._include_dirs == ["kernel/include"]


def test_reused_kernel_compiles_into_each_design_directory(stub_compiler, tmp_path):
    """One ExternalFunction compiled by two designs must land in both directories.

    A kernel built at module scope outlives the design that compiled it, so the
    record of "already compiled" has to name the directory it wrote into.  Keyed
    on a bare flag, the second design skips its own compile and its kernel_dir
    holds no object for aiecc's linker -- and never sees its own -I paths.
    """
    src = tmp_path / "k.cc"
    src.write_text(SOURCE)
    func = _stub_func("k", src)
    dir_a = tmp_path / "design_a"
    dir_a.mkdir()
    dir_b = tmp_path / "design_b"
    dir_b.mkdir()

    compile_external_kernels(
        [func], str(dir_a), "aie2p", include_dirs=[tmp_path / "inc_a"]
    )
    compile_external_kernels(
        [func], str(dir_b), "aie2p", include_dirs=[tmp_path / "inc_b"]
    )

    assert (dir_a / "k.o").exists()
    assert (dir_b / "k.o").exists()
    assert stub_compiler[1]["include_dirs"][-1] == tmp_path / "inc_b"


def test_materialized_source_keeps_umask_permissions(tmp_path):
    """Staging through mkstemp must not narrow the source to 0600."""
    written = tmp_path / "from_string.cc"
    _write_source(str(written), SOURCE)

    copied = tmp_path / "from_file.cc"
    _copy_source(str(copied), str(written))

    expected = 0o666 & ~compile_utils._UMASK
    assert os.stat(written).st_mode & 0o777 == expected
    assert os.stat(copied).st_mode & 0o777 == expected


def test_failed_write_leaves_no_temp_files(tmp_path):
    """A raising write must not litter the kernel directory with .tmp sources."""
    dest = tmp_path / "kernel.cc"
    with pytest.raises(FileNotFoundError):
        _copy_source(str(dest), str(tmp_path / "missing.cc"))

    assert list(tmp_path.iterdir()) == []
