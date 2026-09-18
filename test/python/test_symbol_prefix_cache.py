# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""Symbol-prefix cache and subprocess regression tests; no NPU required."""

from pathlib import Path
from types import SimpleNamespace

import aie.utils.compile.utils as compile_utils
import pytest


@pytest.fixture
def tools(monkeypatch):
    calls = []

    def compile_object(output_path, **kwargs):
        calls.append("compile")
        Path(output_path).write_text("add_one\nhelper\nop0_helper\n")

    def run(args, **kwargs):
        calls.append(args[0])
        obj = Path(args[-1])
        if args[0] == "nm":
            assert args[1:3] == ["--defined-only", "--extern-only"]
            stdout = "".join(f"00000000 T {s}\n" for s in obj.read_text().splitlines())
            return SimpleNamespace(returncode=0, stdout=stdout.encode())
        assert args[0] == "objcopy"
        mapping = Path(args[1].split("=", 1)[1])
        names = dict(line.split() for line in mapping.read_text().splitlines())
        obj.write_text("".join(f"{names[s]}\n" for s in obj.read_text().splitlines()))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compile_utils, "compile_cxx_core_function", compile_object)
    monkeypatch.setattr(compile_utils.config, "nm_path", lambda: "nm")
    monkeypatch.setattr(compile_utils.config, "objcopy_path", lambda: "objcopy")
    monkeypatch.setattr(compile_utils.subprocess, "run", run)
    return calls


@pytest.fixture
def func():
    return SimpleNamespace(
        _name="op0_add_one",
        _original_name="add_one",
        _source_string='extern "C" void add_one() {}',
        _source_file=None,
        _include_dirs=[],
        _compile_flags=[],
        _symbol_prefix="op0",
        _compiled=False,
        object_file_name="op0_add_one.o",
    )


@pytest.mark.parametrize("source_kind", ["string", "file"])
@pytest.mark.parametrize(
    "cache_state",
    [
        "missing",
        "unprefixed",
        "legacy",
        "versionless",
        "prefixed",
        "corrupt",
        "invalid-encoding",
        "stale",
        "wrong-prefix",
    ],
)
def test_untrusted_cache_is_recompiled(tmp_path, tools, func, cache_state, source_kind):
    obj = tmp_path / func.object_file_name
    stamp = Path(compile_utils._symbol_prefix_stamp_path(str(obj), "op0_"))
    if source_kind == "file":
        source = tmp_path / "kernel.cc"
        source.write_text(func._source_string)
        func._source_string = None
        func._source_file = str(source)

    if cache_state != "missing":
        obj.write_text(
            {
                "unprefixed": "add_one\nhelper\n",
                "legacy": "op0_add_one\nhelper\n",
            }.get(cache_state, "op0_add_one\nop0_helper\n")
        )
    if cache_state == "corrupt":
        stamp.write_text("{")
    elif cache_state == "invalid-encoding":
        stamp.write_bytes(b"\xff")
    elif cache_state == "stale":
        compile_utils._write_symbol_prefix_stamp(str(obj), "op0_")
        obj.write_text("op0_add_one\n")
    elif cache_state == "versionless":
        stamp.write_text(
            '{"prefix": "op0_", "object_sha256": "'
            + compile_utils._sha256_file(str(obj))
            + '"}'
        )
    elif cache_state == "wrong-prefix":
        compile_utils._write_symbol_prefix_stamp(str(obj), "other_")
        Path(compile_utils._symbol_prefix_stamp_path(str(obj), "other_")).replace(stamp)

    compile_utils.compile_external_kernel(func, tmp_path, "aie2")

    assert tools == ["compile", "nm", "objcopy", "nm"]
    assert obj.read_text() == "op0_add_one\nop0_helper\nop0_op0_helper\n"
    assert compile_utils._has_current_symbol_prefix_stamp(str(obj), "op0_")
    assert func._compiled
    assert func._compiled_dir == str(tmp_path)

    # A fresh instance reusing this disk cache must not compile or rename again.
    func._compiled = False
    func._compiled_dir = None
    compile_utils.compile_external_kernel(func, tmp_path, "aie2")
    assert tools == ["compile", "nm", "objcopy", "nm"]
    assert func._compiled
    assert func._compiled_dir == str(tmp_path)


@pytest.mark.parametrize(
    "failure", ["compile", "prefix", "stamp-write", "stamp-replace"]
)
def test_failed_update_is_rebuilt_on_retry(tmp_path, monkeypatch, tools, func, failure):
    obj = tmp_path / func.object_file_name
    obj.write_text("op0_add_one\n")
    compile_utils._write_symbol_prefix_stamp(str(obj), "op0_")
    stamp = Path(compile_utils._symbol_prefix_stamp_path(str(obj), "op0_"))
    obj.write_text("changed\n")

    def fail(*args, **kwargs):
        assert not stamp.exists()
        raise OSError("interrupted")

    with monkeypatch.context() as patch:
        if failure == "compile":
            patch.setattr(compile_utils, "compile_cxx_core_function", fail)
        elif failure == "prefix":
            patch.setattr(compile_utils, "prefix_symbols_in_object", fail)
        elif failure == "stamp-write":
            patch.setattr(compile_utils.json, "dump", fail)
        else:
            replace = compile_utils._replace_staged_source

            def fail_stamp_replace(src, dest):
                if Path(dest) == stamp:
                    fail()
                replace(src, dest)

            patch.setattr(compile_utils, "_replace_staged_source", fail_stamp_replace)
        with pytest.raises(OSError, match="interrupted"):
            compile_utils.compile_external_kernel(func, tmp_path, "aie2")

    assert not func._compiled
    assert not stamp.exists()
    assert not list(tmp_path.glob("*.tmp"))
    tools.clear()
    compile_utils.compile_external_kernel(func, tmp_path, "aie2")
    assert tools == ["compile", "nm", "objcopy", "nm"]
    assert obj.read_text() == "op0_add_one\nop0_helper\nop0_op0_helper\n"
    assert compile_utils._has_current_symbol_prefix_stamp(str(obj), "op0_")


def test_unprefixed_cache_hit_is_unchanged(tmp_path, tools, func):
    func._symbol_prefix = None
    obj = tmp_path / func.object_file_name
    obj.write_bytes(b"cached object")
    compile_utils.compile_external_kernel(func, tmp_path, "aie2")
    assert tools == []
    assert obj.read_bytes() == b"cached object"
    assert func._compiled
    assert func._compiled_dir == str(tmp_path)


@pytest.mark.parametrize("mutation", ["object", "stamp"])
def test_shared_owner_revalidates_prefix_stamp(tmp_path, tools, func, mutation):
    from aie.iron.kernel import ExternalFunction

    first = ExternalFunction(
        "add_one", source_string=func._source_string, symbol_prefix="op0"
    )
    second = ExternalFunction(
        "helper",
        object_file_name=first.object_file_name,
        source_string=func._source_string,
        symbol_prefix="op0",
    )
    assert first.object_file is second.object_file
    compile_utils.compile_external_kernel(first, tmp_path, "aie2")
    compile_utils.compile_external_kernel(second, tmp_path, "aie2")
    assert tools == ["compile", "nm", "objcopy", "nm"]

    obj = tmp_path / first.object_file_name
    if mutation == "object":
        obj.write_text("changed\n")
    else:
        Path(compile_utils._symbol_prefix_stamp_path(str(obj), "op0_")).unlink()
    tools.clear()
    compile_utils.compile_external_kernel(second, tmp_path, "aie2")
    assert tools == ["compile", "nm", "objcopy", "nm"]
    assert obj.read_text() == "op0_add_one\nop0_helper\nop0_op0_helper\n"


def test_missing_entry_point_does_not_stamp_object(tmp_path, tools, func):
    func._name = "op0_missing"
    func._original_name = "missing"
    with pytest.raises(RuntimeError, match="does not define 'missing'"):
        compile_utils.compile_external_kernel(func, tmp_path, "aie2")
    obj = tmp_path / func.object_file_name
    assert not obj.exists()
    assert not Path(compile_utils._symbol_prefix_stamp_path(str(obj), "op0_")).exists()
    assert not func._compiled


def test_nm_failure_never_runs_objcopy(tmp_path, monkeypatch, tools):
    calls = []

    def fail_nm(args, **kwargs):
        calls.append(args[0])
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"invalid object")

    monkeypatch.setattr(compile_utils.subprocess, "run", fail_nm)
    with pytest.raises(RuntimeError, match="Symbol listing failed: invalid object"):
        compile_utils.prefix_symbols_in_object(str(tmp_path / "kernel.o"), "op0_")
    assert calls == ["nm"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("outcome", ["success", "nonzero", "exception"])
def test_symbol_map_is_private_and_cleaned(tmp_path, monkeypatch, tools, outcome):
    obj = tmp_path / "kernel with spaces.o"
    obj.write_text("helper\nop0_helper\n")
    existing_map = tmp_path / f"{obj.name}.symbol_map"
    existing_map.write_text("unrelated file")
    run = compile_utils.subprocess.run
    maps = []

    def objcopy(args, **kwargs):
        if args[0] == "objcopy":
            mapping = Path(args[1].split("=", 1)[1])
            maps.append(mapping)
            assert mapping != existing_map
            assert (
                mapping.read_text() == "helper op0_helper\nop0_helper op0_op0_helper\n"
            )
            if outcome == "nonzero":
                return SimpleNamespace(returncode=1, stderr=b"bad object")
            if outcome == "exception":
                raise OSError("cannot execute")
        return run(args, **kwargs)

    monkeypatch.setattr(compile_utils.subprocess, "run", objcopy)
    if outcome == "success":
        compile_utils.prefix_symbols_in_object(str(obj), "op0_")
        assert obj.read_text() == "op0_helper\nop0_op0_helper\n"
    else:
        with pytest.raises((RuntimeError, OSError)):
            compile_utils.prefix_symbols_in_object(str(obj), "op0_")
    assert len(maps) == 1
    assert not maps[0].parent.exists()
    assert existing_map.read_text() == "unrelated file"
