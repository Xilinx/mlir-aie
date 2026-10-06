# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest --noconftest %s

"""Compiler invocation tests without an installed toolchain."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import traceback
import types

import pytest


@pytest.fixture
def compile_utils(monkeypatch):
    source = Path(__file__).resolve().parents[2] / "python/utils/compile/utils.py"
    spec = importlib.util.spec_from_file_location("compile_utils", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "config",
        types.SimpleNamespace(
            aiecc_path=lambda: "tools/aiecc",
            peano_install_dir=lambda: "tools/peano",
        ),
    )
    return module


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("use_chess", [False, True])
@pytest.mark.parametrize("full_elf", [False, True])
def test_compile_uses_work_dir(
    compile_utils, monkeypatch, tmp_path, relative, use_chess, full_elf
):
    monkeypatch.chdir(tmp_path)
    work_dir = tmp_path / "build dir"
    work_dir.mkdir()
    output = tmp_path / "output.bin"
    paths = {"insts_path": output, "xclbin_path": "design.xclbin"}
    if full_elf:
        paths["full_elf_path"] = output

    def run(cmd, *, cwd, capture_output, text):
        assert Path.cwd() == tmp_path
        assert Path(cwd) == work_dir
        assert Path(cmd[0]) == tmp_path / "tools/aiecc"
        assert Path(cmd[1]).read_text() == "module {}"
        assert Path(cmd[1]) == work_dir / "aie.mlir"
        assert f"--tmpdir={work_dir}" in cmd
        assert f"--output-dir={work_dir}" in cmd
        assert "--get-input-with-addresses" in cmd
        if use_chess:
            assert "--xchesscc" in cmd
        else:
            assert f"--peano={tmp_path / 'tools/peano'}" in cmd
        if full_elf:
            assert f"--full-elf-name={output}" in cmd
            assert "--get-npu-insts" not in cmd
            assert "--get-xclbin" not in cmd
        else:
            assert f"--npu-insts-name={output}" in cmd
            assert "--xclbin-name=design.xclbin" in cmd
        assert "--pdi-name=design.pdi" in cmd
        assert "--elf-name=design.elf" in cmd
        assert capture_output and text
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(compile_utils.subprocess, "run", run)
    compile_utils.compile_mlir_module(
        "module {}",
        work_dir=os.path.relpath(work_dir) if relative else work_dir,
        use_chess=use_chess,
        pdi_path="design.pdi",
        elf_path="design.elf",
        **paths,
    )


@pytest.mark.parametrize("failure", [None, "stderr", "stdout"])
def test_no_work_dir_preserves_cwd_and_cleans_up(
    compile_utils, monkeypatch, tmp_path, failure
):
    monkeypatch.chdir(tmp_path)
    inputs = []

    def run(cmd, *, cwd, capture_output, text):
        assert cwd is None
        assert Path.cwd() == tmp_path
        inputs.append(Path(cmd[1]))
        assert inputs[-1].read_text() == "module {}"
        assert "--npu-insts-name=insts.bin" in cmd
        assert not any(arg.startswith("--output-dir=") for arg in cmd)
        return subprocess.CompletedProcess(
            cmd,
            1 if failure else 0,
            "compiler failed" if failure == "stdout" else "",
            "compiler failed" if failure == "stderr" else "",
        )

    monkeypatch.setattr(compile_utils.subprocess, "run", run)
    if failure:
        with pytest.raises(RuntimeError, match="exit code 1:\ncompiler failed"):
            compile_utils.compile_mlir_module("module {}", insts_path="insts.bin")
    else:
        compile_utils.compile_mlir_module("module {}", insts_path="insts.bin")
    assert len(inputs) == 1
    assert not inputs[0].exists()
    assert Path.cwd() == tmp_path


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
@pytest.mark.parametrize("legacy_notes", [False, True])
def test_run_aiecc_preserves_located_error(
    compile_utils, monkeypatch, stream, legacy_notes
):
    output = (
        "design.py:42:3: error: invalid operation\n"
        "note: detail\n"
        "design.py:40:3: note: previous operation"
    )
    error = ValueError("invalid operation")
    if legacy_notes:
        monkeypatch.setattr(error, "add_note", None, raising=False)
    seen = []

    def locate(text):
        seen.append(text)
        return error

    monkeypatch.setattr(compile_utils, "compile_error_from_output", locate)
    monkeypatch.setattr(
        compile_utils.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, **{stream: output}),
    )
    with pytest.raises(ValueError, match="invalid operation") as exc:
        compile_utils._run_aiecc("design.mlir", [])
    assert exc.value is error
    assert seen == [output]
    notes = ["[aiecc] note: detail", "[aiecc] design.py:40:3: note: previous operation"]
    rendered = "".join(traceback.format_exception(error))
    for note in notes:
        assert rendered.count(note) == 1
    assert rendered.index(notes[0]) < rendered.index(notes[1])
    if getattr(error, "add_note", None) is not None:
        assert error.__notes__ == notes
        assert str(error) == "invalid operation"
    else:
        assert str(error) == "invalid operation\n" + "\n".join(notes)


def test_run_aiecc_child_resolves_kernel_in_work_dir(
    compile_utils, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    work_dir = tmp_path / "build dir"
    work_dir.mkdir()
    (tmp_path / "kernel.o").write_text("stale caller object")
    (work_dir / "kernel.o").write_text("build object")
    script = work_dir / "check_cwd.py"
    script.write_text(
        "from pathlib import Path\n"
        "assert Path.cwd() == Path(__file__).parent\n"
        "assert Path('kernel.o').read_text() == 'build object'\n"
    )
    monkeypatch.setattr(
        compile_utils.config, "aiecc_path", lambda: os.path.relpath(sys.executable)
    )

    compile_utils._run_aiecc(os.path.relpath(script), [], cwd=work_dir)

    assert Path.cwd() == tmp_path


@pytest.mark.parametrize("relative", [False, True])
def test_copy_object_files_refreshes_work_dir(
    compile_utils, monkeypatch, tmp_path, relative
):
    monkeypatch.chdir(tmp_path)
    work_dir = tmp_path / "build dir"
    work_dir.mkdir()
    sources = [tmp_path / name for name in ("kernel.o", "helper.o")]
    for source in sources:
        source.write_bytes(b"current object")
        (work_dir / source.name).write_bytes(b"stale object")

    compile_utils._copy_object_files(
        [os.path.relpath(source) if relative else source for source in sources],
        work_dir,
    )

    for source in sources:
        assert (work_dir / source.name).read_bytes() == source.read_bytes()


def test_copy_object_files_already_in_work_dir(compile_utils, tmp_path):
    source = tmp_path / "kernel.o"
    source.write_bytes(b"current object")

    compile_utils._copy_object_files([source], tmp_path)

    assert source.read_bytes() == b"current object"


def test_copy_object_files_missing_source(compile_utils, tmp_path):
    source = tmp_path / "missing" / "kernel.o"
    dest = tmp_path / source.name
    dest.write_bytes(b"stale object")

    with pytest.raises(FileNotFoundError):
        compile_utils._copy_object_files([source], tmp_path)

    assert dest.read_bytes() == b"stale object"


def test_compile_mlir_module_ignores_stale_external_functions(
    compile_utils, monkeypatch, tmp_path
):
    """Only kernels the current module actually declares reach the auto-build
    (and its built_for_arch check) -- ``ExternalFunction._instances`` also holds
    unrelated entries left over from an earlier, unrelated compile in the same
    process, and those must not be treated as belonging to this one."""
    monkeypatch.chdir(tmp_path)
    work_dir = tmp_path / "build"
    work_dir.mkdir()

    class FakeExternalFunction:
        def __init__(self, name, built_for_arch):
            self.name = name
            self.source_file = "kernel.cc"
            self.built_for_arch = built_for_arch

    referenced = FakeExternalFunction("referenced_kernel", "aie2")
    stale = FakeExternalFunction("stale_kernel_from_earlier_compile", "aie2p")
    FakeExternalFunction._instances = {referenced, stale}

    monkeypatch.setitem(
        sys.modules,
        "aie.iron.kernel",
        types.SimpleNamespace(ExternalFunction=FakeExternalFunction),
    )
    monkeypatch.setattr(compile_utils, "resolve_target_arch", lambda device: "aie2")

    captured = {}

    def fake_compile_external_kernels(funcs, kernel_dir, target_arch, **kwargs):
        captured["funcs"] = list(funcs)

    monkeypatch.setattr(
        compile_utils, "compile_external_kernels", fake_compile_external_kernels
    )
    monkeypatch.setattr(
        compile_utils.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "", ""),
    )

    compile_utils.compile_mlir_module(
        "module { func.func private @referenced_kernel() }",
        insts_path="insts.bin",
        work_dir=work_dir,
        device=object(),
    )

    assert [f.name for f in captured["funcs"]] == ["referenced_kernel"]


def test_declared_link_with_picks_among_kernels_sharing_a_symbol(compile_utils):
    """The declaration's ``link_with`` picks among kernels sharing a symbol.

    An inline kernel keeps its bare symbol on every arch. A symbol no instance
    matches keeps them all, so the arch check can still report it.
    """
    aie2 = types.SimpleNamespace(name="setup", object_file_name="setup_aaaa.ll")
    aie2p = types.SimpleNamespace(name="setup", object_file_name="setup_bbbb.ll")
    other = types.SimpleNamespace(name="kernel", object_file_name="kernel.o")

    def select(funcs, text):
        declared = compile_utils._declared_objects(text)
        return compile_utils._select_declared_kernels(funcs, declared)

    text = """module {
      func.func private @setup() attributes {link_with = "setup_bbbb.ll"}
      func.func private @kernel(%arg0: i32)
    }"""
    assert select([aie2, aie2p, other], text) == [aie2p, other]
    stale = """module {
      func.func private @setup() attributes {link_with = "setup_cccc.ll"}
    }"""
    assert select([aie2, aie2p], stale) == [aie2, aie2p]
    both = """module {
      module @a {
        func.func private @setup() attributes {link_with = "setup_aaaa.ll"}
      }
      module @b {
        func.func private @setup() attributes {link_with = "setup_bbbb.ll"}
        func.func private @kernel(%arg0: i32)
      }
    }"""
    assert select([aie2, aie2p, other], both) == [aie2, aie2p, other]
    assert select([aie2, other], "module {}") == []
