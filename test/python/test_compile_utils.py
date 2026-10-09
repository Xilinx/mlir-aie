# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest --noconftest %s
# REQUIRES: peano

"""Compiler invocation tests against a stand-in aiecc."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

import pytest

import aie.utils.config as config
from aie.iron import ExternalFunction
from aie.iron.device import NPU1Col1


@pytest.fixture
def compile_utils():
    source = Path(__file__).resolve().parents[2] / "python/utils/compile/utils.py"
    spec = importlib.util.spec_from_file_location("compile_utils", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def aiecc(tmp_path, monkeypatch):
    """An executable at `AIECC_PATH` that records each call, a JSON line in the
    file this returns.

    It writes `AIECC_STDOUT` and `AIECC_STDERR` (`AIECC_STDERR_TIMES` times); given --await-link-files, it
    then creates `awaiting` beside that file and reads stdin to its end. It
    exits with `AIECC_EXIT`.
    """
    calls = tmp_path / "aiecc_calls.jsonl"
    stand_in = tmp_path / "tools" / "aiecc"
    stand_in.parent.mkdir()
    stand_in.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "call = {'argv': sys.argv, 'cwd': os.getcwd(),\n"
        "        'input': Path(sys.argv[1]).read_text()}\n"
        "sys.stdout.write(os.environ.get('AIECC_STDOUT', ''))\n"
        "sys.stderr.write(os.environ.get('AIECC_STDERR', '')\n"
        "                 * int(os.environ.get('AIECC_STDERR_TIMES', '1')))\n"
        "sys.stdout.flush()\n"
        "sys.stderr.flush()\n"
        "if '--await-link-files' in sys.argv:\n"
        f"    Path({str(tmp_path / 'awaiting')!r}).touch()\n"
        "    call['stdin'] = sys.stdin.read()\n"
        "    call['linked'] = sorted(os.listdir())\n"
        f"with open({str(calls)!r}, 'a') as f:\n"
        "    f.write(json.dumps(call) + '\\n')\n"
        "sys.exit(int(os.environ.get('AIECC_EXIT', '0')))\n"
    )
    stand_in.chmod(0o755)
    monkeypatch.setenv("AIECC_PATH", str(stand_in))
    monkeypatch.chdir(tmp_path)
    return calls


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("use_chess", [False, True])
@pytest.mark.parametrize("full_elf", [False, True])
def test_compile_uses_work_dir(
    compile_utils, aiecc, tmp_path, relative, use_chess, full_elf
):
    work_dir = tmp_path / "build dir"
    work_dir.mkdir()
    output = tmp_path / "output.bin"
    paths = {"insts_path": output, "xclbin_path": "design.xclbin"}
    if full_elf:
        paths["full_elf_path"] = output

    compile_utils.compile_mlir_module(
        "module {}",
        work_dir=os.path.relpath(work_dir) if relative else work_dir,
        use_chess=use_chess,
        pdi_path="design.pdi",
        elf_path="design.elf",
        **paths,
    )

    assert Path.cwd() == tmp_path
    [call] = map(json.loads, aiecc.read_text().splitlines())
    cmd = call["argv"]
    assert Path(call["cwd"]) == work_dir
    assert Path(cmd[0]) == tmp_path / "tools/aiecc"
    assert call["input"] == "module {}"
    assert Path(cmd[1]) == work_dir / "aie.mlir"
    assert f"--tmpdir={work_dir}" in cmd
    assert f"--output-dir={work_dir}" in cmd
    assert "--get-input-with-addresses" in cmd
    assert "--await-link-files" not in cmd
    if use_chess:
        assert "--xchesscc" in cmd
    else:
        assert f"--peano={os.path.abspath(config.peano_install_dir())}" in cmd
    if full_elf:
        assert f"--full-elf-name={output}" in cmd
        assert "--get-npu-insts" not in cmd
        assert "--get-xclbin" not in cmd
    else:
        assert f"--npu-insts-name={output}" in cmd
        assert "--xclbin-name=design.xclbin" in cmd
    assert "--pdi-name=design.pdi" in cmd
    assert "--elf-name=design.elf" in cmd


@pytest.mark.parametrize("failure", [None, "stderr", "stdout"])
def test_no_work_dir_preserves_cwd_and_cleans_up(
    compile_utils, aiecc, monkeypatch, tmp_path, failure
):
    if failure:
        monkeypatch.setenv("AIECC_EXIT", "1")
        monkeypatch.setenv(f"AIECC_{failure.upper()}", "compiler failed")
        with pytest.raises(RuntimeError, match="exit code 1:\ncompiler failed"):
            compile_utils.compile_mlir_module("module {}", insts_path="insts.bin")
    else:
        compile_utils.compile_mlir_module("module {}", insts_path="insts.bin")
    [call] = map(json.loads, aiecc.read_text().splitlines())
    assert Path(call["cwd"]) == tmp_path
    assert call["input"] == "module {}"
    assert "--npu-insts-name=insts.bin" in call["argv"]
    assert not any(arg.startswith("--output-dir=") for arg in call["argv"])
    assert not Path(call["argv"][1]).exists()
    assert Path.cwd() == tmp_path


def test_link_files_are_built_while_aiecc_waits(compile_utils, aiecc, tmp_path):
    work_dir = tmp_path / "build"
    work_dir.mkdir()

    def build_link_files():
        (work_dir / "kernel.o").write_bytes(b"object")

    compile_utils.compile_mlir_module(
        "module {}",
        insts_path="insts.bin",
        work_dir=work_dir,
        build_link_files=build_link_files,
    )

    [call] = map(json.loads, aiecc.read_text().splitlines())
    assert "--await-link-files" in call["argv"]
    assert call["stdin"] == "\n"
    assert call["linked"] == ["aie.mlir", "kernel.o"]


def test_aiecc_output_does_not_stall_it_while_link_files_build(
    compile_utils, aiecc, monkeypatch, tmp_path
):
    work_dir = tmp_path / "build"
    work_dir.mkdir()
    monkeypatch.setenv("AIECC_STDERR", "progress\n")
    monkeypatch.setenv("AIECC_STDERR_TIMES", str(1 << 17))

    def build_link_files():
        deadline = time.monotonic() + 60
        while not (tmp_path / "awaiting").exists():
            assert time.monotonic() < deadline, "aiecc stalled on its output"
            time.sleep(0.01)

    compile_utils.compile_mlir_module(
        "module {}",
        insts_path="insts.bin",
        work_dir=work_dir,
        build_link_files=build_link_files,
    )


def test_link_files_failure_stops_aiecc(compile_utils, aiecc, tmp_path):
    work_dir = tmp_path / "build"
    work_dir.mkdir()

    def build_link_files():
        raise ValueError("kernel failed to compile")

    with pytest.raises(ValueError, match="kernel failed to compile"):
        compile_utils.compile_mlir_module(
            "module {}",
            insts_path="insts.bin",
            work_dir=work_dir,
            build_link_files=build_link_files,
        )
    assert not aiecc.exists()


def test_link_files_need_work_dir(compile_utils, aiecc):
    with pytest.raises(ValueError, match="build_link_files requires work_dir"):
        compile_utils.compile_mlir_module(
            "module {}", insts_path="insts.bin", build_link_files=lambda: None
        )


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
    monkeypatch.setenv("AIECC_PATH", os.path.relpath(sys.executable))

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
    compile_utils, aiecc, tmp_path
):
    """Only kernels the current module declares reach the auto-build and its
    arch check: `ExternalFunction._instances` also holds kernels left over from
    an earlier, unrelated compile in the same process."""
    work_dir = tmp_path / "build"
    work_dir.mkdir()
    referenced_source = tmp_path / "referenced.cc"
    referenced_source.write_text('extern "C" void referenced_kernel() {}\n')
    stale_source = tmp_path / "stale.cc"
    stale_source.write_text("#error a stale kernel is not built\n")
    ExternalFunction._instances.clear()
    referenced = ExternalFunction(
        "referenced_kernel", source_file=str(referenced_source), arg_types=[]
    )
    stale = ExternalFunction(
        "stale_kernel_from_earlier_compile",
        source_file=str(stale_source),
        arg_types=[],
    )
    stale.built_for_arch = "aie2p"

    compile_utils.compile_mlir_module(
        "module { func.func private @referenced_kernel() }",
        insts_path="insts.bin",
        work_dir=work_dir,
        device=NPU1Col1(),
    )

    assert (work_dir / referenced.object_file_name).is_file()
    assert not (work_dir / stale.object_file_name).exists()


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
