# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""Instructions-only builds use the backend ABI named by their key."""

import hashlib
import inspect
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

import aie.utils.compile.jit.compilabledesign as cd
from aie.utils import config
from aie.utils.compile import utils as compile_utils
from aie.utils.compile.jit import _hash, _manifest


@pytest.fixture
def design_factory(tmp_path, monkeypatch, npu2_device):
    monkeypatch.setattr(cd, "NPU_CACHE_HOME", tmp_path / "cache")
    source = tmp_path / "input.mlir"
    source.write_text("module {}")

    def make(fold, **kwargs):
        monkeypatch.setattr(
            cd.CompilableDesign,
            "_resolve_fold_ddr_addr_offset",
            staticmethod(lambda: fold),
        )
        design = cd.CompilableDesign(source, **kwargs)
        monkeypatch.setattr(
            design,
            "_generated_for",
            lambda **kwargs: SimpleNamespace(
                requires_full_elf=False,
                external_kernels=[],
                mlir_text="module {}",
                entry_stream_options=[],
            ),
        )
        return design

    return make


@pytest.fixture
def compiler_calls(monkeypatch):
    signature = inspect.signature(cd.compile_mlir_module)
    seen = []

    def compiler(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        fold = bound.arguments["fold_ddr_addr_offset"]
        seen.append(fold)
        Path(bound.arguments["insts_path"]).write_bytes(
            b"folded" if fold else b"unfolded"
        )

    monkeypatch.setattr(cd, "compile_mlir_module", compiler)
    return seen


@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("use_cache", [False, True])
def test_insts_only_compiler_uses_the_keyed_backend_abi(
    tmp_path, design_factory, compiler_calls, fold, explicit, use_cache
):
    design = design_factory(fold, insts_only=True, use_cache=use_cache)
    output = tmp_path / "insts.bin" if explicit else None
    first = design.compile(inst_path=output)
    assert design.compile(inst_path=output) == first
    assert compiler_calls == [fold] * (1 if use_cache else 2)


def _legacy_path_hash(design, fold):
    # Path-input keys have no Python source identity. Reproduce their pre-ABI
    # artifact digest with the same tools, without depending on the migration salt.
    artifact = hashlib.sha256()
    artifact.update(f"fold_ddr_addr_offset={fold}".encode())
    tools = {
        "peano": config.peano_cxx_path,
        "aiecc": config.aiecc_path,
        "nm": config.nm_path,
        "objcopy": config.objcopy_path,
    }
    optional = {"bootgen", "aiebu-asm"}
    if design.full_elf or not design.insts_only:
        tools["bootgen"] = partial(config.aiecc_tool_path, "bootgen")
    if design.full_elf:
        tools["aiebu-asm"] = partial(config.aiecc_tool_path, "aiebu-asm")
    elif not design.insts_only:
        tools["xclbinutil"] = partial(config.aiecc_tool_path, "xclbinutil")
    for name, resolve in tools.items():
        identity = _hash._tool_identity(name, resolve, expected=name not in optional)
        artifact.update(f"{name}={identity}".encode())
    return hashlib.sha256(
        f"{design.recipe_hash}|{artifact.hexdigest()}".encode()
    ).hexdigest()[:24]


@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize(
    "flow", ["insts_only", "xclbin", "full_elf", "full_elf_insts_only"]
)
def test_abi_migration_changes_only_unfolded_insts_only_keys(
    design_factory, fold, flow
):
    design = design_factory(
        fold,
        insts_only=flow in ("insts_only", "full_elf_insts_only"),
        full_elf=flow in ("full_elf", "full_elf_insts_only"),
    )
    assert (design._compute_cache_hash() != _legacy_path_hash(design, fold)) == (
        flow == "insts_only" and not fold
    )


@pytest.mark.parametrize(
    "flag",
    [
        "--fold-ddr-addr-offset",
        "--fold-ddr-addr-offset=true",
        "--fold-ddr-addr-offset=false",
        "--fold-ddr-addr-offset=1",
        "-fold-ddr-addr-offset",
        "-fold-ddr-addr-offset=0",
    ],
)
def test_explicit_fold_options_keep_legacy_keys(design_factory, flag):
    design = design_factory(False, insts_only=True, aiecc_flags=[flag])
    assert design._compute_cache_hash() == _legacy_path_hash(design, False)


@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize("reuse", ["cache", "explicit", "shared"])
@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("override", [None, False, True])
def test_legacy_insts_only_artifacts_rebuild_then_hit(
    tmp_path, design_factory, compiler_calls, fold, reuse, with_image, override
):
    flags = (
        [] if override is None else [f"--fold-ddr-addr-offset={str(override).lower()}"]
    )
    design = design_factory(fold, insts_only=True, aiecc_flags=flags)
    image = tmp_path / "image.mlir" if with_image else None
    legacy_key = _legacy_path_hash(design, fold)
    if image is not None:
        image.write_text("module {}")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        legacy_key = hashlib.sha256((legacy_key + digest).encode()).hexdigest()[:24]

    output = None if reuse == "cache" else tmp_path / "insts.bin"
    if reuse == "cache":
        directory = cd.NPU_CACHE_HOME / legacy_key
        legacy_insts = directory / "insts.bin"
    else:
        legacy_insts = tmp_path / ("origin.bin" if reuse == "shared" else "insts.bin")
        directory = tmp_path / f"{legacy_insts.stem}.prj"
    directory.mkdir(parents=True)
    legacy_content = b"unfolded" if override is False else b"folded"
    legacy_insts.write_bytes(legacy_content)
    _manifest.record(directory, [], [])
    assert _manifest.is_valid(directory)

    if reuse != "cache":
        _manifest.record_outputs(directory, legacy_key, [legacy_insts])
        assert _manifest.outputs_current(directory, legacy_key, [legacy_insts])
        store = design._build_cache()
        store.store(legacy_key, directory, {"insts": legacy_insts})
        (tmp_path / "eligible.prj").mkdir()
        assert store.fetch(
            legacy_key,
            tmp_path / "eligible.prj",
            {"insts": tmp_path / "eligible.bin"},
        )
        assert (tmp_path / "eligible.bin").read_bytes() == legacy_content

    affected = not fold and override is None
    expected_calls = [False] if affected else []
    result = design.compile(inst_path=output, image=image)
    assert compiler_calls == expected_calls
    assert result[1].read_bytes() == (b"unfolded" if affected else legacy_content)
    assert design.compile(inst_path=output, image=image) == result
    assert compiler_calls == expected_calls
    if reuse != "cache":
        _, sibling = design.compile(inst_path=tmp_path / "sibling.bin", image=image)
        assert sibling.read_bytes() == result[1].read_bytes()
        assert compiler_calls == expected_calls


@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_insts_only_caller_fold_option_has_precedence(
    tmp_path, monkeypatch, design_factory, fold, override, explicit
):
    calls = []

    def aiecc(source, args, **kwargs):
        calls.append(args)
        output = next(
            a.split("=", 1)[1] for a in args if a.startswith("--npu-insts-name=")
        )
        Path(output).write_bytes(b"instructions")

    monkeypatch.setattr(config.config, "peano_install_dir", str(tmp_path))
    monkeypatch.setattr(compile_utils, "_run_aiecc", aiecc)
    flag = f"--fold-ddr-addr-offset={str(override).lower()}"
    design = design_factory(fold, insts_only=True, aiecc_flags=[flag])
    output = tmp_path / "insts.bin" if explicit else None
    result = design.compile(inst_path=output)
    assert design.compile(inst_path=output) == result
    assert len(calls) == 1
    fold_options = [a for a in calls[0] if a.startswith("--fold-ddr-addr-offset=")]
    assert fold_options == ([] if fold else ["--fold-ddr-addr-offset=false"]) + [flag]
