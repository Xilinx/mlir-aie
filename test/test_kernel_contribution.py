# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Write the kernel contribution checklist for synthetic pull requests."""

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "utils/kernel_checks/contribution.py"

FACTORIES = '''
def _helper(name, filename):
    return _make_extern(
        name,
        _kernel_source(f"eltwise/{filename}"),
        [],
        contract=KernelContract(trace=Trace.whole_call(), reference=None),
    )


def add(tile_size: int = 64) -> ExternalFunction:
    """Add two tiles."""
    return _helper("add", "add.cc")
'''

CASES = """
CASES = [
    Case("add", calls=16, smoke=True),
    *[check(name, tag="edge-tiny") for name in ("add",)],
]
"""

NEW = '''

def scale(tile_size: int = 64, dtype=None) -> ExternalFunction:
    return _make_extern(
        "scale",
        _kernel_source("eltwise/scale.cc"),
        [],
        contract=KernelContract(
            trace=Trace.partial("setup outside"), tolerance=Tolerance.exact()
        ),
    )


def bare() -> ExternalFunction:
    """No contract."""
    return _make_extern("bare", _kernel_source("eltwise/bare.cc"), [])
'''


@pytest.fixture(scope="module")
def contribution():
    spec = importlib.util.spec_from_file_location("contribution", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def write(repo, files):
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(textwrap.dedent(text))
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    write(
        tmp_path,
        {
            "python/iron/kernels/__init__.py": '__all__ = ["add", "scale"]\n',
            "python/iron/kernels/eltwise.py": FACTORIES,
            "test/python/npu/kernel_cases.py": CASES,
            "test/python/test_kernel_contracts.py": "NOT_JUDGED = {}\n",
            "docs/api/kernels.md": "::: iron.kernels.eltwise\n",
            "aie_kernels/eltwise/add.cc": '#include "add.h"\n',
            "aie_kernels/eltwise/add.h": "",
            "README.md": "",
        },
    )
    return tmp_path


def checklist(contribution, repo, tmp_path):
    out = tmp_path / "checklist.md"
    args = ["--repo", str(repo), "--base", "HEAD~1", "--head", "HEAD"]
    assert contribution.main(args + ["--out", str(out)]) == 0
    return out.read_text() if out.exists() else None


def test_a_pull_request_without_kernel_code_gets_no_comment(contribution, repo):
    write(repo, {"README.md": "docs only\n"})
    assert checklist(contribution, repo, repo) is None


def test_a_new_factory_is_told_what_it_still_needs(contribution, repo):
    write(
        repo,
        {
            "python/iron/kernels/eltwise.py": FACTORIES + NEW,
            "test/python/npu/kernel_cases.py": CASES
            + 'CASES.append(Case("scale", calls=16))\n',
            "aie_kernels/eltwise/scale.cc": "",
        },
    )
    text = checklist(contribution, repo, repo)
    assert text.startswith(contribution.MARKER)
    scale = [line for line in text.splitlines() if line.startswith("- `scale`")]
    said = "\n".join(scale)
    assert "Trace.whole_call()" in said  # timed, but traced in part
    assert "`note=`" in said
    assert "docstring" in said
    assert "smoke=True" in said
    assert "@dtypes" in said
    assert "export it" not in said  # it is in __all__
    bare = "\n".join(line for line in text.splitlines() if "`bare`" in line)
    assert "contract=KernelContract" in bare
    assert "no case in" in bare
    assert "export it" in bare
    assert '-k "scale or bare"' in text
    assert "--baseline-sources" in text


def test_a_header_change_names_the_factories_that_build_it(contribution, repo):
    write(repo, {"aie_kernels/eltwise/add.h": "// faster\n"})
    text = checklist(contribution, repo, repo)
    assert "#### Changed factories" in text
    assert "| `add` | ✅ | whole call | 2 | ✅ 1 | ✅ 1 |" in text
    # A complete factory gets its row and nothing to fix.
    assert "- `add`:" not in text


def test_a_factory_left_out_on_purpose_is_not_nagged(contribution, repo):
    write(
        repo,
        {
            "python/iron/kernels/eltwise.py": FACTORIES + NEW,
            "test/python/test_kernel_contracts.py": (
                "NOT_JUDGED = {**{n: 'pair' for n in ('bare',)}}\n"
            ),
        },
    )
    text = checklist(contribution, repo, repo)
    bare = "\n".join(line for line in text.splitlines() if "`bare`" in line)
    assert "contract=KernelContract" not in bare
    assert "no case in" not in bare
    assert "| `bare` | — | — | not judged | — | — |" in text


def test_a_moved_source_is_not_unclaimed_but_a_stale_name_is(contribution, repo):
    git(repo, "mv", "aie_kernels/eltwise/add.cc", "aie_kernels/eltwise/sum.cc")
    write(repo, {})
    text = checklist(contribution, repo, repo)
    assert "Sources no factory names" in text  # sum.cc: nothing names it yet
    assert "`aie_kernels/eltwise/sum.cc`" in text
    unclaimed = text.split("Sources no factory names")[1]
    assert "add.cc" not in unclaimed  # it is gone, not unclaimed
    assert "builds `aie_kernels/eltwise/add.cc`, which is not in the tree" in text


def test_a_branch_behind_main_is_not_shown_mains_changes(contribution, repo):
    git(repo, "branch", "fork")
    write(repo, {"python/iron/kernels/eltwise.py": FACTORIES + NEW})  # main moves
    git(repo, "checkout", "-q", "fork")
    write(repo, {"aie_kernels/eltwise/add.h": "// faster\n"})
    args = ["--repo", str(repo), "--base", "master", "--head", "fork"]
    branches = subprocess.run(
        ["git", "-C", str(repo), "branch", "--format=%(refname:short)"],
        capture_output=True,
        text=True,
    ).stdout.split()
    args[3] = next(b for b in branches if b in ("main", "master"))
    out = repo / "x.md"
    assert contribution.main(args + ["--out", str(out)]) == 0
    text = out.read_text()
    assert "`add`" in text
    assert "scale" not in text and "bare" not in text and "Removed" not in text


def test_a_library_wide_change_folds_the_table_and_runs_everything(
    contribution, repo
):
    many = "".join(
        f'''

def k{i}() -> ExternalFunction:
    """Kernel {i}."""
    return _helper("k{i}", "add.cc")
'''
        for i in range(contribution.SHARED + 1)
    )
    write(repo, {"python/iron/kernels/eltwise.py": FACTORIES + many})
    text = checklist(contribution, repo, repo)
    assert f"<summary>{contribution.SHARED + 1} factories</summary>" in text
    assert "-k " not in text
    write(repo, {"aie_kernels/eltwise/add.cc": '#include "add.h"\n// all\n'})
    text = checklist(contribution, repo, repo)
    assert f"`aie_kernels/eltwise/add.cc`: {contribution.SHARED + 2} factories" in text
    assert "Run the whole suite" in text


def test_a_header_nothing_includes_is_called_out(contribution, repo):
    write(repo, {"aie_kernels/eltwise/old.h": "// unused\n"})
    text = checklist(contribution, repo, repo)
    assert "- `aie_kernels/eltwise/old.h`: no source includes it" in text


def test_unreadable_code_never_fails_the_pull_request(contribution, repo):
    write(repo, {"python/iron/kernels/eltwise.py": "def broken(:\n"})
    text = checklist(contribution, repo, repo)
    assert "`python/iron/kernels/eltwise.py` does not parse" in text
    assert "Removed" not in text
    args = ["--repo", str(repo), "--base", "nope", "--head", "HEAD"]
    assert contribution.main(args + ["--out", str(repo / "x.md")]) == 0


def test_it_reads_the_real_kernel_library(contribution):
    """Every kernel source maps to a factory and every factory's contract
    is found, so a refactor of the library does not silently blind it."""
    if subprocess.run(["git", "-C", str(ROOT), "rev-parse"]).returncode:
        pytest.skip("not a git checkout")
    tree = contribution.Tree(ROOT, "HEAD")
    found = contribution.factories(tree)
    judged = contribution.not_judged(tree)
    assert len(found) > 50 and judged
    assert not [n for n, f in found.items() if f.contract == "unknown"]
    assert not [n for n, f in found.items() if f.contract == "no" and n not in judged]
    table = contribution.cases(tree)
    assert not [n for n in found if n not in table and n not in judged]
    units = [p for p in tree.ls("aie_kernels/") if p.endswith(".cc")]
    assert not [p for p in units if not contribution._attributed(found, p)]
