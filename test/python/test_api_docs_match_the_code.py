# test_api_docs_match_the_code.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""The kernel reference docs must resolve against the current public API.

These checks stay intentionally scoped to the kernel documentation this branch
actually advertises: ``docs/api/kernels.md`` and the kernel programming guide.
"""

import importlib
import re
from pathlib import Path

_DOC = Path(__file__).parents[2] / "docs" / "api" / "kernels.md"


def _blocks(doc: str):
    """Yield ``(module, [member, ...])`` for each mkdocstrings block in *doc*."""
    for block in re.split(r"(?m)^::: ", doc)[1:]:
        module = block.split("\n", 1)[0].strip()
        members = re.search(r"^      members:\n((?:        - \S+\n)+)", block, re.M)
        yield module, re.findall(r"        - (\S+)", members.group(1)) if members else []


def test_documented_kernel_modules_and_members_exist():
    missing = []
    for module, members in _blocks(_DOC.read_text()):
        try:
            mod = importlib.import_module(f"aie.{module}")
        except ImportError as exc:  # a doc naming a module that moved
            missing.append(f"aie.{module}: {exc}")
            continue
        missing += [f"aie.{module}.{m}" for m in members if not hasattr(mod, m)]
    assert not missing, f"{_DOC.name} documents symbols that do not exist: {missing}"


def test_kernel_public_all_entries_resolve():
    """Kernel-facing ``__all__`` exports must still resolve.

    ``from module import *`` raises on a missing name, but nothing else does,
    so an entry left behind by a move survives until someone uses the star
    import.
    """
    import importlib

    modules = [
        "aie.iron.kernels",
        "aie.iron.algorithms",
    ]
    missing = []
    for name in modules:
        mod = importlib.import_module(name)
        missing += [
            f"{name}.{n}" for n in getattr(mod, "__all__", []) if not hasattr(mod, n)
        ]
    assert not missing, f"__all__ names symbols that do not exist: {missing}"


_GUIDE = Path(__file__).parents[2] / "programming_guide" / "kernels_library.md"


def test_guide_does_not_name_contract_fields_that_were_removed():
    """Every ``contract.<field>`` the guide shows must still be a field.

    The guide documented ``contract.overflow`` and ``contract.rounding`` for a
    while after both were deleted, because prose describing an attribute is
    not executed by anything. Reading the dataclass is enough to catch it.
    """
    import dataclasses

    from aie.iron.kernels import KernelContract

    real = {f.name for f in dataclasses.fields(KernelContract)}
    real |= {n for n in dir(KernelContract) if not n.startswith("_")}
    named = set(re.findall(r"contract\.([a-z_]+)", _GUIDE.read_text()))
    assert not (named - real), (
        f"{_GUIDE.name} names contract fields that no longer exist: "
        f"{sorted(named - real)}"
    )
