# test_api_docs_match_the_code.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Every symbol the API reference lists must still exist.

``docs/api/*.md`` names the members mkdocstrings renders, one per line under
a ``::: module`` block. Nothing checked those names against the modules, so
deleting or renaming a function left the reference advertising an API that
was gone -- silently, because mkdocs only warns.
"""

import importlib
import re
from pathlib import Path

import pytest

_DOCS = sorted((Path(__file__).parents[2] / "docs" / "api").glob("*.md"))


def _blocks(doc: str):
    """Yield ``(module, [member, ...])`` for each mkdocstrings block in *doc*."""
    for block in re.split(r"(?m)^::: ", doc)[1:]:
        module = block.split("\n", 1)[0].strip()
        members = re.search(r"^      members:\n((?:        - \S+\n)+)", block, re.M)
        if members:
            yield module, re.findall(r"        - (\S+)", members.group(1))


@pytest.mark.parametrize("path", _DOCS, ids=lambda p: p.name)
def test_documented_members_exist(path):
    missing = []
    for module, members in _blocks(path.read_text()):
        try:
            mod = importlib.import_module(f"aie.{module}")
        except ImportError as exc:  # a doc naming a module that moved
            missing.append(f"aie.{module}: {exc}")
            continue
        missing += [f"aie.{module}.{m}" for m in members if not hasattr(mod, m)]
    assert not missing, f"{path.name} documents symbols that do not exist: {missing}"
