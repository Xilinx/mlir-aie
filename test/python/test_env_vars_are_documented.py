# test_env_vars_are_documented.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Every environment variable the library reads is documented somewhere.

An undocumented knob is one only its author knows about. This finds the names
`python/` reads from the environment and requires each to appear in
`programming_guide/`, so adding one without a line about it fails here.
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_READS = re.compile(r'(?:environ(?:\.get)?\(|getenv\()\s*"([A-Z][A-Z_0-9]*)"')

# Read but not ours to document: set by the platform, the CI provider, or a
# third-party toolchain that documents them itself.
_EXTERNAL = {
    "GITHUB_ACTIONS",
    "GITHUB_SHA",
    "LM_LICENSE_FILE",
    "PYTHONPATH",
    "XILINXD_LICENSE_FILE",
    "XILINX_XRT",
    "LLVM_HOST_TRIPLE",
}


def _read_names() -> set[str]:
    names = set()
    for path in (_ROOT / "python").rglob("*.py*"):
        names |= set(_READS.findall(path.read_text(encoding="utf-8", errors="ignore")))
    return names - _EXTERNAL


def _documented() -> str:
    guide = _ROOT / "programming_guide"
    return "\n".join(
        p.read_text(encoding="utf-8", errors="ignore") for p in guide.rglob("*.md")
    )


def test_every_environment_variable_is_documented():
    docs = _documented()
    undocumented = sorted(n for n in _read_names() if n not in docs)
    assert not undocumented, (
        f"these environment variables are read by python/ but appear nowhere in "
        f"programming_guide/: {undocumented}. Document them, or add them to "
        f"_EXTERNAL here if they belong to the platform rather than to us."
    )
