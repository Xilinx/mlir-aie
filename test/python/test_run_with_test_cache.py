# RUN: %pytest %s
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WRAPPER_PATH = _REPO_ROOT / "utils" / "run_with_test_cache.py"


def _load_wrapper_module():
    spec = importlib.util.spec_from_file_location("run_with_test_cache", _WRAPPER_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_lit_wrapper_targets_the_test_file_cache_namespace(monkeypatch):
    monkeypatch.syspath_prepend(str(_REPO_ROOT / "python"))
    from aie_lit_utils.lit_config_helpers import LitConfigHelper

    command = LitConfigHelper._run_with_test_cache_wrap(str(_REPO_ROOT))
    assert "run_with_test_cache.py" in command
    assert '"%s"' in command


def test_wrapper_scopes_cache_home_to_the_current_test_key(tmp_path):
    wrapper_module = _load_wrapper_module()
    base_cache_home = tmp_path / "cache-base"
    test_key = "test/python/npu/test_jit_compilation.py"
    env = os.environ | {"NPU_CACHE_HOME": str(base_cache_home)}

    result = subprocess.run(
        [
            sys.executable,
            str(_WRAPPER_PATH),
            test_key,
            sys.executable,
            "-c",
            (
                "import os; "
                "print(os.environ['NPU_CACHE_HOME']); "
                "print(os.environ['IRON_CACHE_HOME'])"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    expected_cache_home = wrapper_module.cache_home_for_test(
        str(base_cache_home), test_key
    )
    assert result.stdout.splitlines() == [expected_cache_home, expected_cache_home]
    assert Path(expected_cache_home).is_dir()
