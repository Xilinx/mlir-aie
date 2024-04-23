import os
from pathlib import Path
import subprocess

import pytest


@pytest.fixture(autouse=True)
def run_around_tests():
    subprocess.check_call(
        [str(Path(__file__).parent.parent.parent.parent / "utils" / "reset_npu.sh")]
    )
    yield


@pytest.fixture()
def workdir(request):
    workdir_ = os.getenv("workdir")
    if workdir_ is None:
        # will look like file_name/test_name
        workdir_ = Path(request.fspath).parent.absolute() / request.node.nodeid.replace(
            "::", "/"
        ).replace(".py", "")
    else:
        workdir_ = Path(workdir_).absolute()

    workdir_.parent.mkdir(exist_ok=True)
    workdir_.mkdir(exist_ok=True)

    return workdir_
