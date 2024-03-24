import os
from pathlib import Path
import subprocess

import pytest


def _run_around_tests():
    subprocess.call(
        [str(Path(__file__).parent.parent.parent.parent / "utils" / "reset_ipu.sh")]
    )
    yield


run_around_tests = pytest.fixture(autouse=True)(_run_around_tests)


@pytest.fixture()
def workdir(request):
    workdir_ = os.getenv("workdir")
    if workdir_ is None:
        # will look like file_name/test_name
        workdir_ = Path(request.fspath).parent.absolute() / request.node.nodeid.replace(
            "::", "/"
        ).replace(".py", "").replace("[", "-").replace("]", "")
    else:
        workdir_ = Path(workdir_).absolute()

    workdir_.parent.mkdir(exist_ok=True)
    workdir_.mkdir(exist_ok=True)

    return workdir_
