# npu-xrt/e2e/conftest.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import os
from pathlib import Path

import pytest


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
