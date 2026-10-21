#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "extensive: needs the real Llama 3.2 1B weights; deselect with "
        '-m "not extensive"',
    )
