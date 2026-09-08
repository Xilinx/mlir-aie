# benchmarks/kernels/conftest.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Command-line options for the correctness suite.

pytest honours ``pytest_addoption`` only in conftest.py files and plugins;
defined inside a test module it is ignored and ``--seeds 3`` fails with
"unrecognized arguments". This file imports nothing from ``aie`` so the
host-only module in this directory (test_run_driver) collects on a
machine without the package.
"""


def pytest_addoption(parser):
    parser.addoption(
        "--data-cases",
        default=None,
        help="comma-separated subset of data cases to run (test_correctness)",
    )
    parser.addoption(
        "--seeds",
        type=int,
        default=1,
        help="number of random seeds for the 'random' data case (test_correctness)",
    )
