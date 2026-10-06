# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stand-in Python host test for the CMake helper fixture.

Exercises add_aie_run_test's PY (run_py) shape, which checks that the script
exists and passes it --xclbin/--instr. helpers.lit only inspects the registered
ctest command, so this script is never executed.
"""

raise SystemExit("cmake_helpers/host.py is a lit fixture and is never run")
