# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stand-in design script for the CMake helper fixture.

add_aie_design() checks that its PY argument exists on disk and bakes the path
into a custom command. helpers.lit only inspects the generated rule, so this
script is never executed -- it just has to be a real file.
"""

raise SystemExit("cmake_helpers/design.py is a lit fixture and is never run")
