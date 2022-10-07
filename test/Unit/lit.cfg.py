# -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2022, Advanced Micro Devices, Inc.
#

# Configuration file for the 'lit' test runner.

import os

import lit.formats

# name: The name of this test suite.
config.name = 'phy-unit'

# suffixes: A list of file extensions to treat as test files.
config.suffixes = []

# test_source_root: The root path where unit test binaries are located.
# test_exec_root: The root path where tests should be run.
config.test_source_root = os.path.join(config.aie_obj_root, 'unittests')
config.test_exec_root = config.test_source_root

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, 'Tests')

# Tweak the PATH to include the tools dir.
path = os.path.pathsep.join(
    (config.llvm_tools_dir, config.environment['PATH']))
config.environment['PATH'] = path

path = os.path.pathsep.join((config.llvm_libs_dir,
                             config.environment.get('LD_LIBRARY_PATH', '')))
config.environment['LD_LIBRARY_PATH'] = path
