#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import os
import platform

from setuptools import setup


def get_exe_suffix():
    if platform.system() == "Windows":
        suffix = ".exe"
    else:
        suffix = ""
    return suffix


version = os.environ.get("MLIR_WHEEL_VERSION", "0.0.0+DEADBEEF")

data_files = []
for bin in [
    "llvm-tblgen",
    "mlir-tblgen",
    "mlir-linalg-ods-yaml-gen",
    "mlir-pdll",
    "llvm-config",
    "FileCheck",
]:
    data_files.append(bin + get_exe_suffix())

setup(
    version=version,
    name="mlir-native-tools",
    include_package_data=True,
    data_files=[("bin", data_files)],
)
