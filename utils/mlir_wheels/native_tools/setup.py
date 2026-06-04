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
    description=(
        "Host-architecture build of MLIR/LLVM TableGen and related "
        "code-generation tools, used during cross-compiled MLIR distro builds."
    ),
    long_description=(
        "Host-architecture build of MLIR/LLVM TableGen and related "
        "code-generation tools (`llvm-tblgen`, `mlir-tblgen`, `mlir-pdll`, "
        "`mlir-linalg-ods-yaml-gen`, `llvm-config`, `FileCheck`).\n\n"
        "Consumed by the cross-compiled `mlir` wheel to provide a runnable "
        "TableGen during the target build. Not intended for standalone use."
    ),
    long_description_content_type="text/markdown",
    url="https://github.com/Xilinx/mlir-aie",
    license="Apache-2.0 WITH LLVM-exception",
    project_urls={
        "Source": "https://github.com/Xilinx/mlir-aie",
        "Upstream LLVM": "https://github.com/llvm/llvm-project",
        "Issues": "https://github.com/Xilinx/mlir-aie/issues",
    },
    include_package_data=True,
    data_files=[("bin", data_files)],
)
