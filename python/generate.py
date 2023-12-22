import os
from pathlib import Path

from ctypesgen.main import main


def generate_xaiengine_capi():
    INCLUDE_DIR1 = "/home/mlevental/dev_projects/mlir-aie/cmake-build-debug/runtime_lib/x86_64/xaiengine/include"

    paths = []
    for root, dirs, files in os.walk(INCLUDE_DIR1):
        p = Path(root).relative_to(INCLUDE_DIR1)
        for f in files:
            paths.append(f"{INCLUDE_DIR1}/{p}/{f}")

    defines = [
        "-D__AIEARCH__=20",
        "-D__AIESIM__",
        "-D__CDO__",
        "-D__PS_INIT_AIE__",
        "-D__LOCK_FENCE_MODE__=2",
        "-DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR",
        "-DAIE2_FP32_EMULATION_ACCURACY_FAST",
    ]
    args = [
        "-lxaienginecdo",
        "-L/home/mlevental/dev_projects/mlir-aie/cmake-build-debug/runtime_lib/x86_64/xaiengine/cdo",
        "-I/home/mlevental/dev_projects/mlir-aie/cmake-build-debug/runtime_lib/x86_64/xaiengine/cdo/include",
        *defines,
        *paths,
        "-o",
        "xaiengine.py",
    ]
    main(args)


def generate_cdo():

    defines = [
        "-D__AIEARCH__=20",
        "-D__AIESIM__",
        "-D__CDO__",
        "-D__PS_INIT_AIE__",
        "-D__LOCK_FENCE_MODE__=2",
        "-DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR",
        "-DAIE2_FP32_EMULATION_ACCURACY_FAST",
    ]
    args = [
        "-lcdo_driver",
        "-L/home/mlevental/dev_projects/mlir-aie/cmake-build-debug/runtime_lib/x86_64/xaiengine/cdo",
        "-I/media/mlevental/9b03e9bc-780d-4e14-9b46-ec8089db51a6/Xilinx/Vitis/2023.2/aietools/include",
        *defines,
        "/media/mlevental/9b03e9bc-780d-4e14-9b46-ec8089db51a6/Xilinx/Vitis/2023.2/aietools/include/cdo_driver.h",
        "-o",
        "cdo_driver.py",
    ]
    main(args)

generate_cdo()