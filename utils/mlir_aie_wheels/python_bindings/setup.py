import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Union

from pip._internal.req import parse_requirements
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def check_env(build, default=0):
    return os.getenv(build, str(default)) in {"1", "true", "True", "ON", "YES"}


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: Union[str, Path] = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


def get_exe_suffix():
    if platform.system() == "Windows":
        suffix = ".exe"
    else:
        suffix = ""
    return suffix


def get_cross_cmake_args():
    cmake_args = {}

    CIBW_ARCHS = os.getenv("CIBW_ARCHS")
    if CIBW_ARCHS in {"arm64", "aarch64", "ARM64"}:
        ARCH = cmake_args["LLVM_TARGETS_TO_BUILD"] = "AArch64"
    elif CIBW_ARCHS in {"x86_64", "AMD64"}:
        ARCH = cmake_args["LLVM_TARGETS_TO_BUILD"] = "X86"
    else:
        raise ValueError(f"unknown CIBW_ARCHS={CIBW_ARCHS}")
    if CIBW_ARCHS != platform.machine():
        # cmake_args["LLVM_USE_HOST_TOOLS"] = "ON"
        cmake_args["CMAKE_SYSTEM_NAME"] = platform.system()

    if platform.system() == "Darwin":
        if ARCH == "AArch64":
            cmake_args["CMAKE_OSX_ARCHITECTURES"] = "arm64"
        elif ARCH == "X86":
            cmake_args["CMAKE_OSX_ARCHITECTURES"] = "x86_64"

    return cmake_args


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        install_dir = extdir
        cfg = "Debug" if DEBUG else "Release"

        cmake_generator = os.getenv("CMAKE_GENERATOR", "Ninja")

        MLIR_AIE_INSTALL_ABS_PATH = Path(
            os.getenv(
                "MLIR_AIE_INSTALL_ABS_PATH",
                Path(__file__).parent
                / ("mlir_aie" if check_env("ENABLE_RTTI", 1) else "mlir_aie_no_rtti"),
            )
        ).absolute()

        MLIR_INSTALL_ABS_PATH = Path(
            os.getenv(
                "MLIR_INSTALL_ABS_PATH",
                Path(__file__).parent
                / ("mlir" if check_env("ENABLE_RTTI", 1) else "mlir_no_rtti"),
            )
        ).absolute()

        if platform.system() == "Windows":
            # fatal error LNK1170: line in command file contains 131071 or more characters
            if not Path("/tmp/a").exists():
                shutil.move(MLIR_AIE_INSTALL_ABS_PATH, "/tmp/a")
            MLIR_AIE_INSTALL_ABS_PATH = Path("/tmp/a").absolute()
            if not Path("/tmp/m").exists():
                shutil.move(MLIR_INSTALL_ABS_PATH, "/tmp/m")
            MLIR_INSTALL_ABS_PATH = Path("/tmp/m").absolute()

        cmake_args = [
            f"-G {cmake_generator}",
            f"-DMLIR_DIR={MLIR_INSTALL_ABS_PATH / 'lib' / 'cmake' / 'mlir'}",
            f"-DAIE_DIR={MLIR_AIE_INSTALL_ABS_PATH / 'lib' / 'cmake' / 'aie'}",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            # get rid of that annoying af git on the end of .17git of libAIEAggregateCAPI.so
            "-DLLVM_VERSION_SUFFIX=",
            # Disables generation of "version soname" (i.e. libFoo.so.<version>), which
            # causes pure duplication of various shlibs for Python wheels.
            "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DMLIR_DETECT_PYTHON_ENV_PRIME_SEARCH=ON",
            # not used on MSVC, but no harm
            f"-DCMAKE_BUILD_TYPE={cfg}",
            # prevent symbol collision that leads to multiple pass registration and such
            "-DCMAKE_VISIBILITY_INLINES_HIDDEN=ON",
            "-DCMAKE_C_VISIBILITY_PRESET=hidden",
            "-DCMAKE_CXX_VISIBILITY_PRESET=hidden",
        ]

        if os.getenv("CMAKE_MODULE_PATH"):
            cmake_module_path = f"{Path(os.getenv('CMAKE_MODULE_PATH')).absolute()}"
            cmake_args.append(f"-DCMAKE_MODULE_PATH={cmake_module_path}")
        if os.getenv("XRT_ROOT"):
            xrt_dir = f"{Path(os.getenv('XRT_ROOT')).absolute()}"
            cmake_args.append(f"-DXRT_ROOT={xrt_dir}")

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_C_COMPILER=cl",
                "-DCMAKE_CXX_COMPILER=cl",
                "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",
            ]

        cmake_args_dict = get_cross_cmake_args()
        cmake_args += [f"-D{k}={v}" for k, v in cmake_args_dict.items()]

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.getenv("CMAKE_ARGS").split(" ") if item]

        build_args = []
        if self.compiler.compiler_type != "msvc":
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})
            if not single_config and not contains_arch:
                PLAT_TO_CMAKE = {
                    "win32": "Win32",
                    "win-amd64": "x64",
                    "win-arm32": "ARM",
                    "win-arm64": "ARM64",
                }
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            osx_version = os.getenv("OSX_VERSION", "11.6")
            cmake_args += [f"-DCMAKE_OSX_DEPLOYMENT_TARGET={osx_version}"]
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.getenv("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        if "PARALLEL_LEVEL" not in os.environ:
            build_args += [f"-j{str(2 * os.cpu_count())}"]
        else:
            build_args += [f"-j{os.getenv('PARALLEL_LEVEL')}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        print("ENV", pprint(os.environ), file=sys.stderr)
        print("cmake", " ".join(cmake_args), file=sys.stderr)

        if platform.system() == "Windows":
            cmake_args = [c.replace("\\", "\\\\") for c in cmake_args]

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", "--target", "install", *build_args],
            cwd=build_temp,
            check=True,
        )


commit_hash = os.environ.get("AIE_PROJECT_COMMIT", "deadbeef")
release_version = "0.0.1"
now = datetime.now()
datetime = os.environ.get(
    "DATETIME", f"{now.year}{now.month:02}{now.day:02}{now.hour:02}"
)
version = f"{release_version}.{datetime}+{commit_hash}"

DEBUG = check_env("DEBUG")
name = "aie-python-bindings"
if DEBUG:
    name += "-debug"

setup(
    version=os.getenv("MLIR_AIE_WHEEL_VERSION", version),
    author="",
    name=name,
    include_package_data=True,
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension("_aie", sourcedir=Path(__file__).parent.absolute())],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "aiecc=aie.compiler.aiecc.main:main",
        ],
    },
    install_requires=[
        str(ir.requirement)
        for ir in parse_requirements("requirements.txt", session="hack")
    ],
)
