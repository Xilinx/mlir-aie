import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint
from textwrap import dedent
from typing import Union

from importlib_metadata import files
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install


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

    def native_tools():
        nonlocal cmake_args

        mlir_tblgen_host = next(
            f.locate()
            for f in files("mlir-native-tools")
            if f.name.startswith("mlir-tblgen")
        )
        mlir_tblgen_target = next(
            f.locate()
            for f in files("mlir" if check_env("ENABLE_RTTI", 1) else "mlir_no_rtti")
            if f.name.startswith("mlir-tblgen")
        )
        os.remove(mlir_tblgen_target)
        shutil.copy(mlir_tblgen_host, mlir_tblgen_target)
        mlir_pdll_host = next(
            f.locate()
            for f in files("mlir-native-tools")
            if f.name.startswith("mlir-pdll")
        )
        mlir_pdll_target = next(
            f.locate()
            for f in files("mlir" if check_env("ENABLE_RTTI", 1) else "mlir_no_rtti")
            if f.name.startswith("mlir-pdll")
        )
        os.remove(mlir_pdll_target)
        shutil.copy(mlir_pdll_host, mlir_pdll_target)

    CIBW_ARCHS = os.getenv("CIBW_ARCHS")
    if CIBW_ARCHS in {"arm64", "aarch64", "ARM64"}:
        ARCH = cmake_args["LLVM_TARGETS_TO_BUILD"] = "AArch64"
    elif CIBW_ARCHS in {"x86_64", "AMD64"}:
        ARCH = cmake_args["LLVM_TARGETS_TO_BUILD"] = "X86"
    else:
        raise ValueError(f"unknown CIBW_ARCHS={CIBW_ARCHS}")
    if CIBW_ARCHS != platform.machine():
        cmake_args["CMAKE_SYSTEM_NAME"] = platform.system()

    if platform.system() == "Darwin":
        if ARCH == "AArch64":
            cmake_args["CMAKE_OSX_ARCHITECTURES"] = "arm64"
            cmake_args["LLVM_HOST_TRIPLE"] = "arm64-apple-darwin21.6.0"
            native_tools()
        elif ARCH == "X86":
            cmake_args["CMAKE_OSX_ARCHITECTURES"] = "x86_64"
            cmake_args["LLVM_HOST_TRIPLE"] = "x86_64-apple-darwin"
    elif platform.system() == "Linux":
        if ARCH == "AArch64":
            cmake_args["LLVM_HOST_TRIPLE"] = "aarch64-linux-gnu"
            cmake_args["CMAKE_C_COMPILER"] = "aarch64-linux-gnu-gcc"
            cmake_args["CMAKE_CXX_COMPILER"] = "aarch64-linux-gnu-g++"
            cmake_args["CMAKE_CXX_FLAGS"] = "-static-libgcc -static-libstdc++"
            cmake_args["SysrootAarch64"] = "/usr/aarch64-linux-gnu"
            cmake_args["AIE_RUNTIME_TARGETS"] = "aarch64"
            native_tools()
        elif ARCH == "X86":
            cmake_args["LLVM_HOST_TRIPLE"] = "x86_64-unknown-linux-gnu"
            cmake_args["AIE_RUNTIME_TARGETS"] = "x86_64"

    return cmake_args


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        install_dir = extdir / "mlir_aie"
        cfg = "Release"

        cmake_generator = os.getenv("CMAKE_GENERATOR", "Ninja")

        MLIR_INSTALL_ABS_PATH = Path(
            os.getenv(
                "MLIR_INSTALL_ABS_PATH",
                Path(__file__).parent
                / ("mlir" if check_env("ENABLE_RTTI", 1) else "mlir_no_rtti"),
            )
        ).absolute()

        if platform.system() == "Windows":
            # fatal error LNK1170: line in command file contains 131071 or more characters
            if not Path("/tmp/m").exists():
                shutil.move(MLIR_INSTALL_ABS_PATH, "/tmp/m")
            MLIR_INSTALL_ABS_PATH = Path("/tmp/m").absolute()

        cmake_args = [
            f"-G {cmake_generator}",
            f"-DCMAKE_MODULE_PATH={MLIR_AIE_SOURCE_DIR / 'cmake' / 'modulesXilinx'}",
            f"-DCMAKE_PREFIX_PATH={MLIR_INSTALL_ABS_PATH}",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            # prevent symbol collision that leads to multiple pass registration and such
            "-DCMAKE_VISIBILITY_INLINES_HIDDEN=ON",
            "-DCMAKE_C_VISIBILITY_PRESET=hidden",
            "-DCMAKE_CXX_VISIBILITY_PRESET=hidden",
            "-DBUILD_SHARED_LIBS=OFF",
            # get rid of that annoying af git on the end of .17git
            "-DLLVM_VERSION_SUFFIX=",
            # Disables generation of "version soname" (i.e. libFoo.so.<version>), which
            # causes pure duplication of various shlibs for Python wheels.
            "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",
            "-DLLVM_CCACHE_BUILD=ON",
            f"-DLLVM_ENABLE_RTTI={os.getenv('ENABLE_RTTI', 'ON')}",
            f"-DAIE_VITIS_COMPONENTS={os.getenv('AIE_VITIS_COMPONENTS', 'AIE2')}",
            "-DAIE_ENABLE_BINDINGS_PYTHON=ON",
            "-DAIE_ENABLE_PYTHON_PASSES=OFF",
            "-DMLIR_DETECT_PYTHON_ENV_PRIME_SEARCH=ON",
            # not used on MSVC, but no harm
        ]

        if os.getenv("XRT_ROOT"):
            xrt_dir = f"{Path(os.getenv('XRT_ROOT')).absolute()}"
            cmake_args.append(f"-DXRT_ROOT={xrt_dir}")

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_C_COMPILER=cl",
                "-DCMAKE_CXX_COMPILER=cl",
                "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",
                "-DCMAKE_C_FLAGS=/MT",
                "-DCMAKE_CXX_FLAGS=/MT",
                "-DLLVM_USE_CRT_MINSIZEREL=MT",
                "-DLLVM_USE_CRT_RELEASE=MT",
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

        build_args += [f"-j{os.getenv('PARALLEL_LEVEL', 2 * os.cpu_count())}"]
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


class DevelopWithPth(develop):
    """Custom develop command to create a .pth file into the site-packages directory."""

    def run(self):
        super().run()
        pth_target = os.path.join(self.install_dir, "aie.pth")
        with open(pth_target, "w") as pth_file:
            pth_file.write("mlir_aie/python")


class InstallWithPth(install):
    """Custom install command to create a .pth file into the site-packages directory."""

    def run(self):
        super().run()
        pth_target = os.path.join(self.install_lib, "aie.pth")
        with open(pth_target, "w") as pth_file:
            pth_file.write("mlir_aie/python")


def get_version():
    release_version = "0.0.1"
    commit_hash = os.environ.get("AIE_PROJECT_COMMIT", "deadbeef")
    now = datetime.now()
    timestamp = os.environ.get(
        "DATETIME", f"{now.year}{now.month:02}{now.day:02}{now.hour:02}"
    )
    suffix = "" if check_env("ENABLE_RTTI", 1) else "-no_rtti"
    return f"{release_version}.{timestamp}+{commit_hash}{suffix}"


MLIR_AIE_SOURCE_DIR = Path(
    os.getenv(
        "MLIR_AIE_SOURCE_DIR",
        Path(__file__).parent / "mlir-aie",
    )
).absolute()


def parse_requirements(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
        # Remove comments and empty lines
        return [
            line.strip() for line in lines if line.strip() and not line.startswith("#")
        ]


setup(
    version=get_version(),
    author="",
    name="mlir-aie",
    include_package_data=True,
    description=f"An MLIR-based toolchain for Xilinx Versal AIEngine-based devices.",
    long_description=dedent(
        """\
        This repository contains an [MLIR-based](https://mlir.llvm.org/) toolchain for Xilinx Versal
        AIEngine-based devices.  This can be used to generate low-level configuration for the AIEngine portion of the
        device, including processors, stream switches, TileDMA and ShimDMA blocks. Backend code generation is
        included, targetting the LibXAIE library.  This project is primarily intended to support tool builders with
        convenient low-level access to devices and enable the development of a wide variety of programming models
        from higher level abstractions.  As such, although it contains some examples, this project is not intended to
        represent end-to-end compilation flows or to be particularly easy to use for system design.
        """
    ),
    long_description_content_type="text/markdown",
    # note the name here isn't relevant because it's the install (CMake install target) directory that'll be used to
    # actually build the wheel.
    ext_modules=[CMakeExtension("_mlir_aie", sourcedir=MLIR_AIE_SOURCE_DIR)],
    cmdclass={
        "build_ext": CMakeBuild,
        "develop": DevelopWithPth,
        "install": InstallWithPth,
    },
    zip_safe=False,
    python_requires=">=3.10",
    install_requires=parse_requirements(
        Path(MLIR_AIE_SOURCE_DIR) / "python" / "requirements.txt"
    ),
)
