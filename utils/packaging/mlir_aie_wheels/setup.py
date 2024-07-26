import os
import platform
import re
import shutil
import subprocess
import sys
import site
import pdb
import glob
import shutil
from datetime import datetime
from pathlib import Path
from pprint import pprint
from textwrap import dedent
from typing import Union

from importlib_metadata import files
import setuptools
from setuptools.command.build_ext import build_ext
from setuptools.command.install_scripts import install_scripts, log

# --------------------------------------------------------------------------
# Utilities


def check_env(build, default=0):
    return os.getenv(build, str(default)).lower() in {"1", "true", "true", "on", "yes"}


def get_exe_suffix():
    if platform.system() == "Windows":
        suffix = ".exe"
    else:
        suffix = ""
    return suffix


# --------------------------------------------------------------------------
# InstallBin command -- overwrites "install_scripts" stage of "install" cmd
# Copy files into the <packagename>.data/data/scripts directory.
# The wheel package format prescribes that files in <packagename>.data/scripts be added to the PATH on install.
# So whatever is added here will end up in the virtualenv's bin directory upon install.
class InstallBin(install_scripts):

    # The subdirectory of the wheel directory in which binaries can be found.
    # Everything globbed from this directory will be copied into an executable path.
    built_bin_dir = "bin"

    # The subdirectory of the wheel in which the Python modules can be found.
    # This will be copied into an import-able path.
    built_python_modules_dir = "python"

    def run(self):
        # Copy the built binaries back into the source directory.
        # This is needed so that the setup(data_files = [(scripts, ...)]) argument below can
        # pick them up.

        # Not a super clean way of doing this, but:
        #  1. `build_ext`` will call CMake to build mlir_aie in <temp_build_dir>/lib.linux-x86_64.../mlir_aie
        #  2. then, `install_lib`` will copy everything in there to <wheelhouse>/mlir_aie
        #  3. then, this runs -- `install_scripts`, and it will be configured with self.install_dir == <wheelhouse>/<pkgname>.data/scripts
        #     therefore, we can get our hands on the binaries moved into the wheelhouse root by looking at the grandparent dir
        bin_install_dir = Path(self.install_dir)
        wheel_root = bin_install_dir.parent.parent.resolve()
        python_module_install_dir = wheel_root

        built_bin_dir = wheel_root / InstallBin.built_bin_dir
        built_python_modules_dir = wheel_root / InstallBin.built_python_modules_dir

        if not bin_install_dir.exists():
            bin_install_dir.mkdir(parents=True)

        mask = current_umask()
        for bin in built_bin_dir.glob("*"):
            dest = str(bin_install_dir / bin.name)
            log.info(f"Copying binary {bin} -> {dest}")
            shutil.copyfile(bin, dest)
            # The pip wheel installer will only copy files marked executable into the bin dir
            chmod(dest, 0o777 - mask)

        if not python_module_install_dir.exists():
            python_module_install_dir.mkdir(parents=True)

        for mod in built_python_modules_dir.glob("*"):
            dest = str(python_module_install_dir / mod.name)
            log.info(f"Copying Python file {mod} -> {dest}")
            if mod.is_dir():
                shutil.copytree(mod, dest)
            else:
                shutil.copyfile(mod, dest)


# --------------------------------------------------------------------------
# Configuration environment variables

commit_hash = os.environ.get("AIE_PROJECT_COMMIT", "deadbeef")
release_version = "0.0.1"
now = datetime.now()
datetime = os.environ.get(
    "DATETIME", f"{now.year}{now.month:02}{now.day:02}{now.hour:02}"
)
version = f"{release_version}.{datetime}+{commit_hash}"

MLIR_AIE_SOURCE_DIR = Path(
    os.getenv(
        "MLIR_AIE_SOURCE_DIR",
        Path(__file__).parent / "mlir-aie",
    )
).absolute()

REQUIRED_LLVM_WHEEL_VERSION = os.environ["REQUIRED_LLVM_WHEEL_VERSION"]
mlir_prereq = "mlir" if check_env("ENABLE_RTTI", 1) else "mlir-no-rtti"

# A wheel containing precompiled LLVM and MLIR should be installed before running this script, using
# pip install --user mlir
# You can also use the MLIR_INSTALL_ABS_PATH environment variable to specify where it should be found.
default_mlir_install_path = Path(site.getsitepackages()[0]) / mlir_prereq
env_mlir_install_path = os.getenv("MLIR_INSTALL_ABS_PATH", default_mlir_install_path)
MLIR_INSTALL_ABS_PATH = Path(env_mlir_install_path).absolute()


# --------------------------------------------------------------------------
# CMake Build Extension for setuptools build process


class CMakeExtension(setuptools.Extension):
    def __init__(self, name: str, sourcedir: Union[str, Path] = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


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


# --------------------------------------------------------------------------
# CMakeBuild command - overwrites "build_ext" stage of "build" command


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        global MLIR_INSTALL_ABS_PATH, env_mlir_install_path, default_mlir_install_path

        if not os.path.isdir(MLIR_INSTALL_ABS_PATH):
            raise RuntimeError(
                f"Could not find required LLVM/MLIR build prerequisite. Looked in `{env_mlir_install_path}` and `{default_mlir_install_path}`."
            )

        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        install_dir = extdir / (
            "mlir_aie" if check_env("ENABLE_RTTI", 1) else "mlir_aie_no_rtti"
        )
        cfg = "Release"

        cmake_generator = os.getenv("CMAKE_GENERATOR", "Ninja")

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


# --------------------------------------------------------------------------
# Setuptools package configuration

InstallBin.built_bin_dir = Path("mlir_aie") / "bin"
InstallBin.built_python_modules_dir = Path("mlir_aie") / "python"

setuptools.setup(
    version=version,
    author="",
    name="mlir-aie" if check_env("ENABLE_RTTI", 1) else "mlir-aie-no-rtti",
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
    # cmdclass overwrites/defines the Command objects that will be created and called for the named commands as part of the wheel build.
    # We overwrite "build_ext" to customize our CMake build, and "install_scripts" to be able to copy our binaries to the bin directory in the PATH.
    cmdclass={"build_ext": CMakeBuild, "install_scripts": InstallBin},
    # note the name here isn't relevant because it's the install (CMake install target) directory that'll be used to
    # actually build the wheel.
    ext_modules=[CMakeExtension("_mlir_aie", sourcedir=MLIR_AIE_SOURCE_DIR)],
    zip_safe=False,
    python_requires=">=3.8",
    setup_requires=[
        f"{mlir_prereq}=={REQUIRED_LLVM_WHEEL_VERSION}"
        # @ https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels/",
    ],
    install_requires=[
        f"{mlir_prereq}=={REQUIRED_LLVM_WHEEL_VERSION}",
        # @ https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels/"
        "mlir-python-utils",
        # @ https://github.com/makslevental/mlir-python-extras/releases/expanded_assets/0.0.6
        #PyYAML>=5.3.1, <=6.0.1,
        "aiofiles",
        #cmake==3.27.9,
        #dataclasses>=0.6, <=0.8
        #filelock==3.13.1
        #lit
        "numpy>=1.19.5,<=1.26",
        #pandas
        #psutil
        "pybind11>=2.9.0,<=2.10.3",
        "rich",
        #setuptools
        #wheel
    ],
    extras_require={
        "peano": [
            "llvm-aie @ https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly"
            # TODO: Might want to set a fixed a version for llvm-aie here
        ]
    },
)
