import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from pprint import pprint
from typing import Union

from importlib_metadata import files
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install

sys.path.append(os.path.dirname(__file__))
from vendor_eudsl import install_eudsl
from _version_helper import _git, get_version


def check_env(build, default=0):
    return os.getenv(build, str(default)) in {"1", "true", "True", "ON", "YES"}


# Always use forward slashes for CMake paths.
def _cmake_path(p: object) -> str:
    return os.fspath(p).replace("\\", "/")


def _windows_short_root() -> Path:
    return Path(os.getenv("AIE_WHEEL_BUILD_ROOT", "C:/tmp/aiewhls")).absolute()


def _windows_tree_alias_name(tree_name: str) -> str:
    aliases = {
        "mlir": "m",
        "mlir_no_rtti": "mnr",
        "mlir_aie": "a",
        "mlir_aie_no_rtti": "anr",
    }
    return aliases.get(tree_name, tree_name)


def _remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _windows_short_dir(name: str, *, clean: bool = False) -> Path:
    path = _windows_short_root() / name
    if clean and path.exists():
        _remove_tree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _windows_short_alias(name: str, target: Path) -> Path:
    link = _windows_short_root() / name
    target = target.absolute()
    if link.exists():
        try:
            if link.resolve() == target.resolve():
                return link
        except OSError:
            pass
        subprocess.run(
            ["cmd", "/c", "rmdir", str(link)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    link.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return link


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
        cmake_source_dir = Path(ext.sourcedir)
        cmake_module_root = MLIR_AIE_SOURCE_DIR

        if platform.system() == "Windows":
            # Keep source and dependency paths short without moving the installed trees.
            cmake_source_dir = _windows_short_alias("src", cmake_source_dir).absolute()
            cmake_module_root = cmake_source_dir
            MLIR_INSTALL_ABS_PATH = _windows_short_alias(
                _windows_tree_alias_name(MLIR_INSTALL_ABS_PATH.name),
                MLIR_INSTALL_ABS_PATH,
            ).absolute()

        cmake_args = [
            "-G",
            cmake_generator,
            f"-DCMAKE_MODULE_PATH={_cmake_path(cmake_module_root / 'cmake' / 'modulesXilinx')}",
            f"-DCMAKE_PREFIX_PATH={_cmake_path(MLIR_INSTALL_ABS_PATH)}",
            f"-DCMAKE_INSTALL_PREFIX={_cmake_path(install_dir)}",
            f"-DPython3_EXECUTABLE={_cmake_path(sys.executable)}",
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
            "-DAIE_BUILD_LSP_SERVER=OFF",
            "-DAIE_BUILD_VISUALIZE=OFF",
            "-DMLIR_DETECT_PYTHON_ENV_PRIME_SEARCH=ON",
            # not used on MSVC, but no harm
        ]

        if os.getenv("XRT_ROOT"):
            xrt_dir = _cmake_path(Path(os.getenv("XRT_ROOT")).absolute())
            cmake_args.append(f"-DXRT_ROOT={xrt_dir}")

        if shutil.which("ccache"):
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            ]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_C_COMPILER=cl",
                "-DCMAKE_CXX_COMPILER=cl",
                "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",
                "-DCMAKE_C_FLAGS=/MT /wd4065",
                "-DCMAKE_CXX_FLAGS=/MT /wd4065",
                "-DLLVM_USE_CRT_MINSIZEREL=MT",
                "-DLLVM_USE_CRT_RELEASE=MT",
            ]
        elif platform.system() == "Linux":
            # MSVC's Release default is /OPT:REF, which strips unreferenced
            # functions and data. Mirror that on Linux with per-section
            # emission + gc-sections. (MSVC's /OPT:ICF equivalent would be
            # lld's --icf=safe, but GNU ld in the cibuildwheel manylinux
            # container doesn't understand that flag and the configure-time
            # try-compile aborts before lld is even built.)
            size_cflags = "-ffunction-sections -fdata-sections"
            cmake_args += [
                f"-DCMAKE_C_FLAGS={size_cflags}",
                f"-DCMAKE_CXX_FLAGS={size_cflags}",
                "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--gc-sections",
                "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--gc-sections",
                "-DCMAKE_MODULE_LINKER_FLAGS=-Wl,--gc-sections",
            ]

        cmake_args_dict = get_cross_cmake_args()
        cmake_args += [f"-D{k}={v}" for k, v in cmake_args_dict.items()]

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.getenv("CMAKE_ARGS").split(" ") if item]

        build_args = []
        if not cmake_generator or cmake_generator == "Ninja":
            try:
                import ninja

                ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                cmake_args += [
                    f"-DCMAKE_MAKE_PROGRAM:FILEPATH={_cmake_path(ninja_executable_path)}",
                ]
            except ImportError:
                pass

        if self.compiler.compiler_type == "msvc":
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

        build_args += [f"-j{os.getenv('PARALLEL_LEVEL', os.cpu_count() or 2)}"]
        cleanup_build_temp = False
        build_temp = Path(self.build_temp) / ext.name
        if platform.system() == "Windows":
            build_temp = _windows_short_dir("main", clean=True)
            cleanup_build_temp = True
        elif not build_temp.exists():
            build_temp.mkdir(parents=True)

        print("ENV", pprint(os.environ), file=sys.stderr)
        print("cmake", " ".join(cmake_args), file=sys.stderr)

        build_succeeded = False
        try:
            subprocess.run(
                ["cmake", _cmake_path(cmake_source_dir), *cmake_args],
                cwd=build_temp,
                check=True,
            )
            subprocess.run(
                ["cmake", "--build", ".", "--target", "install", *build_args],
                cwd=build_temp,
                check=True,
            )

            # C++-developer-only content: anyone pip installing this wheel
            # reaches the toolchain through Python, never through native
            # linking. aie_api/ and aie_kernels/ headers stay — user AIE
            # kernels #include those at compile time. *.lib on Windows is
            # the MSVC equivalent of Linux *.a — static linker artifacts
            # produced by the LLVM/MLIR install rules that nothing in the
            # runtime path links against (verified: aiecc.exe / aie-opt.exe
            # / AIEAggregateCAPI.dll are self-contained).
            dev_paths = [
                Path(install_dir) / "include" / "aie",
                Path(install_dir) / "include" / "aie-c",
                Path(install_dir) / "include" / "bootgen_c_api.h",
                Path(install_dir) / "include" / "xaienginecdo_static",
                Path(install_dir) / "lib" / "cmake",
                *(Path(install_dir) / "lib").glob("*.a"),
                *(Path(install_dir) / "lib").glob("*.lib"),
            ]
            for p in dev_paths:
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()

            # CMake leaks staging directories, intermediate .o files, and
            # __pycache__ caches into the install prefix; none belong in a
            # shipped wheel.
            for leaked in [
                Path(install_dir) / "src",
                Path(install_dir) / "lib" / "objects-Release",
            ]:
                if leaked.exists():
                    shutil.rmtree(leaked)
            for pycache in Path(install_dir).rglob("__pycache__"):
                shutil.rmtree(pycache, ignore_errors=True)

            # Upstream MLIR's Python install ships bindings for every dialect
            # it knows about. Prune the ones nothing in the AIE Python tree
            # imports (verified via grep across the install). pdl / irdl /
            # nvgpu are kept — they're pulled in transitively by extras /
            # dialects.ext.
            unused_dialects = (
                "spirv",
                "omp",
                "smt",
                "shard",
                "sparse_tensor",
                "x86",
                "amdgpu",
                "shape",
                "emitc",
            )
            dialects_dirs = list(Path(install_dir).rglob("mlir/dialects"))
            for d in dialects_dirs:
                for stem in unused_dialects:
                    for f in d.glob(f"_{stem}_*.py"):
                        f.unlink()
                    front = d / f"{stem}.py"
                    if front.exists():
                        front.unlink()
                async_pkg = d / "async_dialect"
                if async_pkg.is_dir():
                    shutil.rmtree(async_pkg)

            # Vendor eudsl-python-extras
            # Install eudsl to install_dir/python so it merges with mlir-aie's package structure (aie/extras).
            target_dir = Path(install_dir) / "python"
            req_file = Path(MLIR_AIE_SOURCE_DIR) / "python" / "requirements.txt"
            install_eudsl(req_file, target_dir)

            aie_pkg_dir = Path(install_dir) / "python" / "aie"
            aie_pkg_dir.mkdir(parents=True, exist_ok=True)
            sha = _git("rev-parse", "--short=7", "HEAD") or "unknown"
            build_date = (
                datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z")
            )
            (aie_pkg_dir / "_version.py").write_text(
                f'__version__ = "{get_version()}"\n'
                f'__commit__ = "{sha}"\n'
                f'__build_date__ = "{build_date}"\n'
            )

            init_file = aie_pkg_dir / "__init__.py"
            reexport_marker = "# >>> mlir-aie wheel build metadata"
            reexport_block = (
                f"\n{reexport_marker}\n"
                "from ._version import __version__, __commit__, __build_date__\n"
                "# <<< mlir-aie wheel build metadata\n"
            )
            existing_init = (
                init_file.read_text(encoding="utf-8") if init_file.exists() else ""
            )
            if reexport_marker not in existing_init:
                with init_file.open("a", encoding="utf-8") as f:
                    f.write(reexport_block)

            build_succeeded = True
        finally:
            if cleanup_build_temp and build_succeeded:
                _remove_tree(build_temp)


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


class EggInfoWithTopLevel(egg_info):
    """Override the auto-derived top_level.txt.

    setuptools derives top_level.txt from the CMakeExtension's name
    (``_mlir_aie``), which is just a placeholder — the wheel actually
    installs the ``mlir_aie`` package tree. Fix the metadata so
    ``pip uninstall`` and other top-level scanners see reality.
    """

    def run(self):
        super().run()
        top_level_path = os.path.join(self.egg_info, "top_level.txt")
        with open(top_level_path, "w") as f:
            f.write("mlir_aie\n")


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
        # Also remove lines starting with "-" (flags) and eudsl-python-extras
        # because eudsl requires config settings that cannot be passed via install_requires
        # in wheel metadata. It must be installed separately or vendored.
        requirements = []
        for line in lines:
            line = line.strip()
            if (
                line
                and not line.startswith("#")
                and not line.startswith("-")
                and not re.match(r"^eudsl-python-extras\b", line, re.IGNORECASE)
            ):
                requirements.append(line)
        return requirements


_license = MLIR_AIE_SOURCE_DIR / "LICENSE"
if _license.exists():
    shutil.copy(_license, Path(__file__).parent / "LICENSE")

setup(
    name="mlir-aie" if check_env("ENABLE_RTTI", 1) else "mlir-aie-no-rtti",
    version=get_version(),
    description="An MLIR-based toolchain for AMD AI Engine-enabled devices.",
    long_description=(
        (Path(MLIR_AIE_SOURCE_DIR) / "README.md").read_text(encoding="utf-8")
        if (Path(MLIR_AIE_SOURCE_DIR) / "README.md").exists()
        else "An MLIR-based toolchain for AMD AI Engine-enabled devices. "
        "See https://github.com/Xilinx/mlir-aie"
    ),
    long_description_content_type="text/markdown",
    author="AMD Inc.",
    author_email="joseph.melber@amd.com",
    url="https://github.com/Xilinx/mlir-aie",
    license="Apache-2.0 WITH LLVM-exception",
    license_files=["LICENSE"],
    project_urls={
        "Source": "https://github.com/Xilinx/mlir-aie",
        "Issues": "https://github.com/Xilinx/mlir-aie/issues",
        "Documentation": "https://xilinx.github.io/mlir-aie/",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Compilers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
    ],
    entry_points={
        "console_scripts": [
            "aie-opt = aie.tools:aie_opt",
            "aie-reset = aie.tools:aie_reset",
            "aie-translate = aie.tools:aie_translate",
            "aiecc = aie.tools:aiecc",
            "aiecc.py = aie.compiler.aiecc.main:main",
            "bootgen = aie.tools:bootgen",
            "txn2mlir.py = aie.compiler.txn2mlir.main:main",
            "xchesscc_wrapper = aie.tools:xchesscc_wrapper",
        ],
    },
    include_package_data=True,
    ext_modules=[CMakeExtension("_mlir_aie", sourcedir=MLIR_AIE_SOURCE_DIR)],
    cmdclass={
        "build_ext": CMakeBuild,
        "develop": DevelopWithPth,
        "egg_info": EggInfoWithTopLevel,
        "install": InstallWithPth,
    },
    zip_safe=False,
    packages=find_packages(exclude=["wheelhouse", "python_bindings", "mlir-aie"]),
    python_requires=">=3.11",
    install_requires=parse_requirements(
        Path(MLIR_AIE_SOURCE_DIR) / "python" / "requirements.txt"
    ),
)
