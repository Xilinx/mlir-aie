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
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


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
        cmake_src = Path(ext.sourcedir)

        if platform.system() == "Windows":
            # Keep source and dependency paths short without moving the installed trees.
            cmake_src = _windows_short_alias("pb", cmake_src).absolute()
            MLIR_AIE_INSTALL_ABS_PATH = _windows_short_alias(
                _windows_tree_alias_name(MLIR_AIE_INSTALL_ABS_PATH.name), MLIR_AIE_INSTALL_ABS_PATH
            ).absolute()
            MLIR_INSTALL_ABS_PATH = _windows_short_alias(
                _windows_tree_alias_name(MLIR_INSTALL_ABS_PATH.name), MLIR_INSTALL_ABS_PATH
            ).absolute()

        cmake_args = [
            "-G",
            cmake_generator,
            f"-DMLIR_DIR={_cmake_path(MLIR_INSTALL_ABS_PATH / 'lib' / 'cmake' / 'mlir')}",
            f"-DAIE_DIR={_cmake_path(MLIR_AIE_INSTALL_ABS_PATH / 'lib' / 'cmake' / 'aie')}",
            f"-DCMAKE_INSTALL_PREFIX={_cmake_path(install_dir)}",
            # get rid of that annoying af git on the end of .17git of libAIEAggregateCAPI.so
            "-DLLVM_VERSION_SUFFIX=",
            # Disables generation of "version soname" (i.e. libFoo.so.<version>), which
            # causes pure duplication of various shlibs for Python wheels.
            "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",
            f"-DPython3_EXECUTABLE={_cmake_path(sys.executable)}",
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
            cmake_args.append(f"-DCMAKE_MODULE_PATH={_cmake_path(cmake_module_path)}")
        if os.getenv("XRT_ROOT"):
            xrt_dir = _cmake_path(Path(os.getenv("XRT_ROOT")).absolute())
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

        if "PARALLEL_LEVEL" not in os.environ:
            build_args += [f"-j{str(os.cpu_count() or 2)}"]
        else:
            build_args += [f"-j{os.getenv('PARALLEL_LEVEL')}"]

        cleanup_build_temp = False
        build_temp = Path(self.build_temp) / ext.name
        if platform.system() == "Windows":
            build_temp = _windows_short_dir("bind", clean=True)
            cleanup_build_temp = True
        elif not build_temp.exists():
            build_temp.mkdir(parents=True)

        print("ENV", pprint(os.environ), file=sys.stderr)
        print("cmake", " ".join(cmake_args), file=sys.stderr)

        build_succeeded = False
        try:
            subprocess.run(
                ["cmake", _cmake_path(cmake_src), *cmake_args],
                cwd=build_temp,
                check=True,
            )
            subprocess.run(
                ["cmake", "--build", ".", "--target", "install", *build_args],
                cwd=build_temp,
                check=True,
            )
            build_succeeded = True
        finally:
            if cleanup_build_temp and build_succeeded:
                _remove_tree(build_temp)


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


# Read requirements.txt if present (created by the wheel build scripts).
def _read_requirements(req_file: Path) -> list[str]:
    try:
        lines = req_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    out: list[str] = []
    cont = ""
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if cont:
            line = f"{cont} {line}"
            cont = ""
        if line.endswith("\\"):
            cont = line[:-1].rstrip()
            continue

        if " #" in line:
            line = line.split(" #", 1)[0].strip()
        if not line or line.startswith("-"):
            continue

        out.append(line)

    return out


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
    install_requires=_read_requirements(Path(__file__).parent / "requirements.txt"),
)
