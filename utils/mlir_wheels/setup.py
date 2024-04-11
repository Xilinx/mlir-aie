from datetime import datetime
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from pprint import pprint

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def check_env(build, default=0):
    return os.environ.get(build, str(default)) in {"1", "true", "True", "ON", "YES"}


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


def get_cross_cmake_args():
    cmake_args = {}

    def native_tools():
        nonlocal cmake_args

        native_tools_dir = Path(sys.prefix).absolute() / "bin"
        assert native_tools_dir is not None, "native_tools_dir missing"
        assert os.path.exists(native_tools_dir), "native_tools_dir doesn't exist"
        cmake_args["LLVM_USE_HOST_TOOLS"] = "ON"
        cmake_args["LLVM_NATIVE_TOOL_DIR"] = str(native_tools_dir)

    CIBW_ARCHS = os.environ.get("CIBW_ARCHS")
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
            cmake_args["LLVM_DEFAULT_TARGET_TRIPLE"] = "arm64-apple-darwin21.6.0"
            cmake_args["LLVM_HOST_TRIPLE"] = "arm64-apple-darwin21.6.0"
        elif ARCH == "X86":
            cmake_args["CMAKE_OSX_ARCHITECTURES"] = "x86_64"
            cmake_args["LLVM_DEFAULT_TARGET_TRIPLE"] = "x86_64-apple-darwin"
            cmake_args["LLVM_HOST_TRIPLE"] = "x86_64-apple-darwin"
    elif platform.system() == "Linux":
        if ARCH == "AArch64":
            cmake_args["LLVM_DEFAULT_TARGET_TRIPLE"] = "aarch64-linux-gnu"
            cmake_args["LLVM_HOST_TRIPLE"] = "aarch64-linux-gnu"
            cmake_args["CMAKE_C_COMPILER"] = "aarch64-linux-gnu-gcc"
            cmake_args["CMAKE_CXX_COMPILER"] = "aarch64-linux-gnu-g++"
            cmake_args["CMAKE_CXX_FLAGS"] = "-static-libgcc -static-libstdc++"
            native_tools()
        elif ARCH == "X86":
            cmake_args["LLVM_DEFAULT_TARGET_TRIPLE"] = "x86_64-unknown-linux-gnu"
            cmake_args["LLVM_HOST_TRIPLE"] = "x86_64-unknown-linux-gnu"

    cmake_args["LLVM_TARGET_ARCH"] = ARCH

    return cmake_args


def get_exe_suffix():
    if platform.system() == "Windows":
        suffix = ".exe"
    else:
        suffix = ""
    return suffix


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        install_dir = extdir / (
            "mlir" if check_env("ENABLE_RTTI", 1) else "mlir_no_rtti"
        )
        cfg = "Release"

        cmake_generator = os.environ.get("CMAKE_GENERATOR", "Ninja")

        cmake_args = [
            f"-G {cmake_generator}",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DLLVM_BUILD_BENCHMARKS=OFF",
            "-DLLVM_BUILD_EXAMPLES=OFF",
            "-DLLVM_BUILD_RUNTIMES=OFF",
            "-DLLVM_BUILD_TESTS=OFF",
            "-DLLVM_BUILD_UTILS=ON",
            "-DLLVM_CCACHE_BUILD=ON",
            "-DLLVM_ENABLE_ASSERTIONS=ON",
            f"-DLLVM_ENABLE_RTTI={os.getenv('ENABLE_RTTI', 'ON')}",
            "-DLLVM_ENABLE_ZSTD=OFF",
            "-DLLVM_INCLUDE_BENCHMARKS=OFF",
            "-DLLVM_INCLUDE_EXAMPLES=OFF",
            "-DLLVM_INCLUDE_RUNTIMES=OFF",
            "-DLLVM_INCLUDE_TESTS=OFF",
            "-DLLVM_INCLUDE_TOOLS=ON",
            "-DLLVM_INCLUDE_UTILS=ON",
            "-DLLVM_INSTALL_UTILS=ON",
            # Maybe re-enable after the `LLVMContextCreate` thing is fixed?
            "-DMLIR_BUILD_MLIR_C_DYLIB=0",
            "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
            "-DMLIR_ENABLE_EXECUTION_ENGINE=ON",
            "-DMLIR_ENABLE_SPIRV_CPU_RUNNER=ON",
            # get rid of that annoying af git on the end of .17git
            "-DLLVM_VERSION_SUFFIX=",
            # Disables generation of "version soname" (i.e. libFoo.so.<version>), which
            # causes pure duplication of various shlibs for Python wheels.
            "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            # prevent symbol collision that leads to multiple pass registration and such
            "-DCMAKE_VISIBILITY_INLINES_HIDDEN=ON",
            "-DCMAKE_C_VISIBILITY_PRESET=hidden",
            "-DCMAKE_CXX_VISIBILITY_PRESET=hidden",
        ]
        if platform.system() == "Windows":
            cmake_args += [
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

        LLVM_ENABLE_PROJECTS = "llvm;mlir"

        cmake_args += [f"-DLLVM_ENABLE_PROJECTS={LLVM_ENABLE_PROJECTS}"]

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

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
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        if "PARALLEL_LEVEL" not in os.environ:
            build_args += [f"-j{str(2 * os.cpu_count())}"]
        else:
            build_args += [f"-j{os.environ.get('PARALLEL_LEVEL')}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        print("ENV", pprint(os.environ), file=sys.stderr)
        print("cmake", " ".join(cmake_args), file=sys.stderr)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", "--target", "install", *build_args],
            cwd=build_temp,
            check=True,
        )


cmake_txt = open("llvm-project/cmake/Modules/LLVMVersion.cmake").read()
llvm_version = []
for v in ["LLVM_VERSION_MAJOR", "LLVM_VERSION_MINOR", "LLVM_VERSION_PATCH"]:
    vn = re.findall(rf"set\({v} (\d+)\)", cmake_txt)
    assert vn, f"couldn't find {v} in cmake txt"
    llvm_version.append(vn[0])

commit_hash = os.environ.get("LLVM_PROJECT_COMMIT", "DEADBEEF")

now = datetime.now()
llvm_datetime = os.environ.get(
    "DATETIME", f"{now.year}{now.month:02}{now.day:02}{now.hour:02}"
)

version = f"{llvm_version[0]}.{llvm_version[1]}.{llvm_version[2]}.{llvm_datetime}+"
version += commit_hash

llvm_url = f"https://github.com/llvm/llvm-project/commit/{commit_hash}"
setup(
    name="mlir" if check_env("ENABLE_RTTI", 1) else "mlir-no-rtti",
    version=version,
    author="Maksim Levental",
    author_email="maksim.levental@gmail.com",
    description=f"MLIR distribution as wheel. Created at {now} build of {llvm_url}",
    long_description=f"MLIR distribution as wheel. Created at {now} build of [llvm/llvm-project/{commit_hash}]({llvm_url})",
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension("mlir", sourcedir="llvm-project/llvm")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    download_url=llvm_url,
)
