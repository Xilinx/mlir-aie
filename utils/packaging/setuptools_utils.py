import platform
import shutil
import os
import pdb
from pathlib import Path
from setuptools.command.install_scripts import install_scripts, log
from setuptools.command.easy_install import chmod, current_umask

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
