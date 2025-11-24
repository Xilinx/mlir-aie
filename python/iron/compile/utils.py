# utils.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.
import os
import shutil
import subprocess
import aie.compiler.aiecc.main as aiecc
import aie.utils.config as config
from .link import merge_object_files


def compile_cxx_core_function(
    source_path: str,
    target_arch: str,
    output_path: str,
    include_dirs: list[str] | None = None,
    compile_args: list[str] | None = None,
    cwd: str | None = None,
    verbose=False,
):
    """
    Compile a C++ core function.
    This function supports only the Peano compiler.
    Parameters:
        source_path (list[str]): Path to C++ source.
        target_arch (str): Target architecture, e.g., aie2.
        output_path (str): Output object file path.
        include_dirs (list[str], optional): List of include directories to add with -I.
        compile_args (list[str], optional): Additional compile arguments to peano.
        cwd (str, optional): Overrides the current working directory.
        verbose (bool): If True, enable verbose output.
    """
    cmd = [
        config.peano_cxx_path(),
        source_path,
        "-c",
        "-o",
        f"{output_path}",
        f"-I{config.cxx_header_path()}",
        "-std=c++20",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-O2",
        "-DNDEBUG",
        f"--target={target_arch}-none-unknown-elf",
    ]

    # Add include directories
    if include_dirs:
        for include_dir in include_dirs:
            cmd.extend(["-I", include_dir])

    # Add additional compile arguments
    if compile_args:
        cmd.extend(compile_args)

    if verbose:
        print("Compiling with:", " ".join(cmd))
    ret = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
    )
    if verbose and ret.stdout:
        print(f"{ret.stdout.decode()}")
    if ret.returncode != 0:
        if ret.stderr:
            raise RuntimeError(f"[Peano] compilation failed:\n{ret.stderr.decode()}")
        raise RuntimeError("[Peano] compilation failed")


def compile_mlir_module(
    mlir_module: str,
    insts_path: str | None = None,
    pdi_path: str | None = None,
    xclbin_path: str | None = None,
    verbose=False,
    work_dir: str | None = None,
    options=None,
):
    """
    Compile an MLIR module to instruction, PDI, and/or xbclbin files using the aiecc module.
    This function supports only the Peano compiler.
    Parameters:
        mlir_module (str): MLIR module to compile.
        insts_path (str): Path to the instructions binary file.
        pdi_path (str): Path to the PDI file.
        xclbin_path (str): Path to the xclbin file.
        verbose (bool): If True, enable verbose output.
        work_dir (str): Compilation working directory.
        options (list[str]): List of additional options.
    """

    args = [
        "--no-compile-host",
        "--no-xchesscc",
        "--no-xbridge",
        f"--peano={config.peano_install_dir()}",
    ]
    if insts_path:
        args.extend(["--aie-generate-npu-insts", f"--npu-insts-name={insts_path}"])
    if pdi_path:
        args.extend(["--aie-generate-pdi", f"--pdi-name={pdi_path}"])
    if xclbin_path:
        args.extend(["--aie-generate-xclbin", f"--xclbin-name={xclbin_path}"])
    if work_dir:
        args.append(f"--tmpdir={work_dir}")
    if verbose:
        args.append("--verbose")
    if options:
        args.extend(options)
    try:
        aiecc.run(mlir_module, args)
    except Exception as e:
        raise RuntimeError("[aiecc] Compilation failed") from e


def compile_external_kernel(func, kernel_dir, target_arch):
    """
    Compile an ExternalFunction to an object file in the kernel directory.

    Args:
        func: ExternalFunction instance to compile
        kernel_dir: Directory to place the compiled object file
        target_arch: Target architecture (e.g., "aie2" or "aie2p")
    """
    # Skip if already compiled
    if hasattr(func, "_compiled") and func._compiled:
        return

    # Check if object file already exists in kernel directory
    output_file = os.path.join(kernel_dir, func.bin_name)
    if os.path.exists(output_file):
        return

    # Create source file in kernel directory
    source_file = os.path.join(kernel_dir, f"{func._name}.cc")

    # Handle both source_string and source_file cases
    if func._source_string is not None:
        # Use source_string (write to file)
        try:
            with open(source_file, "w") as f:
                f.write(func._source_string)
        except Exception as e:
            raise
    elif func._source_file is not None:
        # Use source_file (copy existing file)
        # Check if source file exists before copying
        if os.path.exists(func._source_file):
            try:
                shutil.copy2(func._source_file, source_file)
            except Exception as e:
                raise
        else:
            return
    else:
        raise ValueError("Neither source_string nor source_file is provided")

    try:
        compile_cxx_core_function(
            source_path=source_file,
            target_arch=target_arch,
            output_path=output_file,
            include_dirs=func._include_dirs,
            compile_args=func._compile_flags,
            cwd=kernel_dir,
            verbose=False,
        )
    except Exception as e:
        raise

    # Mark the function as compiled
    func._compiled = True


def _cleanup_failed_compilation(cache_dir):
    """Clean up cache directory after failed compilation, preserving the lock file."""
    if not os.path.exists(cache_dir):
        return

    for item in os.listdir(cache_dir):
        if item == ".lock":
            continue
        item_path = os.path.join(cache_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
