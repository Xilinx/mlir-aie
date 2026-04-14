# utils.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc.
"""Low-level helpers for compiling MLIR modules and external C++ kernels to NPU artifacts."""

import logging
import os
import shutil
import subprocess
from pathlib import Path
import aie.compiler.aiecc.main as aiecc
import aie.utils.config as config

logger = logging.getLogger(__name__)


def resolve_target_arch(device=None) -> str:
    """Return ``'aie2'`` or ``'aie2p'`` for the given device, or ``'aie2'`` if device is None."""
    if device is None:
        return "aie2"
    from aie.dialects.aie import AIEDevice

    # Normalise to AIEDevice enum if the device is an IRON device class instance.
    device_enum = getattr(device, "_device", device)
    try:
        name = (
            device_enum.name
        )  # e.g. "npu1", "npu1_1col", "npu2", "npu2_1col", "npu2_4col"
    except AttributeError:
        raise RuntimeError(f"Unsupported device type: {type(device)}")

    if name.startswith("npu2"):
        return "aie2p"
    if name.startswith("npu1"):
        return "aie2"
    raise RuntimeError(
        f"Unsupported device type: {type(device)} (AIEDevice name={name!r}). "
        f"Expected name starting with 'npu1' or 'npu2'."
    )


def compile_cxx_core_function(
    source_path: str,
    target_arch: str,
    output_path: str,
    include_dirs: list[str] | None = None,
    compile_args: list[str] | None = None,
    cwd: str | None = None,
):
    """
    Compile a C++ core function.
    This function supports only the Peano compiler.
    Parameters:
        source_path (str): Path to C++ source.
        target_arch (str): Target architecture, e.g., aie2.
        output_path (str): Output object file path.
        include_dirs (list[str], optional): List of include directories to add with -I.
        compile_args (list[str], optional): Additional compile arguments to peano.
        cwd (str, optional): Overrides the current working directory.
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

    logger.debug("Compiling with: %s", " ".join(cmd))
    ret = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
    )
    if ret.stdout:
        logger.debug("%s", ret.stdout.decode())
    if ret.returncode != 0:
        if ret.stderr:
            raise RuntimeError(f"[Peano] compilation failed:\n{ret.stderr.decode()}")
        raise RuntimeError("[Peano] compilation failed")


def compile_mlir_module(
    mlir_module: str,
    insts_path: str | Path | None = None,
    pdi_path: str | Path | None = None,
    xclbin_path: str | Path | None = None,
    verbose=False,
    work_dir: str | Path | None = None,
    options=None,
):
    """
    Compile an MLIR module to instruction, PDI, and/or xclbin files using the aiecc module.
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
    # When work_dir is provided, invoke the aiecc binary as a subprocess so
    # that it resolves relative link_with paths (e.g. "add_one.o") against the
    # same directory where compile_external_kernel placed the compiled objects.
    # The MLIR file is written to work_dir/aie.mlir; callers (e.g. jit.py)
    # may have already written it there, in which case this is a no-op write.
    # If no work_dir is provided, fall back to aiecc.run() which writes to a
    # temporary file internally.
    if work_dir:
        aiecc_bin = shutil.which("aiecc")
        if not aiecc_bin:
            raise RuntimeError(
                "Could not find 'aiecc' binary. Ensure mlir-aie is installed "
                "and its bin directory is in PATH."
            )
        mlir_file = os.path.join(work_dir, "aie.mlir")
        with open(mlir_file, "w") as f:
            f.write(str(mlir_module))
        result = subprocess.run(
            [aiecc_bin, mlir_file] + args, capture_output=True, text=True
        )
        if result.stdout:
            logger.debug("%s", result.stdout)
        if result.stderr:
            logger.debug("%s", result.stderr)
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            raise RuntimeError(
                f"[aiecc] Compilation failed with exit code {result.returncode}:\n"
                f"{error_msg}"
            )
    else:
        try:
            aiecc.run(mlir_module, args)
        except Exception as e:
            raise RuntimeError("[aiecc] Compilation failed") from e


def _rename_symbol_in_object(object_path: str, old_name: str, new_name: str) -> None:
    """Rename a symbol in a compiled object file using llvm-objcopy."""
    objcopy = shutil.which("llvm-objcopy")
    if not objcopy:
        objcopy = shutil.which("objcopy")
    if not objcopy:
        raise RuntimeError(
            "Cannot rename symbol: neither 'llvm-objcopy' nor 'objcopy' found in PATH. "
            "Install the LLVM toolchain or GNU binutils."
        )
    result = subprocess.run(
        [objcopy, f"--redefine-sym={old_name}={new_name}", str(object_path)],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Symbol rename failed: {result.stderr.decode()}")


def compile_external_kernel(func, kernel_dir, target_arch):
    """
    Compile an ExternalFunction to an object file in the kernel directory.

    The output file is named ``func.object_file_name`` and placed in ``kernel_dir``.
    If the object file already exists in ``kernel_dir``, compilation is skipped.

    Args:
        func: ExternalFunction instance to compile.
        kernel_dir: Directory where the compiled object file will be placed.
            Must be the same directory passed as ``work_dir`` to
            ``compile_mlir_module`` so that relative link_with paths resolve
            correctly.
        target_arch: Peano target architecture string (e.g., "aie2", "aie2p").
    """
    # Skip if already compiled in this session.
    if func._compiled:
        return

    # Skip if the object file already exists (cache hit).
    output_file = os.path.join(kernel_dir, func.object_file_name)
    if os.path.exists(output_file):
        if getattr(func, "_symbol_prefix", None):
            # Ensure rename is applied even on cache hit — idempotent with llvm-objcopy
            _rename_symbol_in_object(output_file, func._original_name, func._name)
        return

    original_name = getattr(func, "_original_name", func._name)

    if func._source_string is not None:
        source_file = os.path.join(kernel_dir, f"{original_name}.cc")
        with open(source_file, "w") as f:
            f.write(func._source_string)
        compile_cxx_core_function(
            source_path=source_file,
            target_arch=target_arch,
            output_path=output_file,
            include_dirs=func._include_dirs,
            compile_args=func._compile_flags,
            cwd=str(kernel_dir),
        )

    elif func._source_file is not None:
        source_file = os.path.join(kernel_dir, f"{original_name}.cc")
        # Check if source file exists before copying
        if not os.path.exists(func._source_file):
            raise FileNotFoundError(
                f"ExternalFunction '{func._name}': source file not found: {func._source_file}"
            )
        shutil.copy2(func._source_file, source_file)
        # Include the original source file's directory so relative includes
        # (e.g. "../aie_kernel_utils.h") still resolve after the file is
        # copied into kernel_dir.
        src_dir = os.path.dirname(os.path.abspath(func._source_file))
        include_dirs = list(func._include_dirs)
        if src_dir not in include_dirs:
            include_dirs.append(src_dir)
        compile_cxx_core_function(
            source_path=source_file,
            target_arch=target_arch,
            output_path=output_file,
            include_dirs=include_dirs,
            compile_args=func._compile_flags,
            cwd=kernel_dir,
        )
    else:
        raise ValueError("Neither source_string nor source_file is provided")

    # Rename symbol if a prefix is set.
    if getattr(func, "_symbol_prefix", None):
        original = func._original_name
        prefixed = func._name  # already prefixed
        _rename_symbol_in_object(output_file, original, prefixed)

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
