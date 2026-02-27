# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc.

"""
aiecc.py - AIE Compiler Driver (Python wrapper)

This is a thin wrapper that delegates to the C++ aiecc binary.
The C++ implementation provides better performance through
in-memory MLIR pass execution instead of subprocess calls.

All command-line arguments are passed through unchanged to the
C++ binary, which handles host compilation flags (-I, -L, -l, -o),
host source files (.cpp), and all other options directly.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path


def _find_aiecc_binary():
    """
    Find the C++ aiecc binary.

    Search order:
    1. Relative to this file (installed location: python/compiler/aiecc -> bin/aiecc)
    2. In PATH
    """
    # Check relative to this file (installed location)
    # python/compiler/aiecc/main.py -> bin/aiecc
    bin_dir = Path(__file__).parent.parent.parent.parent / "bin"
    aiecc_path = bin_dir / "aiecc"
    if aiecc_path.exists() and os.access(aiecc_path, os.X_OK):
        return str(aiecc_path)

    # Check PATH
    path = shutil.which("aiecc")
    if path:
        return path

    raise FileNotFoundError(
        "Could not find 'aiecc' binary. Ensure mlir-aie is properly installed "
        "and the bin directory is in your PATH, or use the C++ aiecc directly."
    )


def main():
    """
    Main entry point - delegates to C++ aiecc.

    All command-line arguments are passed directly to the C++ binary unchanged.
    """
    try:
        aiecc_bin = _find_aiecc_binary()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Pass all arguments directly to C++ binary unchanged
    result = subprocess.run([aiecc_bin, *sys.argv[1:]])
    sys.exit(result.returncode)


def run(mlir_module, args=None):
    """
    Programmatic API for compiling MLIR modules.

    DEPRECATED: This function is deprecated. Use the C++ aiecc binary
    directly or the IRON Python API instead.

    Args:
        mlir_module: MLIR module string or object with __str__ method
        args: Optional list of command-line arguments

    Raises:
        RuntimeError: If compilation fails
    """
    warnings.warn(
        "aiecc.run() is deprecated and will be removed in a future release. "
        "Use the C++ aiecc binary directly or the IRON Python API instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        aiecc_bin = _find_aiecc_binary()
    except FileNotFoundError as e:
        raise RuntimeError(str(e))

    # Convert module to string if needed
    mlir_str = str(mlir_module)

    # Write MLIR to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
        f.write(mlir_str)
        mlir_path = f.name

    try:
        cmd = [aiecc_bin, mlir_path]
        if args:
            if isinstance(args, str):
                cmd.extend(args.split())
            else:
                cmd.extend(args)

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            raise RuntimeError(
                f"aiecc failed with exit code {result.returncode}: {error_msg}"
            )

        return result.stdout
    finally:
        try:
            os.unlink(mlir_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
