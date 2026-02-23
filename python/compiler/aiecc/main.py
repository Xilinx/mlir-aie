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


# Features that are deprecated and not supported in C++ aiecc
# These flags are now accepted by C++ aiecc as no-ops for compatibility,
# but we still warn users that they have no effect
_DEPRECATED_HOST_FLAGS = {
    "--compile-host": "Host compilation is deprecated and not supported in C++ aiecc.",
    "--no-compile-host": "Host compilation is deprecated and not supported in C++ aiecc.",
    "--host-target": "Host compilation is deprecated and not supported in C++ aiecc.",
}

# Flags that should be completely filtered out (not passed to C++ aiecc)
_FILTERED_FLAGS = set()

# Host compilation arguments that should be filtered out
# These are compiler/linker flags used for host code compilation
_HOST_COMPILATION_PREFIXES = ["-I", "-L", "-l"]


def _is_host_source_file(arg):
    """
    Check if an argument is a host source file (not an MLIR file).

    Host source files are .c, .cpp, .cc, .cxx files that should be
    compiled with the host compiler, not passed to aiecc.
    """
    if arg.startswith("-"):
        return False
    host_extensions = (".c", ".cpp", ".cc", ".cxx", ".C")
    return arg.endswith(host_extensions)


def _check_deprecated_flags(args):
    """
    Check for deprecated flags and emit warnings.

    Args:
        args: List of command-line arguments

    Returns:
        List of arguments with deprecated host compilation flags removed
    """
    filtered_args = []
    skip_next = False
    warned_host_compilation = False

    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue

        # Check for deprecated flags and warn, but pass them through to C++ aiecc
        # (C++ aiecc now accepts these as no-ops for compatibility)
        for flag, message in _DEPRECATED_HOST_FLAGS.items():
            if arg == flag or arg.startswith(flag + "="):
                warnings.warn(
                    f"{message} This flag will be passed to C++ aiecc but ignored.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                # Note: we don't skip_next for --host-target anymore since we pass it through
                break

        # Filter out host compilation flags (-I, -L, -l prefixes)
        is_host_flag = False
        for prefix in _HOST_COMPILATION_PREFIXES:
            if arg.startswith(prefix):
                is_host_flag = True
                if not warned_host_compilation:
                    warnings.warn(
                        "Host compilation flags (-I, -L, -l) are deprecated and "
                        "not supported in C++ aiecc. These flags will be ignored.",
                        DeprecationWarning,
                        stacklevel=3,
                    )
                    warned_host_compilation = True
                break

        if is_host_flag:
            continue

        # Filter out -o flag and its argument (output file for host compilation)
        if arg == "-o":
            if not warned_host_compilation:
                warnings.warn(
                    "Host compilation output flag (-o) is deprecated and "
                    "not supported in C++ aiecc. This flag will be ignored.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                warned_host_compilation = True
            skip_next = True
            continue

        # Filter out host source files (.c, .cpp, .cc, .cxx)
        if _is_host_source_file(arg):
            if not warned_host_compilation:
                warnings.warn(
                    "Host source files are deprecated and not supported in "
                    "C++ aiecc. These files will be ignored.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                warned_host_compilation = True
            continue

        filtered_args.append(arg)

    return filtered_args


def main():
    """
    Main entry point - delegates to C++ aiecc.

    All command-line arguments are passed directly to the C++ binary,
    except for deprecated host compilation flags which are filtered out
    with a warning.
    """
    try:
        aiecc_bin = _find_aiecc_binary()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Check for deprecated flags and filter them out
    filtered_args = _check_deprecated_flags(sys.argv[1:])

    # Pass filtered arguments to C++ binary
    result = subprocess.run([aiecc_bin, *filtered_args])
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
