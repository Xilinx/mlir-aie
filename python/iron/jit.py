# jit.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import os
import functools
import hashlib
import numpy as np
import pyxrt as xrt
import shutil
import sys
import traceback

from aie.extras.context import mlir_mod_ctx
from ..utils.compile import compile_mlir_module_to_binary
from ..utils.xrt import read_insts_binary
from .device import NPU1, NPU2
from .config import get_current_device


# The `iron.jit` decorator below caches compiled kenrels inside the `IRON_CACHE_DIR` directory.
# Kernels are cached based on their hash value of the MLIR module string. If during compilation,
# we hit in the cache, the `iron.jit` will load the xclbin and instruction binary files from the cache.
IRON_CACHE_DIR = os.path.expanduser("~/.iron/cache")


class NPUKernel:
    """
    NPUKernel class wrapper for NPU kernels.
    """

    def __init__(
        self, xclbin_path, insts_path, device_index=0, kernel_name="PP_FD_PRE"
    ):
        """
        Initialize the NPUKernel object.
        Parameters:
            xclbin_path (str): Path to the XCLBIN file containing the kernel.
            insts_path (str): Path to the instruction binary file for the kernel.
            device_index (int, optional): Index of the device. Defaults to 0.
            kernel_name (str, optional): Name of the kernel. Defaults to "PP_FD_PRE".
        """
        print(f"[DEBUG] NPUKernel.__init__: xclbin_path={xclbin_path}")
        print(f"[DEBUG] NPUKernel.__init__: insts_path={insts_path}")
        print(f"[DEBUG] NPUKernel.__init__: device_index={device_index}")
        print(f"[DEBUG] NPUKernel.__init__: kernel_name={kernel_name}")

        try:
            self.__device = xrt.device(device_index)
            print(
                f"[DEBUG] NPUKernel.__init__: Created device with index {device_index}"
            )
        except Exception as e:
            print(f"[DEBUG] NPUKernel.__init__: Failed to create device: {e}")
            raise

        # Find kernel by name in the xclbin
        try:
            self.__xclbin = xrt.xclbin(xclbin_path)
            print(f"[DEBUG] NPUKernel.__init__: Loaded xclbin from {xclbin_path}")
            kernels = self.__xclbin.get_kernels()
            print(f"[DEBUG] NPUKernel.__init__: Found {len(kernels)} kernels in xclbin")
            for i, k in enumerate(kernels):
                print(f"[DEBUG] NPUKernel.__init__: Kernel {i}: {k.get_name()}")
        except Exception as e:
            print(f"[DEBUG] NPUKernel.__init__: Failed to load xclbin: {e}")
            raise

        try:
            xkernel = [k for k in kernels if kernel_name == k.get_name()][0]
            print(f"[DEBUG] NPUKernel.__init__: Found kernel '{kernel_name}'")
        except (KeyError, IndexError) as e:
            print(
                f"[DEBUG] NPUKernel.__init__: Failed to find kernel '{kernel_name}': {e}"
            )
            raise NPUKernel_Error("No such kernel: " + kernel_name)

        try:
            self.__device.register_xclbin(self.__xclbin)
            print(f"[DEBUG] NPUKernel.__init__: Registered xclbin with device")
            self.__context = xrt.hw_context(self.__device, self.__xclbin.get_uuid())
            print(f"[DEBUG] NPUKernel.__init__: Created hardware context")
            self.__kernel = xrt.kernel(self.__context, xkernel.get_name())
            print(f"[DEBUG] NPUKernel.__init__: Created kernel object")
        except Exception as e:
            print(f"[DEBUG] NPUKernel.__init__: Failed to create kernel: {e}")
            raise

        # Set up instruction stream
        try:
            insts = read_insts_binary(insts_path)
            print(
                f"[DEBUG] NPUKernel.__init__: Read {len(insts)} instructions from {insts_path}"
            )
            self.__n_insts = len(insts)
            insts_buffers_bytes = self.__n_insts * np.dtype(insts.dtype).itemsize
            print(
                f"[DEBUG] NPUKernel.__init__: Instruction buffer size: {insts_buffers_bytes} bytes"
            )
        except Exception as e:
            print(f"[DEBUG] NPUKernel.__init__: Failed to read instructions: {e}")
            raise

        # Magic number for RyzenAI group id that will be fixed in the future. See same code at XRT:
        # https://github.com/Xilinx/XRT/blob/56222ed5cfd119dff0d5bd920735b87024e8c829/src/runtime_src/core/common/api/xrt_module.cpp#L1621
        group_id = 1
        print(f"[DEBUG] NPUKernel.__init__: Using group_id={group_id}")

        try:
            self.__insts_buffer_bo = xrt.bo(
                self.__device,
                insts_buffers_bytes,
                xrt.bo.cacheable,
                group_id,
            )
            print(f"[DEBUG] NPUKernel.__init__: Created instruction buffer object")
        except Exception as e:
            print(
                f"[DEBUG] NPUKernel.__init__: Failed to create instruction buffer: {e}"
            )
            raise

        # Copy into a temporary numpy buffer
        try:
            insts_buffer_bo_np = np.frombuffer(
                self.__insts_buffer_bo.map(), dtype=insts.dtype
            ).reshape(insts.shape)
            insts_buffer_bo_np[:] = insts
            print(f"[DEBUG] NPUKernel.__init__: Copied instructions to buffer")
        except Exception as e:
            print(f"[DEBUG] NPUKernel.__init__: Failed to copy instructions: {e}")
            raise

        # Always sync to the device in the constructor
        try:
            self.__insts_buffer_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            print(f"[DEBUG] NPUKernel.__init__: Synced instructions to device")
        except Exception as e:
            print(f"[DEBUG] NPUKernel.__init__: Failed to sync instructions: {e}")
            raise

    # Blocking call.
    def __call__(self, *args):
        """
        Allows the kernel to be called as a function with the provided arguments.

        Parameters:
            args (IRON Tensors): Arguments to pass to the kernel.
        """
        print(f"[DEBUG] NPUKernel.__call__: Called with {len(args)} arguments")
        for i, arg in enumerate(args):
            print(f"[DEBUG] NPUKernel.__call__: Arg {i}: {type(arg)}")
            if hasattr(arg, "buffer_object"):
                print(
                    f"[DEBUG] NPUKernel.__call__: Arg {i} has buffer_object: {arg.buffer_object()}"
                )

        opcode = 3
        kernel_args = []

        for tensor in args:
            # Skip callable arguments since these are inlined in the kernel
            if callable(tensor):
                print(f"[DEBUG] NPUKernel.__call__: Skipping callable argument")
                continue
            if not hasattr(tensor, "buffer_object"):
                print(
                    f"[DEBUG] NPUKernel.__call__: Argument {type(tensor)} has no buffer_object"
                )
                raise TypeError(
                    f"Expected Tensor with .buffer_object(), got {type(tensor)}"
                )
            kernel_args.append(tensor.buffer_object())
            print(f"[DEBUG] NPUKernel.__call__: Added buffer object to kernel args")

        print(
            f"[DEBUG] NPUKernel.__call__: Calling kernel with opcode={opcode}, n_insts={self.__n_insts}, {len(kernel_args)} buffer args"
        )
        try:
            h = self.__kernel(
                opcode, self.__insts_buffer_bo, self.__n_insts, *kernel_args
            )
            print(f"[DEBUG] NPUKernel.__call__: Kernel execution started")
            r = h.wait()
            print(
                f"[DEBUG] NPUKernel.__call__: Kernel execution completed with result: {r}"
            )
            if r != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                print(f"[DEBUG] NPUKernel.__call__: Kernel failed with state: {r}")
                raise NPUKernel_Error(f"Kernel returned {r}")
        except Exception as e:
            print(f"[DEBUG] NPUKernel.__call__: Exception during kernel execution: {e}")
            print(f"[DEBUG] NPUKernel.__call__: Exception traceback:")
            traceback.print_exc()
            raise

    def __del__(self):
        """
        Destructor to clean up resources and delete the kernel and device objects.
        """
        print(f"[DEBUG] NPUKernel.__del__: Cleaning up resources")
        try:
            del self.__kernel
            del self.__device
        except Exception as e:
            print(f"[DEBUG] NPUKernel.__del__: Exception during cleanup: {e}")


class NPUKernel_Error(Exception):
    """
    Error raised when a NPU kernel encounters an error during execution.
    """

    pass


def jit(function=None, is_placed=True, use_cache=True):
    """
    Decorator to compile an IRON kernel into a binary to run on the NPU.

    Parameters:
    - is_placed (bool): Whether the kernel is using explicit or implicit placement Defaults to True.
    - use_cache (bool): Use cached MLIR module if available. Defaults to True.
    """

    if function is None:
        return functools.partial(jit, is_placed=is_placed, use_cache=use_cache)

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        print(
            f"[DEBUG] jit.decorator: Starting compilation for function {function.__name__}"
        )
        print(f"[DEBUG] jit.decorator: is_placed={is_placed}, use_cache={use_cache}")
        print(
            f"[DEBUG] jit.decorator: args count={len(args)}, kwargs count={len(kwargs)}"
        )

        # Import ExternalKernel at the top
        from .kernel import ExternalKernel

        # Clear any instances from previous runs to make sure if the user provided any broken code we don't try to recompile it
        ExternalKernel._instances.clear()
        print(f"[DEBUG] jit.decorator: Cleared ExternalKernel instances")

        # Find ExternalKernel instances in arguments and kwargs
        external_kernels = []
        for arg in args:
            if isinstance(arg, ExternalKernel):
                external_kernels.append(arg)
                print(
                    f"[DEBUG] jit.decorator: Found ExternalKernel in args: {arg._name}"
                )
        for value in kwargs.values():
            if isinstance(value, ExternalKernel):
                external_kernels.append(value)
                print(
                    f"[DEBUG] jit.decorator: Found ExternalKernel in kwargs: {value._name}"
                )

        # Execute the function to generate MLIR
        try:
            if is_placed:
                print(f"[DEBUG] jit.decorator: Generating MLIR with placement")
                with mlir_mod_ctx() as ctx:
                    function(*args, **kwargs)
                    print(f"[DEBUG] jit.decorator: Function executed successfully")
                    assert (
                        ctx.module.operation.verify()
                    ), f"Verification failed for '{function.__name__}'"
                    print(f"[DEBUG] jit.decorator: Module verification passed")
                    mlir_module = ctx.module
            else:
                print(f"[DEBUG] jit.decorator: Generating MLIR without placement")
                mlir_module = function(*args, **kwargs)
                print(f"[DEBUG] jit.decorator: Function returned MLIR module")
        except Exception as e:
            print(f"[DEBUG] jit.decorator: Exception during MLIR generation: {e}")
            traceback.print_exc()
            raise

        # Compile all ExternalKernel instances that were created during this JIT compilation
        for func in ExternalKernel._instances:
            if (
                not hasattr(func, "_compiled") or not func._compiled
            ):  # Don't compile if already compiled
                external_kernels.append(func)
                print(
                    f"[DEBUG] jit.decorator: Added ExternalKernel to compilation list: {func._name}"
                )

        # Determine target architecture based on device type (hoisted from compile_external_kernel)
        try:
            current_device = get_current_device()
            print(f"[DEBUG] jit.decorator: Current device: {current_device}")
            print(f"[DEBUG] jit.decorator: Device type: {type(current_device)}")

            # Determine target architecture based on device type
            if isinstance(current_device, NPU2):
                target_arch = "aie2p"
            elif isinstance(current_device, NPU1):
                target_arch = "aie2"
            else:
                print(
                    f"[DEBUG] jit.decorator: Unsupported device type: {type(current_device)}"
                )
                raise RuntimeError(f"Unsupported device type: {type(current_device)}")

            print(f"[DEBUG] jit.decorator: Target architecture: {target_arch}")
        except Exception as e:
            print(
                f"[DEBUG] jit.decorator: Failed to determine target architecture: {e}"
            )
            raise

        # Hash of the IR string, ExternalKernel compiler options, and target architecture
        module_hash = hash_module(mlir_module, external_kernels, target_arch)
        kernel_dir = os.path.join(IRON_CACHE_DIR, f"{module_hash}")
        mlir_path = os.path.join(kernel_dir, "aie.mlir")
        print(f"[DEBUG] jit.decorator: Module hash: {module_hash}")
        print(f"[DEBUG] jit.decorator: Kernel directory: {kernel_dir}")

        # Ensure cache directory exists
        os.makedirs(kernel_dir, exist_ok=True)
        print(f"[DEBUG] jit.decorator: Created/verified kernel directory")

        # Write MLIR to file if not already cached
        inst_filename = "insts.bin"
        xclbin_filename = "final.xclbin"
        xclbin_path = os.path.join(kernel_dir, xclbin_filename)
        inst_path = os.path.join(kernel_dir, inst_filename)

        xclbin_exists = os.path.exists(xclbin_path)
        inst_exists = os.path.exists(inst_path)
        print(f"[DEBUG] jit.decorator: xclbin exists: {xclbin_exists}")
        print(f"[DEBUG] jit.decorator: inst file exists: {inst_exists}")

        if not use_cache or not xclbin_exists or not inst_exists:
            print(
                f"[DEBUG] jit.decorator: Need to compile (cache disabled or files missing)"
            )
            try:
                with open(mlir_path, "w", encoding="utf-8") as f:
                    print(mlir_module, file=f)
                print(f"[DEBUG] jit.decorator: Wrote MLIR module to {mlir_path}")

                # Set cache directory for ExternalKernels and compile them
                for func in external_kernels:
                    print(
                        f"[DEBUG] jit.decorator: Compiling ExternalKernel: {func._name}"
                    )
                    # Compile the ExternalKernel directly in the kernel directory
                    compile_external_kernel(func, kernel_dir, target_arch)

                print(f"[DEBUG] jit.decorator: Starting MLIR module compilation")
                # Compile the MLIR module
                compile_mlir_module_to_binary(
                    mlir_module=mlir_module,
                    inst_path=inst_path,
                    xclbin_path=xclbin_path,
                    work_dir=kernel_dir,
                )
                print(f"[DEBUG] jit.decorator: MLIR module compilation completed")
            except Exception as e:
                print(f"[DEBUG] jit.decorator: Exception during compilation: {e}")
                traceback.print_exc()
                # Clean up cache directory on any compilation failure
                if os.path.exists(kernel_dir):
                    shutil.rmtree(kernel_dir)
                    print(
                        f"[DEBUG] jit.decorator: Cleaned up kernel directory after failure"
                    )
                raise e
        else:
            print(f"[DEBUG] jit.decorator: Using cached compilation results")

        kernel_name = "MLIR_AIE"
        print(
            f"[DEBUG] jit.decorator: Creating NPUKernel with kernel_name={kernel_name}"
        )
        try:
            kernel = NPUKernel(xclbin_path, inst_path, kernel_name=kernel_name)
            print(f"[DEBUG] jit.decorator: NPUKernel created successfully")
            result = kernel(*args, **kwargs)
            print(f"[DEBUG] jit.decorator: Kernel execution completed successfully")
            return result
        except Exception as e:
            print(
                f"[DEBUG] jit.decorator: Exception during kernel creation/execution: {e}"
            )
            traceback.print_exc()
            raise

    return decorator


def compile_external_kernel(func, kernel_dir, target_arch):
    """
    Compile an ExternalKernel to an object file in the kernel directory.

    Args:
        func: ExternalKernel instance to compile
        kernel_dir: Directory to place the compiled object file
        target_arch: Target architecture (e.g., "aie2" or "aie2p")
    """
    print(f"[DEBUG] compile_external_kernel: Starting compilation of {func._name}")
    print(f"[DEBUG] compile_external_kernel: kernel_dir={kernel_dir}")
    print(f"[DEBUG] compile_external_kernel: target_arch={target_arch}")

    # Skip if already compiled
    if hasattr(func, "_compiled") and func._compiled:
        print(f"[DEBUG] compile_external_kernel: Function already compiled, skipping")
        return

    # Check if object file already exists in kernel directory
    output_file = os.path.join(kernel_dir, func._object_file_name)
    if os.path.exists(output_file):
        print(
            f"[DEBUG] compile_external_kernel: Object file already exists: {output_file}"
        )
        return

    # Create source file in kernel directory
    source_file = os.path.join(kernel_dir, f"{func._name}.cc")
    print(f"[DEBUG] compile_external_kernel: Source file path: {source_file}")

    # Handle both source_string and source_file cases
    if func._source_string is not None:
        # Use source_string (write to file)
        try:
            with open(source_file, "w") as f:
                f.write(func._source_string)
            print(
                f"[DEBUG] compile_external_kernel: Wrote source from string to {source_file}"
            )
            print(
                f"[DEBUG] compile_external_kernel: Source content:\n{func._source_string}"
            )
        except Exception as e:
            print(f"[DEBUG] compile_external_kernel: Failed to write source file: {e}")
            raise
    elif func._source_file is not None:
        # Use source_file (copy existing file)
        print(
            f"[DEBUG] compile_external_kernel: Copying source file: {func._source_file}"
        )

        # Check if source file exists before copying
        if os.path.exists(func._source_file):
            try:
                shutil.copy2(func._source_file, source_file)
                print(
                    f"[DEBUG] compile_external_kernel: Copied source file successfully"
                )
            except Exception as e:
                print(
                    f"[DEBUG] compile_external_kernel: Failed to copy source file: {e}"
                )
                raise
        else:
            print(
                f"[DEBUG] compile_external_kernel: Source file does not exist: {func._source_file}"
            )
            return
    else:
        print(
            f"[DEBUG] compile_external_kernel: Neither source_string nor source_file provided"
        )
        raise ValueError("Neither source_string nor source_file is provided")

    from .compile.compile import compile_cxx_core_function

    print(f"[DEBUG] compile_external_kernel: Calling compile_cxx_core_function")
    print(f"[DEBUG] compile_external_kernel: source_path={source_file}")
    print(f"[DEBUG] compile_external_kernel: target_arch={target_arch}")
    print(f"[DEBUG] compile_external_kernel: output_path={output_file}")
    print(f"[DEBUG] compile_external_kernel: include_dirs={func._include_dirs}")
    print(f"[DEBUG] compile_external_kernel: compile_args={func._compile_flags}")
    print(f"[DEBUG] compile_external_kernel: cwd={kernel_dir}")

    try:
        compile_cxx_core_function(
            source_path=source_file,
            target_arch=target_arch,
            output_path=output_file,
            include_dirs=func._include_dirs,
            compile_args=func._compile_flags,
            cwd=kernel_dir,
            verbose=True,
        )
        print(f"[DEBUG] compile_external_kernel: Compilation completed successfully")
    except Exception as e:
        print(f"[DEBUG] compile_external_kernel: Compilation failed: {e}")
        traceback.print_exc()
        raise

    # Mark the function as compiled
    func._compiled = True
    print(f"[DEBUG] compile_external_kernel: Marked function as compiled")


def hash_module(module, external_kernels=None, target_arch=None):
    """
    Hash the MLIR module and ExternalKernel compiler options to create a unique identifier.
    """
    print(f"[DEBUG] hash_module: Starting hash computation")
    mlir_str = str(module)
    print(f"[DEBUG] hash_module: MLIR string length: {len(mlir_str)}")
    print(f"[DEBUG] hash_module: Target architecture: {target_arch}")

    # Include ExternalKernel compiler options in the hash
    if external_kernels:
        print(
            f"[DEBUG] hash_module: Processing {len(external_kernels)} external kernels"
        )
        compiler_options = []
        for func in external_kernels:
            # Include include_dirs and compile_flags in the hash
            compiler_options.extend(func._include_dirs)
            compiler_options.extend(func._compile_flags)
            print(
                f"[DEBUG] hash_module: Added options for {func._name}: include_dirs={func._include_dirs}, compile_flags={func._compile_flags}"
            )

        # Create a combined string for hashing
        combined_str = mlir_str + "|" + "|".join(compiler_options)
    else:
        combined_str = mlir_str

    # Include target architecture in the hash
    if target_arch:
        combined_str += f"|target_arch={target_arch}"
        print(f"[DEBUG] hash_module: Added target architecture to hash: {target_arch}")

    hash_result = hashlib.sha256(combined_str.encode("utf-8")).hexdigest()[:16]
    print(f"[DEBUG] hash_module: Computed hash: {hash_result}")
    return hash_result
