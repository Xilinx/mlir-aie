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
import shutil

from aie.extras.context import mlir_mod_ctx
from ..utils.xrt import read_insts_binary
from .device import NPU1, NPU2, NPU1Col1, NPU2Col1
from .compile import compile_mlir_module
from .config import get_current_device
from .tensor import Tensor
from aie.dialects.aie import AIEDevice


# The `iron.jit` decorator below caches compiled kenrels inside the `IRON_CACHE_DIR` directory.
# Kernels are cached based on their hash value of the MLIR module string. If during compilation,
# we hit in the cache, the `iron.jit` will load the xclbin and instruction binary files from the cache.
IRON_CACHE_DIR = os.path.expanduser("~/.iron/cache")

# Global cache for compiled kernels at the function level
# Key: (function_name, args_signature) -> NPUKernel instance
_compiled_kernels = {}


class Promise:
    """
    Promise class for asynchronous execution.
    This class is used to wait for the kernel to complete.
    """
    def __init__(self, kernel, opcode, insts_buffer_bo, n_insts, hw_context, *kernel_args):
        from .graph import is_graph_capture_enabled

        # Store the kernel execution parameters
        self.kernel = kernel
        self.opcode = opcode
        self.insts_buffer_bo = insts_buffer_bo
        self.n_insts = n_insts
        self.hw_context = hw_context
        self.kernel_args = list(kernel_args)
        self.kernel_handle = None
        self.output_tensor = None

        # If graph capture is enabled, don't execute immediately
        # The Promise will be collected and executed during graph replay
        if not is_graph_capture_enabled():
            # Execute immediately if not in graph capture mode
            self._execute_kernel()

    def _execute_kernel(self):
        """Execute the kernel and return the result."""
        # Create a run object
        run = xrt.run(self.kernel)
        run.set_arg(0, self.opcode)
        run.set_arg(1, self.insts_buffer_bo)
        run.set_arg(2, self.n_insts)

        # Set additional kernel arguments
        for i, arg in enumerate(self.kernel_args):
            run.set_arg(3 + i, arg)

        # Start the run
        run.start()
        self.kernel_handle = run
        return self

    def get_hw_context(self):
        """
        Get the hardware context of the kernel.
        """
        return self.hw_context

    def done(self):
        """
        Wait for the kernel to complete.
        """
        if self.kernel_handle is not None:
            result = self.kernel_handle.wait()
            if result != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(f"Kernel returned {result}")


class KernelObject:
    """
    KernelObject class wrapper for NPU kernels.
    """
    def __init__(self, mlir_module, external_kernels, target_arch, kernel_name, *kernel_args):
        self.mlir_module = mlir_module
        self.external_kernels = external_kernels
        self.target_arch = target_arch
        self.kernel_name = kernel_name
        self.opcode = 3

        self.kernel_args = []
        for tensor in kernel_args:
            # Skip callable arguments since these are inlined in the kernel
            if callable(tensor):
                continue
            if not hasattr(tensor, "buffer_object"):
                raise TypeError(
                    f"Expected Tensor with .buffer_object(), got {type(tensor)}"
                )
            self.kernel_args.append(tensor.buffer_object())        



class UberKernel:
    """
    UberKernel class wrapper for NPU kernels.
    """
    def __init__(self, xclbin_path, inst_files, kernel_objects, device_index=0):
        self.xclbin_path = xclbin_path
        self.kernel_objects = kernel_objects
        self.inst_files = inst_files

        self.__device = xrt.device(device_index)

        # Find kernel by name in the xclbin
        self.__xclbin = xrt.xclbin(xclbin_path)

        self.__device.register_xclbin(self.__xclbin)
        self.__context = xrt.hw_context(self.__device, self.__xclbin.get_uuid())
        

        self.__kernels = {}
        for kernel_object in kernel_objects:
            name = kernel_object.kernel_name
            self.__kernels[name] = {"kernel": xrt.kernel(self.__context, name), "args": kernel_object.kernel_args}

        self.__insts = {}
        
        # Set up instruction stream
        for kernel_name, insts_path in self.inst_files.items():
            insts = read_insts_binary(insts_path)
            
            n_insts = len(insts)
            insts_buffers_bytes = n_insts * np.dtype(insts.dtype).itemsize

            # Magic number for RyzenAI group id that will be fixed in the future. See same code at XRT:
            # https://github.com/Xilinx/XRT/blob/56222ed5cfd119dff0d5bd920735b87024e8c829/src/runtime_src/core/common/api/xrt_module.cpp#L1621
            group_id = 1

            buffer_bo = xrt.bo(
                self.__device,
                insts_buffers_bytes,
                xrt.bo.cacheable,
                group_id,
            )

            # Copy into a temporary numpy buffer
            insts_buffer_bo_np = np.frombuffer(
                buffer_bo.map(), dtype=insts.dtype
            ).reshape(insts.shape)
            insts_buffer_bo_np[:] = insts

            # Always sync to the device in the constructor
            buffer_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)    

            self.__insts[kernel_name] = {"insts": insts_path, "n_insts": n_insts, "buffer_bo": buffer_bo}
            
    def get_kernel(self, kernel_name):
        """
        Get the kernel by name.
        """
        if kernel_name not in self.__kernels:
            raise ValueError(f"Kernel {kernel_name} not found")
        return self.__kernels[kernel_name]
    
    def get_insts(self, kernel_name):
        """
        Get the instruction file of the kernel.
        """
        if kernel_name not in self.__insts:
            raise ValueError(f"Kernel {kernel_name} not found")
        return self.__insts[kernel_name]
    
    def get_hw_context(self):
        """
        Get the hardware context of the kernel.
        """
        return self.__context
    def get_insts_buffer_bo(self):
        """
        Get the instruction buffer object of the kernel.
        """
        return self.__insts_buffer_bo
    
    
def compile_kernel_objects(kernel_objects: list[KernelObject]):
    """
    Compile multiple kernel objects into a single combined xclbin.
    Following the pattern from the example where kernels are combined sequentially.
    """
    if not kernel_objects:
        raise ValueError("No kernel objects provided")
    
    # Calculate total hash for all kernels
    total_hash = ""
    for kernel_object in kernel_objects:
        total_hash = hash_module(kernel_object.mlir_module, kernel_object.external_kernels, kernel_object.target_arch, total_hash)
    
    # Create combined kernel directory
    combined_kernel_dir = os.path.join(IRON_CACHE_DIR, f"{total_hash}")
    os.makedirs(combined_kernel_dir, exist_ok=True)
    
    # Check if already compiled
    combined_xclbin_path = os.path.join(combined_kernel_dir, "combined.xclbin")
    
    inst_files = {}
    for kernel_object in kernel_objects:
        inst_files[kernel_object.kernel_name] = os.path.join(combined_kernel_dir, f"{kernel_object.kernel_name}.bin")
    
    if os.path.exists(combined_xclbin_path) and all(os.path.exists(inst_file) for inst_file in inst_files.values()):
        # Return the combined NPUKernel
        return UberKernel(combined_xclbin_path, inst_files, kernel_objects)
    
    # Compile kernels sequentially, combining them into a single xclbin
    current_xclbin_path = None
    
    for i, kernel_object in enumerate(kernel_objects):
        # Write MLIR to file in the combined directory
        mlir_path = os.path.join(combined_kernel_dir, f"{kernel_object.kernel_name}.mlir")
        with open(mlir_path, "w", encoding="utf-8") as f:
            print(kernel_object.mlir_module, file=f)
        
        # Compile ExternalFunctions in the combined directory
        for func in kernel_object.external_kernels:
            compile_external_kernel(func, combined_kernel_dir, kernel_object.target_arch)
        
        # Determine output paths - all in the same directory
        xclbin_path = os.path.join(combined_kernel_dir, f"{kernel_object.kernel_name}.xclbin")
        insts_path = os.path.join(combined_kernel_dir, f"{kernel_object.kernel_name}.bin")
            
        xclbin_input = None if i == 0 else current_xclbin_path
        
        # Compile options for aiecc
        options = [
            f"--xclbin-kernel-name={kernel_object.kernel_name}",
            f"--xclbin-kernel-id={hex(0x900 + i)}",  # Start from 0x900 like in example
            f"--xclbin-instance-name={kernel_object.kernel_name}_INST",
        ]
        
        
        if xclbin_input:
            options.append(f"--xclbin-input={xclbin_input}")
        inst_files[kernel_object.kernel_name] = insts_path   
        # Compile the kernel
        try:
            compile_mlir_module(
                mlir_module=kernel_object.mlir_module,
                insts_path=insts_path,
                xclbin_path=xclbin_path,
                work_dir=combined_kernel_dir,
                options=options,
            )
        except Exception as e:
            # Clean up on failure
            if os.path.exists(combined_kernel_dir):
                shutil.rmtree(combined_kernel_dir)
            raise e
        
        # Update current xclbin path for next iteration
        current_xclbin_path = xclbin_path
    
    # Copy final xclbin and insts to combined location
    shutil.copy2(current_xclbin_path, combined_xclbin_path)
    
    # Return the combined NPUKernel
    return UberKernel(combined_xclbin_path, inst_files, kernel_objects)
    

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

        self.__device = xrt.device(device_index)

        # Find kernel by name in the xclbin
        self.__xclbin = xrt.xclbin(xclbin_path)
        kernels = self.__xclbin.get_kernels()

        try:
            xkernel = [k for k in kernels if kernel_name == k.get_name()][0]
        except KeyError:
            raise NPUKernel_Error("No such kernel: " + kernel_name)

        self.__device.register_xclbin(self.__xclbin)
        self.__context = xrt.hw_context(self.__device, self.__xclbin.get_uuid())
        self.__kernel = xrt.kernel(self.__context, xkernel.get_name())

        # Set up instruction stream
        insts = read_insts_binary(insts_path)
        self.__n_insts = len(insts)
        insts_buffers_bytes = self.__n_insts * np.dtype(insts.dtype).itemsize

        # Magic number for RyzenAI group id that will be fixed in the future. See same code at XRT:
        # https://github.com/Xilinx/XRT/blob/56222ed5cfd119dff0d5bd920735b87024e8c829/src/runtime_src/core/common/api/xrt_module.cpp#L1621
        group_id = 1

        self.__insts_buffer_bo = xrt.bo(
            self.__device,
            insts_buffers_bytes,
            xrt.bo.cacheable,
            group_id,
        )

        # Copy into a temporary numpy buffer
        insts_buffer_bo_np = np.frombuffer(
            self.__insts_buffer_bo.map(), dtype=insts.dtype
        ).reshape(insts.shape)
        insts_buffer_bo_np[:] = insts

        # Always sync to the device in the constructor
        self.__insts_buffer_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def __call__(self, *args):
        """
        Allows the kernel to be called as a function with the provided arguments.

        Parameters:
            args (IRON Tensors): Arguments to pass to the kernel.
        """

        opcode = 3
        kernel_args = []

        for tensor in args:
            # Skip callable arguments since these are inlined in the kernel
            if callable(tensor):
                continue
            if not hasattr(tensor, "buffer_object"):
                raise TypeError(
                    f"Expected Tensor with .buffer_object(), got {type(tensor)}"
                )
            kernel_args.append(tensor.buffer_object())



        h = self.__kernel(opcode, self.__insts_buffer_bo, self.__n_insts, *kernel_args)
        r = h.wait()
        if r != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise NPUKernel_Error(f"Kernel returned {r}")

    def __del__(self):
        """
        Destructor to clean up resources and delete the kernel and device objects.
        """
        del self.__kernel
        del self.__device


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
        from .kernel import ExternalFunction

        # Check if we already have a compiled kernel for this function signature
        cache_key = _create_function_cache_key(function, args, kwargs)
        if cache_key in _compiled_kernels:
            cached_kernel = _compiled_kernels[cache_key]
            return cached_kernel(*args, **kwargs)

        # Clear any instances from previous runs to make sure if the user provided any broken code we don't try to recompile it
        ExternalFunction.clear()

        # Find ExternalFunction instances in arguments and kwargs
        external_kernels = []
        for arg in args:
            if isinstance(arg, ExternalFunction):
                external_kernels.append(arg)
        for value in kwargs.values():
            if isinstance(value, ExternalFunction):
                external_kernels.append(value)

        # Execute the function to generate MLIR
        try:
            if is_placed:
                with mlir_mod_ctx() as ctx:
                    function(*args, **kwargs)
                    assert (
                        ctx.module.operation.verify()
                    ), f"Verification failed for '{function.__name__}'"
                    mlir_module = ctx.module
            else:
                mlir_module = function(*args, **kwargs)
        except Exception as e:
            raise

        # Compile all ExternalFunction instances that were created during this JIT compilation
        for func in ExternalFunction._instances:
            external_kernels.append(func)
        
        # Determine target architecture based on device type
        try:
            current_device = get_current_device()

            # Determine target architecture based on device type
            if isinstance(current_device, (NPU2, NPU2Col1)):
                target_arch = "aie2p"
            elif isinstance(current_device, (NPU1, NPU1Col1)):
                target_arch = "aie2"
            elif current_device in (AIEDevice.npu2, AIEDevice.npu2_1col):
                target_arch = "aie2p"
            elif current_device in (AIEDevice.npu1, AIEDevice.npu1_1col):
                target_arch = "aie2"
            else:
                raise RuntimeError(f"Unsupported device type: {type(current_device)}")
        except Exception as e:
            raise
        
        from .graph import is_graph_capture_enabled
        if is_graph_capture_enabled():
            kernel_name = function.__name__
            return KernelObject(mlir_module, external_kernels, target_arch, kernel_name, *args)        

        # Hash of the IR string, ExternalFunction compiler options, and target architecture
        module_hash = hash_module(mlir_module, external_kernels, target_arch)
        kernel_dir = os.path.join(IRON_CACHE_DIR, f"{module_hash}")
        mlir_path = os.path.join(kernel_dir, "aie.mlir")

        # Ensure cache directory exists
        os.makedirs(kernel_dir, exist_ok=True)

        # Write MLIR to file if not already cached
        inst_filename = "insts.bin"
        xclbin_filename = "final.xclbin"
        xclbin_path = os.path.join(kernel_dir, xclbin_filename)
        inst_path = os.path.join(kernel_dir, inst_filename)

        xclbin_exists = os.path.exists(xclbin_path)
        inst_exists = os.path.exists(inst_path)

        if not use_cache or not xclbin_exists or not inst_exists:
            try:
                with open(mlir_path, "w", encoding="utf-8") as f:
                    print(mlir_module, file=f)

                # Compile ExternalFunctions from inside the JIT compilation directory
                for func in external_kernels:
                    compile_external_kernel(func, kernel_dir, target_arch)

                # Compile the MLIR module
                compile_mlir_module(
                    mlir_module=mlir_module,
                    insts_path=inst_path,
                    xclbin_path=xclbin_path,
                    work_dir=kernel_dir,
                )
            except Exception as e:
                # Clean up cache directory on any compilation failure to avoid any corrupted objects in the cache
                if os.path.exists(kernel_dir):
                    shutil.rmtree(kernel_dir)
                raise e

        kernel_name = "MLIR_AIE"
        try:
            kernel = NPUKernel(xclbin_path, inst_path, kernel_name=kernel_name)
            
            # Cache the kernel for this function signature
            _compiled_kernels[cache_key] = kernel
            
            result = kernel(*args, **kwargs)
            return result
        except Exception as e:
            raise

    return decorator


def compile_external_kernel(func, kernel_dir, target_arch):
    """
    Compile an ExternalFunction to an object file in the kernel directory.

    Args:
        func: ExternalFunction instance to compile
        kernel_dir: Directory to place the compiled object file
        target_arch: Target architecture (e.g., "aie2" or "aie2p")
    """
    # Check if we can reuse a cached object file
    if (
        hasattr(func, "_cache_key")
        and func._cache_key
        and func._cache_key in func._cache
    ):
        cached_info = func._cache[func._cache_key]
        cached_object_file = cached_info["object_file_name"]

        # Check if object file already exists in current kernel directory
        current_output_path = os.path.join(kernel_dir, cached_object_file)
        if os.path.exists(current_output_path):
            # Object file already exists locally, just use it
            func._object_file_name = cached_object_file
            return

        # Copy the cached object file to the current kernel directory
        # This happens when we have a code object that is already cached from previous
        # runs. The issue is that ExternalFunction objects get resolved during MLIR
        # generation, but each JIT call creates different MLIR modules (different hashes),
        # so they end up in different cache directories. However, the ExternalFunction
        # cache is global and contains object files from previous directories. We can't
        # just reference the old path because the linker runs in the new directory,
        # so we must copy the cached object file to the current kernel directory.
        # We also can't simply clear the code object cache between JIT compilations
        # because ExternalFunctions get resolved when the worker gets resolved during
        # MLIR generation. If we clear the instances, they no longer exist and we
        # don't see the ExternalFunction anymore, breaking the compilation pipeline.
        cached_source_dir = cached_info.get("source_dir", kernel_dir)
        cached_source_path = os.path.join(cached_source_dir, cached_object_file)

        if os.path.exists(cached_source_path):
            # Copy object file to current kernel directory
            shutil.copy2(cached_source_path, current_output_path)

            # Update the function to use the local copy
            func._object_file_name = cached_object_file
            return
        else:
            # Cached object file doesn't exist, remove from cache and recompile
            del func._cache[func._cache_key]

    # Check if object file already exists in kernel directory
    output_file = os.path.join(kernel_dir, func._object_file_name)
    if os.path.exists(output_file):
        return

    # Create source file in kernel directory
    source_file = os.path.join(kernel_dir, f"{func._name}.cc")

    # Get source content
    try:
        source_content = func._get_source_content()
        with open(source_file, "w") as f:
            f.write(source_content)
    except Exception as e:
        raise

    from .compile.compile import compile_cxx_core_function

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

        # Only add to cache after successful compilation
        if hasattr(func, "_cache_key") and func._cache_key:
            # Store both object file name and source directory for future copying
            func.add_to_cache(func._cache_key, func._object_file_name, kernel_dir)

    except Exception as e:
        # Don't add to cache if compilation failed
        raise


def _create_function_cache_key(function, args, kwargs):
    """
    Create a cache key for a function call based on function name and argument types/shapes.
    This allows us to cache compiled kernels at the function level.
    """
    # Get function name
    func_name = function.__name__

    # Create signature from argument types and shapes
    signature_parts = []

    for arg in args:
        if hasattr(arg, "shape") and hasattr(arg, "dtype"):
            # Tensor argument - include shape and dtype
            signature_parts.append(f"tensor_{arg.shape}_{arg.dtype}")
        elif hasattr(arg, "__class__"):
            # Other argument - include type name
            signature_parts.append(f"{arg.__class__.__name__}")
        else:
            # Fallback
            signature_parts.append(str(type(arg)))

    for key, value in sorted(kwargs.items()):
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            signature_parts.append(f"{key}_tensor_{value.shape}_{value.dtype}")
        else:
            signature_parts.append(f"{key}_{type(value).__name__}")

    signature = "_".join(signature_parts)
    return (func_name, signature)


def hash_module(module, external_kernels=None, target_arch=None, starting_hash=None):
    """
    Hash the MLIR module and ExternalFunction compiler options to create a unique identifier.
    """
    mlir_str = str(module)

    # Include ExternalFunction compiler options in the hash
    if external_kernels:
        compiler_options = []
        for func in external_kernels:
            compiler_options.extend(func._include_dirs)
            compiler_options.extend(func._compile_flags)

        # Create a combined string for hashing
        combined_str = mlir_str + "|" + "|".join(compiler_options)
    else:
        combined_str = mlir_str

    # Include target architecture in the hash
    if target_arch:
        combined_str += f"|target_arch={target_arch}"

    if starting_hash:
        combined_str += f"|{starting_hash}"
        
    hash_result = hashlib.sha256(combined_str.encode("utf-8")).hexdigest()[:16]
    return hash_result
