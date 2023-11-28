# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ctypes
import gc
import os
import pathlib
import sys
import tempfile

import numpy as np
import aie.dialects.aie
from aie.execution_engine import ExecutionEngine
from aie.ir import Context, Module
from aie.passmanager import PassManager
from aie.runtime import get_ranked_memref_descriptor


# RUN: %PYTHON %s 2>&1 | FileCheck %s
# REQUIRES: has_mlir_runtime_libraries


# Log everything to stderr and flush so that we have a unified stream to match
# errors/info emitted by MLIR to stderr.
def log(*args):
    print(*args, file=sys.stderr)
    sys.stderr.flush()


def run(f):
    log("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0


def lowerToLLVM(module):
    pm = PassManager.parse(
        "builtin.module(convert-complex-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)"
    )
    pm.run(module.operation)
    return module


#  Test loading of shared libraries.
# CHECK-LABEL: TEST: testSharedLibLoad
def testSharedLibLoad():
    with Context():
        module = Module.parse(
            """\
            module  {
              func.func @main(%arg0: memref<1xf32>) attributes { llvm.emit_c_interface } {
                %c0 = arith.constant 0 : index
                %cst42 = arith.constant 42.0 : f32
                memref.store %cst42, %arg0[%c0] : memref<1xf32>
                %u_memref = memref.cast %arg0 : memref<1xf32> to memref<*xf32>
                call @printMemrefF32(%u_memref) : (memref<*xf32>) -> ()
                return
              }
              func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }
            }
        """
        )
        arg0 = np.array([0.0]).astype(np.float32)

        arg0_memref_ptr = ctypes.pointer(
            ctypes.pointer(get_ranked_memref_descriptor(arg0))
        )

        if sys.platform == "win32":
            lib_dir = "bin"
        else:
            lib_dir = "lib"
        mlir_runner_utils = next(
            pathlib.Path(f"../../{lib_dir}/").glob("*mlir_runner_utils.*")
        )
        mlir_c_runner_utils = next(
            pathlib.Path(f"../../{lib_dir}/").glob("*mlir_c_runner_utils.*")
        )
        execution_engine = ExecutionEngine(
            lowerToLLVM(module),
            opt_level=3,
            shared_libs=[str(mlir_runner_utils), str(mlir_c_runner_utils)],
        )
        execution_engine.invoke("main", arg0_memref_ptr)

        # CHECK: Unranked Memref
        # CHECK-NEXT: [42]


run(testSharedLibLoad)


#  Test that nano time clock is available.
# CHECK-LABEL: TEST: testNanoTime
def testNanoTime():
    with Context():
        module = Module.parse(
            """\
            module {
              func.func @main() attributes { llvm.emit_c_interface } {
                %now = call @nanoTime() : () -> i64
                %memref = memref.alloca() : memref<1xi64>
                %c0 = arith.constant 0 : index
                memref.store %now, %memref[%c0] : memref<1xi64>
                %u_memref = memref.cast %memref : memref<1xi64> to memref<*xi64>
                call @printMemrefI64(%u_memref) : (memref<*xi64>) -> ()
                return
              }
              func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
              func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }
            }
            """
        )

        if sys.platform == "win32":
            lib_dir = "bin"
        else:
            lib_dir = "lib"
        mlir_runner_utils = next(
            pathlib.Path(f"../../{lib_dir}/").glob("*mlir_runner_utils.*")
        )
        mlir_c_runner_utils = next(
            pathlib.Path(f"../../{lib_dir}/").glob("*mlir_c_runner_utils.*")
        )
        execution_engine = ExecutionEngine(
            lowerToLLVM(module),
            opt_level=3,
            shared_libs=[str(mlir_runner_utils), str(mlir_c_runner_utils)],
        )
        execution_engine.invoke("main")
        # CHECK: Unranked Memref
        # CHECK: [{{.*}}]


run(testNanoTime)


#  Test that nano time clock is available.
# CHECK-LABEL: TEST: testDumpToObjectFile
def testDumpToObjectFile():
    fd, object_path = tempfile.mkstemp(suffix=".o")

    try:
        with Context():
            module = Module.parse(
                """
                module {
                  func.func @main() attributes { llvm.emit_c_interface } {
                    return
                  }
                }
            """
            )

            execution_engine = ExecutionEngine(lowerToLLVM(module), opt_level=3)

            # CHECK: Object file exists: True
            print(f"Object file exists: {os.path.exists(object_path)}")
            # CHECK: Object file is empty: True
            print(f"Object file is empty: {os.path.getsize(object_path) == 0}")

            execution_engine.dump_to_object_file(object_path)

            # CHECK: Object file exists: True
            print(f"Object file exists: {os.path.exists(object_path)}")
            # CHECK: Object file is empty: False
            print(f"Object file is empty: {os.path.getsize(object_path) == 0}")

    finally:
        os.close(fd)
        os.remove(object_path)


run(testDumpToObjectFile)
