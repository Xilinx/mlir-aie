# Copyright (C) 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from aie.ir import Context, Location, Module, InsertionPoint


# Create and print ModuleOp.
def construct_and_print_module(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            args = inspect.getfullargspec(f).args
            if args:
                if args == ["module"]:
                    module = f(module)
                else:
                    raise Exception(f"only `module` arg supported {args=}")
            else:
                f()
        if module is not None:
            assert module.operation.verify()
            print(module)
