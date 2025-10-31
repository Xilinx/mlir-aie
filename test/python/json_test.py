# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s

import pytest
import aie.iron as iron
import json


def my_func():
    pass


def test_json_serialization():
    compilable = iron.compileconfig(
        my_func,
        compile_flags=["-O3"],
        source_files=["a.cpp", "b.cpp"],
        aiecc_flags=["--verbose"],
        metaargs={"my_var": 42},
        object_files=["a.o", "b.o"],
    )
    json_str = compilable.to_json()
    data = json.loads(json_str)

    assert data["mlir_generator"] == "my_func"
    assert data["compile_flags"] == ["-O3"]
    assert data["source_files"] == ["a.cpp", "b.cpp"]
    assert data["aiecc_flags"] == ["--verbose"]
    assert data["metaargs"] == {"my_var": 42}
    assert data["object_files"] == ["a.o", "b.o"]


def test_json_deserialization():
    json_str = """
    {
        "mlir_generator": "my_func",
        "use_cache": true,
        "compile_flags": ["-O3"],
        "source_files": ["a.cpp", "b.cpp"],
        "include_paths": null,
        "aiecc_flags": ["--verbose"],
        "metaargs": {
            "my_var": 42
        },
        "object_files": ["a.o", "b.o"]
    }
    """
    compilable = iron.Compilable.from_json(json_str, my_func)
    assert compilable.mlir_generator.__name__ == "my_func"
    assert compilable.compile_flags == ["-O3"]
    assert compilable.source_files == ["a.cpp", "b.cpp"]
    assert compilable.aiecc_flags == ["--verbose"]
    assert compilable.metaargs == {"my_var": 42}
    assert compilable.object_files == ["a.o", "b.o"]


def test_compilable_round_trip():
    compilable1 = iron.compileconfig(
        my_func,
        compile_flags=["-O3"],
        source_files=["a.cpp", "b.cpp"],
        aiecc_flags=["--verbose"],
        metaargs={"my_var": 42},
        object_files=["a.o", "b.o"],
    )
    json_str = compilable1.to_json()
    compilable2 = iron.Compilable.from_json(json_str, my_func)
    assert compilable1.to_json() == compilable2.to_json()


def test_iron_callable_round_trip():
    callable1 = iron.jit(
        my_func,
        compile_flags=["-O3"],
        source_files=["a.cpp", "b.cpp"],
        aiecc_flags=["--verbose"],
        metaargs={"my_var": 42},
        object_files=["a.o", "b.o"],
    )
    json_str = callable1.to_json()
    callable2 = iron.Callable.from_json(json_str, my_func)
    assert callable1.to_json() == callable2.to_json()


def test_get_json_schema():
    schema_str = iron.Callable.get_json_schema()
    schema = json.loads(schema_str)
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "function" in schema["properties"]


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        iron.compileconfig(
            my_func,
            source_files=["non_existent_file.cpp"],
        )
    with pytest.raises(FileNotFoundError):
        iron.compileconfig(
            my_func,
            object_files=["non_existent_file.o"],
        )
