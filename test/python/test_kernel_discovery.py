# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""The JIT builds exactly the objects a design links, and names never collide.

Walks every kernel case the device tests run, on both NPU generations, and
generates each design's MLIR (no compiler or NPU). The objects the JIT would
compile are the kernels generation discovered; the objects the design links are
the ``link_with`` of its declarations. A discovered object nothing links is a
wasted compile; a linked object nothing discovered is a link failure.
"""

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest
from aie._mlir_libs._mlir import ir
from aie.iron import kernels
from aie.iron.algorithms import kernel_design as kd
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.kernel import ExternalFunction
from aie.utils import get_current_device
from aie.utils.hostruntime import set_current_device

sys.path.insert(0, str(Path(__file__).parent / "npu"))
from kernel_cases import CASES  # noqa: E402

_DEVICES = {"npu1": NPU1Col1, "npu2": NPU2Col1}


@dataclass
class _Design:
    declarations: list[tuple[str, str, str, str]]
    kernels: list
    prebuilt: set[str]


def _label(device, case):
    kwargs = ",".join(
        f"{k}={getattr(v, '__name__', v)}" for k, v in case.kwargs.items()
    )
    return f"{device}/{case.factory}({kwargs})/calls={case.calls}{case.tag or ''}"


def _declarations(module):
    """``(device, symbol, signature, object)`` for every linked declaration."""
    found = []

    def visit(op):
        op = op.operation
        if op.name == "func.func" and "link_with" in op.attributes:
            attrs = op.attributes
            found.append(
                (
                    str(op.parent.attributes["sym_name"]),
                    attrs["sym_name"].value,
                    str(attrs["function_type"].value),
                    attrs["link_with"].value,
                )
            )
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    return found


def _generate(case):
    factory = getattr(kernels, case.factory)
    fn = factory(**case.kwargs)
    design = kd.design(
        factory,
        params=fn.param_values(kd.sample_inputs(fn, calls=case.calls)),
        calls=case.calls,
        scalars=case.scalars,
        **case.kwargs,
    ).compilable
    ExternalFunction._instances.clear()
    module = design.generate_mlir()
    discovered = list(ExternalFunction._instances)
    ExternalFunction._instances.clear()
    return _Design(
        _declarations(module),
        discovered,
        {Path(f).name for f in design.object_files},
    )


@pytest.fixture(scope="module")
def corpus():
    previous = get_current_device(probe_runtime=False)
    designs = {}
    try:
        for name, device in _DEVICES.items():
            set_current_device(device())
            for case in CASES:
                if not case.devices or name in case.devices:
                    designs[_label(name, case)] = _generate(case)
    finally:
        set_current_device(previous)
    return designs


def test_corpus_covers_both_generations(corpus):
    assert {label.split("/")[0] for label in corpus} == set(_DEVICES)
    assert all(design.declarations for design in corpus.values())


def test_every_linked_object_is_built_and_every_built_object_linked(corpus):
    wrong = {}
    for label, design in corpus.items():
        linked = {obj for *_, obj in design.declarations}
        built = {k.object_file_name for k in design.kernels if k.object_file._source}
        unbuildable = {
            k.object_file_name for k in design.kernels if not k.object_file._source
        }
        if linked != built | design.prebuilt or not unbuildable <= design.prebuilt:
            wrong[label] = (sorted(linked), sorted(built), sorted(unbuildable))
    assert not wrong


def test_each_symbol_is_declared_once_per_device(corpus):
    repeated = {}
    for label, design in corpus.items():
        keys = [(device, symbol) for device, symbol, *_ in design.declarations]
        if len(keys) != len(set(keys)):
            repeated[label] = keys
    assert not repeated


def test_a_symbol_always_names_one_signature_and_object(corpus):
    """So any two kernels for one NPU can share a design without conflict."""
    meanings = defaultdict(set)
    for label, design in corpus.items():
        npu = label.split("/")[0]
        for _, symbol, signature, obj in design.declarations:
            meanings[npu, symbol].add((signature, obj))
    assert not {s: m for s, m in meanings.items() if len(m) > 1}


def test_an_object_name_names_one_recipe_per_generation(corpus):
    """Two recipes behind one object name would share a design directory's .o.

    Across generations a name may differ in recipe (``exp_bf16_1024.o`` wraps
    an aie2 source in ``lut_kernel.cc``): a design targets one device, and the
    object cache keys on the recipe and the target architecture.
    """
    recipes = defaultdict(set)
    for label, design in corpus.items():
        generation = label.split("/")[0]
        for kernel in design.kernels:
            recipes[generation, kernel.object_file_name].add(kernel.object_file._source)
    assert not {name: r for name, r in recipes.items() if len(r) > 1}
