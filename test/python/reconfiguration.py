# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
# REQUIRES: peano
import shutil
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import (
    CompileTime,
    ExternalFunction,
    In,
    Out,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_

_TILE, _N = 16, 64


def _add_const_program(add_value):
    tile_ty = np.ndarray[(_TILE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(_N,), np.dtype[np.int32]]
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(a, b):
        for _ in range_(_N // _TILE):
            e = a.acquire(1)
            o = b.acquire(1)
            for i in range_(_TILE):
                o[i] = e[i] + add_value
            a.release(1)
            b.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod()])

    def sequence(inp, out, in_h, out_h):
        in_h.fill(inp)
        out_h.drain(out, wait=True)

    rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


@iron.jit(name="add_a", add_value=5)
def _design_a(inp: In, out: Out, *, add_value: CompileTime[int]):
    return _add_const_program(add_value)


@iron.jit(name="add_b", add_value=9)
def _design_b(inp: In, out: Out, *, add_value: CompileTime[int]):
    return _add_const_program(add_value)


def main():
    inp = iron.arange(_N, dtype=np.int32, device="npu")
    out_a = iron.zeros(_N, dtype=np.int32, device="npu")
    out_b = iron.zeros(_N, dtype=np.int32, device="npu")
    work = Path("/tmp/jkimko/recfg_unit")
    r = iron.Reconfiguration("prog", method="ctrlpkt", output_dir=str(work))
    r.add(_design_a, inp, out_a)
    r.add(_design_b, inp, out_b)
    elf = r.compile()
    assert Path(elf.path).exists(), f"no ELF at {elf.path}"
    # CHECK: entrypoints=['main:init', 'main:add_a', 'main:add_b']
    print(f"entrypoints={elf.entrypoints}")
    # CHECK: init=main:init needs_ctrl_bo=True
    print(f"init={elf.init} needs_ctrl_bo={elf.needs_ctrl_bo}")
    print("OK")  # CHECK: OK


main()


def test_single_no_method():
    inp = iron.arange(_N, dtype=np.int32, device="npu")
    out = iron.zeros(_N, dtype=np.int32, device="npu")
    r = iron.Reconfiguration("prog_n1", output_dir="/tmp/jkimko/recfg_n1")
    r.add(_design_a, inp, out)
    elf = r.compile()
    assert Path(elf.path).exists()
    assert elf.entrypoints == ["main:add_a"]
    assert elf.init is None and elf.needs_ctrl_bo is False
    print("N1-OK")  # CHECK: N1-OK


test_single_no_method()


def test_duplicate_names():
    inp = iron.arange(_N, dtype=np.int32, device="npu")
    out1 = iron.zeros(_N, dtype=np.int32, device="npu")
    out2 = iron.zeros(_N, dtype=np.int32, device="npu")
    r = iron.Reconfiguration(
        "prog_dup", method="ctrlpkt", output_dir="/tmp/jkimko/recfg_dup"
    )
    r.add(_design_a, inp, out1)
    r.add(_design_a, inp, out2)
    try:
        r.compile()
    except ValueError as e:
        assert "duplicate design name" in str(e), str(e)
        print("DUP-OK")  # CHECK: DUP-OK
    else:
        raise AssertionError("expected ValueError on duplicate names")


test_duplicate_names()


_KERNEL_SRC_TMPL = """extern "C" {{
    void {sym}(int* input, int* output, int tile_size) {{
        for (int i = 0; i < tile_size; i++) {{
            output[i] = input[i] + {delta};
        }}
    }}
}}"""


def _add_one_program(func, suffix=""):
    tile_ty = np.ndarray[(_TILE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(_N,), np.dtype[np.int32]]
    of_in = ObjectFifo(tile_ty, name=f"ein{suffix}")
    of_out = ObjectFifo(tile_ty, name=f"eout{suffix}")

    def core_body(a, b, func_to_apply):
        for _ in range_(_N // _TILE):
            e = a.acquire(1)
            o = b.acquire(1)
            func_to_apply(e, o, _TILE)
            a.release(1)
            b.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), func])

    def sequence(inp, out, in_h, out_h):
        in_h.fill(inp)
        out_h.drain(out, wait=True)

    rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def test_two_external_designs_fold():
    """Two designs, each with its own DISTINCT external kernel, fold together.

    Reconfiguration builds each design's own kernels via
    CompilableDesign._build_kernels (keyed off that design's own cached
    generation, not a process-global registry), so N externally-kerneled
    designs fold with no generation-order requirement.
    """
    work = Path("/tmp/jkimko/recfg_external")
    if work.exists():
        shutil.rmtree(work)
    tile_ty = np.ndarray[(_TILE,), np.dtype[np.int32]]

    func_c = ExternalFunction(
        "add_c_ext",
        source_string=_KERNEL_SRC_TMPL.format(sym="add_c_ext", delta=1),
        arg_types=[tile_ty, tile_ty, np.int32],
    )

    @iron.jit(name="ext_design_c", func=func_c)
    def _design_c(inp: In, out: Out, *, func: CompileTime[object]):
        return _add_one_program(func, suffix="_c")

    func_d = ExternalFunction(
        "add_d_ext",
        source_string=_KERNEL_SRC_TMPL.format(sym="add_d_ext", delta=2),
        arg_types=[tile_ty, tile_ty, np.int32],
    )

    @iron.jit(name="ext_design_d", func=func_d)
    def _design_d(inp: In, out: Out, *, func: CompileTime[object]):
        return _add_one_program(func, suffix="_d")

    inp = iron.arange(_N, dtype=np.int32, device="npu")
    out_c = iron.zeros(_N, dtype=np.int32, device="npu")
    out_d = iron.zeros(_N, dtype=np.int32, device="npu")

    r = iron.Reconfiguration("prog_ext", method="ctrlpkt", output_dir=str(work))
    r.add(_design_c, inp, out_c)
    r.add(_design_d, inp, out_d)
    elf = r.compile()
    assert Path(elf.path).exists(), f"no ELF at {elf.path}"
    # CHECK: entrypoints=['main:init', 'main:ext_design_c', 'main:ext_design_d']
    print(f"entrypoints={elf.entrypoints}")
    assert (work / func_c.object_file_name).exists(), f"no {func_c.object_file_name}"
    assert (work / func_d.object_file_name).exists(), f"no {func_d.object_file_name}"
    print("EXT-OK")  # CHECK: EXT-OK


test_two_external_designs_fold()


def test_object_file_collision_rejected():
    """Two designs whose kernels share an explicit object_file_name but have
    different symbol names/source must be rejected BEFORE any kernel builds.

    ExternalFunction's own __init__ collision guard only fires when the
    symbol name AND object_file_name both match; two differently-named
    kernels sharing one explicit object_file_name slip past that guard, and
    would otherwise silently overwrite one another in the flat-staged
    output_dir. Reconfiguration.compile() catches this fold-wide, before
    staging or building anything.
    """
    work = Path("/tmp/jkimko/recfg_collision")
    if work.exists():
        shutil.rmtree(work)
    tile_ty = np.ndarray[(_TILE,), np.dtype[np.int32]]

    func_e = ExternalFunction(
        "add_e_ext",
        object_file_name="shared.o",
        source_string=_KERNEL_SRC_TMPL.format(sym="add_e_ext", delta=3),
        arg_types=[tile_ty, tile_ty, np.int32],
    )

    @iron.jit(name="ext_design_e", func=func_e)
    def _design_e(inp: In, out: Out, *, func: CompileTime[object]):
        return _add_one_program(func, suffix="_e")

    func_f = ExternalFunction(
        "add_f_ext",
        object_file_name="shared.o",
        source_string=_KERNEL_SRC_TMPL.format(sym="add_f_ext", delta=4),
        arg_types=[tile_ty, tile_ty, np.int32],
    )

    @iron.jit(name="ext_design_f", func=func_f)
    def _design_f(inp: In, out: Out, *, func: CompileTime[object]):
        return _add_one_program(func, suffix="_f")

    inp = iron.arange(_N, dtype=np.int32, device="npu")
    out_e = iron.zeros(_N, dtype=np.int32, device="npu")
    out_f = iron.zeros(_N, dtype=np.int32, device="npu")

    r = iron.Reconfiguration("prog_collide", output_dir=str(work))
    r.add(_design_e, inp, out_e)
    r.add(_design_f, inp, out_f)
    try:
        r.compile()
    except ValueError as e:
        assert "object_file_name" in str(e), str(e)
        assert not (work / "prog_collide.elf").exists()
        print("COLLIDE-OK")  # CHECK: COLLIDE-OK
    else:
        raise AssertionError("expected ValueError for object_file_name collision")


test_object_file_collision_rejected()


def test_object_file_identical_digest_allowed():
    """Two designs whose kernels share an explicit object_file_name AND
    identical source (same symbol name/body -- same _content_digest()) must
    be ALLOWED: the guard only rejects a same-name/different-source clash,
    it does not reject every same-name pair. Proves the guard is idempotent,
    not match-all.
    """
    work = Path("/tmp/jkimko/recfg_same_digest")
    if work.exists():
        shutil.rmtree(work)
    tile_ty = np.ndarray[(_TILE,), np.dtype[np.int32]]

    func_g = ExternalFunction(
        "add_g_ext",
        object_file_name="shared_same.o",
        source_string=_KERNEL_SRC_TMPL.format(sym="add_g_ext", delta=5),
        arg_types=[tile_ty, tile_ty, np.int32],
    )

    @iron.jit(name="ext_design_g", func=func_g)
    def _design_g(inp: In, out: Out, *, func: CompileTime[object]):
        return _add_one_program(func, suffix="_g")

    func_h = ExternalFunction(
        "add_g_ext",
        object_file_name="shared_same.o",
        source_string=_KERNEL_SRC_TMPL.format(sym="add_g_ext", delta=5),
        arg_types=[tile_ty, tile_ty, np.int32],
    )

    @iron.jit(name="ext_design_h", func=func_h)
    def _design_h(inp: In, out: Out, *, func: CompileTime[object]):
        return _add_one_program(func, suffix="_h")

    inp = iron.arange(_N, dtype=np.int32, device="npu")
    out_g = iron.zeros(_N, dtype=np.int32, device="npu")
    out_h = iron.zeros(_N, dtype=np.int32, device="npu")

    r = iron.Reconfiguration("prog_same_digest", method="ctrlpkt", output_dir=str(work))
    r.add(_design_g, inp, out_g)
    r.add(_design_h, inp, out_h)
    elf = r.compile()
    assert Path(elf.path).exists(), f"no ELF at {elf.path}"
    # CHECK: entrypoints=['main:init', 'main:ext_design_g', 'main:ext_design_h']
    print(f"entrypoints={elf.entrypoints}")
    print("SAME-DIGEST-OK")  # CHECK: SAME-DIGEST-OK


test_object_file_identical_digest_allowed()


# Two symbols in ONE source file, mirroring reduce_max.cc's
# reduce_max_vector + compute_max companions.
_COMPANION_SRC = """extern "C" {
    void comp_add(int* input, int* output, int tile_size) {
        for (int i = 0; i < tile_size; i++) {
            output[i] = input[i] + 1;
        }
    }
    void comp_mul(int* output, int tile_size) {
        for (int i = 0; i < tile_size; i++) {
            output[i] = output[i] * 2;
        }
    }
}"""


def _add_two_program(func1, func2, suffix=""):
    tile_ty = np.ndarray[(_TILE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(_N,), np.dtype[np.int32]]
    of_in = ObjectFifo(tile_ty, name=f"cin{suffix}")
    of_out = ObjectFifo(tile_ty, name=f"cout{suffix}")

    def core_body(a, b, f1, f2):
        for _ in range_(_N // _TILE):
            e = a.acquire(1)
            o = b.acquire(1)
            f1(e, o, _TILE)
            f2(o, _TILE)
            a.release(1)
            b.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), func1, func2])

    def sequence(inp, out, in_h, out_h):
        in_h.fill(inp)
        out_h.drain(out, wait=True)

    rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def test_companion_kernels_shared_object_allowed():
    """A design's two COMPANION external kernels -- different symbol names /
    arg-types compiled from ONE source file, sharing an explicit
    object_file_name (the shared_object_file_name idiom, e.g. reduce_max.cc's
    reduce_max_vector + compute_max) -- must fold.

    Their compiled .o is byte-identical (same source, flags, toolchain, no
    symbol_prefix), so the fold-wide overwrite guard keys on
    _object_content_digest, NOT the ExternalFunction *identity*
    (_content_digest, which also folds in name + arg_types and so differs
    here). Regression for the old identity-keyed guard that wrongly rejected
    every real shared-.o design (all vector_reduce_max variants) under
    write32/ctrlpkt.
    """
    work = Path("/tmp/jkimko/recfg_companion")
    if work.exists():
        shutil.rmtree(work)
    tile_ty = np.ndarray[(_TILE,), np.dtype[np.int32]]

    func_k1 = ExternalFunction(
        "comp_add",
        object_file_name="companion.o",
        source_string=_COMPANION_SRC,
        arg_types=[tile_ty, tile_ty, np.int32],
    )
    func_k2 = ExternalFunction(
        "comp_mul",
        object_file_name="companion.o",
        source_string=_COMPANION_SRC,
        arg_types=[tile_ty, np.int32],
    )
    # The invariant the guard now depends on, asserted explicitly: identical
    # compiled object, distinct ExternalFunction identity.
    assert func_k1._object_content_digest() == func_k2._object_content_digest()
    assert func_k1._content_digest() != func_k2._content_digest()

    @iron.jit(name="ext_design_companion", func1=func_k1, func2=func_k2)
    def _design_comp(
        inp: In,
        out: Out,
        *,
        func1: CompileTime[object],
        func2: CompileTime[object],
    ):
        return _add_two_program(func1, func2, suffix="_comp")

    inp = iron.arange(_N, dtype=np.int32, device="npu")
    out = iron.zeros(_N, dtype=np.int32, device="npu")

    r = iron.Reconfiguration("prog_companion", method="write32", output_dir=str(work))
    r.add(_design_comp, inp, out)
    elf = r.compile()
    assert Path(elf.path).exists(), f"no ELF at {elf.path}"
    assert (work / "companion.o").exists(), "companion.o not built"
    print("COMPANION-OK")  # CHECK: COMPANION-OK


test_companion_kernels_shared_object_allowed()
