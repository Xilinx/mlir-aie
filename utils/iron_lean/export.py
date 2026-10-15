# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Export an IRON design's data movement from its live objects to a Lean `Design`."""

import argparse
import functools
import importlib.util
import math
import re
from typing import get_origin
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie import ir
from aie.helpers.npdtypes import np_ndarray_type_get_dtype, np_ndarray_type_get_shape
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Program, Worker, str_to_dtype
from aie.iron.kernel import Kernel
from aie.iron.kernels._common import TensorLayout
from aie.iron.dataflow.objectfifo import ObjectFifo, ObjectFifoLink
from aie.iron.device import from_name
from aie.iron.runtime.endpoint import RuntimeEndpoint

REPO = Path(__file__).resolve().parents[2]
WHOLE_ARRAY = (
    REPO / "programming_examples/basic/matrix_multiplication/whole_array/whole_array.py"
)
LEAN_ROOT = Path(__file__).parent / "IronVerify"
# REUSE-IgnoreStart
SPDX = (
    "-- Copyright (C) 2026 Advanced Micro Devices, Inc.\n"
    "-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n"
)
# REUSE-IgnoreEnd

DTYPE_NAMES = {
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "uint8": "ui8",
    "uint16": "ui16",
    "bfloat16": "bf16",
    "float32": "f32",
}

CMPI = {0: "=", 1: "≠", 2: "<", 3: "≤", 4: ">", 5: "≥", 6: "<", 7: "≤", 8: ">", 9: "≥"}
BINARY = {
    "arith.addi": "({a} + {b})",
    "arith.subi": "({a} - {b})",
    "arith.muli": "({a} * {b})",
    "arith.divsi": "(Int.tdiv {a} {b})",
    "arith.divui": "(Int.tdiv {a} {b})",
    "arith.floordivsi": "(Int.fdiv {a} {b})",
    "arith.ceildivsi": "(-(Int.fdiv (-{a}) {b}))",
    "arith.remsi": "(Int.tmod {a} {b})",
    "arith.remui": "(Int.emod {a} {b})",
    "arith.maxsi": "(max {a} {b})",
    "arith.minsi": "(min {a} {b})",
    "arith.andi": "({a} && {b})",
    "arith.ori": "({a} || {b})",
    "arith.xori": "({a} != {b})",
}
CASTS = {
    "arith.extsi",
    "arith.extui",
    "arith.trunci",
    "arith.index_cast",
    "arith.index_castui",
}


class LeanTerms:
    """Translates the SSA values a design's taps and control flow use into Lean `Int`/`Bool` terms.

    Casts are dropped, so a term means the exact integer only while no intermediate
    leaves its type's range.
    """

    def __init__(self, dispatch: dict[str, ir.Value]):
        self.dispatch = [(ir.Value(v), name) for name, v in dispatch.items()]
        self.ivs: list[tuple[ir.Block, str]] = []

    def bind_iv(self, block: ir.Block) -> str:
        name = f"i{len(self.ivs)}"
        self.ivs.append((block, name))
        return name

    def term(self, x) -> str:
        if isinstance(x, (int, np.integer)):
            return f"({int(x)} : Int)"
        v = ir.Value(x)
        owner = v.owner
        if isinstance(owner, ir.Block):
            for dv, name in self.dispatch:
                if dv == v:
                    return name
            arg = ir.BlockArgument(v)
            for block, name in self.ivs:
                if block == owner and arg.arg_number == 0:
                    return name
            raise NotImplementedError(f"block argument {v} has no Lean meaning")
        op = owner.operation if isinstance(owner, ir.OpView) else owner
        name = op.name
        if name == "arith.constant":
            value = ir.IntegerAttr(op.attributes["value"]).value
            if isinstance(v.type, ir.IntegerType) and v.type.width == 1:
                return "true" if value else "false"
            return f"({value} : Int)"
        if name in CASTS:
            return self.term(op.operands[0])
        if name in BINARY:
            a, b = (self.term(o) for o in op.operands)
            return BINARY[name].format(a=a, b=b)
        if name == "arith.cmpi":
            predicate = ir.IntegerAttr(op.attributes["predicate"]).value
            a, b = (self.term(o) for o in op.operands)
            return f"(decide ({a} {CMPI[predicate]} {b}))"
        if name == "arith.select":
            c, a, b = (self.term(o) for o in op.operands)
            return f"(if {c} then {a} else {b})"
        raise NotImplementedError(f"no Lean translation for {name}: {op}")

    def walk(self, tap: TensorAccessPattern) -> str:
        if tap.padding is not None:
            raise NotImplementedError(f"padded walk {tap}")
        dims = ", ".join(
            f"({self.term(s)}, {self.term(t)})" for s, t in tap.transformation_dims
        )
        return f"{{ offset := {self.term(tap.offset)}, dims := [{dims}] }}"


class DesignExporter:
    """Reads a resolved Program's fifo graph and runtime sequence and writes them as Lean."""

    def __init__(
        self, program: Program, canonical: bool = False, contracts: bool = False
    ):
        self.program = program
        self.contracts = contracts
        self.runtime = program.entry
        self.workers = [w for c in program.configurations for w in c.workers]
        self.terms = LeanTerms(self.runtime.dispatch_values)
        self.fifos: list[ObjectFifo] = []
        self.links: list[ObjectFifoLink] = []
        self.collect()
        if canonical:
            # Natural order by name, so a parametric schema can list the same fifos.
            self.fifos.sort(key=lambda of: natural_key(of.name))
            self.links.sort(key=lambda l: natural_key(l.dsts[0].object_fifo.name))

    def collect(self) -> None:
        pending = [h.object_fifo for h in self.runtime.fifos]
        pending += [h.object_fifo for w in self.workers for h in w.fifos]
        while pending:
            of = pending.pop(0)
            if any(of is f for f in self.fifos):
                continue
            self.fifos.append(of)
            handles = [of.producer, *of.consumers]
            for h in handles:
                if isinstance(h.endpoint, ObjectFifoLink):
                    link = h.endpoint
                    if not any(link is l for l in self.links):
                        self.links.append(link)
                    pending += [x.object_fifo for x in link.srcs + link.dsts]

    def endpoint(self, ep) -> str:
        if isinstance(ep, RuntimeEndpoint):
            return ".shim"
        if isinstance(ep, Worker):
            return f"(.worker {next(i for i, w in enumerate(self.workers) if w is ep)})"
        if isinstance(ep, ObjectFifoLink):
            return f"(.link {next(i for i, l in enumerate(self.links) if l is ep)})"
        raise NotImplementedError(f"endpoint {ep}")

    def option_walk(self, tap: TensorAccessPattern | None) -> str:
        return "none" if tap is None else f"(some {self.terms.walk(tap)})"

    def fifo(self, of: ObjectFifo) -> str:
        cons = ", ".join(
            f"({self.endpoint(c.endpoint)}, {self.option_walk(c.from_stream)})"
            for c in of.consumers
        )
        return (
            f'{{ name := "{of.name}", objSize := {math.prod(of.shape)}, '
            f"consSize := {math.prod(np_ndarray_type_get_shape(of.consumer_obj_type))}, "
            f"toStream := {self.option_walk(of.to_stream)}, "
            f"repeatCount := {of.repeat_count or 1}, "
            f"prod := {self.endpoint(of.producer.endpoint)}, cons := [{cons}] }}"
        )

    def layout(self, layout: TensorLayout | None) -> str:
        if layout is None:
            return "none"
        if len(layout.shape) not in (1, 2) or layout.axes not in (None, (0, 1), (1, 0)):
            raise NotImplementedError(f"layout {layout.shape} axes {layout.axes}")
        rows, cols = (1, *layout.shape) if len(layout.shape) == 1 else layout.shape
        return (
            f"(some {{ rows := {rows}, cols := {cols}, "
            f"transposed := {str(layout.axes == (1, 0)).lower()}, "
            f"stream := {self.option_walk(layout.stream)}, "
            f"inverse := {str(layout.inverse).lower()} }})"
        )

    def dtype(self, arg_type) -> str:
        if get_origin(arg_type) is np.ndarray:
            arg_type = np_ndarray_type_get_dtype(arg_type)
        return DTYPE_NAMES[np.dtype(arg_type).name]

    def kernel(self, fn: Kernel) -> str:
        c = fn.contract
        args = []
        for i, ty in enumerate(fn.arg_types()):
            role = c.roles[i].__name__ if c is not None else "In"
            layout = c.layouts[i] if c is not None and c.layouts else None
            args.append(
                f'{{ dtype := "{self.dtype(ty)}", role := "{role}", '
                f"layout := {self.layout(layout)} }}"
            )
        acc = (
            "none"
            if c is None or c.acc_dtype is None
            else f'(some "{self.dtype(c.acc_dtype)}")'
        )
        red = "none" if c is None or c.reduction is None else f"(some {c.reduction})"
        setup = "none"
        if c is not None and c.setup is not None:
            # A setup may be a factory bound to its argument, e.g. conv_even.
            factory = (
                c.setup.func if isinstance(c.setup, functools.partial) else c.setup
            )
            setup = f'(some "{factory.__name__}")'
        sat = (
            "none"
            if c is None or c.saturation is None
            else f'(some "{c.saturation.value}")'
        )
        leaves = (
            "none"
            if c is None or c.leaves_saturation is None
            else f'(some "{c.leaves_saturation.value}")'
        )
        return (
            f'{{ name := "{fn.name}", args := [{", ".join(args)}], '
            f"accDtype := {acc}, reduction := {red}, setup := {setup}, "
            f"declared := {str(c is not None).lower()}, saturation := {sat}, "
            f"leavesSaturation := {leaves} }}"
        )

    def kernels(self) -> str:
        entries = []
        for w, worker in enumerate(self.workers):
            fns = [a for a in worker.flat_fn_args if isinstance(a, Kernel)]
            entries.append(f"({w}, [{', '.join(self.kernel(f) for f in fns)}])")
        return ",\n    ".join(entries)

    def link(self, link: ObjectFifoLink) -> str:
        srcs = ", ".join(f'"{h.object_fifo.name}"' for h in link.srcs)
        dsts = ", ".join(f'"{h.object_fifo.name}"' for h in link.dsts)
        return (
            f"{{ srcs := [{srcs}], dsts := [{dsts}], "
            f"srcOffsets := {list(link.src_offsets)}, dstOffsets := {list(link.dst_offsets)} }}"
        )

    def block(self, block: ir.Block, indent: str) -> str:
        items = []
        for op in block.operations:
            items += self.op(op.operation, indent)
        out = ".skip"
        for item in reversed(items):
            out = item if out == ".skip" else f"(.seq {item}\n{indent}{out})"
        return out

    def op(self, op: ir.Operation, indent: str) -> list[str]:
        inner = indent + "  "
        if op.name == "aiex.dma_configure_task_for":
            task = next(t for t in self.runtime.transfers if t.task.operation == op)
            buf = ir.BlockArgument(ir.Value(task.rt_data.op)).arg_number
            return [f'(.xfer "{task.alloc}" {buf} {self.terms.walk(task.tap)})']
        if op.name == "scf.if":
            cond = self.terms.term(op.operands[0])
            then = self.block(op.regions[0].blocks[0], inner)
            other = (
                self.block(op.regions[1].blocks[0], inner)
                if len(op.regions[1].blocks)
                else ".skip"
            )
            return [f"(.ite {cond}\n{inner}{then}\n{inner}{other})"]
        if op.name == "scf.for":
            lo, hi, step = (self.terms.term(o) for o in op.operands[:3])
            if step != "(1 : Int)":
                raise NotImplementedError(f"scf.for step {step}")
            body = op.regions[0].blocks[0]
            iv = self.terms.bind_iv(body)
            return [f"(.loop {lo} {hi} fun {iv} =>\n{inner}{self.block(body, inner)})"]
        if op.name == "cf.assert":
            return [f"(.check {self.terms.term(op.operands[0])})"]
        if len(op.regions):
            raise NotImplementedError(f"region op {op.name} in the runtime sequence")
        return []

    def lean(self, namespace: str, prefix: str = "IronVerify.Designs") -> str:
        params = list(self.runtime.dispatch_values)
        binders = " ".join(f"({p} : Int)" for p in params)
        param_type = " × ".join(["Int"] * len(params)) if params else "Unit"
        pattern = "(" + ", ".join(params) + ")" if params else "()"
        seq = self.block(self.runtime.sequence_op.body.blocks[0], "  ")
        fifos = ",\n    ".join(self.fifo(of) for of in self.fifos)
        links = ",\n    ".join(self.link(l) for l in self.links)
        list_pattern = "[" + ", ".join(params) + "]"
        kernel_import = "import IronVerify.Kernel\n" if self.contracts else ""
        kernels = (
            f"\n/-- Each worker's kernels and their contracts. -/\n"
            f"def kernels : List (Nat × List KernelD) := [\n    {self.kernels()}]\n"
            if self.contracts
            else ""
        )
        return f"""{SPDX}
import IronVerify.Net
{kernel_import}
namespace {prefix}.{namespace}

open IronVerify

-- Wide designs nest their runtime sequence deeper than Lean's default limit.
set_option maxRecDepth 20000

def params : List String := [{", ".join(f'"{p}"' for p in params)}]

def net : Net where
  fifos := [
    {fifos}]
  links := [
    {links}]

def seq {binders} : Seq :=
  {seq}

def design : Design ({param_type}) where
  net := net
  seq := fun {pattern} => seq {" ".join(params)}

def seqOfList : List Int → Option Seq
  | {list_pattern} => some (seq {" ".join(params)})
  | _ => none
{kernels}
end {prefix}.{namespace}
"""


class WrapChecks:
    """The `--wraps` obligations: every integer `arith` result in the runtime sequence fits its width.

    Mirrors the sequence's region tree (transfers and asserts dropped) with one
    `.check (fitsS w e)` per result of width `w` (`index` is 64), where `e` is the exact
    term the exporter writes for that result, plus `0 ≤ x` for each operand of an unsigned
    op. When they all hold, `IExpr.machine_eq_exact` makes the exported terms the machine's.
    """

    UNSIGNED = {"arith.divui", "arith.remui", "arith.extui", "arith.index_castui"}

    def __init__(self, exporter: "DesignExporter"):
        self.terms = exporter.terms

    def width(self, v: ir.Value) -> int | None:
        if isinstance(v.type, ir.IndexType):
            return 64
        if isinstance(v.type, ir.IntegerType) and v.type.width > 1:
            return v.type.width
        return None

    def block(self, block: ir.Block, indent: str) -> str:
        items = []
        for op in block.operations:
            items += self.op(op.operation, indent)
        out = ".skip"
        for item in reversed(items):
            out = item if out == ".skip" else f"(.seq {item}\n{indent}{out})"
        return out

    def op(self, op: ir.Operation, indent: str) -> list[str]:
        inner = indent + "  "
        if op.name == "scf.if":
            cond = self.terms.term(op.operands[0])
            then = self.block(op.regions[0].blocks[0], inner)
            other = (
                self.block(op.regions[1].blocks[0], inner)
                if len(op.regions[1].blocks)
                else ".skip"
            )
            return [f"(.ite {cond}\n{inner}{then}\n{inner}{other})"]
        if op.name == "scf.for":
            lo, hi = (self.terms.term(o) for o in op.operands[:2])
            body = op.regions[0].blocks[0]
            iv = self.terms.term(body.arguments[0])
            return [f"(.loop {lo} {hi} fun {iv} =>\n{inner}{self.block(body, inner)})"]
        if not op.name.startswith("arith.") or op.name == "arith.constant":
            if len(op.regions) and op.name != "aiex.dma_configure_task_for":
                raise NotImplementedError(
                    f"region op {op.name} in the runtime sequence"
                )
            return []
        checks = []
        if op.name in self.UNSIGNED:
            checks += [
                f"(.check (decide (0 ≤ {self.terms.term(o)})))" for o in op.operands
            ]
        for result in op.results:
            w = self.width(result)
            if w is not None:
                checks.append(f"(.check (fitsS {w} {self.terms.term(result)}))")
        return checks

    def lean(self, exporter: "DesignExporter") -> str:
        params = list(exporter.runtime.dispatch_values)
        binders = " ".join(f"({p} : Int)" for p in params)
        body = self.block(exporter.runtime.sequence_op.body.blocks[0], "  ")
        return (
            "/-- Every integer `arith` result of the runtime sequence fits its width. -/\n"
            f"def wraps {binders} : Seq :=\n  {body}\n"
        )


def natural_key(name: str) -> tuple:
    """Sort key ordering embedded numbers numerically: ``A_L2L1_2`` before ``A_L2L1_10``."""
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", name))


def load_whole_array(source: Path):
    """The `whole_array` design defined in `source` (the example, or a mutated copy of it)."""
    spec = importlib.util.spec_from_file_location(f"_design_{source.stem}", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.whole_array


def whole_array_program(
    source: Path,
    shape: str | None,
    m: int = 64,
    k: int = 64,
    n: int = 32,
    n_aie_cols: int = 1,
    device: str = "npu2",
    dtype_in: str = "i16",
    dtype_out: str = "i16",
    b_col_maj: int = 0,
    c_col_maj: int = 0,
    emulate_bf16: int = 0,
) -> Program:
    """Resolve whole_array for `device`; `shape` (`MxKxN`) specializes M, K and N, None leaves them dispatch-time."""
    iron.set_current_device(from_name(device, n_cols=None))
    big_in = np.ndarray[(1 << 22,), np.dtype[str_to_dtype(dtype_in)]]
    big_out = np.ndarray[(1 << 22,), np.dtype[str_to_dtype(dtype_out)]]
    kwargs = dict(
        A=big_in,
        B=big_in,
        C=big_out,
        m=m,
        k=k,
        n=n,
        n_aie_cols=n_aie_cols,
        b_col_maj=b_col_maj,
        c_col_maj=c_col_maj,
        emulate_bf16_mmul_with_bfp16=bool(emulate_bf16),
    )
    if shape:
        kwargs.update(zip("MKN", (int(x) for x in shape.split("x"))))
    return load_whole_array(source).specialize(**kwargs).compilable.program


def write_design(
    program: Program,
    name: str,
    lean_root: Path,
    canonical: bool = False,
    contracts: bool = False,
    wraps: bool = False,
) -> Path:
    """Write `program` as `IronVerify.Designs.<name>` and regenerate the design registry."""
    out = lean_root / "IronVerify/Designs" / f"{name}.lean"
    out.parent.mkdir(exist_ok=True)
    exporter = DesignExporter(program, canonical, contracts)
    text = exporter.lean(name)
    if wraps:
        end = f"end IronVerify.Designs.{name}\n"
        text = text.replace(
            "import IronVerify.Net\n",
            "import IronVerify.Net\nimport IronVerify.Schema.Overflow\n",
            1,
        ).replace(end, WrapChecks(exporter).lean(exporter) + "\n" + end)
    out.write_text(text)
    names = sorted(f.stem for f in out.parent.glob("*.lean"))
    imports = "".join(f"import IronVerify.Designs.{n}\n" for n in names)
    entries = ",\n    ".join(
        f'("{n}", Designs.{n}.net, Designs.{n}.seqOfList)' for n in names
    )
    (lean_root / "IronVerify/Designs.lean").write_text(
        f"{SPDX}\n{imports}\nnamespace IronVerify.Designs\n\n"
        f"def registry : List (String × Net × (List Int → Option Seq)) := [\n    {entries}]\n\n"
        f"end IronVerify.Designs\n"
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source", type=Path, default=WHOLE_ARRAY, help="whole_array.py to export"
    )
    p.add_argument(
        "--lean-root", type=Path, default=LEAN_ROOT, help="Lake project to write into"
    )
    p.add_argument("--device", choices=["npu1", "npu2"], default="npu2")
    p.add_argument("--n-aie-cols", type=int, choices=range(1, 9), default=1)
    p.add_argument("--m", type=int, default=64)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--dtype-in", choices=["i8", "i16", "bf16"], default="i16")
    p.add_argument(
        "--dtype-out", choices=["i8", "i16", "i32", "bf16", "f32"], default="i16"
    )
    p.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    p.add_argument("--c-col-maj", type=int, choices=[0, 1], default=0)
    p.add_argument("--emulate-bf16", type=int, choices=[0, 1], default=0)
    p.add_argument(
        "--order",
        choices=["discovery", "canonical"],
        default="discovery",
        help="fifo/link order: as the exporter meets them, or by name (schema instances)",
    )
    p.add_argument(
        "--contracts",
        action="store_true",
        help="also export each worker's kernel contracts (layouts, dtypes, roles)",
    )
    p.add_argument(
        "--wraps",
        action="store_true",
        help="also export the width obligations (`wraps`): every arith result fits its type",
    )
    p.add_argument("--shape", type=str, default=None, help="MxKxN to specialize")
    p.add_argument("--name", type=str, required=True, help="Lean namespace")
    opts = p.parse_args()

    program = whole_array_program(
        opts.source,
        opts.shape,
        opts.m,
        opts.k,
        opts.n,
        opts.n_aie_cols,
        opts.device,
        opts.dtype_in,
        opts.dtype_out,
        opts.b_col_maj,
        opts.c_col_maj,
        opts.emulate_bf16,
    )
    out = write_design(
        program,
        opts.name,
        opts.lean_root,
        opts.order == "canonical",
        opts.contracts,
        opts.wraps,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
