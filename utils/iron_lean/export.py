# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Export an IRON design's data movement from its live objects to a Lean `Design`."""

import argparse
import importlib.util
import math
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie import ir
from aie.helpers.npdtypes import np_ndarray_type_get_shape
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Program, Worker
from aie.iron.dataflow.objectfifo import ObjectFifo, ObjectFifoLink
from aie.iron.device import from_name
from aie.iron.runtime.endpoint import RuntimeEndpoint

REPO = Path(__file__).resolve().parents[2]
WHOLE_ARRAY = (
    REPO / "programming_examples/basic/matrix_multiplication/whole_array/whole_array.py"
)
LEAN_ROOT = Path(__file__).parent / "IronVerify"
SPDX = (
    "-- Copyright (C) 2026 Advanced Micro Devices, Inc.\n"
    "-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n"
)

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

    def __init__(self, program: Program):
        self.program = program
        self.runtime = program.entry
        self.workers = [w for c in program.configurations for w in c.workers]
        self.terms = LeanTerms(self.runtime.dispatch_values)
        self.fifos: list[ObjectFifo] = []
        self.links: list[ObjectFifoLink] = []
        self.collect()

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

    def lean(self, namespace: str) -> str:
        params = list(self.runtime.dispatch_values)
        binders = " ".join(f"({p} : Int)" for p in params)
        param_type = " × ".join(["Int"] * len(params)) if params else "Unit"
        pattern = "(" + ", ".join(params) + ")" if params else "()"
        seq = self.block(self.runtime.sequence_op.body.blocks[0], "  ")
        fifos = ",\n    ".join(self.fifo(of) for of in self.fifos)
        links = ",\n    ".join(self.link(l) for l in self.links)
        list_pattern = "[" + ", ".join(params) + "]"
        return f"""{SPDX}
import IronVerify.Net

namespace IronVerify.Designs.{namespace}

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

end IronVerify.Designs.{namespace}
"""


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
) -> Program:
    """Resolve whole_array for `device`; `shape` (`MxKxN`) specializes M, K and N, None leaves them dispatch-time."""
    iron.set_current_device(from_name(device, n_cols=None))
    big = np.ndarray[(1 << 22,), np.dtype[np.int16]]
    kwargs = dict(A=big, B=big, C=big, m=m, k=k, n=n, n_aie_cols=n_aie_cols)
    if shape:
        kwargs.update(zip("MKN", (int(x) for x in shape.split("x"))))
    return load_whole_array(source).specialize(**kwargs).compilable.program


def write_design(program: Program, name: str, lean_root: Path) -> Path:
    """Write `program` as `IronVerify.Designs.<name>` and regenerate the design registry."""
    out = lean_root / "IronVerify/Designs" / f"{name}.lean"
    out.parent.mkdir(exist_ok=True)
    out.write_text(DesignExporter(program).lean(name))
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
    p.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=1)
    p.add_argument("--m", type=int, default=64)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--shape", type=str, default=None, help="MxKxN to specialize")
    p.add_argument("--name", type=str, required=True, help="Lean namespace")
    opts = p.parse_args()

    program = whole_array_program(
        opts.source, opts.shape, opts.m, opts.k, opts.n, opts.n_aie_cols, opts.device
    )
    print(f"wrote {write_design(program, opts.name, opts.lean_root)}")


if __name__ == "__main__":
    main()
