#!/usr/bin/env python3
# contribution.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Remind a pull request that adds or changes a kernel what a kernel needs.

Reads the factories in ``python/iron/kernels/``, the case table in
``test/python/npu/kernel_cases.py`` and the sources in ``aie_kernels/`` at
two revisions, and writes a Markdown checklist for each factory the pull
request adds or changes. It only reads the files (``git show`` and ``ast``),
never imports or runs them, so it is safe on a pull request from a fork.

It is advice, not a check: it always exits 0. The host contract test
(``test/python/test_kernel_contracts.py``) is what enforces a contract.

    contribution.py --base origin/main --head HEAD [--out comment.md]

Writes no file when the pull request touches no kernel.
"""

import argparse
import ast
import itertools
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

MARKER = "<!-- kernel-contribution-check -->"
KERNELS = "python/iron/kernels/"
CASES = "test/python/npu/kernel_cases.py"
SOURCES = "aie_kernels/"
INIT = KERNELS + "__init__.py"
API_DOCS = "docs/api/kernels.md"
CONTRACT_TEST = "test/python/test_kernel_contracts.py"
GUIDE = "https://xilinx.github.io/mlir-aie/dev/programming_guide/kernels_library/"
ADDING = GUIDE + "#adding-a-kernel"
TESTING = GUIDE + "#testing-performance-and-static-checks"
# The calls that build a kernel and so take a contract.
BUILDERS = {"_make_extern", "ExternalFunction", "MatrixKernel"}
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
UNKNOWN = object()
# A header more factories build than this is listed, not each factory.
SHARED = 12


class Tree:
    """Files at one revision, read with git; nothing is checked out or run."""

    def __init__(self, repo: Path, rev: str):
        self.repo, self.rev = repo, rev
        self.unreadable: list[str] = []  # files that do not parse

    def _git(self, *args: str) -> str | None:
        done = subprocess.run(
            ["git", "-C", str(self.repo), *args], capture_output=True, text=True
        )
        return done.stdout if done.returncode == 0 else None

    def read(self, path: str) -> str | None:
        return self._git("show", f"{self.rev}:{path}")

    def ls(self, prefix: str) -> list[str]:
        out = self._git("ls-tree", "-r", "--name-only", self.rev, "--", prefix)
        return out.split() if out else []


def changed_files(repo: Path, base: str, head: str) -> list[str]:
    # A rename is listed as its old and new path, so a factory still naming
    # the old one is found.
    diff = ["diff", "--no-renames", "--name-only", f"{base}...{head}"]
    done = subprocess.run(
        ["git", "-C", str(repo), *diff],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.split()


def _parse(text: str | None) -> ast.Module | None:
    try:
        return ast.parse(text) if text else None
    except SyntaxError:
        return None


def _callee(call: ast.Call) -> str:
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")


def _assigns(mod: ast.Module | None, name: str) -> ast.expr | None:
    for node in mod.body if mod else []:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, "id", "") == name for t in targets):
                return node.value
    return None


def _source_pattern(arg: ast.expr, env: dict) -> tuple[re.Pattern, bool] | None:
    """The aie_kernels/ paths a ``_kernel_source(...)`` argument can name,
    and whether it names exactly one."""
    value = _value(arg, env)
    if isinstance(value, str):
        return re.compile(re.escape(value) + "$"), True
    if isinstance(arg, ast.JoinedStr):
        parts = []
        for v in arg.values:
            part = _part(v, env)
            parts.append(re.escape(part) if isinstance(part, str) else "[^/]*")
        return re.compile("".join(parts) + "$"), False
    return None


@dataclass
class Factory:
    name: str
    module: str
    text: str
    contract: str = "unknown"  # yes, no, unknown, or "via <factory>"
    trace: str | None = None  # whole_call, partial, none, computed; None: unset
    tolerance: str | None = None  # noted, unnoted, computed; None: dtype default
    docstring: bool = False
    takes_dtype: bool = False
    dtypes_table: bool = False
    sources: list = field(default_factory=list)
    node: ast.FunctionDef | None = None

    def builds(self, path: str, exact: bool | None = None) -> bool:
        rel = path[len(SOURCES) :]
        return any(
            p.match(rel) for p, e in self.sources if exact is None or e == exact
        )


def _kernel_classes(modules: dict[str, ast.Module]) -> set[str]:
    """ExternalFunction and the classes in the package that extend it."""
    found = {"ExternalFunction"}
    classes = [
        n for m in modules.values() for n in m.body if isinstance(n, ast.ClassDef)
    ]
    grew = True
    while grew:
        grew = False
        for c in classes:
            bases = {getattr(b, "id", getattr(b, "attr", "")) for b in c.bases}
            if c.name not in found and bases & found:
                found.add(c.name)
                grew = True
    return found


def factories(tree: Tree) -> dict[str, Factory]:
    """Every public top-level function in the package that returns a kernel."""
    texts = {
        Path(p).stem: tree.read(p)
        for p in tree.ls(KERNELS)
        if p.endswith(".py") and Path(p).name != "__init__.py"
    }
    modules = {m: mod for m, text in texts.items() if (mod := _parse(text))}
    tree.unreadable = [KERNELS + m + ".py" for m in texts if m not in modules]
    kinds = _kernel_classes(modules)
    every: dict[str, Factory] = {}
    public = []
    for module, mod in modules.items():
        for fn in mod.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            text = ast.get_source_segment(texts[module], fn) or ""
            every.setdefault(fn.name, _describe(fn, module, text))
            returns = ast.unparse(fn.returns) if fn.returns else ""
            if not fn.name.startswith("_") and set(re.findall(r"\w+", returns)) & kinds:
                public.append(fn.name)
    for name in public:
        _inherit(every[name], every, set())
        every[name].sources = _sources(every[name], every, {})
    return {name: every[name] for name in public}


def _sources(f: Factory, every: dict[str, Factory], env: dict, depth=0) -> list:
    """The source patterns a factory builds, through the functions it calls,
    with the literal arguments it passes them (``_norm_extern(..., "rms_norm.cc")``
    builds ``norm/rms_norm.cc``)."""
    found = []
    for call in ast.walk(f.node):
        if not isinstance(call, ast.Call):
            continue
        name = _callee(call)
        if name == "_kernel_source" and call.args:
            if pattern := _source_pattern(call.args[0], env):
                found.append(pattern)
        elif name in every and name != f.name and depth < 4:
            callee = every[name].node
            params = [a.arg for a in callee.args.args + callee.args.kwonlyargs]
            passed = dict(zip(params, call.args))
            passed |= {k.arg: k.value for k in call.keywords if k.arg}
            bound = {}
            for param, arg in passed.items():
                value = _value(arg, env)
                bound[param] = UNKNOWN if value is UNKNOWN else ast.Constant(value)
            found += _sources(every[name], every, bound, depth + 1)
    return found


def _inherit(f: Factory, every: dict[str, Factory], seen: set) -> None:
    """Fill in what a factory leaves to a function it calls: ``add`` returns
    ``_eltwise_bf16_kernel(...)``, ``gelu`` passes ``_unary_lut_contract(...)``."""
    if f.trace != "unknown" or f.name in seen:
        return
    seen.add(f.name)
    calls = [n for n in ast.walk(f.node) if isinstance(n, ast.Call)]
    for other in [every[n] for c in calls if (n := _callee(c)) in every]:
        if other is f:
            continue
        _inherit(other, every, seen)
        if other.trace == "unknown":
            continue
        if f.contract == "unknown":
            public = not other.name.startswith("_") and other.contract == "yes"
            f.contract = f"via {other.name}" if public else other.contract
        f.trace, f.tolerance = other.trace, other.tolerance
        return


def _describe(fn: ast.FunctionDef, module: str, text: str) -> Factory:
    f = Factory(fn.name, module, text, node=fn)
    f.docstring = bool(ast.get_docstring(fn))
    f.takes_dtype = any(a.arg == "dtype" for a in fn.args.args + fn.args.kwonlyargs)
    f.dtypes_table = any(
        isinstance(d, ast.Call) and _callee(d) == "dtypes" for d in fn.decorator_list
    )
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    contracts = [c for c in calls if _callee(c) == "KernelContract"]
    passes = any(k.arg == "contract" for c in calls for k in c.keywords)
    if contracts or passes:
        f.contract = "yes"
    elif any(_callee(c) in BUILDERS for c in calls):
        f.contract = "no"
    if not contracts:
        f.trace = f.tolerance = "unknown"
        return f
    kw = {k.arg: k.value for c in contracts for k in c.keywords}
    trace = kw.get("trace")
    if isinstance(trace, ast.Call) and isinstance(trace.func, ast.Attribute):
        f.trace = trace.func.attr
    elif trace is not None:
        f.trace = "computed"
    tol = kw.get("tolerance")
    if isinstance(tol, ast.Constant) and tol.value is None:
        tol = None
    if isinstance(tol, ast.Call) and "Tolerance" in ast.unparse(tol.func):
        noted = any(k.arg == "note" for k in tol.keywords)
        f.tolerance = "noted" if noted else "unnoted"
    elif tol is not None:
        f.tolerance = "computed"
    return f


@dataclass
class Cases:
    total: int = 0
    smoke: int = 0
    timed: int = 0
    edge: int = 0
    devices: set = field(default_factory=set)


def _bindings(comp: ast.expr) -> list[dict]:
    """Each assignment of a literal comprehension's loop names, as far as the
    text says; a name it cannot read is UNKNOWN."""
    per_loop = []
    for gen in getattr(comp, "generators", []):
        names = gen.target.elts if isinstance(gen.target, ast.Tuple) else [gen.target]
        it = gen.iter
        if isinstance(it, ast.Call) and _callee(it) == "range" and it.args:
            stop = it.args[-1]
            n = stop.value if isinstance(stop, ast.Constant) else 1
            items = [ast.Constant(i) for i in range(n)]
        else:
            items = getattr(it, "elts", None) or [None]
        loop = []
        for item in items:
            if len(names) == 1:
                values = [item]
            else:
                values = getattr(item, "elts", None) or [None] * len(names)
            loop.append(
                {
                    getattr(n, "id", ""): (v if v is not None else UNKNOWN)
                    for n, v in zip(names, values)
                }
            )
        per_loop.append(loop)
    return [
        {k: v for d in combo for k, v in d.items()}
        for combo in itertools.product(*per_loop)
    ] or [{}]


def _value(node, env: dict, default=UNKNOWN):
    if node is None:
        return default
    if isinstance(node, ast.Name) and node.id in env:
        node = env[node.id]
        if node is UNKNOWN:
            return UNKNOWN
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = [_part(v, env) for v in node.values]
        return UNKNOWN if UNKNOWN in parts else "".join(map(str, parts))
    if isinstance(node, ast.Tuple):
        return tuple(_value(e, env) for e in node.elts)
    return UNKNOWN


def _part(node: ast.expr, env: dict):
    """One piece of an f-string: its text, or the value it formats."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.FormattedValue) and not node.format_spec:
        return _value(node.value, env)
    return UNKNOWN


def cases(tree: Tree) -> dict[str, Cases]:
    """The case table, per factory: how many, which run on PRs and are timed."""
    table: dict[str, Cases] = {}
    mod = _parse(tree.read(CASES)) or ast.Module(body=[], type_ignores=[])
    comps = (ast.ListComp, ast.GeneratorExp)
    inside = {id(n.elt): n for n in ast.walk(mod) if isinstance(n, comps)}
    for call in ast.walk(mod):
        if not isinstance(call, ast.Call) or _callee(call) not in {"Case", "check"}:
            continue
        if not call.args:
            continue
        comp = inside.get(id(call))
        for env in _bindings(comp) if comp else [{}]:
            _count(call, env, table)
    return table


def _count(call: ast.Call, env: dict, table: dict[str, Cases]) -> None:
    name = _value(call.args[0], env)
    if not isinstance(name, str):
        return
    kw = {k.arg: k.value for k in call.keywords}
    entry = table.setdefault(name, Cases())
    entry.total += 1
    entry.smoke += _value(kw.get("smoke"), env, False) is True
    entry.timed += _value(kw.get("perf"), env, _callee(call) == "Case") is True
    entry.edge += str(_value(kw.get("tag"), env, "")).startswith("edge")
    devices = _value(kw.get("devices"), env, ())
    if isinstance(devices, tuple) and devices and UNKNOWN not in devices:
        entry.devices |= set(devices)
    else:
        entry.devices |= {"npu1", "npu2"}


def not_judged(tree: Tree) -> set[str]:
    """Factories the contract test names as deliberately without a case."""
    table = _assigns(_parse(tree.read(CONTRACT_TEST)), "NOT_JUDGED")
    names = set()
    for key, value in zip(*(getattr(table, a, []) for a in ("keys", "values"))):
        if key is not None:  # "name": "reason"
            names.add(_value(key, {}))
        elif isinstance(value, ast.DictComp):  # **{name: ... for name in (...)}
            names |= {_value(value.key, env) for env in _bindings(value)}
    return {n for n in names if isinstance(n, str)}


def exported(tree: Tree) -> set[str]:
    """The names ``aie.iron.kernels`` exports."""
    names = getattr(_assigns(_parse(tree.read(INIT)), "__all__"), "elts", [])
    return {e.value for e in names if isinstance(e, ast.Constant)}


def _includers(tree: Tree, header: str, files: list[str]) -> list[str]:
    """The .cc files that include ``header``, directly or through headers."""
    texts = {p: tree.read(p) or "" for p in files}
    found, todo = set(), [header]
    while todo:
        name = re.escape(todo.pop().rsplit("/", 1)[-1])
        include = re.compile(rf'#\s*include\s*"([^"]*/)?{name}"')
        for p, text in texts.items():
            if p not in found and include.search(text):
                found.add(p)
                todo += [p] if p.endswith(".h") else []
    return sorted(p for p in found if p.endswith(".cc"))


@dataclass
class Context:
    exports: set
    judged: set
    api_docs: str
    files: set = field(default_factory=set)
    docs_noted: set = field(default_factory=set)  # modules already told


def missing(f: Factory, files: set) -> list[str]:
    """The sources ``f`` names exactly that the tree does not have."""
    gone = []
    for pattern, exact in f.sources:
        if exact and not any(pattern.fullmatch(p[len(SOURCES) :]) for p in files):
            gone.append(SOURCES + pattern.pattern.removesuffix("$").replace("\\", ""))
    return gone


def advice(f: Factory, c: Cases | None, new: bool, ctx: Context) -> list[str]:
    """What to look at for one factory, the things that fail a test first."""
    warn, note = [], []
    if f.contract == "no" and f.name not in ctx.judged:
        warn.append(
            "no `contract=KernelContract(...)`; the host contract test fails "
            "without one"
        )
    if not c and f.name not in ctx.judged:
        warn.append(
            f"no case in `{CASES}`; the host contract test fails until there "
            "is one, or until it is in `NOT_JUDGED` with a reason"
        )
    if f.contract == "yes" and f.trace is None:
        warn.append("say what its trace markers measure with `trace=Trace...`")
    elif c and c.timed and f.trace in ("partial", "none"):
        warn.append(
            f"it has timed cases but `trace=Trace.{f.trace}(...)`; a timed "
            "kernel needs `Trace.whole_call()`, one marker pair around each call"
        )
    for path in missing(f, ctx.files):
        warn.append(f"it builds `{path}`, which is not in the tree")
    if f.tolerance == "unnoted":
        warn.append("give the tolerance's evidence in `note=`")
    if new:
        if f.name not in ctx.exports:
            warn.append(
                "export it from `aie.iron.kernels` (import it and add it to "
                "`__all__` in `kernels/__init__.py`)"
            )
        if not f.docstring:
            warn.append("add a docstring; the API docs are built from it")
        if f"iron.kernels.{f.module}" not in ctx.api_docs + " ".join(ctx.docs_noted):
            ctx.docs_noted.add(f"iron.kernels.{f.module}")
            warn.append(
                f"add `::: iron.kernels.{f.module}` to `{API_DOCS}` so the API "
                "docs show the new module"
            )
    if c:
        if not c.smoke:
            note.append(
                "no `smoke=True` case, so pull requests never run it on an NPU; "
                "only the nightly does"
            )
        if not c.timed:
            note.append(
                "no timed case (`Case(...)`; `check(...)` is not timed), so the "
                "nightly records no cycles for it"
            )
        if new and not c.edge:
            note.append(
                'consider an edge case (`tag="edge-..."`): one vector, the '
                "smallest tile, a tail"
            )
    if new and f.contract == "yes" and f.tolerance is None and c:
        note.append(
            "it uses the dtype's default tolerance; if it needs another, say why "
            "in `Tolerance(..., note=...)`"
        )
    if new and f.takes_dtype and not f.dtypes_table:
        note.append("if it builds more than one dtype, list them with `@dtypes(...)`")
    return [f"⚠️ {w}" for w in warn] + [f"ℹ️ {n}" for n in note]


def _row(f: Factory, c: Cases | None, ctx: Context) -> list[str]:
    exempt = f.name in ctx.judged
    contract = {"yes": "✅", "no": "—" if exempt else "⚠️ none", "unknown": "?"}
    contract = contract.get(f.contract, f.contract)
    trace = {"whole_call": "whole call", None: "unset", "unknown": "?"}.get(
        f.trace, f.trace
    )
    if f.contract == "no":
        trace = "—"
    if not c:
        return [contract, trace, "not judged" if exempt else "⚠️ none", "—", "—"]
    only = sorted(c.devices) if c.devices < {"npu1", "npu2"} else []
    devices = f" ({', '.join(only)} only)" if only else ""
    return [
        contract,
        trace,
        f"{c.total}{devices}",
        f"✅ {c.smoke}" if c.smoke else "—",
        f"✅ {c.timed}" if c.timed else "—",
    ]


def _attributed(after: dict[str, Factory], path: str) -> list[str]:
    """The factories that build ``path``: by its exact name if any factory
    names it, else by an f-string pattern."""
    exact = [n for n, f in after.items() if f.builds(path, exact=True)]
    return exact or [n for n, f in after.items() if f.builds(path, exact=False)]


def report(repo: Path, base: str, head: str) -> str | None:
    files = changed_files(repo, base, head)
    if not [p for p in files if p.startswith((KERNELS, SOURCES)) or p == CASES]:
        return None
    # Before is where the branch left main, as the file list is: a branch
    # behind main must not be shown main's own changes as its own.
    fork = subprocess.run(
        ["git", "-C", str(repo), "merge-base", base, head],
        capture_output=True,
        text=True,
    ).stdout.strip()
    old, new = Tree(repo, fork or base), Tree(repo, head)
    before, after = factories(old), factories(new)
    table = cases(new)
    present = set(new.ls(SOURCES))
    ctx = Context(exported(new), not_judged(new), new.read(API_DOCS) or "", present)

    # A source this change deleted or moved away is not a kernel to look for;
    # a factory that still names it is warned about below.
    sources = [p for p in files if p.startswith(SOURCES) and p in present]
    every_unit = [p for p in present if p.endswith((".cc", ".h"))]
    built: dict[str, list[str]] = {}
    unclaimed, shared = [], []
    for p in sources:
        units = [p, *(_includers(new, p, every_unit) if p.endswith(".h") else [])]
        names = {n for u in units for n in _attributed(after, u)}
        if len(names) > SHARED:
            shared.append(f"- `{p}`: {len(names)} factories build it")
            continue
        for n in names:
            built.setdefault(n, []).append(p)
        if not names and p.endswith((".cc", ".h")):
            unclaimed.append(p)
    gone = [p for p in files if p.startswith(SOURCES) and p not in present]
    for n, f in after.items():
        if any(f.builds(p, exact=True) for p in gone):
            built.setdefault(n, [])
    added = [n for n in after if n not in before]
    edited = [
        n
        for n, f in after.items()
        if n in before and (f.text != before[n].text or n in built)
    ]
    unread = {Path(p).stem for p in new.unreadable}
    removed = [n for n in before if n not in after and before[n].module not in unread]

    lines = [
        MARKER,
        "### Kernel contribution checklist",
        "",
        "This pull request changes kernel code, so here is what the kernel "
        "library asks of a kernel, read from the files. It is a reminder, not a "
        f"required check; [Adding a kernel]({ADDING}) has the details.",
        "",
    ]
    intro = len(lines)
    groups = (("New factories", added, True), ("Changed factories", edited, False))
    for title, names, is_new in groups:
        if not names:
            continue
        rows = [
            "| Factory | Contract | Trace | Cases | Run on PRs | Timed nightly |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        warns, notes = [], []
        for n in names:
            f, c = after[n], table.get(n)
            rows.append(f"| `{n}` | " + " | ".join(_row(f, c, ctx)) + " |")
            for a in advice(f, c, is_new, ctx):
                (warns if a.startswith("⚠️") else notes).append(f"- `{n}`: {a}")
        lines += [f"#### {title}", ""]
        if len(names) > SHARED:
            # A refactor touches most of the library: keep what needs doing
            # in view and fold the rest.
            lines += warns + ([""] if warns else [])
            lines += [
                "<details>",
                f"<summary>{len(names)} factories</summary>",
                "",
                *rows,
                "",
                *notes,
                "",
                "</details>",
                "",
            ]
        else:
            lines += rows + [""] + warns + notes + ([""] if warns or notes else [])
    if removed:
        names = ", ".join(f"`{n}`" for n in removed)
        lines += [
            "#### Removed factories",
            "",
            f"{names}: remove their cases from `{CASES}` and their names from "
            "`kernels/__init__.py`.",
            "",
        ]
    if shared:
        lines += [
            "#### Shared headers",
            "",
            *shared,
            "",
            "Run the whole suite, not just one kernel's cases, before merging.",
            "",
        ]
    if unclaimed:
        lines += [
            "#### Sources no factory names",
            "",
            *[
                f"- `{p}`" + (": no source includes it" if p.endswith(".h") else "")
                for p in unclaimed
            ],
            "",
            "A kernel is only tested and timed through a factory and its cases; "
            "a header nothing includes can go.",
            "",
        ]

    if new.unreadable:
        files = ", ".join(f"`{p}`" for p in new.unreadable)
        lines += [
            "#### Not read",
            "",
            f"{files} does not parse, so its factories are left out above.",
            "",
        ]

    names = [n for n in added + edited if NAME.match(n)]
    if names:
        # Past a dozen kernels, or with a shared header, a -k filter is
        # both unreadable and most of the suite: run all of it.
        whole = len(names) > SHARED or bool(shared)
        lines += _commands("" if whole else " or ".join(names), bool(sources))
    if len(lines) == intro:
        return None  # e.g. only unused headers deleted: nothing to ask
    lines.append(
        f"More in [Testing, performance and static checks]({TESTING}). "
        "This comment is updated on each push."
    )
    return "\n".join(lines) + "\n"


def _commands(k: str, sources: bool) -> list[str]:
    """The commands to run, for the kernels ``k`` names, or for all."""
    e2e = "test/python/npu/test_kernels_e2e.py"
    only = f' -k "{k}"' if k else ""
    lines = [
        "#### Before you merge",
        "",
        "```bash",
        "# On any machine: contract, reference and lowering",
        f"pytest {CONTRACT_TEST}{only}",
        "# On an NPU: the smoke cases a pull request runs, then the nightly sweep",
        f"pytest {e2e}{only}",
        f"pytest {e2e} -m extensive --seeds 3{only}",
    ]
    if sources:
        lines += [
            "# How the source change moves cycles, against main",
            "mkdir ../base && git archive origin/main aie_kernels aie_runtime_lib"
            " | tar -x -C ../base",
            f"pytest test/python/npu/test_kernels_perf.py -m perf{only} "
            "--baseline-sources ../base",
        ]
    return lines + ["```", ""]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--out", type=Path, help="default: print it")
    args = parser.parse_args(argv)
    try:
        text = report(args.repo, args.base, args.head)
    except Exception as e:  # advice never fails a pull request
        print(f"::warning::kernel contribution check skipped: {e}", file=sys.stderr)
        return 0
    if text and args.out:
        args.out.write_text(text)
    elif not args.out:
        print(text or "No kernel changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
