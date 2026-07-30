# test_inline_ir_rewrite.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Grammar coverage for ``_make_ir_inlinable`` (``ExternalFunction(inline=True)``).

The inline path post-processes Peano's ``-emit-llvm`` output to mark the kernel
``alwaysinline`` / ``linkonce_odr``.  The LLVM ``define`` grammar is::

    define [linkage] [preemption] [visibility] [dll] [cconv] [ret attrs]
           <ty> @<name>(<params>) [unnamed_addr] [addrspace(N)] [fn attrs]
           [section] [partition] [comdat] [align] [gc] [prefix] [prologue]
           [personality] (!name !N)* { ...

so the attribute has to land in the ``[fn attrs]`` slot.  Appending it next to
the opening brace instead is a parse error whenever a later clause is present
-- most reachably the ``!dbg`` attachment that ``-g`` in ``compile_flags``
adds.  A second linkage keyword is likewise a parse error, and pairing
``alwaysinline`` with the ``noinline``/``optnone`` clang emits at ``-O0`` is a
verifier error.  Each of those is pinned below.

The textual cases run everywhere; ``llvm-as`` is additionally used to check the
result really parses when a Peano install is present.
"""

import subprocess
from pathlib import Path

import pytest

from aie.utils.compile.utils import _make_ir_inlinable

TAIL = """
  ret void
}

attributes #0 = { mustprogress noinline nounwind optnone "no-trapping-math"="true" }
"""

# Attribute-group-free tail, for the cases that assert on plain insertion.
PLAIN_TAIL = """
  ret void
}
"""


def _rewrite(tmp_path, define_line, tail=TAIL, symbol="add_one", extra=""):
    """Run the rewriter over a one-function module; return its lines."""
    ir = tmp_path / "kernel.ll"
    ir.write_text(extra + define_line + tail)
    _make_ir_inlinable(str(ir), symbol)
    return ir, ir.read_text().splitlines()


def _define_of(lines, symbol="add_one"):
    return next(
        line for line in lines if line.startswith("define") and f"@{symbol}(" in line
    )


def _llvm_as():
    """Path to Peano's llvm-as, or None when no Peano install is available."""
    try:
        import aie.utils.config as config

        candidate = Path(config.peano_cxx_path()).with_name("llvm-as")
    except Exception:
        return None
    return str(candidate) if candidate.is_file() else None


requires_llvm_as = pytest.mark.skipif(
    _llvm_as() is None, reason="no Peano llvm-as available"
)


# ---------------------------------------------------------------------------
# Insertion position
# ---------------------------------------------------------------------------

# Each case is (define line, the trailing clause `alwaysinline` must be
# inserted *before*, any module-level declarations the define needs to be a
# self-contained module).  `extra` keeps the llvm-as check below honest -- it
# has to parse a complete module, not just the one line under test.
DEBUG_INFO_PREAMBLE = """\
!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !2, \
producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!2 = !DIFile(filename: "kernel.cc", directory: "/")
!9 = distinct !DISubprogram(name: "add_one", scope: !2, file: !2, line: 1, \
type: !10, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !1)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
"""

POSITION_CASES = [
    pytest.param(
        "define dso_local void @add_one(ptr noundef %0, i32 noundef %1)"
        " local_unnamed_addr #0 {",
        None,
        "",
        id="plain",
    ),
    pytest.param(
        "define dso_local void @add_one(ptr %0) local_unnamed_addr #0 !dbg !9 {",
        "!dbg",
        DEBUG_INFO_PREAMBLE,
        id="debug-info",  # reachable via compile_flags=["-g"]
    ),
    pytest.param(
        "define dso_local void @add_one(ptr %0) local_unnamed_addr #0"
        " personality ptr @__gxx_personality_v0 {",
        "personality",
        "declare i32 @__gxx_personality_v0(...)\n",
        id="personality",
    ),
    pytest.param(
        "define dso_local void @add_one(ptr %0) #0 comdat {",
        "comdat",
        "$add_one = comdat any\n",
        id="comdat",
    ),
    pytest.param(
        'define dso_local void @add_one(ptr %0) #0 section "foo" align 16 {',
        "section",
        "",
        id="section-align",
    ),
    pytest.param(
        "define dso_local void @add_one(ptr byval(%struct.S) align 4 %0,"
        " ptr sret({ i32, i32 }) %1) unnamed_addr #0 {",
        None,
        "%struct.S = type { i32 }\n",
        id="nested-parens",
    ),
    pytest.param(
        "define void @add_one(ptr %0) {",
        None,
        "",
        id="no-attribute-group",
    ),
]


@pytest.mark.parametrize("define_line,trailing,extra", POSITION_CASES)
def test_alwaysinline_lands_in_the_fn_attrs_slot(
    tmp_path, define_line, trailing, extra
):
    _, lines = _rewrite(tmp_path, define_line, extra=extra)
    define = _define_of(lines)

    assert "alwaysinline" in define
    # It must follow the parameter list...
    assert define.index("alwaysinline") > define.index(")")
    # ...and precede any clause that the grammar puts after [fn attrs].
    if trailing is not None:
        assert define.index("alwaysinline") < define.index(trailing), define


@requires_llvm_as
@pytest.mark.parametrize("define_line,trailing,extra", POSITION_CASES)
def test_rewritten_ir_parses(tmp_path, define_line, trailing, extra):
    ir, _ = _rewrite(tmp_path, define_line, extra=extra)
    result = subprocess.run(
        [_llvm_as(), str(ir), "-o", "/dev/null"], capture_output=True
    )
    assert result.returncode == 0, result.stderr.decode()


# ---------------------------------------------------------------------------
# Linkage
# ---------------------------------------------------------------------------


def test_linkonce_odr_is_added_when_no_linkage_is_present(tmp_path):
    _, lines = _rewrite(
        tmp_path, "define dso_local void @add_one(ptr %0) {", PLAIN_TAIL
    )
    assert _define_of(lines).startswith("define linkonce_odr dso_local void")


@pytest.mark.parametrize("linkage", ["internal", "weak_odr", "linkonce_odr", "private"])
def test_existing_discardable_linkage_is_left_alone(tmp_path, linkage):
    """A second linkage keyword is a parse error; these are already discardable."""
    _, lines = _rewrite(
        tmp_path, f"define {linkage} void @add_one(ptr %0) {{", PLAIN_TAIL
    )
    define = _define_of(lines)
    assert define.startswith(f"define {linkage} void")
    assert define.count("linkonce_odr") == (1 if linkage == "linkonce_odr" else 0)


def test_explicit_external_linkage_is_replaced(tmp_path):
    """`external` is a strong definition -- swap it for a discardable one."""
    _, lines = _rewrite(tmp_path, "define external void @add_one(ptr %0) {", PLAIN_TAIL)
    define = _define_of(lines)
    assert define.startswith("define linkonce_odr void")
    assert "external" not in define


# ---------------------------------------------------------------------------
# noinline / optnone conflict (clang emits both at -O0)
# ---------------------------------------------------------------------------


def test_optnone_group_is_copied_not_mutated(tmp_path):
    """`alwaysinline` + `optnone` fails the verifier, so the define is
    repointed at a private copy of the group -- the shared group other
    functions reference must survive untouched."""
    _, lines = _rewrite(tmp_path, "define dso_local void @add_one(ptr %0) #0 {")
    define = _define_of(lines)
    groups = {line.split()[1]: line for line in lines if line.startswith("attributes ")}

    assert "#0" not in define, define
    assert "noinline" in groups["#0"] and "optnone" in groups["#0"]

    new_group = define.split("#")[1].split()[0]
    copied = groups[f"#{new_group}"]
    assert "noinline" not in copied and "optnone" not in copied
    assert "nounwind" in copied and '"no-trapping-math"="true"' in copied


def test_conflicting_attributes_on_the_define_itself_are_dropped(tmp_path):
    _, lines = _rewrite(
        tmp_path,
        "define dso_local void @add_one(ptr %0) local_unnamed_addr noinline optnone {",
        PLAIN_TAIL,
    )
    define = _define_of(lines)
    assert "alwaysinline" in define
    assert "noinline" not in define and "optnone" not in define


# ---------------------------------------------------------------------------
# General behaviour
# ---------------------------------------------------------------------------


def test_rewrite_is_idempotent(tmp_path):
    ir, lines = _rewrite(
        tmp_path, "define dso_local void @add_one(ptr %0) local_unnamed_addr #0 {"
    )
    first = ir.read_text()
    _make_ir_inlinable(str(ir), "add_one")
    assert ir.read_text() == first


def test_other_functions_are_untouched(tmp_path):
    ir = tmp_path / "kernel.ll"
    ir.write_text(
        "define dso_local void @helper(ptr %0) {\n  ret void\n}\n"
        "define dso_local void @add_one(ptr %0) {\n  ret void\n}\n"
    )
    _make_ir_inlinable(str(ir), "add_one")
    lines = ir.read_text().splitlines()

    assert _define_of(lines, "helper") == "define dso_local void @helper(ptr %0) {"
    assert "alwaysinline" in _define_of(lines, "add_one")


def test_similarly_named_symbol_is_not_matched(tmp_path):
    ir = tmp_path / "kernel.ll"
    ir.write_text("define dso_local void @add_one_helper(ptr %0) {\n  ret void\n}\n")
    with pytest.raises(RuntimeError, match="no `define` for symbol 'add_one'"):
        _make_ir_inlinable(str(ir), "add_one")


def test_missing_definition_fails_loudly(tmp_path):
    """A mangled (non-`extern \"C\"`) kernel would otherwise silently not inline."""
    ir = tmp_path / "kernel.ll"
    ir.write_text("define dso_local void @_Z7add_onePi(ptr %0) {\n  ret void\n}\n")
    with pytest.raises(RuntimeError, match='extern "C"'):
        _make_ir_inlinable(str(ir), "add_one")
