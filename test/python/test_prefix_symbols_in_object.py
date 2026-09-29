# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
# REQUIRES: peano
"""prefix_symbols_in_object and its cache, against a real toolchain.

Every test compiles a real kernel with Peano and inspects the resulting object
with the real llvm-nm, because the contract under test is about a real ELF. A
stand-in for nm or objcopy would agree with any implementation, including one
that hands the job to a tool unable to do it -- and that failure is not
hypothetical: GNU objcopy cannot auto-detect the AIEngine e_machine and bails,
which test_gnu_objcopy_declines_to_autodetect_an_aie_object pins down, along
with the fact that GNU nm and ar handle the same object fine.

Failures are induced with real broken inputs -- a source that does not compile,
a file that is not an object, a read-only directory -- rather than by injecting
exceptions, so what is exercised is the recovery path the build actually takes.

These compile only; no NPU is required.
"""

import json
import os
import shutil
import stat
import subprocess
from types import SimpleNamespace

import aie.utils.compile.utils as compile_utils
import aie.utils.config as config
import pytest

_KERNEL_SOURCE = """
extern "C" void helper_fn(int *p) { *p += 1; }
extern "C" void add_one(int *p) { helper_fn(p); }
"""

_UNCOMPILABLE_SOURCE = 'extern "C" void add_one(int *p) { this is not c++ }\n'


def _symbols(object_path):
    """Return the defined external symbols of `object_path` via the real llvm-nm."""
    out = subprocess.run(
        [config.nm_path(), "--defined-only", "--extern-only", str(object_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(
        line.split()[-1] for line in out.splitlines() if len(line.split()) >= 3
    )


def _compile(source_path, object_path):
    compile_utils.compile_cxx_core_function(str(source_path), "aie2p", str(object_path))


@pytest.fixture
def kernel_object(tmp_path):
    """Compile a real AIE object with two defined external symbols."""
    source = tmp_path / "add_one.cc"
    source.write_text(_KERNEL_SOURCE)
    obj = tmp_path / "add_one.o"
    _compile(source, obj)
    assert _symbols(obj) == ["add_one", "helper_fn"]
    return obj


@pytest.fixture
def func(tmp_path):
    """Create an ExternalFunction-shaped stand-in with real, compilable source."""
    return SimpleNamespace(
        _name="op0_add_one",
        _original_name="add_one",
        _source_string=_KERNEL_SOURCE,
        _source_file=None,
        _include_dirs=[],
        _compile_flags=[],
        _symbol_prefix="op0",
        _compiled=False,
        _compiled_dir=None,
        object_file_name="op0_add_one.o",
    )


def _gnu_tool(name):
    """Return a GNU binutils `name`, or None if only an LLVM one is installed."""
    found = shutil.which(name)
    if found is None or "llvm" in os.path.realpath(found).lower():
        return None
    version = subprocess.run([found, "--version"], capture_output=True, text=True)
    return found if "GNU" in version.stdout else None


def test_gnu_objcopy_declines_to_autodetect_an_aie_object(kernel_object):
    """The precise reason objcopy_path() insists on the LLVM spelling.

    Not that the object is malformed, and not that BFD cannot represent it:
    with an explicit -I the very same GNU objcopy rewrites it correctly. What
    fails is target auto-detection, because no GNU backend claims e_machine
    0x108. Pinning that down here keeps the rationale in aie.utils.config
    honest -- if GNU ever grows an AIEngine backend, this test says so.
    """
    gnu_objcopy = _gnu_tool("objcopy")
    if gnu_objcopy is None:
        pytest.skip("no GNU binutils objcopy available to contrast against")

    auto = subprocess.run(
        [gnu_objcopy, "--redefine-sym=add_one=p_add_one", str(kernel_object)],
        capture_output=True,
        text=True,
    )
    assert auto.returncode != 0
    assert "recognise" in auto.stderr or "recognize" in auto.stderr
    assert _symbols(kernel_object) == ["add_one", "helper_fn"]

    # Told which target to use, it succeeds -- so the object is fine.
    forced = subprocess.run(
        [
            gnu_objcopy,
            "-I",
            "elf32-little",
            "--redefine-sym=add_one=p_add_one",
            str(kernel_object),
        ],
        capture_output=True,
        text=True,
    )
    assert forced.returncode == 0, forced.stderr
    assert _symbols(kernel_object) == ["helper_fn", "p_add_one"]


@pytest.mark.parametrize("tool", ["nm", "ar"])
def test_gnu_nm_and_ar_read_an_aie_object(tmp_path, kernel_object, tool):
    """Check that GNU nm and ar tolerate the unknown architecture.

    Kept alongside the objcopy case so the asymmetry is recorded rather than
    rediscovered: only objcopy is a hard requirement.
    """
    gnu = _gnu_tool(tool)
    if gnu is None:
        pytest.skip(f"no GNU binutils {tool} available to contrast against")

    if tool == "nm":
        result = subprocess.run(
            [gnu, "--defined-only", "--extern-only", str(kernel_object)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "add_one" in result.stdout
    else:
        archive = tmp_path / "libadd_one.a"
        assert (
            subprocess.run(
                [gnu, "rcs", str(archive), str(kernel_object)], capture_output=True
            ).returncode
            == 0
        )
        listed = subprocess.run(
            [gnu, "t", str(archive)], capture_output=True, text=True
        )
        assert listed.returncode == 0
        assert "add_one.o" in listed.stdout


def test_every_defined_symbol_is_prefixed(kernel_object):
    """Not just the entry point: helpers must move into the namespace too.

    A sibling object compiled from the same source under another prefix is
    linked alongside this one, so a helper left unprefixed collides at the
    per-core link.
    """
    compile_utils.prefix_symbols_in_object(str(kernel_object), "op0_")
    assert _symbols(kernel_object) == ["op0_add_one", "op0_helper_fn"]


def test_prefixing_is_literal_and_repeatable(kernel_object):
    """Applying a prefix twice stacks it; the function infers no prior state.

    Callers that must apply a prefix once across cache hits therefore have to
    track that themselves, which is what the stamp below exists for.
    """
    compile_utils.prefix_symbols_in_object(str(kernel_object), "op0_")
    compile_utils.prefix_symbols_in_object(str(kernel_object), "op0_")
    assert _symbols(kernel_object) == ["op0_op0_add_one", "op0_op0_helper_fn"]


def _embedded_ir(obj, tmp_path):
    bitcode = tmp_path / "extracted.bc"
    subprocess.run(
        [
            config.objcopy_path(),
            f"--dump-section=.llvmbc={bitcode}",
            str(obj),
            os.devnull,
        ],
        check=True,
        capture_output=True,
    )
    opt = os.path.join(
        os.path.dirname(config.peano_cxx_path()),
        "opt.exe" if os.name == "nt" else "opt",
    )
    return (
        bitcode,
        subprocess.run(
            [opt, "-S", str(bitcode), "-o", "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout,
    )


def test_ir_renaming_preserves_local_and_metadata_identifiers():
    ir = """
@table = global i32 0
define i32 @kernel() {
$kernel:
  %$table = load i32, ptr @table
  %foo$table = add i32 %$table, 1
  ret i32 %foo$table
}
!foo$table = !{!0}
!0 = !{!"table"}
"""
    opt = os.path.join(
        os.path.dirname(config.peano_cxx_path()),
        "opt.exe" if os.name == "nt" else "opt",
    )
    renamed = compile_utils._rename_ir_symbols(ir, ["table", "kernel"], "op0_")
    for text in (ir, renamed):
        subprocess.run(
            [opt, "-disable-output"],
            input=text,
            text=True,
            check=True,
            capture_output=True,
        )
    assert '@"op0_table"' in renamed
    assert '@"op0_kernel"' in renamed
    for identifier in ("%$table", "%foo$table", "!foo$table", "$kernel:"):
        assert identifier in renamed


@pytest.mark.parametrize("prefix_count", [1, 2])
def test_embedded_bitcode_uses_native_symbol_names(tmp_path, func, prefix_count):
    func._source_string = """
extern "C" {
int table[4] = {1, 2, 3, 4};
int external_fn(int);
int helper_fn(int i) { return external_fn(table[i]); }
int add_one(int i) { return helper_fn(i) + 1; }
}
"""
    func._compile_flags = ["-O0"]
    obj = tmp_path / func.object_file_name
    compile_utils.compile_external_kernel(
        func, str(tmp_path), "aie2p", embed_bitcode=True
    )
    if prefix_count == 2:
        compile_utils.prefix_symbols_in_object(str(obj), "op0_")
    bitcode, ir = _embedded_ir(obj, tmp_path)
    prefix = "op0_" * prefix_count
    assert (
        _symbols(obj)
        == _symbols(bitcode)
        == [f"{prefix}add_one", f"{prefix}helper_fn", f"{prefix}table"]
    )
    assert f"@{prefix}helper_fn(" in ir
    assert f"@{prefix}table" in ir
    assert "@external_fn(" in ir
    assert "@op0_external_fn(" not in ir


def test_embedded_bitcode_preserves_aliases_and_comdats(tmp_path, func):
    func._source_string = """
template <typename T> __attribute__((noinline)) T helper(T i) { return i + 1; }
extern "C" int add_one(int i) { return helper(i); }
extern "C" int alias(int i) __attribute__((alias("add_one")));
"""
    func._compile_flags = ["-O0"]
    obj = tmp_path / func.object_file_name
    compile_utils.compile_external_kernel(
        func, str(tmp_path), "aie2p", embed_bitcode=True
    )
    bitcode, ir = _embedded_ir(obj, tmp_path)
    assert (
        _symbols(obj)
        == _symbols(bitcode)
        == ["op0__Z6helperIiET_S0_", "op0_add_one", "op0_alias"]
    )
    assert "$op0__Z6helperIiET_S0_ = comdat any" in ir
    assert "@op0_alias = " in ir
    assert "ptr @op0_add_one" in ir


@pytest.mark.parametrize("stamp_version", [1, 2, 3])
def test_bitcode_prefix_cache_version(tmp_path, func, stamp_version):
    obj = tmp_path / func.object_file_name
    compile_utils.compile_external_kernel(
        func, str(tmp_path), "aie2p", embed_bitcode=True
    )
    stamp = tmp_path / os.path.basename(
        compile_utils._symbol_prefix_stamp_path(str(obj), "op0_")
    )
    if stamp_version < 3:
        # Recreate the legacy bug: native names were prefixed, IR was untouched.
        subprocess.run(
            [
                config.objcopy_path(),
                f"--update-section=.llvmbc={obj}.bc",
                str(obj),
            ],
            check=True,
            capture_output=True,
        )
        stamp.write_text(
            json.dumps(
                {
                    "version": stamp_version,
                    "prefix": "op0_",
                    "object_sha256": compile_utils._sha256_file(str(obj)),
                }
            )
        )
    before = obj.read_bytes()
    untouched = obj.stat().st_mtime_ns
    func._compiled = False
    func._compiled_dir = None
    compile_utils.compile_external_kernel(
        func, str(tmp_path), "aie2p", embed_bitcode=True
    )
    bitcode, _ = _embedded_ir(obj, tmp_path)
    assert _symbols(obj) == _symbols(bitcode) == ["op0_add_one", "op0_helper_fn"]
    assert compile_utils._has_current_symbol_prefix_stamp(str(obj), "op0_")
    if stamp_version == 3:
        assert obj.read_bytes() == before
        assert obj.stat().st_mtime_ns == untouched
    else:
        assert obj.read_bytes() != before


@pytest.mark.parametrize("embed_bitcode", [False, True])
def test_bitcode_detection_preserves_object(tmp_path, embed_bitcode):
    source = tmp_path / "kernel.cc"
    source.write_text(_KERNEL_SOURCE)
    obj = tmp_path / "kernel.o"
    compile_utils.compile_cxx_core_function(
        str(source), "aie2p", str(obj), embed_bitcode=embed_bitcode
    )
    before = obj.read_bytes()
    mtime = obj.stat().st_mtime_ns
    assert compile_utils._object_has_bitcode(obj) == embed_bitcode
    assert obj.read_bytes() == before
    assert obj.stat().st_mtime_ns == mtime


def test_bad_embedded_bitcode_leaves_native_object_untouched(tmp_path, kernel_object):
    invalid = tmp_path / "invalid.bc"
    invalid.write_bytes(b"not LLVM bitcode")
    subprocess.run(
        [
            config.objcopy_path(),
            f"--add-section=.llvmbc={invalid}",
            str(kernel_object),
        ],
        check=True,
        capture_output=True,
    )
    before = kernel_object.read_bytes()
    with pytest.raises(RuntimeError, match="Embedded bitcode symbol prefixing failed"):
        compile_utils.prefix_symbols_in_object(str(kernel_object), "op0_")
    assert kernel_object.read_bytes() == before
    assert not list(tmp_path.glob("aie-symbol-map-*"))


def test_listing_failure_leaves_the_object_untouched(tmp_path):
    """A real nm failure must abort before objcopy runs.

    Ignoring it would yield an empty rename map, and the pass would silently
    become a no-op that only surfaces as an undefined symbol at final link.
    """
    not_an_object = tmp_path / "kernel.o"
    not_an_object.write_text("this is not an ELF file\n")
    before = not_an_object.read_bytes()

    with pytest.raises(RuntimeError, match="Symbol listing failed"):
        compile_utils.prefix_symbols_in_object(str(not_an_object), "op0_")

    assert not_an_object.read_bytes() == before
    assert list(tmp_path.iterdir()) == [not_an_object]


def test_symbol_map_does_not_disturb_a_sibling_file(tmp_path):
    """The rename map is private, even for an object path containing spaces."""
    source = tmp_path / "add_one.cc"
    source.write_text(_KERNEL_SOURCE)
    obj = tmp_path / "kernel with spaces.o"
    _compile(source, obj)

    sibling = tmp_path / f"{obj.name}.symbol_map"
    sibling.write_text("unrelated file")

    compile_utils.prefix_symbols_in_object(str(obj), "op0_")

    assert _symbols(obj) == ["op0_add_one", "op0_helper_fn"]
    assert sibling.read_text() == "unrelated file"
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("aie-symbol-map-*"))


@pytest.mark.parametrize("source_kind", ["string", "file"])
def test_compile_external_kernel_prefixes_once_and_caches(tmp_path, func, source_kind):
    """The object is prefixed exactly once, and a cache hit re-prefixes nothing."""
    if source_kind == "file":
        source = tmp_path / "add_one.cc"
        source.write_text(func._source_string)
        func._source_string = None
        func._source_file = str(source)

    obj = tmp_path / func.object_file_name
    compile_utils.compile_external_kernel(func, str(tmp_path), "aie2p")

    assert _symbols(obj) == ["op0_add_one", "op0_helper_fn"]
    assert compile_utils._has_current_symbol_prefix_stamp(str(obj), "op0_")
    assert func._compiled

    # A fresh instance over the same directory must reuse the object as-is;
    # re-running the prefix pass would stack a second "op0_".
    untouched = obj.stat().st_mtime_ns
    func._compiled = False
    func._compiled_dir = None
    compile_utils.compile_external_kernel(func, str(tmp_path), "aie2p")

    assert obj.stat().st_mtime_ns == untouched
    assert _symbols(obj) == ["op0_add_one", "op0_helper_fn"]
    assert func._compiled


@pytest.mark.parametrize(
    "cache_state",
    ["unprefixed", "legacy", "no-stamp", "corrupt", "invalid-encoding", "wrong-prefix"],
)
def test_untrusted_cache_is_rebuilt(tmp_path, func, cache_state):
    """Any cache entry whose stamp does not vouch for these exact bytes is rebuilt.

    Trusting one would leave the object either unprefixed or double-prefixed,
    both of which fail at link rather than here.
    """
    obj = tmp_path / func.object_file_name
    stamp = tmp_path / os.path.basename(
        compile_utils._symbol_prefix_stamp_path(str(obj), "op0_")
    )

    source = tmp_path / "seed.cc"
    source.write_text(_KERNEL_SOURCE)
    _compile(source, obj)
    if cache_state in ("legacy", "no-stamp", "corrupt", "invalid-encoding"):
        # Entry-point-only rename, as the pre-bulk-prefix implementation left it.
        compile_utils.prefix_symbols_in_object(str(obj), "op0_")

    if cache_state == "corrupt":
        stamp.write_text("{")
    elif cache_state == "invalid-encoding":
        stamp.write_bytes(b"\xff")
    elif cache_state == "wrong-prefix":
        compile_utils._write_symbol_prefix_stamp(str(obj), "other_")

    compile_utils.compile_external_kernel(func, str(tmp_path), "aie2p")

    assert _symbols(obj) == ["op0_add_one", "op0_helper_fn"]
    assert compile_utils._has_current_symbol_prefix_stamp(str(obj), "op0_")


def test_stale_stamp_for_changed_bytes_is_rebuilt(tmp_path, func):
    """A stamp is bound to the object's digest, not merely to its existence."""
    obj = tmp_path / func.object_file_name
    source = tmp_path / "seed.cc"
    source.write_text(_KERNEL_SOURCE)
    _compile(source, obj)
    compile_utils.prefix_symbols_in_object(str(obj), "op0_")
    compile_utils._write_symbol_prefix_stamp(str(obj), "op0_")

    # Same path, different bytes: the stamp must no longer be believed.
    _compile(source, obj)
    assert not compile_utils._has_current_symbol_prefix_stamp(str(obj), "op0_")

    compile_utils.compile_external_kernel(func, str(tmp_path), "aie2p")
    assert _symbols(obj) == ["op0_add_one", "op0_helper_fn"]
    assert compile_utils._has_current_symbol_prefix_stamp(str(obj), "op0_")


def test_unprefixed_kernel_cache_hit_is_left_alone(tmp_path, func):
    """With no prefix requested there is nothing to stamp, so bytes stand."""
    func._symbol_prefix = None
    obj = tmp_path / func.object_file_name
    obj.write_bytes(b"cached object")

    compile_utils.compile_external_kernel(func, str(tmp_path), "aie2p")

    assert obj.read_bytes() == b"cached object"
    assert func._compiled


def test_compile_failure_leaves_no_trusted_cache(tmp_path, func):
    """A real compile error must not leave a stamp a later run would trust."""
    func._source_string = _UNCOMPILABLE_SOURCE
    obj = tmp_path / func.object_file_name
    stamp = tmp_path / os.path.basename(
        compile_utils._symbol_prefix_stamp_path(str(obj), "op0_")
    )

    with pytest.raises(Exception):
        compile_utils.compile_external_kernel(func, str(tmp_path), "aie2p")

    assert not func._compiled
    assert not stamp.exists()
    assert not list(tmp_path.glob("*.tmp"))

    # The same directory must still build cleanly once the source is fixed.
    func._source_string = _KERNEL_SOURCE
    compile_utils.compile_external_kernel(func, str(tmp_path), "aie2p")
    assert _symbols(obj) == ["op0_add_one", "op0_helper_fn"]
    assert compile_utils._has_current_symbol_prefix_stamp(str(obj), "op0_")


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="requires POSIX directory permissions and a non-root user",
)
def test_unwritable_directory_leaves_no_trusted_cache(tmp_path, func):
    """If the stamp cannot be written, the entry must not be trusted later."""
    obj = tmp_path / func.object_file_name
    source = tmp_path / "seed.cc"
    source.write_text(_KERNEL_SOURCE)
    _compile(source, obj)

    mode = tmp_path.stat().st_mode
    os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(OSError):
            compile_utils.compile_external_kernel(func, str(tmp_path), "aie2p")
    finally:
        os.chmod(tmp_path, mode)

    assert not compile_utils._has_current_symbol_prefix_stamp(str(obj), "op0_")
    assert not func._compiled
