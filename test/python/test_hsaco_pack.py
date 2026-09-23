# test_hsaco_pack.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s

"""aie.compiler.hsaco -- section packing, the three kernel input forms, injection.

Each of the three ``--kernel`` forms gets its own group of tests:

  * PDI + insts is exercised on raw bytes, since that form has no external
    dependency and is where the layout and validation rules live.
  * Full ELF is exercised against a synthetic ELF64 built in-test. The COMDAT
    pairing it relies on is unusual enough (see ``elf.kernel_names_from_full_elf``)
    that a hand-built fixture is clearer than a compiled artifact, and it keeps
    the test independent of Peano.
  * xclbin is exercised through a stub ``xclbinutil`` so the real subprocess,
    glob and read path runs without needing XRT or a compiled design.

Injection itself needs a real ``llvm-objcopy`` and is skipped without one.
"""

import argparse
import os
import re
import stat
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from aie.compiler.hsaco import dump, elf, pack
from aie.compiler.hsaco import format as hsaco_format

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Restated here on purpose rather than imported from elf.py: a fixture built
# from the parser's own layout constants can only ever agree with itself, so it
# could not catch a wrong constant. These are the second, independent statement
# of the ELF64 layout -- keep them literal.
_SHDR = "<IIQQQQIIQQ"
_SHDR_SIZE = struct.calcsize(_SHDR)
_SYM = "<IBBHQQ"
_SYM_SIZE = struct.calcsize(_SYM)
_EHDR_SIZE = 64


def _strtab(names):
    """Return (blob, {name: offset}) for a NUL-separated string table."""
    blob = bytearray(b"\x00")
    offsets = {}
    for n in names:
        offsets[n] = len(blob)
        blob += n.encode() + b"\x00"
    return bytes(blob), offsets


def make_full_elf(pairs):
    """Return a synthetic full ELF64 with one COMDAT group per (kernel, instance).

    The instance symbol signs the group (``sh_info``), and its ``st_shndx``
    holds the symbol index of the kernel symbol -- the encoding the real AIE
    full-ELF producer uses.
    """
    section_names = [".shstrtab", ".strtab", ".symtab"] + [".group"] * len(pairs)
    shstrtab, sh_off = _strtab([".shstrtab", ".strtab", ".symtab", ".group"])

    symbol_names = []
    for kernel, instance in pairs:
        symbol_names += [kernel, instance]
    strtab, st_off = _strtab(symbol_names)

    # Symbol 0 is the mandatory null entry; then kernel/instance pairs.
    symbols = [(0, 0)]
    signature_indices = []
    for kernel, instance in pairs:
        kernel_index = len(symbols)
        symbols.append((st_off[kernel], 0))
        signature_indices.append(len(symbols))
        symbols.append((st_off[instance], kernel_index))
    symtab = b"".join(
        struct.pack(_SYM, name, 0, 0, shndx, 0, 0) for name, shndx in symbols
    )

    # One 4-byte GRP_COMDAT flag word per group; contents are not read.
    group = struct.pack("<I", 0x1)
    bodies = [shstrtab, strtab, symtab] + [group] * len(pairs)

    offset = _EHDR_SIZE
    offsets = []
    for body in bodies:
        offsets.append(offset)
        offset += len(body)
    shoff = offset

    headers = [struct.pack(_SHDR, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
    for i, (name, body, off) in enumerate(zip(section_names, bodies, offsets)):
        index = i + 1
        if name == ".symtab":
            sh_type, link, info, entsize, align = 2, 2, 1, _SYM_SIZE, 8
        elif name == ".group":
            # SHT_GROUP holds 4-byte words, and llvm-objcopy rejects any other
            # alignment on one outright -- so this must match real ELFs.
            sh_type, link, info, entsize, align = (
                17,
                3,
                signature_indices[index - 4],
                4,
                4,
            )
        else:
            sh_type, link, info, entsize, align = 3, 0, 0, 0, 1
        headers.append(
            struct.pack(
                _SHDR,
                sh_off[name],
                sh_type,
                0,
                0,
                off,
                len(body),
                link,
                info,
                align,
                entsize,
            )
        )

    ehdr = (
        b"\x7fELF\x02\x01\x01\x00"
        + b"\x00" * 8
        + struct.pack(
            "<HHIQQQIHHHHHH",
            1,  # ET_REL
            224,  # EM_AMDGPU
            1,
            0,
            0,
            shoff,
            0,
            _EHDR_SIZE,
            0,
            0,
            _SHDR_SIZE,
            len(headers),
            1,  # .shstrtab is section 1
        )
    )
    return ehdr + b"".join(bodies) + b"".join(headers)


def _kernel(name, insts, pdi=None, **kw):
    k = {"name": name, "insts": insts, "pdi": pdi}
    k.update(kw)
    return k


def _write(path, data):
    with open(path, "wb") as f:
        f.write(data)
    return str(path)


def _have_objcopy():
    """Resolve objcopy exactly as the code under test does.

    Not ``shutil.which``: in a wheel or CMake install llvm-objcopy is bundled
    under the mlir-aie/Peano bin dirs and is usually absent from PATH, so a
    which-based guard would skip every injection test in precisely the
    environment they exist to cover, and still report green.
    """
    try:
        return pack.objcopy_path() is not None
    except Exception:
        return False


needs_objcopy = pytest.mark.skipif(
    not _have_objcopy(), reason="llvm-objcopy could not be resolved"
)
needs_posix = pytest.mark.skipif(
    os.name == "nt", reason="stub tool script is POSIX-only"
)


# ---------------------------------------------------------------------------
# form 1: PDI + insts
# ---------------------------------------------------------------------------


def test_pdi_insts_round_trip():
    kernels = [
        _kernel("alpha", b"\x01\x02\x03\x04", b"PDI-A", kernarg_size=64, num_cols=2),
        _kernel("beta", b"\x05\x06", None, kernarg_size=32, num_cols=1),
    ]
    info = dump.parse_section(pack.build_section("aie2p", kernels))

    assert info["arch_version"] == (
        hsaco_format.VERSION_MAJOR,
        hsaco_format.VERSION_MINOR,
    )
    assert info["kernel_count"] == 2
    alpha, beta = info["kernels"]
    assert alpha["name"] == "alpha"
    assert alpha["kind_name"] == "PdiInsts"
    assert (alpha["insts_size"], alpha["pdi_size"]) == (4, 5)
    assert (alpha["kernarg_size"], alpha["num_cols"]) == (64, 2)
    assert beta["name"] == "beta"
    assert beta["has_pdi"] is False
    assert (beta["kernarg_size"], beta["num_cols"]) == (32, 1)


def test_blob_bytes_survive_the_round_trip():
    section = pack.build_section(
        "aie2", [_kernel("k", b"\xde\xad\xbe\xef", b"\xca\xfe")]
    )
    k = dump.parse_section(section)["kernels"][0]
    insts = section[k["insts_offset"] : k["insts_offset"] + k["insts_size"]]
    pdi = section[k["pdi_offset"] : k["pdi_offset"] + k["pdi_size"]]
    assert insts == b"\xde\xad\xbe\xef"
    assert pdi == b"\xca\xfe"


def test_identical_blobs_are_pooled_once():
    shared = b"X" * 4096
    two = pack.build_section("aie2p", [_kernel("a", shared), _kernel("b", shared)])
    one = pack.build_section("aie2p", [_kernel("a", shared)])
    # The second kernel adds a table entry and a name, not another copy of the blob.
    assert len(two) - len(one) < len(shared)

    a, b = dump.parse_section(two)["kernels"]
    assert a["insts_offset"] == b["insts_offset"]


def test_defaults_when_optional_fields_are_omitted():
    k = dump.parse_section(pack.build_section("aie2", [_kernel("k", b"\x00")]))[
        "kernels"
    ][0]
    assert (k["kernarg_size"], k["num_cols"], k["kind"]) == (
        0,
        1,
        hsaco_format.KIND_PDI_INSTS,
    )


@pytest.mark.parametrize(
    "kernels, message",
    [
        ([_kernel("k", b"")], "insts must be non-empty"),
        ([_kernel("k", b"\x01", kind=7)], "unknown kind"),
        (
            [_kernel("k", b"\x01", b"pdi", kind=hsaco_format.KIND_FULL_ELF)],
            "carry no separate PDI",
        ),
    ],
)
def test_build_section_rejects_bad_kernels(kernels, message):
    with pytest.raises(ValueError, match=message):
        pack.build_section("aie2", kernels)


def test_build_section_rejects_unknown_arch():
    with pytest.raises(ValueError, match="unknown arch"):
        pack.build_section("aie7", [_kernel("k", b"\x01")])


# ---------------------------------------------------------------------------
# section validation
# ---------------------------------------------------------------------------


def test_parse_section_rejects_bad_magic():
    section = bytearray(pack.build_section("aie2", [_kernel("k", b"\x01")]))
    section[0:4] = b"XXXX"
    with pytest.raises(ValueError, match="bad magic"):
        dump.parse_section(bytes(section))


def test_parse_section_rejects_a_future_major_version():
    """The layout mirrors a header in another repo; a bumped major must not be guessed at."""
    section = bytearray(pack.build_section("aie2", [_kernel("k", b"\x01")]))
    struct.pack_into("<H", section, 4, hsaco_format.VERSION_MAJOR + 1)
    with pytest.raises(ValueError, match="unsupported section version"):
        dump.parse_section(bytes(section))


def test_parse_section_rejects_a_truncated_section():
    section = pack.build_section("aie2", [_kernel("k", b"\x01\x02\x03\x04")])
    with pytest.raises(ValueError):
        dump.parse_section(section[: len(section) - 2])


def test_parse_section_rejects_a_header_only_runt():
    with pytest.raises(ValueError, match="smaller than header"):
        dump.parse_section(b"\x00" * 4)


def test_blob_offsets_must_point_into_the_blob_pool():
    """insts_offset=0 is inside the section but is the header, not a blob."""
    section = bytearray(pack.build_section("aie2", [_kernel("k", b"\x01\x02\x03\x04")]))
    hdr_size = struct.unpack_from(hsaco_format.HDR, section, 0)[3]
    struct.pack_into("<I", section, hdr_size + 4, 0)  # insts_offset
    with pytest.raises(ValueError, match="insts out of bounds/overrun"):
        dump.parse_section(bytes(section))


def test_writer_and_reader_agree_on_which_kinds_exist():
    """Both sides bound `kind` by KIND_COUNT, so adding one stays a single edit."""
    with pytest.raises(ValueError, match="unknown kind"):
        pack.build_section(
            "aie2", [_kernel("k", b"\x01", kind=hsaco_format.KIND_COUNT)]
        )
    section = bytearray(pack.build_section("aie2", [_kernel("k", b"\x01")]))
    hdr_size = struct.unpack_from(hsaco_format.HDR, section, 0)[3]
    struct.pack_into("<I", section, hdr_size + 28, hsaco_format.KIND_COUNT)
    with pytest.raises(ValueError, match="unknown kind"):
        dump.parse_section(bytes(section))


def test_parse_section_rejects_a_header_size_below_the_real_header():
    """The kernel table is indexed from header_size; too small overlaps the header."""
    section = bytearray(pack.build_section("aie2", [_kernel("k", b"\x01\x02\x03\x04")]))
    struct.pack_into("<I", section, 8, 8)
    with pytest.raises(ValueError, match="header_size smaller than the header"):
        dump.parse_section(bytes(section))


def test_kernel_name_cannot_point_outside_the_string_table():
    """A name offset past the table would otherwise read out of the blob pool."""
    section = bytearray(pack.build_section("aie2", [_kernel("k", b"AB\x00CD")]))
    hdr = struct.unpack_from(hsaco_format.HDR, section, 0)
    hdr_size, st_size = hdr[3], hdr[7]
    struct.pack_into("<I", section, hdr_size, st_size)
    with pytest.raises(ValueError, match="name offset outside the string table"):
        dump.parse_section(bytes(section))


def test_duplicate_kernel_names_are_rejected():
    """ROCR resolves by name, so a duplicate makes one entry unreachable."""
    with pytest.raises(ValueError, match="duplicate kernel name\\(s\\): dup"):
        pack.build_section(
            "aie2",
            [_kernel("dup", b"\x01"), _kernel("ok", b"\x02"), _kernel("dup", b"\x03")],
        )


# ---------------------------------------------------------------------------
# form 3: full ELF
# ---------------------------------------------------------------------------


def test_full_elf_names_every_comdat_group(tmp_path):
    blob = make_full_elf([("_Z4mainPcPcPc", "instance0"), ("vadd", "instance1")])
    path = _write(tmp_path / "final.elf", blob)

    kernels = pack.kernels_from_full_elf(path, kernarg_size=64, num_cols=4)
    assert [k["name"] for k in kernels] == ["main:instance0", "vadd:instance1"]
    assert all(k["kind"] == hsaco_format.KIND_FULL_ELF for k in kernels)
    assert all(k["pdi"] is None for k in kernels)
    assert all(k["insts"] == blob for k in kernels)
    assert all((k["kernarg_size"], k["num_cols"]) == (64, 4) for k in kernels)


def test_full_elf_image_is_embedded_once(tmp_path):
    blob = make_full_elf([("k0", "i0"), ("k1", "i1"), ("k2", "i2")])
    path = _write(tmp_path / "final.elf", blob)

    section = pack.build_section("aie2p", pack.kernels_from_full_elf(path))
    info = dump.parse_section(section)
    assert info["kernel_count"] == 3
    assert len({k["insts_offset"] for k in info["kernels"]}) == 1
    assert all(k["kind_name"] == "FullElf" for k in info["kernels"])
    assert all(not k["has_pdi"] for k in info["kernels"])
    # One copy of the ELF plus table/strings, not three copies.
    assert len(section) < 2 * len(blob)


def test_elf_without_comdat_groups_is_rejected(tmp_path):
    path = _write(tmp_path / "plain.elf", make_full_elf([]))
    with pytest.raises(ValueError, match="no COMDAT groups"):
        pack.kernels_from_full_elf(path)


def test_elf_without_a_symbol_table_is_rejected():
    with pytest.raises(ValueError, match="missing .symtab"):
        elf.ElfFile(elf.make_empty_elf64()).symbol_at(0)


def test_non_elf_input_is_rejected(tmp_path):
    path = _write(tmp_path / "not.elf", b"this is not an ELF file at all" * 8)
    with pytest.raises(ValueError, match="bad magic"):
        pack.kernels_from_full_elf(path)


def test_elf_without_a_section_name_table_is_rejected():
    """e_shstrndx 0 is SHN_UNDEF; resolving names against offset 0 yields garbage."""
    blob = bytearray(elf.make_empty_elf64())
    struct.pack_into("<H", blob, 62, 0)
    with pytest.raises(ValueError, match="no section-name table"):
        elf.ElfFile(bytes(blob))


def _symtab_shdr_offset(blob):
    """Return the file offset of the .symtab section header in a fixture ELF."""
    shoff = struct.unpack_from("<Q", blob, 40)[0]
    for index, section in enumerate(elf.ElfFile(bytes(blob)).sections):
        if section.type == elf.SHT_SYMTAB:
            return shoff + index * _SHDR_SIZE
    raise AssertionError("fixture has no .symtab")


@pytest.mark.parametrize(
    "field_offset, field_format, value, message",
    [
        (40, "<I", 99, "sh_link 99 out of range"),
        (56, "<Q", 1, "sh_entsize 1 too small"),
    ],
    ids=["sh_link", "sh_entsize"],
)
def test_corrupt_symtab_fields_raise_valueerror(
    field_offset, field_format, value, message
):
    """Every failure in this module is a ValueError, so callers can add context."""
    blob = bytearray(make_full_elf([("k", "i")]))
    base = _symtab_shdr_offset(blob)
    struct.pack_into(field_format, blob, base + field_offset, value)
    with pytest.raises(ValueError, match=message):
        elf.ElfFile(bytes(blob)).symbol_at(0)


def test_section_name_offset_is_bounded_by_the_string_table():
    """An out-of-range sh_name must not read a name out of whatever follows."""
    blob = bytearray(elf.make_empty_elf64())
    shoff = struct.unpack_from("<Q", blob, 40)[0]
    struct.pack_into("<I", blob, shoff + _SHDR_SIZE, 100)  # section 1 sh_name
    # Well inside the file, but past the end of the 11-byte .shstrtab: without
    # the table bound this resolved to '' rather than raising.
    with pytest.raises(ValueError, match="string offset .* out of bounds"):
        [s.name for s in elf.ElfFile(bytes(blob)).sections]


def test_symtab_running_past_end_of_file_is_rejected():
    """symbol_at unpacks straight out of the image, bypassing Section.data."""
    blob = bytearray(make_full_elf([("k", "i")]))
    base = _symtab_shdr_offset(blob)
    struct.pack_into("<Q", blob, base + 32, 1 << 20)  # sh_size
    with pytest.raises(ValueError, match="runs past end of file"):
        elf.ElfFile(bytes(blob)).symbol_at(1)


def test_non_utf8_names_raise_valueerror_not_unicodeerror():
    blob = bytearray(elf.make_empty_elf64())
    blob[blob.index(b".shstrtab") + 1] = 0xFF
    with pytest.raises(ValueError, match="non-UTF-8"):
        [s.name for s in elf.ElfFile(bytes(blob)).sections]


@pytest.mark.parametrize(
    "symbol, expected",
    [
        ("_Z4mainPcPcPc", "main"),
        ("_Z11long_kernelv", "long_kernel"),
        ("plain_name", "plain_name"),
        ("_Znotmangled", "_Znotmangled"),
        ("_Z99truncated", "_Z99truncated"),
    ],
)
def test_demangle_kernel_name(symbol, expected):
    assert elf.demangle_kernel_name(symbol) == expected


# ---------------------------------------------------------------------------
# form 2: xclbin + insts
# ---------------------------------------------------------------------------


_STUB_XCLBINUTIL = """#!{python}
import os, sys
args = sys.argv[1:]
spec = args[args.index("--dump-section") + 1]
out_dir = os.path.dirname(spec.split(":", 2)[2])
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "partition.pdi"), "wb") as f:
    f.write({payload!r})
"""


def _make_stub_xclbinutil(tmp_path, payload=b"PDI-FROM-XCLBIN", count=1):
    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / "xclbinutil"
    body = _STUB_XCLBINUTIL.format(python=sys.executable, payload=payload)
    if count != 1:
        body += "".join(
            f'\nopen(os.path.join(out_dir, "extra{i}.pdi"), "wb").write(b"x")'
            for i in range(count - 1)
        )
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(bin_dir), str(script)


@needs_posix
def test_pdi_is_extracted_from_an_xclbin(tmp_path, monkeypatch):
    _, script = _make_stub_xclbinutil(tmp_path)
    monkeypatch.setattr(pack, "xclbinutil_path", lambda: script)
    xclbin = _write(tmp_path / "final.xclbin", b"not really an xclbin")

    assert pack.pdi_from_xclbin(xclbin) == b"PDI-FROM-XCLBIN"


@needs_posix
def test_xclbin_kernel_spec_packs_the_extracted_pdi(tmp_path, monkeypatch):
    _, script = _make_stub_xclbinutil(tmp_path)
    monkeypatch.setattr(pack, "xclbinutil_path", lambda: script)
    xclbin = _write(tmp_path / "final.xclbin", b"stub")
    insts = _write(tmp_path / "insts.bin", b"\x11\x22\x33\x44")

    kernels = pack.parse_kernel_arg(f"xclbin:MLIR_AIE:{xclbin}:{insts}:64:2")
    assert len(kernels) == 1
    assert kernels[0]["insts"] == b"\x11\x22\x33\x44"
    assert kernels[0]["pdi"] == b"PDI-FROM-XCLBIN"

    k = dump.parse_section(pack.build_section("aie2p", kernels))["kernels"][0]
    assert k["name"] == "MLIR_AIE"
    assert (k["kernarg_size"], k["num_cols"], k["pdi_size"]) == (64, 2, 15)


@needs_posix
def test_xclbin_with_several_pdis_is_rejected(tmp_path, monkeypatch):
    _, script = _make_stub_xclbinutil(tmp_path, count=2)
    monkeypatch.setattr(pack, "xclbinutil_path", lambda: script)
    xclbin = _write(tmp_path / "final.xclbin", b"stub")

    with pytest.raises(ValueError, match="expected exactly one PDI, found 2"):
        pack.pdi_from_xclbin(xclbin)


@needs_posix
def test_xclbinutil_falls_back_to_path(tmp_path, monkeypatch):
    bin_dir, script = _make_stub_xclbinutil(tmp_path)
    monkeypatch.setattr(pack, "_bundled_tool", lambda name: None)
    monkeypatch.setenv("PATH", bin_dir, prepend=os.pathsep)
    assert os.path.samefile(pack.xclbinutil_path(), script)


def test_bundled_xclbinutil_wins_over_path(monkeypatch):
    monkeypatch.setattr(pack, "_bundled_tool", lambda name: f"/bundled/{name}")
    assert pack.xclbinutil_path() == "/bundled/xclbinutil"


def test_missing_xclbinutil_explains_the_alternatives(monkeypatch):
    monkeypatch.setattr(pack, "_bundled_tool", lambda name: None)
    monkeypatch.setattr(pack.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="PDI\\+insts --kernel form"):
        pack.xclbinutil_path()


# ---------------------------------------------------------------------------
# --kernel argument parsing
# ---------------------------------------------------------------------------


def test_pdi_insts_kernel_specs(tmp_path):
    insts = _write(tmp_path / "insts.bin", b"\x01\x02")
    pdi = _write(tmp_path / "main.pdi", b"\x03\x04")

    (with_pdi,) = pack.parse_kernel_arg(f"k:{insts}:{pdi}:64:2")
    assert (with_pdi["insts"], with_pdi["pdi"]) == (b"\x01\x02", b"\x03\x04")
    assert (with_pdi["kernarg_size"], with_pdi["num_cols"]) == (64, 2)

    (without_pdi,) = pack.parse_kernel_arg(f"k:{insts}:32:1")
    assert without_pdi["pdi"] is None
    assert (without_pdi["kernarg_size"], without_pdi["num_cols"]) == (32, 1)


def test_elf_kernel_spec_accepts_optional_trailing_fields(tmp_path):
    path = _write(tmp_path / "final.elf", make_full_elf([("k", "i")]))

    (bare,) = pack.parse_kernel_arg(f"elf:{path}")
    assert (bare["kernarg_size"], bare["num_cols"]) == (0, 1)
    (sized,) = pack.parse_kernel_arg(f"elf:{path}:96")
    assert (sized["kernarg_size"], sized["num_cols"]) == (96, 1)
    (full,) = pack.parse_kernel_arg(f"elf:{path}:96:4")
    assert (full["kernarg_size"], full["num_cols"]) == (96, 4)


@pytest.mark.parametrize(
    "spec",
    ["elf", "elf:a:1:2:3", "xclbin:too:few:fields", "only_two:fields", "a:b:c:d:e:f"],
)
def test_malformed_kernel_specs_are_rejected(spec):
    with pytest.raises(Exception, match="bad --kernel spec"):
        pack.parse_kernel_arg(spec)


def test_unreadable_input_becomes_a_usage_error_not_a_traceback():
    """Argparse converts only TypeError/ValueError/ArgumentTypeError from type=."""
    with pytest.raises(argparse.ArgumentTypeError, match="No such file"):
        pack.parse_kernel_arg("k:/nonexistent/insts.bin:64:1")
    with pytest.raises(argparse.ArgumentTypeError, match="No such file"):
        pack.parse_kernel_arg("elf:/nonexistent/final.elf")


# ---------------------------------------------------------------------------
# long-form kernel options
# ---------------------------------------------------------------------------


def test_long_form_pdi_insts(tmp_path):
    insts = _write(tmp_path / "insts.bin", b"\x01\x02")
    pdi = _write(tmp_path / "main.pdi", b"\x03")

    (kernel,) = pack.kernels_from_options(
        [
            ("kernel_name", "k"),
            ("kernel_insts", insts),
            ("kernel_pdi", pdi),
            ("kernel_kernarg", 64),
            ("kernel_cols", 2),
        ]
    )
    assert kernel["name"] == "k"
    assert (kernel["insts"], kernel["pdi"]) == (b"\x01\x02", b"\x03")
    assert (kernel["kernarg_size"], kernel["num_cols"]) == (64, 2)
    assert kernel["kind"] == hsaco_format.KIND_PDI_INSTS


def test_long_form_defaults_match_the_colon_form(tmp_path):
    insts = _write(tmp_path / "insts.bin", b"\x01")
    (kernel,) = pack.kernels_from_options(
        [("kernel_name", "k"), ("kernel_insts", insts)]
    )
    assert (kernel["kernarg_size"], kernel["num_cols"]) == (0, 1)
    assert kernel["pdi"] is None


def test_long_form_full_elf(tmp_path):
    path = _write(tmp_path / "final.elf", make_full_elf([("k0", "i0"), ("k1", "i1")]))
    kernels = pack.kernels_from_options(
        [("kernel_elf", path), ("kernel_kernarg", 96), ("kernel_cols", 4)]
    )
    assert [k["name"] for k in kernels] == ["k0:i0", "k1:i1"]
    assert all(k["kind"] == hsaco_format.KIND_FULL_ELF for k in kernels)
    assert all((k["kernarg_size"], k["num_cols"]) == (96, 4) for k in kernels)


@needs_posix
def test_long_form_xclbin(tmp_path, monkeypatch):
    _, script = _make_stub_xclbinutil(tmp_path)
    monkeypatch.setattr(pack, "xclbinutil_path", lambda: script)
    xclbin = _write(tmp_path / "final.xclbin", b"stub")
    insts = _write(tmp_path / "insts.bin", b"\x01")

    (kernel,) = pack.kernels_from_options(
        [
            ("kernel_name", "k"),
            ("kernel_xclbin", xclbin),
            ("kernel_insts", insts),
        ]
    )
    assert kernel["pdi"] == b"PDI-FROM-XCLBIN"


def test_long_form_describes_several_kernels(tmp_path):
    """Each starter opens a new kernel; following options attach to it."""
    a = _write(tmp_path / "a.bin", b"\xaa")
    b = _write(tmp_path / "b.bin", b"\xbb")
    elf_path = _write(tmp_path / "final.elf", make_full_elf([("k", "i")]))

    kernels = pack.kernels_from_options(
        [
            ("kernel_name", "first"),
            ("kernel_insts", a),
            ("kernel_cols", 2),
            ("kernel_elf", elf_path),
            ("kernel_name", "second"),
            ("kernel_insts", b),
            ("kernel_kernarg", 32),
        ]
    )
    assert [k["name"] for k in kernels] == ["first", "k:i", "second"]
    assert kernels[0]["num_cols"] == 2
    assert kernels[1]["num_cols"] == 1  # not inherited from the previous kernel
    assert kernels[2]["kernarg_size"] == 32
    assert kernels[0]["kernarg_size"] == 0


@pytest.mark.parametrize(
    "ops, message",
    [
        ([("kernel_insts", "x.bin")], "must follow --kernel-name"),
        (
            [("kernel_name", "k"), ("kernel_insts", "a"), ("kernel_insts", "b")],
            "given twice for the same kernel",
        ),
        ([("kernel_name", "k")], "requires --kernel-insts"),
        (
            [
                ("kernel_name", "k"),
                ("kernel_insts", "a"),
                ("kernel_pdi", "p"),
                ("kernel_xclbin", "x"),
            ],
            "not both",
        ),
        (
            [("kernel_elf", "e"), ("kernel_insts", "a")],
            "self-contained and cannot be combined",
        ),
    ],
)
def test_malformed_long_form_is_rejected(ops, message):
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        pack.kernels_from_options(ops)


def test_long_form_unreadable_file_is_a_usage_error():
    with pytest.raises(argparse.ArgumentTypeError, match="No such file"):
        pack.kernels_from_options(
            [("kernel_name", "k"), ("kernel_insts", "/nonexistent.bin")]
        )


@needs_objcopy
def test_long_form_handles_a_path_containing_a_colon(tmp_path):
    """The whole point: a colon in a path breaks --kernel but not --kernel-*."""
    odd = tmp_path / "C:weird"
    odd.mkdir()
    insts = _write(odd / "insts.bin", b"\x01\x02\x03")
    hsaco = str(tmp_path / "out.hsaco")

    with pytest.raises(SystemExit):
        pack.main(["--hsaco", hsaco, "--arch", "aie2", "--kernel", f"k:{insts}:0:1"])

    assert (
        pack.main(
            [
                "--hsaco",
                hsaco,
                "--arch",
                "aie2",
                "--kernel-name",
                "k",
                "--kernel-insts",
                insts,
            ]
        )
        == 0
    )
    ((_, data),) = dump.read_sections_from_hsaco(hsaco)
    assert dump.parse_section(data)["kernels"][0]["name"] == "k"


@needs_objcopy
def test_mixed_grammars_keep_argv_order(tmp_path):
    """The README calls the two forms freely mixable; the kernel table is ordered."""
    a = _write(tmp_path / "a.bin", b"\xaa")
    b = _write(tmp_path / "b.bin", b"\xbb")
    elf_path = _write(tmp_path / "final.elf", make_full_elf([("k", "i")]))
    hsaco = str(tmp_path / "out.hsaco")

    # Long form first, colon form second: the output must follow argv, not the
    # grammar the kernel happened to be written in.
    assert (
        pack.main(
            ["--hsaco", hsaco, "--arch", "aie2"]
            + ["--kernel-name", "first", "--kernel-insts", a]
            + [f"--kernel=elf:{elf_path}"]
            + ["--kernel-name", "last", "--kernel-insts", b]
        )
        == 0
    )
    ((_, data),) = dump.read_sections_from_hsaco(hsaco)
    assert [k["name"] for k in dump.parse_section(data)["kernels"]] == [
        "first",
        "k:i",
        "last",
    ]


@needs_objcopy
def test_main_mixes_long_form_and_colon_form(tmp_path):
    elf_path = _write(tmp_path / "final.elf", make_full_elf([("k0", "i0")]))
    insts = _write(tmp_path / "insts.bin", b"\x07\x08")
    hsaco = str(tmp_path / "out.hsaco")

    assert (
        pack.main(
            [
                "--hsaco",
                hsaco,
                "--arch",
                "aie2p",
                f"--kernel=elf:{elf_path}",
                "--kernel-name",
                "direct",
                "--kernel-insts",
                insts,
                "--kernel-kernarg",
                "16",
            ]
        )
        == 0
    )
    ((_, data),) = dump.read_sections_from_hsaco(hsaco)
    info = dump.parse_section(data)
    assert [k["name"] for k in info["kernels"]] == ["k0:i0", "direct"]
    assert info["kernels"][1]["kernarg_size"] == 16


def test_main_requires_at_least_one_kernel(tmp_path, capsys):
    with pytest.raises(SystemExit):
        pack.main(["--hsaco", str(tmp_path / "x.hsaco"), "--arch", "aie2"])
    assert "no kernels given" in capsys.readouterr().err


@needs_objcopy
def test_kernel_ops_do_not_leak_between_runs(tmp_path):
    """A list default on the action would accumulate across parse_args calls."""
    insts = _write(tmp_path / "insts.bin", b"\x01")
    args = ["--hsaco", str(tmp_path / "a.hsaco"), "--arch", "aie2"]
    argv = args + ["--kernel-name", "k", "--kernel-insts", insts]

    first = pack.main(argv)
    # A second identical run must not see the first run's kernel again, which
    # would trip the duplicate-name check.
    second = pack.main(argv)
    assert (first, second) == (0, 0)


@pytest.mark.parametrize("field", ["kernarg_size", "num_cols"])
@pytest.mark.parametrize("value", [-1, 2**32])
def test_out_of_range_counts_are_rejected(field, value):
    """These land in uint32 fields; struct.error is not a usable diagnosis."""
    with pytest.raises(ValueError, match=f"{field} {value} out of range"):
        pack.build_section("aie2", [_kernel("k", b"\x01", **{field: value})])


def test_out_of_range_counts_are_reachable_from_the_cli(tmp_path, capsys):
    insts = _write(tmp_path / "insts.bin", b"\x01")
    with pytest.raises(SystemExit):
        pack.main(
            [
                "--hsaco",
                str(tmp_path / "x.hsaco"),
                "--arch",
                "aie2",
                "--kernel-name",
                "k",
                "--kernel-insts",
                insts,
                "--kernel-cols",
                "-1",
            ]
        )
    assert "num_cols -1 out of range" in capsys.readouterr().err


def test_both_kernel_grammars_report_a_bad_elf_the_same_way(tmp_path, capsys):
    """The long form is what the README recommends; it must not be the worse path."""
    plain = _write(tmp_path / "plain.elf", b"\x7fELF\x02\x01\x01" + b"\x00" * 80)
    base = ["--hsaco", str(tmp_path / "x.hsaco"), "--arch", "aie2"]

    for extra in (["--kernel", f"elf:{plain}"], ["--kernel-elf", plain]):
        with pytest.raises(SystemExit) as excinfo:
            pack.main(base + extra)
        assert excinfo.value.code == 2
        # The real diagnosis must survive, not argparse's "invalid value".
        assert "ELF has no section headers" in capsys.readouterr().err


def test_a_non_numeric_count_keeps_its_diagnosis(tmp_path, capsys):
    insts = _write(tmp_path / "insts.bin", b"\x01")
    with pytest.raises(SystemExit):
        pack.main(
            ["--hsaco", str(tmp_path / "x.hsaco"), "--arch", "aie2"]
            + ["--kernel", f"k:{insts}:notanumber:1"]
        )
    assert "invalid literal for int" in capsys.readouterr().err


def test_main_reports_a_missing_file_as_usage(tmp_path, capsys):
    hsaco = str(tmp_path / "out.hsaco")
    with pytest.raises(SystemExit) as excinfo:
        pack.main(
            ["--hsaco", hsaco, "--arch", "aie2", "--kernel", "k:/nonexistent.bin:64:1"]
        )
    assert excinfo.value.code == 2
    assert "--kernel" in capsys.readouterr().err
    assert not os.path.exists(hsaco)


# ---------------------------------------------------------------------------
# injection into an hsaco
# ---------------------------------------------------------------------------


@needs_objcopy
def test_ensure_hsaco_creates_an_injectable_container(tmp_path):
    path = str(tmp_path / "out.hsaco")
    pack.ensure_hsaco(path)
    assert os.path.exists(path)

    section = pack.build_section("aie2p", [_kernel("k", b"\x01\x02\x03\x04")])
    pack.inject(path, "aie2p", section)

    assert dump.read_sections_from_hsaco(path) == [("aie2p", section)]


def test_ensure_hsaco_leaves_an_existing_file_alone(tmp_path):
    path = str(tmp_path / "out.hsaco")
    _write(tmp_path / "out.hsaco", elf.make_empty_elf64() + b"\x00" * 16)
    before = open(path, "rb").read()
    pack.ensure_hsaco(path)
    assert open(path, "rb").read() == before


@needs_objcopy
def test_injecting_into_a_populated_elf_keeps_its_sections(tmp_path):
    """The AIE section rides alongside whatever the code object already carries.

    Uses a group-free ELF: ``make_full_elf`` synthesizes COMDAT groups with no
    member sections, and llvm-objcopy prunes empty groups. Real full ELFs have
    populated groups, and nothing injects into one anyway -- they are only ever
    read.
    """
    path = _write(tmp_path / "existing.hsaco", make_full_elf([]))
    with open(path, "rb") as f:
        before = [s.name for s in elf.ElfFile(f.read()).sections]
    assert ".symtab" in before

    section = pack.build_section("aie2", [_kernel("added", b"\x01\x02")])
    pack.inject(path, "aie2", section)

    with open(path, "rb") as f:
        image = elf.ElfFile(f.read())
    after = [s.name for s in image.sections]
    assert after[: len(before)] == before
    assert "aie2" in after
    assert image.section_by_name("aie2").data == section


@needs_objcopy
def test_a_failed_injection_leaves_the_previous_section_intact(tmp_path, monkeypatch):
    """Objcopy runs on a scratch copy, so a mid-way failure must not strip the file."""
    path = str(tmp_path / "out.hsaco")
    pack.ensure_hsaco(path)
    good = pack.build_section("aie2p", [_kernel("original", b"\x01\x02")])
    pack.inject(path, "aie2p", good)
    before = open(path, "rb").read()

    def fail(cmd, *a, **kw):
        raise subprocess.CalledProcessError(1, cmd, stderr=b"objcopy said no")

    monkeypatch.setattr(pack.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="objcopy said no") as excinfo:
        pack.inject(
            path, "aie2p", pack.build_section("aie2p", [_kernel("new", b"\x03")])
        )
    # capture_output would otherwise swallow the only text explaining why.
    assert "aie2p" in str(excinfo.value)

    assert open(path, "rb").read() == before
    assert dump.read_sections_from_hsaco(path) == [("aie2p", good)]
    assert not [p for p in os.listdir(tmp_path) if ".tmp" in p]


def test_a_rejected_kernel_set_leaves_no_stray_container(tmp_path, capsys):
    """A bad kernel set is bad input: a usage error, not a traceback."""
    path = str(tmp_path / "never.hsaco")
    empty = _write(tmp_path / "empty.bin", b"")
    with pytest.raises(SystemExit) as excinfo:
        pack.main(["--hsaco", path, "--arch", "aie2", "--kernel", f"k:{empty}:64:1"])
    assert excinfo.value.code == 2
    assert "insts must be non-empty" in capsys.readouterr().err
    assert not os.path.exists(path)


@needs_objcopy
def test_reinjection_replaces_rather_than_duplicates(tmp_path):
    path = str(tmp_path / "out.hsaco")
    pack.ensure_hsaco(path)

    pack.inject(path, "aie2p", pack.build_section("aie2p", [_kernel("a", b"\x01")]))
    second = pack.build_section("aie2p", [_kernel("b", b"\x02\x03")])
    pack.inject(path, "aie2p", second)

    with open(path, "rb") as f:
        image = elf.ElfFile(f.read())
    assert [s.name for s in image.sections].count("aie2p") == 1
    assert image.section_by_name("aie2p").data == second
    assert dump.parse_section(second)["kernels"][0]["name"] == "b"


@needs_objcopy
def test_dump_reports_a_damaged_arch_without_hiding_the_intact_one(tmp_path, capsys):
    """Which arch is readable is exactly what the user needs; don't abort on the first."""
    path = str(tmp_path / "partly.hsaco")
    pack.ensure_hsaco(path)
    good = pack.build_section("aie2", [_kernel("survivor", b"\x01\x02")])
    pack.inject(path, "aie2", good)
    broken = bytearray(pack.build_section("aie2p", [_kernel("damaged", b"\x03")]))
    struct.pack_into("<I", broken, 0, 0xDEADBEEF)  # clobber the magic
    pack.inject(path, "aie2p", bytes(broken))

    assert dump.main(["--hsaco", path]) == 1
    out, err = capsys.readouterr()
    assert "kernel survivor:" in out
    assert "aie2p: bad magic" in err


@needs_objcopy
def test_a_failed_injection_leaves_no_stray_container(tmp_path, monkeypatch, capsys):
    """The build-failure path already promises this; the objcopy path must too."""
    path = str(tmp_path / "never.hsaco")
    insts = _write(tmp_path / "insts.bin", b"\x01")

    def fail(cmd, *a, **kw):
        raise subprocess.CalledProcessError(1, cmd, stderr=b"nope")

    monkeypatch.setattr(pack.subprocess, "run", fail)
    with pytest.raises(SystemExit):
        pack.main(
            ["--hsaco", path, "--arch", "aie2"]
            + ["--kernel-name", "k", "--kernel-insts", insts]
        )
    assert "nope" in capsys.readouterr().err
    assert not os.path.exists(path)


def test_dump_main_reports_a_malformed_hsaco_instead_of_crashing(tmp_path, capsys):
    """Reporting on malformed input is this tool's job, not a crash."""
    bare = _write(tmp_path / "bare.hsaco", elf.make_empty_elf64())
    assert dump.main(["--hsaco", bare]) == 1
    assert "no aie2/aie2p section found" in capsys.readouterr().err

    missing = str(tmp_path / "nope.hsaco")
    assert dump.main(["--hsaco", missing]) == 1
    assert "No such file" in capsys.readouterr().err


def test_read_sections_from_hsaco_without_an_arch_section(tmp_path):
    path = _write(tmp_path / "bare.hsaco", elf.make_empty_elf64())
    with pytest.raises(ValueError, match="no aie2/aie2p section found"):
        dump.read_sections_from_hsaco(path)


@needs_objcopy
def test_every_arch_section_is_reported(tmp_path):
    """One hsaco can carry a section per arch; none of them may be hidden."""
    path = str(tmp_path / "both.hsaco")
    pack.ensure_hsaco(path)
    aie2 = pack.build_section("aie2", [_kernel("on_aie2", b"\x01")])
    aie2p = pack.build_section("aie2p", [_kernel("on_aie2p", b"\x02\x03")])
    pack.inject(path, "aie2p", aie2p)
    pack.inject(path, "aie2", aie2)

    # Reported in ARCHES order, whatever order they were injected in.
    assert dump.read_sections_from_hsaco(path) == [("aie2", aie2), ("aie2p", aie2p)]


@needs_objcopy
def test_main_prints_a_block_per_arch_section(tmp_path, capsys):
    path = str(tmp_path / "both.hsaco")
    pack.ensure_hsaco(path)
    pack.inject(path, "aie2", pack.build_section("aie2", [_kernel("on_aie2", b"\x01")]))
    pack.inject(
        path, "aie2p", pack.build_section("aie2p", [_kernel("on_aie2p", b"\x02")])
    )

    assert dump.main(["--hsaco", path]) == 0
    out = capsys.readouterr().out
    assert out.count("arch section:") == 2
    assert "arch section: aie2\n" in out
    assert "arch section: aie2p\n" in out
    assert "kernel on_aie2:" in out
    assert "kernel on_aie2p:" in out


@needs_objcopy
def test_main_packs_a_full_elf_end_to_end(tmp_path, capsys):
    source = _write(tmp_path / "final.elf", make_full_elf([("_Z4mainv", "i0")]))
    hsaco = str(tmp_path / "out.hsaco")

    assert (
        pack.main(["--hsaco", hsaco, "--arch", "aie2p", f"--kernel=elf:{source}:64"])
        == 0
    )
    assert dump.main(["--hsaco", hsaco]) == 0

    out = capsys.readouterr().out
    assert "arch section: aie2p" in out
    assert "kernel main:i0" in out
    assert "kind=FullElf" in out
    assert "kernarg=64" in out


def test_command_names_agree_across_the_three_registrations(tmp_path):
    """A tool's name is asserted in three files with nothing tying them together.

    The launcher filename, the CMake install loop and the wheel console_scripts
    entry must spell the same command, or a source build and a wheel expose
    different names -- silently, since neither install runs the other's path.
    """
    repo = Path(__file__).resolve().parents[2]
    cmake_path = repo / "python" / "CMakeLists.txt"
    setup_path = repo / "utils" / "mlir_aie_wheels" / "setup.py"
    if not (cmake_path.exists() and setup_path.exists()):
        pytest.skip("build-system files are not present outside a source checkout")
    cmake = cmake_path.read_text()
    setup = setup_path.read_text()

    (foreach,) = re.findall(r"foreach\(_hsaco_tool ([^)]*)\)", cmake)
    from_cmake = set(foreach.split())
    from_console = set(
        re.findall(r'"(aie-hsaco[\w-]*) = aie\.compiler\.hsaco\.', setup)
    )
    from_disk = {p.stem for p in (repo / "python" / "compiler").glob("aie-hsaco*.py")}

    assert from_cmake == from_console == from_disk
    assert from_cmake == {"aie-hsaco", "aie-hsaco-dump"}


def test_objcopy_falls_back_to_path(monkeypatch):
    monkeypatch.setattr(pack.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setitem(sys.modules, "aie.utils.config", None)
    assert pack.objcopy_path().endswith("llvm-objcopy")


@needs_objcopy
def test_injected_section_is_not_loaded_into_memory(tmp_path):
    """The arch section is metadata for the loader, not part of the program image."""
    path = str(tmp_path / "out.hsaco")
    pack.ensure_hsaco(path)
    pack.inject(path, "aie2", pack.build_section("aie2", [_kernel("k", b"\x01")]))

    with open(path, "rb") as f:
        section = elf.ElfFile(f.read()).section_by_name("aie2")
    # SHF_ALLOC (0x2) would make the loader map it; --set-section-flags=noload
    # is what keeps it off the image.
    assert section is not None
    assert section.flags & 0x2 == 0
