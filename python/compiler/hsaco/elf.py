# elf.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

"""Minimal ELF reading and writing for the hsaco tools.

Deliberately hand-rolled with :mod:`struct`: only three narrow things
are needed here - look a section up by name, walk a full AIE ELF's COMDAT
groups, and emit an empty container for the packer to inject into.

Both ELF classes are read, because the tools need both: the hsaco container is
ELF64, while a real AIE full ELF is ELF32 -- which is not incidental, it is the
class ROCr's nested-ELF reader (``core/runtime/amd_aie_elf.cpp``) requires.
Only ELF64 is *written*, by :func:`make_empty_elf64`.
"""

import functools
import struct

# e_ident indices and the values this reader accepts.
_ELF_MAGIC = b"\x7fELF"
_ELFCLASS32 = 1
_ELFCLASS64 = 2
_ELFDATA2LSB = 1

# Section header types used here.
SHT_SYMTAB = 2
SHT_NOBITS = 8
SHT_GROUP = 17

# e_shstrndx escape: the real index lives in section header 0's sh_link.
_SHN_XINDEX = 0xFFFF

# Elf64_Shdr: sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link,
# sh_info, sh_addralign, sh_entsize.
_SHDR = "<IIQQQQIIQQ"
_SHDR_SIZE = struct.calcsize(_SHDR)

# Elf64_Sym: st_name, st_info, st_other, st_shndx, st_value, st_size.
_SYM = "<IBBHQQ"
_SYM_SIZE = struct.calcsize(_SYM)

_EHDR_SIZE = 64

# Elf32_Shdr: the same fields in the same order, 4 bytes each -- so one unpack
# order serves both classes and Section needs no per-class code.
_SHDR32 = "<IIIIIIIIII"

# Elf32_Sym: st_name, st_value, st_size, st_info, st_other, st_shndx. The
# value/size pair moves *ahead* of info/other/shndx, so unlike the section
# header this is a genuine reordering, not just narrower members. Narrowing the
# widths alone would read st_shndx out of what is really st_size.
_SYM32 = "<IIIBBH"

_EHDR32_SIZE = 52

# Per-class layout, keyed by e_ident[EI_CLASS]:
#   (shdr format, sym format, index of st_shndx within the unpacked symbol,
#    file offset of e_shoff, format reading from e_shoff, ELF header size)
# The "10x" in the e_shoff formats skips e_flags, e_ehsize, e_phentsize and
# e_phnum -- 10 bytes in both classes -- to reach e_shentsize/e_shnum/e_shstrndx.
_LAYOUTS = {
    _ELFCLASS32: (_SHDR32, _SYM32, 5, 32, "<I10xHHH", _EHDR32_SIZE),
    _ELFCLASS64: (_SHDR, _SYM, 3, 40, "<Q10xHHH", _EHDR_SIZE),
}


def _cstr(blob, offset, limit=None):
    """Return the NUL-terminated string at ``offset`` in ``blob``.

    ``limit`` bounds the search at the end of the string table the offset is
    supposed to index. Without it an out-of-range offset reads whatever
    follows the table and yields a plausible-looking name rather than an error.

    Raises:
        ValueError: If the offset is out of range, the string is unterminated,
            or it is not valid UTF-8. (UnicodeDecodeError is already a
            ValueError; it is re-raised only to say which offset was at fault.)
    """
    end_limit = len(blob) if limit is None else min(limit, len(blob))
    if offset >= end_limit:
        raise ValueError(f"string offset {offset} out of bounds")
    end = blob.find(b"\x00", offset, end_limit)
    if end == -1:
        raise ValueError(f"unterminated string at offset {offset}")
    try:
        return blob[offset:end].decode()
    except UnicodeDecodeError as e:
        raise ValueError(f"non-UTF-8 string at offset {offset}") from e


class Section:
    """One parsed section header, plus access to its bytes."""

    def __init__(self, elf, index, fields):
        self._elf = elf
        self.index = index
        (
            self.name_offset,
            self.type,
            self.flags,
            self.addr,
            self.offset,
            self.size,
            self.link,
            self.info,
            self.addralign,
            self.entsize,
        ) = fields

    @functools.cached_property
    def name(self):
        """Return the section's name, resolved through the section-name table."""
        return _cstr(
            self._elf.blob,
            self._elf.shstrtab_offset + self.name_offset,
            self._elf.shstrtab_end,
        )

    @property
    def data(self):
        """Return the section's bytes (empty for ``SHT_NOBITS``)."""
        if self.type == SHT_NOBITS:
            return b""
        end = self.offset + self.size
        if end > len(self._elf.blob):
            raise ValueError(f"section {self.index} runs past end of file")
        return self._elf.blob[self.offset : end]


class ElfFile:
    """Read-only view of a little-endian ELF32 or ELF64 image held in memory."""

    def __init__(self, blob):
        # EI_CLASS picks the layout, so it has to be read before the header size
        # it determines can be checked. 6 bytes covers e_ident up to EI_DATA.
        if len(blob) < 6:
            raise ValueError("too small to be an ELF file")
        if blob[:4] != _ELF_MAGIC:
            raise ValueError("not an ELF file (bad magic)")
        layout = _LAYOUTS.get(blob[4])
        if layout is None:
            raise ValueError("not an ELF32 or ELF64 file")
        if blob[5] != _ELFDATA2LSB:
            raise ValueError("not a little-endian ELF file")
        (
            self._shdr,
            self._sym,
            self._sym_shndx_index,
            shoff_offset,
            shoff_format,
            ehdr_size,
        ) = layout
        self._shdr_size = struct.calcsize(self._shdr)
        self._sym_size = struct.calcsize(self._sym)
        if len(blob) < ehdr_size:
            raise ValueError("too small to be an ELF file")
        self.blob = blob

        shoff, shentsize, shnum, shstrndx = struct.unpack_from(
            shoff_format, blob, shoff_offset
        )
        if shoff == 0:
            raise ValueError("ELF has no section headers")
        if shentsize < self._shdr_size:
            raise ValueError("section header entry size too small")
        self._shoff = shoff
        self._shentsize = shentsize

        # Section header 0 carries the extended counts when the 16-bit fields
        # in the ELF header overflow; read it before trusting shnum/shstrndx.
        zero = self._read_shdr(0)
        if shnum == 0:
            shnum = zero[5]
        if shstrndx == _SHN_XINDEX:
            shstrndx = zero[6]
        self.sections = [
            Section(self, i, self._read_shdr(i)) for i in range(int(shnum))
        ]
        # 0 is SHN_UNDEF: no section-name table at all. Accepting it would
        # resolve every name against file offset 0 and yield garbage names,
        # which reads downstream as "section not found" on a file whose
        # section is present.
        if shstrndx == 0:
            raise ValueError("ELF has no section-name table (e_shstrndx is 0)")
        if shstrndx >= len(self.sections):
            raise ValueError("section-name table index out of range")
        shstrtab = self.sections[shstrndx]
        self.shstrtab_offset = shstrtab.offset
        self.shstrtab_end = shstrtab.offset + shstrtab.size

    def _read_shdr(self, index):
        offset = self._shoff + index * self._shentsize
        if offset + self._shdr_size > len(self.blob):
            raise ValueError(f"section header {index} out of bounds")
        return struct.unpack_from(self._shdr, self.blob, offset)

    @functools.cached_property
    def _by_name(self):
        """Map resolvable section names to their first :class:`Section`.

        A section whose own name cannot be resolved is skipped rather than
        raising: an hsaco can carry unrelated sections alongside the AIE one,
        and a single damaged sh_name elsewhere must not make the intact
        section unreachable. First wins, so a duplicate name resolves the way
        a scan in section order would.
        """
        by_name = {}
        for section in self.sections:
            try:
                name = section.name
            except ValueError:
                continue
            by_name.setdefault(name, section)
        return by_name

    def section_by_name(self, name):
        """Return the named :class:`Section`, or ``None`` if absent."""
        return self._by_name.get(name)

    @functools.cached_property
    def _symtab(self):
        """Return ``(section, strtab_offset, strtab_end, entsize)`` for ``.symtab``.

        Raises:
            ValueError: If the file has no symbol table, or its header is
                inconsistent.
        """
        for section in self.sections:
            if section.type != SHT_SYMTAB:
                continue
            if section.link >= len(self.sections):
                raise ValueError(f".symtab sh_link {section.link} out of range")
            entsize = section.entsize or self._sym_size
            # A stride below one symbol would walk the table at a misaligned
            # step and yield overlapping garbage rather than failing.
            if entsize < self._sym_size:
                raise ValueError(f".symtab sh_entsize {entsize} too small")
            # symbol_at() unpacks straight out of the image, bypassing
            # Section.data's own extent check, so the extent is checked here.
            if section.offset + section.size > len(self.blob):
                raise ValueError(".symtab runs past end of file")
            strtab = self.sections[section.link]
            return section, strtab.offset, strtab.offset + strtab.size, entsize
        raise ValueError("missing .symtab")

    def symbol_at(self, index):
        """Return the ``(name, shndx)`` of one ``.symtab`` entry.

        Decodes just the entry asked for, straight out of the image. Callers
        want a handful of symbols out of a table that can hold many thousands,
        so materializing the whole table would dominate the work.

        Raises:
            ValueError: If ``index`` is past the end of the symbol table.
        """
        section, strtab_offset, strtab_end, entsize = self._symtab
        if index >= section.size // entsize:
            raise ValueError(f"symbol index {index} out of range")
        # Indexed rather than destructured: st_shndx sits at a different
        # position in Elf32_Sym than in Elf64_Sym, so one tuple shape does not
        # describe both.
        fields = struct.unpack_from(
            self._sym, self.blob, section.offset + index * entsize
        )
        st_name, st_shndx = fields[0], fields[self._sym_shndx_index]
        return _cstr(self.blob, strtab_offset + st_name, strtab_end), st_shndx

    def group_signature_indices(self):
        """Return the symbol index signing each ``SHT_GROUP`` section, in order."""
        return [s.info for s in self.sections if s.type == SHT_GROUP]


def demangle_kernel_name(symbol):
    """Return the unqualified name in an Itanium-mangled symbol.

    ``'_Z4mainPcPcPc'`` becomes ``'main'``. Anything that is not a mangled
    name with a leading length-prefixed identifier is returned unchanged.

    Args:
        symbol (str): The possibly-mangled symbol name.

    Returns:
        str: The demangled kernel name, or ``symbol`` unchanged.
    """
    if not symbol.startswith("_Z"):
        return symbol
    i, length = 2, 0
    while i < len(symbol) and symbol[i].isdigit():
        length = length * 10 + int(symbol[i])
        i += 1
    if length == 0 or i + length > len(symbol):
        return symbol
    return symbol[i : i + length]


def kernel_names_from_full_elf(blob):
    """Return the ``kernel:instance`` names of every COMDAT group in a full ELF.

    Each group's ``sh_info`` names the *instance* symbol that signs it. That
    symbol's ``st_shndx`` is in turn read as an index back into the symbol
    table to reach the *kernel* symbol -- this is how the AIE full-ELF producer
    encodes the pairing, and is not the usual meaning of ``st_shndx``. It
    mirrors what ROCr's own packer does; changing it here would desynchronise
    the two.

    Args:
        blob (bytes): The whole ELF image.

    Returns:
        list[str]: Sorted ``kernel:instance`` names.

    Raises:
        ValueError: If the ELF contains no COMDAT groups.
    """
    elf = ElfFile(blob)
    names = []
    for signature_index in elf.group_signature_indices():
        instance_name, instance_shndx = elf.symbol_at(signature_index)
        kernel_name, _ = elf.symbol_at(instance_shndx)
        names.append(f"{demangle_kernel_name(kernel_name)}:{instance_name}")

    if not names:
        raise ValueError("no COMDAT groups; not a full ELF")
    return sorted(names)


def make_empty_elf64():
    """Return a minimal ELF64 relocatable with a NULL section and a ``.shstrtab``.

    ``llvm-objcopy`` needs a real ``.shstrtab`` to write new section names
    into. An ELF with ``e_shnum == 0`` has nowhere to put one, so
    ``--add-section`` silently no-ops on it instead of failing.
    """
    shstrtab = b"\x00.shstrtab\x00"
    # Elf64_Shdr has 8-byte members, so the section header table has to start
    # 8-byte aligned or a reader that maps the image and casts to Elf64_Shdr*
    # performs misaligned loads.
    shstrtab += b"\x00" * (-(_EHDR_SIZE + len(shstrtab)) % 8)
    shstrtab_off = _EHDR_SIZE
    shoff = shstrtab_off + len(shstrtab)
    null_sh = struct.pack(_SHDR, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    shstrtab_sh = struct.pack(
        _SHDR, 1, 3, 0, 0, shstrtab_off, len(shstrtab), 0, 0, 1, 0
    )
    ehdr = (
        _ELF_MAGIC
        + bytes([_ELFCLASS64, _ELFDATA2LSB, 1, 0])
        + b"\x00" * 8  # rest of e_ident
        + struct.pack(
            "<HHIQQQIHHHHHH",
            1,  # e_type = ET_REL
            224,  # e_machine = EM_AMDGPU
            1,  # e_version
            0,  # e_entry
            0,  # e_phoff
            shoff,  # e_shoff
            0,  # e_flags
            _EHDR_SIZE,  # e_ehsize
            0,  # e_phentsize
            0,  # e_phnum
            _SHDR_SIZE,  # e_shentsize
            2,  # e_shnum
            1,  # e_shstrndx
        )
    )
    return ehdr + shstrtab + null_sh + shstrtab_sh
