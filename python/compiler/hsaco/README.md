<!---//===- README.md ---------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# hsaco packer

Packs compiled AIE kernels into a **hsaco** (an HSA code object), so ROCR
can load an NPU kernel through the same machinery it uses for GPU kernels. This
is a separate post-processing step: compile a design first with `aiecc`, then
run this tool over the artifacts it emitted.

The counterpart to this tool is the HSA/ROCR host runtime at
[`python/utils/hostruntime/hsaruntime`](../../utils/hostruntime/hsaruntime/README.md),
which dispatches the packed kernels.

## Commands

| command | does |
|---|---|
| `aie-hsaco` | packs kernels into an hsaco (creating it if absent) |
| `aie-hsaco-dump` | parses and validates every arch section in an existing hsaco |

Both are installed into `bin/` and are also importable as
`aie.compiler.hsaco.pack` / `aie.compiler.hsaco.dump`.

## Usage

```
aie-hsaco --hsaco OUT.hsaco --arch {aie2,aie2p} --kernel SPEC [--kernel SPEC ...]
```

`--hsaco` is created as an empty AMDGPU relocatable if it does not exist, so
packing into a fresh file and adding to an existing code object are the same
command. Repacking the same arch replaces the previous section rather than
appending a second one.

`--kernel` is repeatable and takes one of three forms. Every form also has a
delimiter-free spelling — see [Long-form options](#long-form-options-paths-containing-a-colon),
which is what you need if any path contains a colon.

### 1. PDI + instruction stream

```
NAME:INSTS[:PDI]:KERNARG_SIZE:NUM_COLS
```

The direct form: `INSTS` is the raw `insts.bin` and `PDI` the `main.pdi` that
`aiecc` emitted next to the xclbin. The PDI is optional — omit it for a design
whose instruction stream is self-contained.

```bash
aie-hsaco --hsaco vector_add.hsaco --arch aie2p \
  --kernel 'MLIR_AIE:build/insts.bin:build/main.pdi:64:1'
```

### 2. xclbin + instruction stream

```
xclbin:NAME:XCLBIN:INSTS:KERNARG_SIZE:NUM_COLS
```

Same as above, but the PDI is extracted from the xclbin's `AIE_PARTITION`
section instead of being passed directly. All six fields are required.

```bash
aie-hsaco --hsaco vector_add.hsaco --arch aie2p \
  --kernel 'xclbin:MLIR_AIE:build/final.xclbin:build/insts.bin:64:1'
```

This shells out to `xclbinutil`, resolved from mlir-aie's own `bin/` first
(built from `tools/hrx-xclbinutil`, which needs no system XRT install) and
then from `PATH`. The xclbin must contain exactly one PDI.

### 3. Full ELF

```
elf:PATH[:KERNARG_SIZE[:NUM_COLS]]
```

A full ELF is self-contained — it carries its own control code, so there is no
separate PDI. The tool reads the ELF's COMDAT groups and emits one kernel entry
per group, named `kernel:instance`. Every entry embeds the same ELF image, and
the section's blob pool stores it once.

```bash
aie-hsaco --hsaco vector_add.hsaco --arch aie2p --kernel 'elf:build/final.elf:64:1'
```

`KERNARG_SIZE` and `NUM_COLS` default to `0` and `1`.

### Two limits of the colon grammar

It is inherited verbatim from ROCR's packer, so it is kept as-is rather than
fixed, and the long-form options below avoid both problems:

- **A path containing a colon cannot be expressed** — the fields are split on
  `:` with no escaping, so every absolute Windows path misparses.
- **A kernel cannot be named `elf` or `xclbin`** — the first field is tested as
  a form tag before the field count disambiguates, so
  `--kernel 'elf:insts.bin:64:1'` is read as a full-ELF spec, not as a kernel
  named `elf`. The error then points at your insts file as a malformed ELF.

### Long-form options (paths containing a colon)

The colon-separated grammar above is inherited from ROCR's packer and has no
escaping, so it cannot express a path that itself contains a colon — which
means **every absolute Windows path**. `--kernel 'k:C:\build\insts.bin:64:1'`
splits into the wrong fields and fails with `No such file or directory: 'C'`.

Each field therefore also has a long-form option. A kernel *starts* at
`--kernel-name` (forms 1 and 2) or `--kernel-elf` (form 3) and absorbs the
`--kernel-*` options that follow it, so the group can be repeated for several
kernels and freely mixed with `--kernel`.

| option | field |
|---|---|
| `--kernel-name NAME` | starts a PDI+insts kernel |
| `--kernel-elf PATH` | starts a self-contained full ELF |
| `--kernel-insts PATH` | instruction stream (`insts.bin`) |
| `--kernel-pdi PATH` | PDI (`main.pdi`) |
| `--kernel-xclbin PATH` | xclbin to extract the PDI from |
| `--kernel-kernarg N` | kernarg buffer size (default `0`) |
| `--kernel-cols N` | column count (default `1`) |

```bash
aie-hsaco --hsaco vector_add.hsaco --arch aie2p \
  --kernel-name MLIR_AIE \
  --kernel-insts 'C:\build\insts.bin' \
  --kernel-pdi   'C:\build\main.pdi' \
  --kernel-kernarg 64 --kernel-cols 1
```

Two kernels and a full ELF in one invocation:

```bash
aie-hsaco --hsaco design.hsaco --arch aie2 \
  --kernel-name first  --kernel-insts build/a.bin --kernel-cols 2 \
  --kernel-name second --kernel-insts build/b.bin --kernel-kernarg 32 \
  --kernel-elf build/final.elf
```

Nothing is inherited between groups: each kernel's `--kernel-kernarg` and
`--kernel-cols` fall back to their defaults, not to the previous kernel's
values. Giving the same option twice within one group, using `--kernel-pdi`
and `--kernel-xclbin` together, or attaching anything to `--kernel-elf` (which
is self-contained) is rejected with a usage error.

### Inspecting the result

```bash
aie-hsaco-dump --hsaco vector_add.hsaco
```

```
arch section: aie2
version: 1.0
  kernel MLIR_AIE: kind=PdiInsts insts=2304B pdi=yes kernarg=64 cols=1
arch section: aie2p
version: 1.0
  kernel main:instance0: kind=FullElf insts=41216B pdi=no kernarg=64 cols=1
```

One hsaco can carry a section per architecture -- ROCR selects the one matching
the running device -- so every arch is packed into the same file by repeating
`aie-hsaco` with a different `--arch`, and the dump prints a block for each.

## ELF structure

Two different ELFs are involved, and it is easy to conflate them: the **hsaco**
this tool writes, and the **full ELF** that input form 3 reads.

### The hsaco container

The hsaco is an ELF64 relocatable (`ET_REL`) with `e_machine = EM_AMDGPU`. When
`--hsaco` names a file that does not exist, `ensure_hsaco` synthesizes the
smallest container that can be injected into — a header, a NULL section, and a
`.shstrtab`:

```
  [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
  [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
  [ 1] .shstrtab         STRTAB          0000000000000000 000040 00000b 00      0   0  1
```

The `.shstrtab` is not optional padding. `llvm-objcopy` needs a real
section-name table to write a new section's name into; an ELF with
`e_shnum == 0` has nowhere to put one, so `--add-section` silently no-ops
instead of failing.

Each `aie-hsaco` run adds one `PROGBITS` section named after its `--arch`.
Packing two architectures into one file gives:

```
  [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
  [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
  [ 1] .shstrtab         STRTAB          0000000000000000 000040 000016 00      0   0  1
  [ 2] aie2p             PROGBITS        0000000000000000 000056 000331 00      0   0  1
  [ 3] aie2              PROGBITS        0000000000000000 000387 0000d5 00      0   0  1
```

The `Flg` column is empty on both, and that is deliberate: the sections are
written `noload,readonly`, so `SHF_ALLOC` is clear and the loader does not map
them into the program image. They are metadata ROCR reads, not code. Sections
appear in injection order; `aie-hsaco-dump` reports them in `ARCHES` order
regardless.

Because the payload is an ordinary ELF section, an hsaco that already carries
GPU code objects can be packed into as-is — the AIE section rides alongside
whatever else is in the file.

### The full-ELF input

A full ELF is a self-contained AIE program, and it is an **ELF32** — not
incidentally, but because that is the class ROCR's nested-ELF reader requires
(`core/runtime/amd_aie_elf.cpp` rejects anything but `ELFCLASS32`). So the two
ELFs in play differ in class: an ELF32 payload embedded in an ELF64 container.
`elf.py` reads both.

The packer does not interpret the payload's contents; it only enumerates the
kernels inside, from the COMDAT groups:

```
COMDAT group section [    4] `.group' [inst_a] contains 0 sections:
COMDAT group section [    5] `.group' [inst_b] contains 0 sections:

Symbol table '.symtab' contains 5 entries:
   Num:    Value          Size Type    Bind   Vis       Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
     1: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND _Z6vecaddPiS_
     2: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT     1 inst_a
     3: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND _Z6vecmulPiS_
     4: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT     3 inst_b
```

Each `SHT_GROUP` section's `sh_info` is the index of the *instance* symbol that
signs it (`inst_a` at index 2, `inst_b` at index 4). **That instance symbol's
`st_shndx` is then read as an index back into the symbol table**, not as a
section index, to reach the *kernel* symbol — `inst_a`'s `Ndx` of 1 means
symbol 1, `_Z6vecaddPiS_`. Note that `readelf` prints that column as a section
index, so the dump above is misleading unless you know the convention.

This is not standard ELF; it is how the AIE full-ELF producer encodes the
pairing, and `elf.kernel_names_from_full_elf` mirrors ROCR's own packer rather
than "correcting" it. The kernel name is demangled and joined to the instance
name, giving `vecadd:inst_a` and `vecmul:inst_b`.

## Section layout

The section is named after the architecture (`aie2` or `aie2p`) and is laid out
as a versioned header, a kernel table, a string table, and a blob pool:

```
[header][kernel table][string table][blob pool]
```

Blobs are deduplicated on their raw bytes, and every offset in the table is
absolute within the section. `aie-hsaco-dump` bounds-checks all of them, so a
truncated section fails loudly rather than parsing into plausible garbage.

## Keeping the layout in sync

`format.py` is a hand-maintained mirror of ROCr's `core/inc/amd_aie_section.h`
(in the rocm-systems repo, under `projects/rocr-runtime/runtime/hsa-runtime/`).
**The C++ header is authoritative.** Because the two now live in separate
repositories the mirror can drift silently, so `parse_section` rejects a
section whose `version_major` it does not recognise instead of misreading it.
When a field is added or a reserved word repurposed on the ROCr side, update
`format.py` to match and bump the version there.

The header is self-describing — it carries its own `header_size` and
`kernel_entry_size`, and the reader indexes the kernel table using those rather
than its own idea of the sizes. So **additive** growth (new trailing fields, a
reserved word given a meaning) is readable by an older tool and only needs a
`version_minor` bump. Anything an older reader would misinterpret needs
`version_major`, and that includes **adding a new `AieKernelKind`**: the reader
rejects a `kind` it does not know rather than guessing, so a new kind is a
breaking change to every tool that predates it.

## Dependencies

- `llvm-objcopy` — injects the section. Resolved via
  `aie.utils.config.objcopy_path()` (honouring `AIE_OBJCOPY_PATH` and the
  wheel-bundled copy), falling back to `PATH`.
- `xclbinutil` — only for input form 2.

ELF reading and writing is done directly with `struct` in `elf.py`.
