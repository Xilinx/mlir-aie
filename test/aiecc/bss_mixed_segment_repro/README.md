# aie-rt ELF loader out-of-bounds read — minimal reproducer

`aie-rt`'s `_XAie_LoadDataMemSection`
(`driver/src/core/xaie_elfloader.c`) writes `p_memsz` bytes starting from a
segment's file offset, but only substitutes a zeroed buffer when
`p_filesz == 0`:

```c
SectionSize = Phdr->p_memsz;
if (Phdr->p_filesz == 0U) Buffer = calloc(Phdr->p_memsz, sizeof(char));
while (SectionSize > 0U) { ...Write(..., Buffer, BytesToWrite); Buffer += BytesToWrite; }
```

For the ordinary **mixed** `data`+`bss` segment (`0 < p_filesz < p_memsz`) it
therefore reads `p_memsz - p_filesz` bytes past the segment's file contents and
DMAs whatever follows in the ELF image (`.comment`, `.symtab`, …) into the tile.

Per the System V gABI (`man 5 elf`, `PT_LOAD`): *"If the segment's memory size
`p_memsz` is larger than the file size `p_filesz`, the 'extra' bytes are defined
to hold the value 0."* Zeroing that gap is required of the loader.

## Run

```bash
export PEANO_INSTALL_DIR=/path/to/llvm-aie
./run.sh
```

Expected:

```
--- bug ---
    segment: FileSiz 0x01000  MemSiz 0x01200
    ELF .comment strings baked into the CDO: 1
    >>> BUG: ...
--- control ---
    segment: FileSiz 0x00000  MemSiz 0x00200
    ELF .comment strings baked into the CDO: 0
    OK: no metadata leaked.
```

No hardware needed — the defect is visible in the CDO. To see the exact bytes
that will be written where `zero_state[512]` should be all zeros:

```python
elf = open('work_bug/elfs_main_core_0_2/elfs_main_core_0_2.elf','rb').read()
print(elf[0x12b0:0x12b0+0x200][:56])
# b'\x00Linker: LLD 21.0.0 (https://github.com/Xilinx/llvm-aie '
```

## Fix

Split the write at `p_filesz`: copy the real bytes, then write zeros for the
remaining `p_memsz - p_filesz`.

## Real-world impact

In a Gemma4 decode layer this silently inverted a kernel's ping/pong selector
(`static bool is_ksc_ping` received `'k'` from the string `"Linker"`), so every
call read the wrong buffer — correct arithmetic on stale data. See
`docs/AIE_RT_ELF_LOADER_BUG.md`.
