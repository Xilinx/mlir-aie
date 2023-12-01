import json
import sys
from pathlib import Path

from elftools.elf.elffile import ELFFile


def process_file(filename):
    p = Path(filename)
    print(p.parent.name, p.name)
    with open(filename, "rb") as f:
        elffile = ELFFile(f)

        assert elffile.structs.e_ident_osabi == "ELFOSABI_LINUX"
        # assert elffile.structs.e_machine == "EM_AMDGPU"
        assert elffile.structs.e_type == "ET_NONE"
        assert elffile.structs.elfclass == 64
        assert elffile.structs.little_endian

        print(json.dumps(dict(elffile.header.e_ident), indent=2))

        for i in range(elffile.num_sections()):
            section = elffile.get_section(i)
            if not section.is_null():
                print(f"{section.name=}")


if __name__ == "__main__":
    process_file(sys.argv[1])
