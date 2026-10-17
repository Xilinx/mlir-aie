#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Build a register database (aie_registers_<arch>.json) from aie-rt's register header.

aie-rt's ``xaie<arch>gbl_params.h`` defines every register of a generation: ``<PREFIX>_<MODULE>_<NAME>`` its offset,
with ``_WIDTH`` and ``_MASK``, and each bit field ``<register>_<FIELD>`` by ``_LSB``, ``_WIDTH``, ``_MASK`` and
``_DEFVAL``. Registers and fields are spelled as an existing database (``--names-from``, the AM025 one) spells them
where that database has the register in the same module, so a register keeps one name across generations; the rest
keep the header's spelling.

Usage:
    python parse_aie_rt_registers.py --header xaie2pgbl_params.h --prefix XAIE2PGBL \
        --names-from aie_registers_aie2.json --source aie-rt@8849e208 --output aie_registers_aie2p.json
"""

import argparse
import json
import re

# The header's modules, as the database groups them: a shim tile is its NoC and PL modules.
MODULES = {
    "CORE_MODULE": "core",
    "MEMORY_MODULE": "memory",
    "MEM_TILE_MODULE": "memory_tile",
    "NOC_MODULE": "shim",
    "PL_MODULE": "shim",
}

DESCRIPTIONS = {
    "core": "Core Module Control and Status Registers",
    "memory": "Memory Module Control and Status Registers",
    "memory_tile": "Memory Tile Control and Status Registers",
    "shim": "Shim Tile Control and Status Registers (NOC and PL Interface)",
}


def parse_header(text: str, prefix: str) -> dict[str, list[dict]]:
    """Return each database module's registers, in header order, with their bit fields."""
    defines = {}
    order = []
    for name, value in re.findall(rf"^#define\s+{prefix}_(\w+)\s+(\S+)", text, re.M):
        defines[name] = value
        order.append(name)
    modules: dict[str, list[dict]] = {m: [] for m in DESCRIPTIONS}
    current = None
    for define in order:
        # A register is its offset with _WIDTH and _MASK beside it, whatever
        # its own name ends in; a field is named only through its suffixed
        # defines, _LSB first.
        if f"{define}_WIDTH" in defines and f"{define}_LSB" not in defines:
            name = define
        elif define.endswith("_LSB"):
            name = define[: -len("_LSB")]
        else:
            continue
        header_module = next((m for m in MODULES if name.startswith(f"{m}_")), None)
        if header_module is None:
            continue
        if f"{name}_LSB" in defines:
            if current is None or not name.startswith(f"{current['_full']}_"):
                raise ValueError(f"bit field {name} outside its register")
            lsb = int(defines[f"{name}_LSB"], 0)
            width = int(defines[f"{name}_WIDTH"], 0)
            current["bit_fields"].append(
                {
                    "name": name[len(current["_full"]) + 1 :],
                    "bits": f"{lsb + width - 1}:{lsb}" if width > 1 else f"{lsb}",
                    "bit_range": [lsb, lsb + width - 1],
                    "reset": hex(int(defines[f"{name}_DEFVAL"], 0)),
                }
            )
        elif f"{name}_WIDTH" in defines:
            current = {
                "_full": name,
                "name": name[len(header_module) + 1 :],
                "offset": f"0x{int(defines[name], 0):010X}",
                "width": int(defines[f"{name}_WIDTH"], 0),
                "bit_fields": [],
            }
            modules[MODULES[header_module]].append(current)
    for registers in modules.values():
        for register in registers:
            del register["_full"]
            reset = sum(
                int(f["reset"], 16) << f["bit_range"][0] for f in register["bit_fields"]
            )
            register["reset"] = f"0x{reset:08X}"
    return modules


def respell(modules: dict[str, list[dict]], reference: dict) -> int:
    """Spell registers and fields as ``reference`` does where it has them; return how many registers took it."""
    taken = 0
    for module, registers in modules.items():
        known = {
            r["name"].upper(): r
            for r in reference["modules"].get(module, {}).get("registers", [])
        }
        seen = set()
        for register in registers:
            ref = known.get(register["name"].upper())
            if ref is None:
                continue
            seen.add(ref["name"])
            register["name"] = ref["name"]
            fields = {f["name"].upper(): f["name"] for f in ref.get("bit_fields", [])}
            for field in register["bit_fields"]:
                field["name"] = fields.get(field["name"].upper(), field["name"])
            taken += 1
    return taken


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--header", required=True, help="aie-rt's xaie<arch>gbl_params.h"
    )
    parser.add_argument(
        "--prefix", required=True, help="its define prefix, such as XAIE2PGBL"
    )
    parser.add_argument(
        "--names-from", required=True, help="a database whose spellings to keep"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="the header's provenance, recorded as the version",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.header) as f:
        modules = parse_header(f.read(), args.prefix)
    with open(args.names_from) as f:
        reference = json.load(f)
    taken = respell(modules, reference)
    database = {
        "version": args.source,
        "source": f"aie-rt {args.prefix} register definitions",
        "source_path": "driver/src/global/" + args.header.rsplit("/", 1)[-1],
        "modules": {
            m: {"base_address_info": {"description": DESCRIPTIONS[m]}, "registers": r}
            for m, r in modules.items()
        },
    }
    with open(args.output, "w") as f:
        # Compact: indented, a generation's database is over a megabyte.
        json.dump(database, f, separators=(",", ":"))
        f.write("\n")
    counts = {m: len(r) for m, r in modules.items()}
    print(f"{args.output}: {counts}, {taken} spelled as {args.names_from}")


if __name__ == "__main__":
    main()
