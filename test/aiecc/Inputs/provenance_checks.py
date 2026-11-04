# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import pathlib
import re
import struct
import sys

NOTE = ".note.mlir-aie.provenance"


def sections(data):
    offset = struct.unpack_from("<I", data, 32)[0]
    size, count, names_index = struct.unpack_from("<HHH", data, 46)
    headers = [
        struct.unpack_from("<10I", data, offset + size * i) for i in range(count)
    ]
    h = headers[names_index]
    names = data[h[4] : h[4] + h[5]]
    return [(names[h[0] :].split(b"\0")[0].decode(), h) for h in headers]


def provenance(data):
    _, h = next(s for s in sections(data) if s[0] == NOTE)
    assert h[1] == 7 and h[2] == 0 and h[8] == 4, h
    payload = data[h[4] : h[4] + h[5]]
    namesz, descsz, kind = struct.unpack_from("<III", payload)
    assert kind == 1 and payload[12 : 12 + namesz] == b"MLIR-AIE\0"
    start = 12 + ((namesz + 3) & ~3)
    description = payload[start : start + descsz]
    assert description.endswith(b"\0")
    return json.loads(description[:-1])


def compare(before, after):
    original = bytearray(before[:52])
    modified = bytearray(after[:52])
    for region in [slice(32, 36), slice(48, 50)]:
        original[region] = modified[region]
    assert original == modified, "unrelated ELF header fields changed"
    old_sections, new_sections = sections(before), sections(after)
    assert len(new_sections) == len(old_sections) + 1
    for (name, old), (new_name, new) in zip(old_sections, new_sections):
        assert name == new_name
        if name == ".shstrtab":
            continue
        assert old == new, name
        if old[1] != 8:
            assert (
                before[old[4] : old[4] + old[5]] == after[new[4] : new[4] + new[5]]
            ), name
    assert before[52:] == after[52 : len(before)], "assembled bytes moved"
    return provenance(after)


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "compare":
        info = compare(*(pathlib.Path(p).read_bytes() for p in args))
        assert info["producer"] == "mlir-aie/aiecc"
        assert info["schema"] == 1
        assert info["revision"] == "unknown" or re.fullmatch(
            r"[0-9a-f]{40}", info["revision"]
        )
        assert info["dirty"] in (True, False, None)
        assert info["devices"]
        assert info["core_compiler"]["name"] == "llvm-aie"
        assert info["core_compiler"]["version"]
    elif mode == "xclbin":
        info = provenance(pathlib.Path(args[0]).read_bytes())
        metadata = json.loads(pathlib.Path(args[1]).read_text())
        entries = {
            e["key"]: e["value"] for e in metadata["keyvalue_metadata"]["key_values"]
        }
        assert json.loads(entries["mlir_aie.provenance"]) == info
        if len(args) > 2:
            assert entries["unrelated"] == "keep"
    else:
        raise ValueError(mode)
