# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import pathlib
import re
import struct
import subprocess
import sys
import tempfile
import unittest

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


def fixture():
    names = b"\0.shstrtab\0.pdi.1\0.note.xrt.UID\0"
    data = bytearray(128)
    ident = b"\x7fELF\x01\x01\x01\x45\x10" + bytes(7)
    struct.pack_into(
        "<16sHHIIIIIHHHHHH",
        data,
        0,
        ident,
        2,
        1,
        1,
        0,
        52,
        128,
        0,
        52,
        32,
        1,
        40,
        4,
        1,
    )
    struct.pack_into("<8I", data, 52, 1, 96, 0, 0, 16, 16, 5, 16)
    data[96:112] = b"device payload!!"
    data[112:128] = bytes(range(16))
    headers = [
        (0,) * 10,
        (1, 3, 0, 0, 288, len(names), 0, 0, 1, 0),
        (11, 1, 6, 0, 96, 16, 0, 0, 16, 0),
        (18, 7, 0, 0, 112, 16, 0, 0, 4, 0),
    ]
    for header in headers:
        data.extend(struct.pack("<10I", *header))
    data.extend(names)
    return data


class ProvenanceTests(unittest.TestCase):
    def test_preserves_payloads_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "input.elf"
            source.write_bytes(fixture())
            outputs = []
            for i in range(2):
                target = root / f"output{i}.elf"
                subprocess.run([TOOL, source, target], check=True)
                outputs.append(target.read_bytes())
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(
                compare(source.read_bytes(), outputs[0])["producer"], "mlir-aie/aiecc"
            )
            duplicate = subprocess.run(
                [TOOL, root / "output0.elf", root / "duplicate.elf"],
                capture_output=True,
            )
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn(b"already exists", duplicate.stderr)

    def test_rejects_malformed_elf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            bad = [b"", b"\x7fELF", b"not an elf", fixture()[:128]]
            big_endian = fixture()
            big_endian[5] = 2
            bad.append(big_endian)
            for data in bad:
                source = root / "bad.elf"
                source.write_bytes(data)
                result = subprocess.run(
                    [TOOL, source, root / "out"], capture_output=True
                )
                self.assertNotEqual(result.returncode, 0)

    def test_selected_compiler_and_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            compiler = pathlib.Path(directory) / "clang"
            for version, revision in [("22.0.0", "a" * 40), ("23.0.0git", "b" * 40)]:
                compiler.write_text(
                    "#!/bin/sh\n"
                    f"echo 'clang version {version} (https://github.com/Xilinx/llvm-aie {revision})'\n"
                    "echo 'InstalledDir: /private/build/path'\n"
                )
                compiler.chmod(0o755)
                result = json.loads(
                    subprocess.check_output([TOOL, "--compiler", "llvm-aie", compiler])
                )
                self.assertEqual(
                    result,
                    {"name": "llvm-aie", "version": version, "revision": revision},
                )
            result = json.loads(
                subprocess.check_output([TOOL, "--compiler", "llvm-aie", ""])
            )
            self.assertIsNone(result["version"])
            self.assertIsNone(result["revision"])
            compiler.write_text(
                "#!/bin/sh\n"
                "echo 'chesscc version V-2024.06#84922c0d9f#241219 built on 20-Dec-2024 at 09:00:49'\n"
            )
            result = json.loads(
                subprocess.check_output([TOOL, "--compiler", "chess", compiler])
            )
            self.assertEqual(
                result,
                {"name": "chess", "version": "V-2024.06", "revision": "84922c0d9f"},
            )


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "unit":
        TOOL = args[0]
        unittest.main(argv=[sys.argv[0]])
    elif mode == "compare":
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
