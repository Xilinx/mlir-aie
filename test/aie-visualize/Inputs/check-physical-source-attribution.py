# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import pathlib
import re
import sys
import xml.etree.ElementTree as ET


def source_lines(element: dict) -> set[int]:
    return {
        line
        for source in element["sources"]
        if source["file"] == "design.py"
        for line in source["lines"]
    }


def main() -> None:
    dot = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    svg = ET.parse(sys.argv[2]).getroot()
    graph = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
    assert graph["format"] == "aie-semantic-graph"
    assert graph["version"] == 2
    semantic = {
        element["id"]: element for element in [*graph["nodes"], *graph["edges"]]
    }
    groups = {
        group.attrib["id"]: group for group in svg.iter() if "id" in group.attrib
    }

    positions = set()
    for argument in sys.argv[4:]:
        semantic_id, expected_line = argument.rsplit("=", 1)
        alias_id = f"physical_alias_{semantic_id}"
        assert alias_id in groups, f"SVG has no group with id {alias_id}"
        classes = set(groups[alias_id].attrib.get("class", "").split())
        assert "physical-alias" in classes
        assert f"semantic-id-{semantic_id}" in classes
        assert semantic_id in semantic, f"JSON has no element with id {semantic_id}"
        assert int(expected_line) in source_lines(semantic[semantic_id]), (
            f"{semantic_id} does not resolve to design.py:{expected_line}"
        )
        match = re.search(
            rf"^\s*{re.escape(alias_id)}\s+\[.*?\bpos=\"([^\"]+)\"",
            dot,
            re.MULTILINE,
        )
        assert match, f"DOT has no fixed position for {alias_id}"
        positions.add(match.group(1))

    assert len(positions) == 1, "physical aliases do not share one SVG coordinate"


if __name__ == "__main__":
    main()
