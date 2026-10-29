# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import http.server
import json
import pathlib
import socketserver
import sys
import tempfile
import threading
import xml.etree.ElementTree as ET

from playwright.sync_api import sync_playwright


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format_string: str, *args: object) -> None:
        pass


def main() -> None:
    dot = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")
    graph = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
    svg = ET.parse(sys.argv[4])

    aliases = [
        node
        for node in graph["nodes"]
        if node.get("semanticId") == "runtime_arg_0"
    ]
    assert [node["id"] for node in aliases] == [
        "runtime_arg_0_col_0",
        "runtime_arg_0_col_1",
    ]
    assert aliases[0]["sources"] == aliases[1]["sources"]
    assert "runtime_arg_0_col_0 -> p_0_0_1_0_m" in dot
    assert "runtime_arg_0_col_1 -> tile_1_2" in dot
    assert "runtime_arg_0 ->" not in dot
    assert dot.count("semantic-id-runtime_arg_0") == 2

    ids = [element.attrib["id"] for element in svg.iter() if "id" in element.attrib]
    assert len(ids) == len(set(ids))

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        (root / "index.html").write_text(
            pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"), encoding="utf-8"
        )
        (root / "design.svg").write_bytes(pathlib.Path(sys.argv[4]).read_bytes())
        source = pathlib.Path(aliases[0]["sources"][0]["file"])
        for element in [*graph["nodes"], *graph["edges"]]:
            for location in element.get("sources", []):
                location["file"] = pathlib.Path(location["file"]).name
        (root / "design.json").write_text(json.dumps(graph), encoding="utf-8")
        (root / source.name).write_bytes(source.read_bytes())

        handler = functools.partial(QuietHandler, directory=directory)
        with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(
                        f"http://127.0.0.1:{server.server_address[1]}/index.html"
                        "?svg=design.svg&json=design.json&source-root=."
                    )
                    first = page.locator("#runtime_arg_0_col_0 ellipse")
                    second = page.locator("#runtime_arg_0_col_1 ellipse")
                    first.dispatch_event("pointerover")
                    assert page.locator("g.semantic-hovered").count() == 2
                    source_line = aliases[0]["sources"][0]["lines"][0]
                    page.locator(
                        f'.source-line.highlighted[data-line="{source_line}"]'
                    ).wait_for()
                    first.dispatch_event(
                        "pointerdown", {"pointerId": 1, "clientX": 100, "clientY": 100}
                    )
                    first.dispatch_event(
                        "pointerup", {"pointerId": 1, "clientX": 100, "clientY": 100}
                    )
                    first.dispatch_event("pointerleave")
                    selected = page.locator("g.semantic-selected").count()
                    assert selected == 2, page.locator(
                        "#runtime_arg_0_col_0, #runtime_arg_0_col_1"
                    ).evaluate_all("elements => elements.map(element => element.className.baseVal)")
                    second.dispatch_event(
                        "pointerdown", {"pointerId": 2, "clientX": 200, "clientY": 100}
                    )
                    second.dispatch_event(
                        "pointerup", {"pointerId": 2, "clientX": 200, "clientY": 100}
                    )
                    assert page.locator("g.semantic-selected").count() == 0
                    browser.close()
            finally:
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    main()
