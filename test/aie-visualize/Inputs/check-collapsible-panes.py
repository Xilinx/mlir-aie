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

from playwright.sync_api import sync_playwright


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format_string: str, *args: object) -> None:
        pass


def rectangles_overlap(first: dict[str, float], second: dict[str, float]) -> bool:
    return not (
        first["x"] + first["width"] <= second["x"]
        or second["x"] + second["width"] <= first["x"]
        or first["y"] + first["height"] <= second["y"]
        or second["y"] + second["height"] <= first["y"]
    )


def main() -> None:
    screenshot_directory = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if screenshot_directory:
        screenshot_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        (root / "index.html").write_text(
            pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"), encoding="utf-8"
        )
        (root / "design.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500">'
            '<g class="graph"><g id="source-node" class="node semantic-id-node-0">'
            '<title>Major topology label</title><rect x="40" y="40" width="300" '
            'height="100" fill="#dceae3"/><text x="60" y="100" font-size="28">'
            'Major topology label</text></g><g id="flow-edge" '
            'class="edge flow src-0 fate-node-0"><path d="M 340 90 L 700 90" '
            'stroke="#247451"/></g></g></svg>',
            encoding="utf-8",
        )
        (root / "design.json").write_text(
            json.dumps(
                {
                    "format": "aie-semantic-graph",
                    "version": 2,
                    "nodes": [
                        {
                            "id": "node-0",
                            "sources": [{"file": "design.mlir", "lines": [2]}],
                        }
                    ],
                    "edges": [],
                }
            ),
            encoding="utf-8",
        )
        (root / "design.mlir").write_text(
            "module {\n  // selected source line\n}\n", encoding="utf-8"
        )

        handler = functools.partial(QuietHandler, directory=directory)
        with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True)
                    page = browser.new_page(viewport={"width": 1440, "height": 900})
                    page.goto(
                        f"http://127.0.0.1:{server.server_address[1]}/index.html"
                        "?svg=design.svg&source-root=."
                    )
                    page.locator("#source-node rect").dispatch_event("pointerover")
                    page.locator('.source-line.highlighted[data-line="2"]').wait_for()
                    page.locator("#auto-scroll").uncheck()
                    active_tab = page.locator(".source-tab.active").text_content()
                    original_view_box = page.locator("#viewport svg").get_attribute(
                        "viewBox"
                    )
                    initial_width = page.locator("#viewport").bounding_box()["width"]

                    page.locator("#toggle-source").click()
                    source_width = page.locator("#viewport").bounding_box()["width"]
                    assert source_width > initial_width + 500
                    assert page.locator("#source-pane").is_hidden()
                    page.locator("#toggle-source").click()
                    assert page.locator(".source-tab.active").text_content() == active_tab
                    assert not page.locator("#auto-scroll").is_checked()

                    page.locator(".flow-row").click()
                    assert page.locator(".flow-row.selected").count() == 1
                    page.locator("#toggle-flow").click()
                    flow_width = page.locator("#viewport").bounding_box()["width"]
                    assert flow_width >= initial_width + 140
                    assert page.locator("#toggle-flow").get_attribute(
                        "aria-expanded"
                    ) == "false"
                    page.locator("#toggle-source").click()
                    expanded_width = page.locator("#viewport").bounding_box()["width"]
                    assert expanded_width >= 1400
                    assert page.locator(".flow-row.selected").count() == 1
                    assert page.locator("#viewport svg").get_attribute(
                        "viewBox"
                    ) == original_view_box
                    page.locator("#reset").click()
                    assert page.locator("#flow-sidebar").is_hidden()
                    assert page.locator("#source-pane").is_hidden()
                    assert page.locator("#viewport svg").get_attribute(
                        "viewBox"
                    ) == original_view_box
                    if screenshot_directory:
                        page.screenshot(
                            path=screenshot_directory / "desktop-collapsed.png"
                        )

                    page.set_viewport_size({"width": 390, "height": 844})
                    controls = [
                        page.locator(selector)
                        for selector in (
                            "#toggle-flow",
                            "#open",
                            "#reset",
                            "#toggle-source",
                        )
                    ]
                    boxes = [control.bounding_box() for control in controls]
                    assert all(control.is_visible() for control in controls)
                    assert all(boxes)
                    assert not any(
                        rectangles_overlap(first, second)
                        for index, first in enumerate(boxes)
                        for second in boxes[index + 1 :]
                    )
                    assert page.locator("#viewport").bounding_box()["width"] >= 380
                    if screenshot_directory:
                        page.screenshot(path=screenshot_directory / "mobile-collapsed.png")

                    page.locator("#toggle-flow").click()
                    page.locator("#toggle-source").click()
                    assert page.locator("#flow-sidebar").is_visible()
                    assert page.locator("#source-pane").is_visible()
                    browser.close()
            finally:
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    main()