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


def main() -> None:
    screenshot_directory = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if screenshot_directory:
        screenshot_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        (root / "index.html").write_text(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"), encoding="utf-8")
        flows = "".join(
            f'<g id="flow-{index}" class="edge flow authored-{index} src-{index} fate-destination" '
            f'opacity="{0.8 - index * 0.1}">'
            f'<path d="M 190 {100 + index * 18} L 700 {100 + index * 18}" '
            f'stroke="{color}" stroke-width="{width}"/></g>'
            for index, (color, width) in enumerate((("#b2182b", 4), ("#2166ac", 3), ("#1b7837", 2), ("#762a83", 3)))
        )
        (root / "design.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500">'
            '<g class="graph uniform-width-1"><g id="source" class="node">'
            '<rect x="40" y="70" width="150" height="120" fill="#dceae3"/>'
            '<text x="65" y="135" font-size="24">Sources</text></g>'
            f'{flows}<g id="destination" class="node semantic-id-destination">'
            '<rect x="700" y="70" width="70" height="120" fill="#fff0a8"/>'
            '<text x="712" y="135" font-size="18">Tile</text></g></g></svg>',
            encoding="utf-8",
        )
        (root / "design.json").write_text(
            json.dumps(
                {
                    "format": "aie-semantic-graph",
                    "version": 2,
                    "nodes": [
                        {
                            "id": "destination",
                            "sources": [{"file": "design.mlir", "lines": [2]}],
                        }
                    ],
                    "edges": [],
                }
            ),
            encoding="utf-8",
        )
        (root / "design.mlir").write_text("module {\n  aie.tile(0, 2)\n}\n", encoding="utf-8")

        handler = functools.partial(QuietHandler, directory=directory)
        with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True)
                    page = browser.new_page(viewport={"width": 1440, "height": 900})
                    page.set_default_timeout(5000)
                    page.goto(f"http://127.0.0.1:{server.server_address[1]}/index.html?svg=design.svg&source-root=.")
                    routes = page.locator("#viewport svg g.flow")
                    paths = page.locator("#viewport svg g.flow path")
                    original_paths = paths.evaluate_all("elements => elements.map(element => element.getAttribute('d'))")
                    original_classes = routes.evaluate_all("elements => elements.map(element => element.getAttribute('class'))")
                    original_opacities = routes.evaluate_all("elements => elements.map(element => element.getAttribute('opacity'))")

                    def assert_authored_opacity() -> None:
                        rendered = routes.evaluate_all(
                            "elements => elements.map(element => getComputedStyle(element).opacity)"
                        )
                        assert all(
                            abs(float(value) - float(authored)) < 1e-6
                            for value, authored in zip(rendered, original_opacities)
                        ), f"rendered opacity {rendered}, authored opacity {original_opacities}"
                    assert routes.count() == 4
                    assert all(
                        float(opacity) == 0.12 for opacity in routes.evaluate_all("elements => elements.map(element => getComputedStyle(element).opacity)")
                    )
                    assert set(paths.evaluate_all("elements => elements.map(element => getComputedStyle(element).stroke)")) == {"rgb(102, 115, 109)"}
                    assert set(paths.evaluate_all("elements => elements.map(element => getComputedStyle(element).strokeWidth)")) == {"1px"}
                    checkbox = page.get_by_role("checkbox", name="Show all routes", exact=True)
                    assert not checkbox.is_checked()
                    checkbox.focus()
                    page.keyboard.press("Space")
                    assert checkbox.is_checked()
                    assert_authored_opacity()
                    assert routes.evaluate_all("elements => elements.map(element => element.getAttribute('class'))") == original_classes
                    assert routes.evaluate_all("elements => elements.map(element => element.getAttribute('opacity'))") == original_opacities
                    if screenshot_directory:
                        page.screenshot(path=screenshot_directory / "focus-show-all.png")
                    page.keyboard.press("Space")
                    assert not checkbox.is_checked()

                    page.locator("#destination rect").dispatch_event("pointerover")
                    page.locator('.source-line.highlighted[data-line="2"]').wait_for()
                    page.locator("#destination rect").dispatch_event("pointerleave")
                    page.locator(".flow-row").first.click()
                    assert page.locator(".flow-row.selected").count() == 1
                    assert page.locator("#viewport svg.highlighting").count() == 1
                    assert_authored_opacity()
                    assert paths.evaluate_all("elements => elements.map(element => getComputedStyle(element).stroke)") == [
                        "rgb(178, 24, 43)",
                        "rgb(33, 102, 172)",
                        "rgb(27, 120, 55)",
                        "rgb(118, 42, 131)",
                    ]
                    assert paths.evaluate_all("elements => elements.map(element => getComputedStyle(element).strokeWidth)") == ["4px", "3px", "2px", "3px"]
                    page.locator("#destination rect").dispatch_event("pointerover")
                    assert page.locator(".flow-row.selected").count() == 1
                    assert paths.evaluate_all("elements => elements.map(element => element.getAttribute('d'))") == original_paths
                    page.locator("#reset").click()
                    assert page.locator(".flow-row.selected").count() == 0
                    assert page.locator("#viewport svg.route-focus").count() == 1
                    assert not checkbox.is_checked()
                    assert page.get_by_role("region", name="Source code").is_visible()
                    if screenshot_directory:
                        page.screenshot(path=screenshot_directory / "focus-reset.png")
                        page.locator(".flow-row").first.click()
                        page.locator("#viewport svg").dispatch_event("wheel", {"deltaY": -300, "clientX": 700, "clientY": 300})
                        page.screenshot(path=screenshot_directory / "focus-selected.png")

                    page.locator("#reset").click()
                    page.emulate_media(media="print")
                    assert all(float(opacity) == 0.12 for opacity in routes.evaluate_all("elements => elements.map(element => getComputedStyle(element).opacity)"))
                    page.emulate_media(media="screen")
                    page.set_viewport_size({"width": 390, "height": 844})
                    assert page.get_by_role("checkbox", name="Show all routes", exact=True).is_visible()
                    page.get_by_role("checkbox", name="Show all routes", exact=True).check()
                    assert_authored_opacity()
                    assert page.get_by_role("region", name="Source code").is_visible()
                    if screenshot_directory:
                        page.screenshot(path=screenshot_directory / "focus-mobile-show-all.png")

                    physical_svg = (
                        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500">'
                        '<g class="graph"><g id="physical-track" class="edge physical-track used">'
                        '<path d="M 100 100 L 700 100" stroke="#c51b32" stroke-width="1.6"/>'
                        '</g></g></svg>'
                    )
                    page.evaluate("svg => showSVG(svg, 'physical.svg')", physical_svg)
                    assert page.locator("#viewport svg.route-focus").count() == 1
                    physical_path = page.locator("#physical-track path")
                    assert physical_path.evaluate("element => getComputedStyle(element).stroke") == "rgb(197, 27, 50)"
                    assert physical_path.evaluate("element => getComputedStyle(element).opacity") == "1"
                    assert page.get_by_role("checkbox", name="Show all routes", exact=True).is_checked() is False
                    browser.close()
            finally:
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    main()
