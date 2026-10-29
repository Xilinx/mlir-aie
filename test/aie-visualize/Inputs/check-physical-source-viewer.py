# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import http.server
import pathlib
import socketserver
import sys
import tempfile
import threading
import xml.etree.ElementTree as ET

from playwright.sync_api import sync_playwright


SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


def write_alias_fixture(source: pathlib.Path, destination: pathlib.Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    alias = next(
        element
        for element in root.iter()
        if element.attrib.get("id") == "physical_alias_p_0_2_0_0_k3"
    )
    title = next((child for child in alias if child.tag == f"{{{SVG}}}title"), None)
    for child in list(alias):
        if child is not title:
            alias.remove(child)
    alias.attrib["transform"] = "translate(100 100)"
    ET.SubElement(alias, f"{{{SVG}}}path", {"d": "M -6 -6 L 6 -6 L 0 6 Z"})

    duplicate = ET.fromstring(ET.tostring(alias))
    duplicate.attrib["id"] = "physical_alias_p_0_2_0_0_k3_rect"
    for child in list(duplicate):
        if child.tag != f"{{{SVG}}}title":
            duplicate.remove(child)
    ET.SubElement(
        duplicate,
        f"{{{SVG}}}rect",
        {"x": "-6", "y": "-6", "width": "12", "height": "12"},
    )
    parent = next(element for element in root.iter() if alias in list(element))
    parent.append(duplicate)

    polygon = ET.fromstring(ET.tostring(alias))
    polygon.attrib["id"] = "physical_alias_p_0_2_0_0_k3_polygon"
    for child in list(polygon):
        if child.tag != f"{{{SVG}}}title":
            polygon.remove(child)
    ET.SubElement(polygon, f"{{{SVG}}}polygon", {"points": "-6,6 0,-6 6,6"})
    parent.append(polygon)

    for index, x in enumerate((120, 140)):
        flow = ET.SubElement(
            parent,
            f"{{{SVG}}}g",
            {
                "id": f"test_flow_{index}",
                "class": f"edge flow src-{index} fate-test-{index}",
            },
        )
        ET.SubElement(
            flow,
            f"{{{SVG}}}path",
            {"d": f"M {x} 100 L {x + 10} 100", "stroke": "#000000"},
        )
    tree.write(destination, encoding="unicode")


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format_string: str, *args: object) -> None:
        pass


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        (root / "index.html").write_text(
            pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"), encoding="utf-8"
        )
        write_alias_fixture(pathlib.Path(sys.argv[2]), root / "design.svg")
        (root / "design.json").write_text(
            pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"), encoding="utf-8"
        )
        (root / "design.py").write_text(
            "".join(f"source line {line}\n" for line in range(1, 61)),
            encoding="utf-8",
        )

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
                        "?svg=design.svg&source-root=."
                    )
                    assert page.evaluate("sourceURL('design.py')").endswith(
                        "/design.py"
                    )
                    for source in (
                        "/tmp/design.py",
                        "../design.py",
                        "%2ftmp%2fdesign.py",
                        "dir%2f..%2f..%2fdesign.py",
                        "dir%5c..%5c..%5cdesign.py",
                        "https://example.com/design.py",
                    ):
                        assert page.evaluate(
                            "source => { try { sourceURL(source); return false; } "
                            "catch (error) { return error.message.includes("
                            "'escapes the configured source root'); } }",
                            source,
                        )
                    page.evaluate(
                        "() => { const url = new URL(location.href); "
                        "url.searchParams.set('source./tmp/design.py', 'design.py'); "
                        "history.replaceState(null, '', url); }"
                    )
                    assert page.evaluate("sourceURL('/tmp/design.py')").endswith(
                        "/design.py"
                    )
                    ids = page.locator("[id]").evaluate_all(
                        "elements => elements.map(element => element.id)"
                    )
                    assert len(ids) == len(set(ids))
                    path_alias = page.locator("#physical_alias_p_0_2_0_0_k3 path")
                    rectangle_alias = page.locator(
                        "#physical_alias_p_0_2_0_0_k3_rect rect"
                    )
                    polygon_alias = page.locator(
                        "#physical_alias_p_0_2_0_0_k3_polygon polygon"
                    )
                    path_box = path_alias.bounding_box()
                    rectangle_box = rectangle_alias.bounding_box()
                    polygon_box = polygon_alias.bounding_box()
                    assert path_box and rectangle_box and polygon_box
                    assert path_box["x"] == rectangle_box["x"]
                    assert path_box["y"] == rectangle_box["y"]
                    assert path_box["x"] == polygon_box["x"]
                    assert path_box["y"] == polygon_box["y"]
                    path_alias.dispatch_event("pointerover")
                    highlighted_line = page.locator(
                        '.source-line.highlighted[data-line="30"]'
                    )
                    highlighted_line.wait_for()
                    assert not page.locator("#viewport svg").evaluate(
                        "element => element.classList.contains('highlighting')"
                    )
                    rectangle_alias.dispatch_event("pointerover")
                    highlighted_line.wait_for()
                    polygon_alias.dispatch_event(
                        "pointerdown", {"pointerId": 1, "clientX": 100, "clientY": 100}
                    )
                    polygon_alias.dispatch_event(
                        "pointerup", {"pointerId": 1, "clientX": 100, "clientY": 100}
                    )
                    polygon_alias.dispatch_event("pointerleave")
                    highlighted_line.wait_for()
                    assert page.locator(".source-line.highlighted").count() == 1
                    page.locator("#reset").click()
                    runtime_alias = page.locator("#physical_alias_runtime_arg_0 ellipse")
                    runtime_alias.dispatch_event("pointerover")
                    runtime_line = page.locator(
                        '.source-line.highlighted[data-line="50"]'
                    )
                    runtime_line.wait_for()
                    edge_alias = page.locator(
                        "#physical_alias_edge__runtime_arg_0__p_0_0_1_0_m ellipse"
                    )
                    edge_alias.dispatch_event("pointerover")
                    runtime_line.wait_for()
                    edge_box = edge_alias.bounding_box()
                    assert edge_box
                    edge_alias.dispatch_event(
                        "pointerdown",
                        {
                            "pointerId": 3,
                            "clientX": edge_box["x"],
                            "clientY": edge_box["y"],
                        },
                    )
                    edge_alias.dispatch_event(
                        "pointerup",
                        {
                            "pointerId": 3,
                            "clientX": edge_box["x"],
                            "clientY": edge_box["y"],
                        },
                    )
                    edge_alias.dispatch_event("pointerleave")
                    runtime_line.wait_for()
                    page.locator("#reset").click()
                    flow = page.locator("#test_flow_0 path")
                    flow.dispatch_event("pointerover")
                    assert page.locator("#viewport svg.highlighting").count() == 1
                    flow.dispatch_event(
                        "pointerdown", {"pointerId": 2, "clientX": 120, "clientY": 100}
                    )
                    flow.dispatch_event(
                        "pointerup", {"pointerId": 2, "clientX": 120, "clientY": 100}
                    )
                    flow.dispatch_event("pointerleave")
                    assert page.locator("#viewport svg.highlighting").count() == 1
                    assert page.locator(".flow-row.selected").count() == 1
                    browser.close()
            finally:
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    main()
