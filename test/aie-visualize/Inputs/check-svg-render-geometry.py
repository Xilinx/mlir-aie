# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from argparse import ArgumentParser

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
PATH_TOKEN = re.compile(rf"[A-Za-z]|{NUMBER}")
TRANSFORM = re.compile(r"([A-Za-z]+)\s*\(([^)]*)\)")
EPSILON = 1e-6
COLLINEAR_TOLERANCE = 0.05
MINIMUM_GAP = 1.2


def multiply(left, right):
    a, b, c, d, e, f = left
    g, h, i, j, k, offset_y = right
    return (
        a * g + c * h,
        b * g + d * h,
        a * i + c * j,
        b * i + d * j,
        a * k + c * offset_y + e,
        b * k + d * offset_y + f,
    )


def transform_matrix(text):
    result = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    for name, arguments in TRANSFORM.findall(text or ""):
        values = [float(value) for value in re.findall(NUMBER, arguments)]
        if name == "matrix" and len(values) == 6:
            operation = tuple(values)
        elif name == "translate" and 1 <= len(values) <= 2:
            operation = (1.0, 0.0, 0.0, 1.0, values[0], values[1] if len(values) == 2 else 0.0)
        elif name == "scale" and 1 <= len(values) <= 2:
            operation = (values[0], 0.0, 0.0, values[-1], 0.0, 0.0)
        elif name == "rotate" and len(values) in (1, 3):
            angle = math.radians(values[0])
            rotation = (math.cos(angle), math.sin(angle), -math.sin(angle), math.cos(angle), 0.0, 0.0)
            if len(values) == 1:
                operation = rotation
            else:
                cx, cy = values[1:]
                operation = multiply(
                    multiply((1.0, 0.0, 0.0, 1.0, cx, cy), rotation),
                    (1.0, 0.0, 0.0, 1.0, -cx, -cy),
                )
        else:
            raise ValueError(f"unsupported SVG transform: {name}({arguments})")
        result = multiply(result, operation)
    return result


def apply(matrix, point):
    a, b, c, d, e, f = matrix
    x, y = point
    return (a * x + c * y + e, b * x + d * y + f)


def parse_path(text):
    tokens = PATH_TOKEN.findall(text)
    index = 0
    command = None
    current = (0.0, 0.0)
    start = current
    segments = []
    widths = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2}
    while index < len(tokens):
        if tokens[index].isalpha():
            command = tokens[index]
            index += 1
        if command is None:
            raise ValueError("path data starts without a command")
        upper = command.upper()
        relative = command.islower()
        if upper == "Z":
            segments.append(("L", (current, start)))
            current = start
            command = None
            continue
        if upper == "A":
            raise ValueError("elliptical arc path commands are unsupported")
        width = widths.get(upper)
        if width is None or index + width > len(tokens):
            raise ValueError(f"unsupported or incomplete path command: {command}")
        values = [float(value) for value in tokens[index : index + width]]
        index += width

        def point(x, y):
            return (x + current[0], y + current[1]) if relative else (x, y)

        if upper == "M":
            current = point(*values)
            start = current
            command = "l" if relative else "L"
            continue
        if upper == "H":
            end = (values[0] + current[0], current[1]) if relative else (values[0], current[1])
            geometry = (current, end)
        elif upper == "V":
            end = (current[0], values[0] + current[1]) if relative else (current[0], values[0])
            geometry = (current, end)
        elif upper in ("L", "T"):
            end = point(*values[-2:])
            geometry = (current, end)
        elif upper in ("S", "Q"):
            control = point(*values[:2])
            end = point(*values[-2:])
            geometry = (current, control, end)
        else:
            first = point(*values[:2])
            second = point(*values[2:4])
            end = point(*values[4:6])
            geometry = (current, first, second, end)
        segments.append((upper, geometry))
        current = end
    return segments


def flatten(segment, steps=16):
    command, points = segment
    if command == "L":
        return list(points)
    output = []
    for step in range(steps + 1):
        t = step / steps
        if len(points) == 3:
            first, control, second = points
            x = (1 - t) ** 2 * first[0] + 2 * (1 - t) * t * control[0] + t**2 * second[0]
            y = (1 - t) ** 2 * first[1] + 2 * (1 - t) * t * control[1] + t**2 * second[1]
        else:
            first, control1, control2, second = points
            x = ((1 - t) ** 3 * first[0] + 3 * (1 - t) ** 2 * t * control1[0]
                 + 3 * (1 - t) * t**2 * control2[0] + t**3 * second[0])
            y = ((1 - t) ** 3 * first[1] + 3 * (1 - t) ** 2 * t * control1[1]
                 + 3 * (1 - t) * t**2 * control2[1] + t**3 * second[1])
        output.append((x, y))
    return output


def is_axis_aligned(points):
    return all(abs(point[0] - points[0][0]) <= EPSILON for point in points[1:]) or all(
        abs(point[1] - points[0][1]) <= EPSILON for point in points[1:]
    )


def is_straight(points):
    first, second = points[0], points[-1]
    delta_x, delta_y = second[0] - first[0], second[1] - first[1]
    length = math.hypot(delta_x, delta_y)
    if length <= EPSILON:
        return True
    return all(
        abs((point[0] - first[0]) * delta_y - (point[1] - first[1]) * delta_x) / length
        <= COLLINEAR_TOLERANCE
        for point in points[1:-1]
    )


def classes(element):
    return frozenset(element.get("class", "").split())


def descendants(element, name):
    return [child for child in element.iter() if child.tag.rsplit("}", 1)[-1] == name]


def title(element):
    values = descendants(element, "title")
    return values[0].text if values else None


def shape_points(element):
    name = element.tag.rsplit("}", 1)[-1]
    if name == "path":
        return [point for segment in parse_path(element.get("d", "")) for point in segment[1]]
    if name in ("polygon", "polyline"):
        values = [float(value) for value in re.findall(NUMBER, element.get("points", ""))]
        return list(zip(values[::2], values[1::2]))
    if name == "ellipse":
        cx, cy = float(element.get("cx", 0)), float(element.get("cy", 0))
        rx, ry = float(element.get("rx", 0)), float(element.get("ry", 0))
        return [(cx - rx, cy - ry), (cx + rx, cy + ry)]
    return []


def segment_hits_box(first, second, box, inset=0.25):
    low_x, low_y, high_x, high_y = box
    low_x += inset
    low_y += inset
    high_x -= inset
    high_y -= inset
    if low_x >= high_x or low_y >= high_y:
        return False
    delta_x, delta_y = second[0] - first[0], second[1] - first[1]
    lower, upper = 0.0, 1.0
    for start, delta, minimum, maximum in (
        (first[0], delta_x, low_x, high_x),
        (first[1], delta_y, low_y, high_y),
    ):
        if abs(delta) <= EPSILON:
            if start < minimum or start > maximum:
                return False
            continue
        entry, exit = (minimum - start) / delta, (maximum - start) / delta
        if entry > exit:
            entry, exit = exit, entry
        lower, upper = max(lower, entry), min(upper, exit)
        if lower > upper:
            return False
    return upper - lower > EPSILON


def endpoint_tile(node_id):
    match = re.match(r"(?:p|tile)_(\d+)_(\d+)(?:_|$)", node_id)
    return f"tile_{match.group(1)}_{match.group(2)}" if match else None


def dot_endpoints(path):
    text = open(path, encoding="utf-8").read()
    mapping = {}
    edge_pattern = r'^\s*("[^"]+"|\w+)\s*->\s*("[^"]+"|\w+)\s*\[(.*)\];$'
    for match in re.finditer(edge_pattern, text, re.MULTILINE):
        attributes = match.group(3)
        edge_id = re.search(r'\bid="([^"]+)"', attributes)
        if edge_id:
            mapping[edge_id.group(1)] = tuple(value.strip('"') for value in match.group(1, 2))
    return mapping


def json_endpoints(path):
    document = json.load(open(path, encoding="utf-8"))
    mapping = {}

    def visit(value):
        if isinstance(value, dict):
            edge_id = value.get("id") or value.get("edge_id")
            source = value.get("source") or value.get("from")
            target = value.get("target") or value.get("to")
            if edge_id is not None and source is not None and target is not None:
                mapping[str(edge_id)] = (str(source), str(target))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(document)
    return mapping


def edge_endpoints(group, external_mapping):
    edge_title = title(group) or ""
    if "->" in edge_title:
        return tuple(edge_title.split("->", 1))
    return external_mapping.get(group.get("id"), ())


def axis_run(first, second, edge_id, flow_id, endpoints):
    endpoint_set = frozenset(endpoints)
    if abs(first[1] - second[1]) <= EPSILON:
        return ("h", first[1], *sorted((first[0], second[0])), flow_id, edge_id, endpoint_set)
    if abs(first[0] - second[0]) <= EPSILON:
        return ("v", first[0], *sorted((first[1], second[1])), flow_id, edge_id, endpoint_set)
    return None


def main():
    parser = ArgumentParser()
    parser.add_argument("svg")
    parser.add_argument("--dot")
    parser.add_argument("--json")
    parser.add_argument("--allow-defects", action="store_true")
    arguments = parser.parse_args()
    external_mapping = {}
    if arguments.dot:
        external_mapping.update(dot_endpoints(arguments.dot))
    if arguments.json:
        external_mapping.update(json_endpoints(arguments.json))
    root = ET.parse(arguments.svg).getroot()
    parent = {child: node for node in root.iter() for child in node}

    def matrix(element):
        lineage = []
        while element is not None:
            lineage.append(element)
            element = parent.get(element)
        result = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        for ancestor in reversed(lineage):
            result = multiply(result, transform_matrix(ancestor.get("transform")))
        return result

    nodes = []
    for group in root.iter():
        if group.tag.rsplit("}", 1)[-1] != "g" or "node" not in classes(group) or "legend" in classes(group):
            continue
        points = []
        for shape_name in ("path", "polygon", "polyline", "ellipse"):
            for shape in descendants(group, shape_name):
                if shape.get("fill", "none") == "none":
                    continue
                points.extend(apply(matrix(shape), point) for point in shape_points(shape))
        if points:
            nodes.append((group.get("id", "<unnamed-node>"), title(group), (
                min(point[0] for point in points), min(point[1] for point in points),
                max(point[0] for point in points), max(point[1] for point in points),
            )))
    diagonal_defects = []
    curved_defects = []
    intersection_defects = []
    runs = []
    edge_count = 0
    for group in root.iter():
        edge_classes = classes(group)
        if group.tag.rsplit("}", 1)[-1] != "g" or "edge" not in edge_classes or "legend" in edge_classes:
            continue
        edge_count += 1
        edge_id = group.get("id", "<unnamed-edge>")
        class_text = group.get("class", "")
        endpoints = edge_endpoints(group, external_mapping)
        endpoint_regions = set(endpoints)
        endpoint_regions.update(filter(None, (endpoint_tile(endpoint) for endpoint in endpoints)))
        flow_classes = sorted(value for value in edge_classes if re.fullmatch(r"src-\d+", value))
        flow_id = flow_classes[0] if flow_classes else edge_id
        for path in descendants(group, "path"):
            path_segments = parse_path(path.get("d", ""))
            path_matrix = matrix(path)
            transformed = [(command, tuple(apply(path_matrix, point) for point in points))
                           for command, points in path_segments]
            for command_index, segment in enumerate(transformed):
                if not is_axis_aligned(segment[1]):
                    defect = (edge_id, class_text, command_index, segment[1][0], segment[1][-1])
                    (diagonal_defects if is_straight(segment[1]) else curved_defects).append(defect)
                elif is_straight(segment[1]):
                    run = axis_run(segment[1][0], segment[1][-1], edge_id, flow_id, endpoints)
                    if run:
                        runs.append(run)
            lines = []
            for segment in transformed:
                points = flatten(segment)
                lines.extend(zip(points, points[1:]))
            path_box = (
                min(point[0] for line in lines for point in line),
                min(point[1] for line in lines for point in line),
                max(point[0] for line in lines for point in line),
                max(point[1] for line in lines for point in line),
            )
            for node_id, node_title, box in nodes:
                if node_id in endpoint_regions or node_title in endpoint_regions:
                    continue
                if (box[2] < path_box[0] or box[0] > path_box[2]
                        or box[3] < path_box[1] or box[1] > path_box[3]):
                    continue
                hits = [index for index, line in enumerate(lines) if segment_hits_box(*line, box)]
                if not hits:
                    continue
                intersection_defects.append((edge_id, class_text, node_id, node_title))

    overlap_defects = []
    narrow_gap_defects = []
    for index, first in enumerate(runs):
        first_axis, first_cross, first_low, first_high = first[:4]
        first_flow, first_edge, first_endpoints = first[4:]
        for second in runs[index + 1:]:
            second_axis, second_cross, second_low, second_high = second[:4]
            second_flow, second_edge, second_endpoints = second[4:]
            if (first_edge == second_edge or first_flow == second_flow or first_axis != second_axis
                    or first_endpoints & second_endpoints):
                continue
            overlap = min(first_high, second_high) - max(first_low, second_low)
            if overlap <= EPSILON:
                continue
            gap = abs(first_cross - second_cross)
            defect = (first_edge, second_edge, overlap, gap)
            if gap <= EPSILON:
                overlap_defects.append(defect)
            elif gap < MINIMUM_GAP:
                narrow_gap_defects.append(defect)

    print(
        f"edges={edge_count} diagonals={len(diagonal_defects)} "
        f"curved_bends={len(curved_defects)} node_intersections={len(intersection_defects)} "
        f"collinear_overlaps={len(overlap_defects)} narrow_gaps={len(narrow_gap_defects)}"
    )
    for edge_id, class_text, command_index, first, second in diagonal_defects:
        print(f"non-orthogonal edge={edge_id} class={class_text!r} command={command_index} from={first} to={second}")
    for edge_id, class_text, node_id, node_title in intersection_defects:
        print(f"node-intersection edge={edge_id} class={class_text!r} node={node_id} title={node_title!r}")
    for first_edge, second_edge, overlap, gap in overlap_defects + narrow_gap_defects:
        print(f"edge-separation first={first_edge} second={second_edge} overlap={overlap:.2f} gap={gap:.2f}")
    defects = diagonal_defects or intersection_defects or overlap_defects or narrow_gap_defects
    if defects and not arguments.allow_defects:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())