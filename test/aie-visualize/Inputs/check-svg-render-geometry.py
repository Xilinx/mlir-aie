# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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


def main():
    parser = ArgumentParser()
    parser.add_argument("svg")
    parser.add_argument("--allow-defects", action="store_true")
    arguments = parser.parse_args()
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
    node_boxes = {node_title: box for _, node_title, box in nodes}
    tile_boxes = {
        node_title: box for node_id, node_title, box in nodes if node_id.startswith("tile_")
    }

    diagonal_defects = []
    curved_defects = []
    intersection_defects = []
    edge_count = 0
    for group in root.iter():
        edge_classes = classes(group)
        if group.tag.rsplit("}", 1)[-1] != "g" or "edge" not in edge_classes or "legend" in edge_classes:
            continue
        edge_count += 1
        edge_id = group.get("id", "<unnamed-edge>")
        class_text = group.get("class", "")
        edge_title = title(group) or ""
        endpoints = edge_title.split("->", 1) if "->" in edge_title else []
        endpoint_regions = set(endpoints)
        for endpoint in endpoints:
            box = node_boxes.get(endpoint)
            if box is None:
                continue
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            endpoint_regions.update(
                tile_title
                for tile_title, tile_box in tile_boxes.items()
                if tile_box[0] <= center[0] <= tile_box[2] and tile_box[1] <= center[1] <= tile_box[3]
            )
        for path in descendants(group, "path"):
            path_segments = parse_path(path.get("d", ""))
            path_matrix = matrix(path)
            transformed = [(command, tuple(apply(path_matrix, point) for point in points))
                           for command, points in path_segments]
            for command_index, segment in enumerate(transformed):
                if not is_axis_aligned(segment[1]):
                    defect = (edge_id, class_text, command_index, segment[1][0], segment[1][-1])
                    (diagonal_defects if is_straight(segment[1]) else curved_defects).append(defect)
            lines = []
            for segment in transformed:
                points = flatten(segment)
                lines.extend(zip(points, points[1:]))
            for node_id, node_title, box in nodes:
                if node_title in endpoint_regions:
                    continue
                hits = [index for index, line in enumerate(lines) if segment_hits_box(*line, box)]
                if not hits:
                    continue
                intersection_defects.append((edge_id, class_text, node_id, node_title))

    print(
        f"edges={edge_count} diagonals={len(diagonal_defects)} "
        f"curved_bends={len(curved_defects)} node_intersections={len(intersection_defects)}"
    )
    for edge_id, class_text, command_index, first, second in diagonal_defects:
        print(f"non-orthogonal edge={edge_id} class={class_text!r} command={command_index} from={first} to={second}")
    for edge_id, class_text, node_id, node_title in intersection_defects:
        print(f"node-intersection edge={edge_id} class={class_text!r} node={node_id} title={node_title!r}")
    if (diagonal_defects or intersection_defects) and not arguments.allow_defects:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())