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
    return (a * point[0] + c * point[1] + e, b * point[0] + d * point[1] + f)


def flatten_path(text, steps=16):
    tokens = PATH_TOKEN.findall(text)
    index = 0
    command = None
    current = (0.0, 0.0)
    start = current
    paths = []
    points = []
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
            if points and points[-1] != start:
                points.append(start)
            command = None
            continue
        width = widths.get(upper)
        if width is None or index + width > len(tokens):
            raise ValueError(f"unsupported or incomplete path command: {command}")
        values = [float(value) for value in tokens[index:index + width]]
        index += width

        def point(x, y):
            return (x + current[0], y + current[1]) if relative else (x, y)

        if upper == "M":
            if points:
                paths.append(points)
            current = point(*values)
            start = current
            points = [current]
            command = "l" if relative else "L"
            continue
        if upper == "H":
            end = (values[0] + current[0], current[1]) if relative else (values[0], current[1])
            controls = ()
        elif upper == "V":
            end = (current[0], values[0] + current[1]) if relative else (current[0], values[0])
            controls = ()
        elif upper in ("L", "T"):
            end = point(*values[-2:])
            controls = ()
        elif upper in ("S", "Q"):
            controls = (point(*values[:2]),)
            end = point(*values[-2:])
        else:
            controls = (point(*values[:2]), point(*values[2:4]))
            end = point(*values[4:6])
        if not controls:
            points.append(end)
        else:
            first = current
            for step in range(1, steps + 1):
                t = step / steps
                if len(controls) == 1:
                    control = controls[0]
                    sample = (
                        (1 - t) ** 2 * first[0] + 2 * (1 - t) * t * control[0] + t**2 * end[0],
                        (1 - t) ** 2 * first[1] + 2 * (1 - t) * t * control[1] + t**2 * end[1],
                    )
                else:
                    control1, control2 = controls
                    sample = (
                        (1 - t) ** 3 * first[0] + 3 * (1 - t) ** 2 * t * control1[0]
                        + 3 * (1 - t) * t**2 * control2[0] + t**3 * end[0],
                        (1 - t) ** 3 * first[1] + 3 * (1 - t) ** 2 * t * control1[1]
                        + 3 * (1 - t) * t**2 * control2[1] + t**3 * end[1],
                    )
                points.append(sample)
        current = end
    if points:
        paths.append(points)
    return paths


def local_name(element):
    return element.tag.rsplit("}", 1)[-1]


def classes(element):
    return frozenset(element.get("class", "").split())


def descendants(element, name):
    return [child for child in element.iter() if local_name(child) == name]


def title(element):
    values = descendants(element, "title")
    return values[0].text if values else None


def point_distance(first, second):
    return math.hypot(first[0] - second[0], first[1] - second[1])


def segment_distance(point, first, second):
    delta_x, delta_y = second[0] - first[0], second[1] - first[1]
    length_squared = delta_x**2 + delta_y**2
    if length_squared <= EPSILON:
        return point_distance(point, first)
    projection = ((point[0] - first[0]) * delta_x + (point[1] - first[1]) * delta_y) / length_squared
    projection = min(1.0, max(0.0, projection))
    return point_distance(point, (first[0] + projection * delta_x, first[1] + projection * delta_y))


def geometry_distance(point, geometries):
    if any(len(geometry) >= 4 and geometry[0] == geometry[-1]
           and point_in_polygon(point, geometry[:-1]) for geometry in geometries):
        return 0.0
    return min(
        (segment_distance(point, first, second)
         for geometry in geometries for first, second in zip(geometry, geometry[1:])),
        default=math.inf,
    )


def point_in_polygon(point, polygon):
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if ((current[1] > point[1]) != (previous[1] > point[1])
                and point[0] < (previous[0] - current[0]) * (point[1] - current[1])
                / (previous[1] - current[1]) + current[0]):
            inside = not inside
        previous = current
    return inside


def parse_attributes(text):
    return {
        match.group(1): match.group(3) if match.group(3) is not None else match.group(4).strip()
        for match in re.finditer(r'(\w+)\s*=\s*("((?:\\.|[^"])*)"|([^,\]]+))', text)
    }


def metadata_from_dot(path):
    text = open(path, encoding="utf-8").read()
    endpoints = {}
    edge_classes = {}
    pattern = r'^\s*("[^"]+"|\w+)\s*->\s*("[^"]+"|\w+)\s*\[(.*)\];$'
    for match in re.finditer(pattern, text, re.MULTILINE):
        source, target = (value.strip('"') for value in match.group(1, 2))
        attributes = parse_attributes(match.group(3))
        edge_id = attributes.get("id")
        if edge_id:
            endpoints[edge_id] = (source, target)
            edge_classes[edge_id] = frozenset(attributes.get("class", "").split())
    return endpoints, edge_classes


def metadata_from_json(path):
    document = json.load(open(path, encoding="utf-8"))
    endpoints = {}

    def visit(value):
        if isinstance(value, dict):
            edge_id = value.get("id") or value.get("edge_id")
            source = value.get("source") or value.get("from")
            target = value.get("target") or value.get("to")
            if edge_id is not None and source is not None and target is not None:
                endpoints[str(edge_id)] = (str(source), str(target))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(document)
    return endpoints


def element_geometry(element, matrix):
    name = local_name(element)
    if name == "path":
        paths = [[apply(matrix, point) for point in path] for path in flatten_path(element.get("d", ""))]
        if element.get("fill", "none") != "none":
            for path in paths:
                if path and path[0] != path[-1]:
                    path.append(path[0])
        return paths
    if name in ("polygon", "polyline"):
        values = [float(value) for value in re.findall(NUMBER, element.get("points", ""))]
        points = [apply(matrix, point) for point in zip(values[::2], values[1::2])]
        if name == "polygon" and points and points[0] != points[-1]:
            points.append(points[0])
        return [points]
    if name == "ellipse":
        center = (float(element.get("cx", 0)), float(element.get("cy", 0)))
        radius = (float(element.get("rx", 0)), float(element.get("ry", 0)))
        points = [
            apply(matrix, (center[0] + radius[0] * math.cos(index * math.tau / 64),
                           center[1] + radius[1] * math.sin(index * math.tau / 64)))
            for index in range(65)
        ]
        return [points]
    return []


def aligned_gap(first, second, tolerance):
    delta_x, delta_y = abs(first[0] - second[0]), abs(first[1] - second[1])
    distance = point_distance(first, second)
    return distance if distance > tolerance and (delta_x <= tolerance or delta_y <= tolerance) else None


def main():
    parser = ArgumentParser()
    parser.add_argument("dot")
    parser.add_argument("json")
    parser.add_argument("svg")
    parser.add_argument("--tolerance", type=float, default=1.5)
    parser.add_argument("--join-limit", type=float, default=24.0)
    parser.add_argument("--allow-defects", action="store_true")
    arguments = parser.parse_args()

    dot_endpoints, dot_classes = metadata_from_dot(arguments.dot)
    json_endpoints = metadata_from_json(arguments.json)
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

    node_geometry = {}
    svg_edges = {}
    for group in root.iter():
        if local_name(group) != "g":
            continue
        group_classes = classes(group)
        if "node" in group_classes and "legend" not in group_classes:
            geometry = []
            for element in group.iter():
                if local_name(element) in ("path", "polygon", "polyline", "ellipse"):
                    geometry.extend(element_geometry(element, matrix(element)))
            node_id = group.get("id")
            if node_id and geometry:
                node_geometry[node_id] = geometry
                if title(group):
                    node_geometry[title(group)] = geometry
        if "edge" in group_classes and "legend" not in group_classes:
            paths = []
            arrows = []
            for element in group.iter():
                name = local_name(element)
                if name == "path" and element.get("fill", "none") == "none":
                    paths.extend(element_geometry(element, matrix(element)))
                elif name == "polygon":
                    arrows.extend(element_geometry(element, matrix(element)))
            svg_edges[group.get("id", "<unnamed-edge>")] = (group_classes, paths, arrows)

    format_defects = []
    all_edge_ids = set(dot_endpoints) | set(json_endpoints) | set(svg_edges)
    for edge_id in sorted(all_edge_ids):
        if edge_id not in dot_endpoints or edge_id not in json_endpoints or edge_id not in svg_edges:
            format_defects.append((edge_id, edge_id in dot_endpoints, edge_id in json_endpoints, edge_id in svg_edges))
        elif dot_endpoints[edge_id] != json_endpoints[edge_id]:
            format_defects.append((edge_id, dot_endpoints[edge_id], json_endpoints[edge_id], True))

    endpoint_defects = []
    open_endpoints = []
    checked_edges = 0
    for edge_id in sorted(set(dot_endpoints) & set(json_endpoints) & set(svg_edges)):
        source, target = dot_endpoints[edge_id]
        edge_classes, paths, arrows = svg_edges[edge_id]
        if not paths:
            endpoint_defects.append((edge_id, "path", source, target, None, math.inf))
            continue
        checked_edges += 1
        source_geometry = node_geometry.get(source, ())
        target_geometry = node_geometry.get(target, ())
        starts = [path[0] for path in paths if len(path) >= 2]
        ends = [path[-1] for path in paths if len(path) >= 2]
        source_distance = min((geometry_distance(point, source_geometry) for point in starts), default=math.inf)
        target_distance = min((geometry_distance(point, target_geometry) for point in ends), default=math.inf)
        arrow_bridge = False
        for arrow in arrows:
            polygon = arrow[:-1] if len(arrow) > 1 and arrow[0] == arrow[-1] else arrow
            path_touches_arrow = any(
                point_in_polygon(point, polygon) or geometry_distance(point, [arrow]) <= arguments.tolerance
                for point in ends
            )
            arrow_touches_target = any(
                geometry_distance(point, target_geometry) <= arguments.tolerance for point in polygon
            )
            arrow_bridge = arrow_bridge or (path_touches_arrow and arrow_touches_target)
        if source_distance > arguments.tolerance:
            endpoint_defects.append((edge_id, "source", source, target, starts[0], source_distance))
            open_endpoints.extend((edge_id, "source", point, edge_classes | dot_classes.get(edge_id, frozenset()))
                                  for point in starts)
        if target_distance > arguments.tolerance and not arrow_bridge:
            endpoint_defects.append((edge_id, "target", source, target, ends[-1], target_distance))
            open_endpoints.extend((edge_id, "target", point, edge_classes | dot_classes.get(edge_id, frozenset()))
                                  for point in ends)
        for path in paths[1:]:
            open_endpoints.append((edge_id, "subpath-start", path[0], edge_classes))
        for path in paths[:-1]:
            open_endpoints.append((edge_id, "subpath-end", path[-1], edge_classes))

    disconnected = []
    paired = set()
    for index, first in enumerate(open_endpoints):
        first_groups = {value for value in first[3] if re.fullmatch(r"(?:src-\d+|fate-.+)", value)}
        for second_index in range(index + 1, len(open_endpoints)):
            second = open_endpoints[second_index]
            if first[0] == second[0]:
                continue
            second_groups = {value for value in second[3] if re.fullmatch(r"(?:src-\d+|fate-.+)", value)}
            shared = sorted(first_groups & second_groups)
            gap = aligned_gap(first[2], second[2], arguments.tolerance)
            if shared and gap is not None and gap <= arguments.join_limit:
                disconnected.append((first, second, gap, shared))
                paired.update((index, second_index))
    dangling = [endpoint for index, endpoint in enumerate(open_endpoints) if index not in paired]

    print(
        f"edges={checked_edges} format_defects={len(format_defects)} "
        f"endpoint_defects={len(endpoint_defects)} disconnected_segments={len(disconnected)} "
        f"dangling_endpoints={len(dangling)}"
    )
    for defect in format_defects:
        print(f"format-defect edge={defect[0]} dot={defect[1]!r} json={defect[2]!r} svg={defect[3]!r}")
    for edge_id, side, source, target, point, distance in endpoint_defects:
        print(f"endpoint-defect edge={edge_id} side={side} source={source} target={target} "
              f"point={point} distance={distance:.3f}")
    for first, second, gap, shared in disconnected:
        print(f"disconnected-segments first={first[0]}:{first[1]} second={second[0]}:{second[1]} "
              f"gap={gap:.3f} groups={','.join(shared)}")
    for edge_id, side, point, endpoint_classes in dangling:
        groups = sorted(value for value in endpoint_classes if re.fullmatch(r"(?:src-\d+|fate-.+)", value))
        print(f"dangling-endpoint edge={edge_id} side={side} point={point} groups={','.join(groups)}")
    defects = format_defects or endpoint_defects or disconnected or dangling
    return 1 if defects and not arguments.allow_defects else 0


if __name__ == "__main__":
    sys.exit(main())