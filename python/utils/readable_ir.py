# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Print MLIR with each op's location as a short trailing comment.

An IR file printed with debug info ends each op with a `loc(...)`, usually an
alias (`loc(#loc42)`) defined at the bottom of the file as a chain of
callsites, names and fused locations. This rewrites the text so each op line
instead ends in what a reader wants from that location:

    aie.objectfifo @act_in(...)   // act_in · design.py:12

the IRON object or Python function names on it (`iron:` dropped, `py:` kept)
and the innermost source line. The alias definitions are dropped. The input is
read as text, so any stage aiecc writes works, with or without
`--mlir-print-local-scope`, and no dialect needs to be registered.

    python -m aie.utils.readable_ir input_with_symbols.mlir
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# MLIR names its own location aliases #locN; a hand-written file may use any
# alias name.
_ALIAS_DEF = re.compile(r"^(#[A-Za-z_][\w$.-]*)\s*=\s*loc\((.*)\)\s*$")
_TOKEN = re.compile(
    r'\s*(?:(?P<str>"(?:[^"\\]|\\.)*")|(?P<alias>#[A-Za-z_][\w$.-]*)'
    r"|(?P<num>[0-9]+)|(?P<word>[A-Za-z_][A-Za-z0-9_]*)|(?P<punct>[()\[\],:<>]))"
)


class _Parser:
    """Parse the text of a location into nested tuples.

    ("file", filename, line), ("name", name, child or None),
    ("callsite", callee, caller), ("fused", [children]), ("alias", "#locN"),
    or ("unknown",).
    """

    def __init__(self, text: str):
        self.tokens = []
        pos = 0
        while pos < len(text):
            m = _TOKEN.match(text, pos)
            if not m or m.end() == pos:
                # Something this reader does not model (opaque metadata):
                # keep what came before.
                break
            self.tokens.append((m.lastgroup, m[0].lstrip()))
            pos = m.end()
        self.i = 0

    def peek(self, value=None):
        if self.i >= len(self.tokens):
            return None
        tok = self.tokens[self.i]
        return tok if value is None or tok[1] == value else None

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def skip_balanced(self, open_, close):
        depth = 0
        while (tok := self.take()) is not None:
            if tok[1] == open_:
                depth += 1
            elif tok[1] == close:
                depth -= 1
                if depth == 0:
                    return

    def parse(self):
        tok = self.take()
        if tok is None:
            return ("unknown",)
        kind, value = tok
        if kind == "alias":
            return ("alias", value)
        if kind == "word" and value == "callsite" and self.peek("("):
            self.take()
            callee = self.parse()
            if self.peek("at"):
                self.take()
            caller = self.parse()
            if self.peek(")"):
                self.take()
            return ("callsite", callee, caller)
        if kind == "word" and value == "fused":
            if self.peek("<"):
                self.skip_balanced("<", ">")
            children = []
            if self.peek("["):
                self.take()
                while self.peek() and not self.peek("]"):
                    children.append(self.parse())
                    if self.peek(","):
                        self.take()
                self.take()
            return ("fused", children)
        if kind == "str":
            text = _unescape(value[1:-1])
            if self.peek(":"):
                self.take()
                line = int(tok[1]) if (tok := self.take()) else 0
                # The column, and any `to` range end, are not shown.
                while self.peek(":") or self.peek("to"):
                    self.take()
                    if (tok := self.peek()) and tok[0] == "num":
                        self.take()
                return ("file", text, line)
            if self.peek("("):
                self.take()
                child = self.parse()
                if self.peek(")"):
                    self.take()
                return ("name", text, child)
            return ("name", text, None)
        return ("unknown",)


_ESCAPE = re.compile(r"\\(?:([0-9A-Fa-f]{2})|(.))", re.DOTALL)


def _unescape(text: str) -> str:
    r"""Undo MLIR's string escapes: `\XX` hex bytes, `\n`, `\t`, `\"`, `\\`.

    A non-ASCII path prints as its UTF-8 bytes, `d\C3\A9sign.py`.
    """
    out = bytearray()
    pos = 0
    for m in _ESCAPE.finditer(text):
        out += text[pos : m.start()].encode()
        hex_byte, char = m.groups()
        if hex_byte is not None:
            out.append(int(hex_byte, 16))
        else:
            out += {"n": "\n", "t": "\t"}.get(char, char).encode()
        pos = m.end()
    out += text[pos:].encode()
    return out.decode("utf-8", "replace")


def label(loc, aliases: dict) -> str:
    """Return the short form of a parsed location.

    Args:
        loc: A location from `_Parser.parse`.
        aliases: Each alias name, as `#loc3`, to its parsed location.

    Returns:
        The names on the location and its innermost source line, as
        "act_in · design.py:12"; "" when it has neither.
    """
    names: list[str] = []
    files: list[tuple[str, int]] = []

    def visit(node, seen):
        kind = node[0]
        if kind == "alias":
            if node[1] in aliases and node[1] not in seen:
                visit(aliases[node[1]], seen | {node[1]})
        elif kind == "name":
            name = node[1].removeprefix("iron:")
            if name not in names:
                names.append(name)
            if node[2] is not None:
                visit(node[2], seen)
        elif kind == "callsite":
            # The callee is where the op was built; callers only add lines.
            visit(node[1], seen)
            caller_files = len(files)
            visit(node[2], seen)
            del files[caller_files:]
        elif kind == "fused":
            for child in node[1]:
                visit(child, seen)
        elif kind == "file":
            files.append((os.path.basename(node[1]), node[2]))

    visit(loc, frozenset())
    where = f"{files[0][0]}:{files[0][1]}" if files else ""
    return " · ".join(part for part in (", ".join(names), where) if part)


def _locs(line: str):
    """Yield (start, end, body) of each `loc(...)` outside a string literal."""
    i, in_string = 0, False
    while i < len(line):
        c = line[i]
        if in_string:
            if c == "\\":
                i += 1
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif line.startswith("loc(", i) and (
            # Not the end of another name, as in `@get_loc(`.
            i == 0
            or not (line[i - 1].isalnum() or line[i - 1] in "_.$@")
        ):
            depth, j, quoted = 0, i + 3, False
            while j < len(line):
                ch = line[j]
                if quoted:
                    if ch == "\\":
                        j += 1
                    elif ch == '"':
                        quoted = False
                elif ch == '"':
                    quoted = True
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            yield i, j + 1, line[i + 4 : j]
            i = j
        i += 1


def readable(text: str) -> str:
    """Return `text`, MLIR printed with debug info, with locations as comments.

    Args:
        text: The IR, e.g. an aiecc intermediate.

    Returns:
        The IR with each line's locations removed, the last one on a line
        (the op's own) appended as a `// ...` comment, and the location alias
        definitions dropped. An op with regions prints its location after
        its closing `}`; that comment goes on the line that names the op.
    """
    aliases = {}
    body = []
    for line in text.splitlines():
        m = _ALIAS_DEF.match(line)
        if m:
            aliases[m.group(1)] = _Parser(m.group(2)).parse()
        else:
            body.append(line)

    out, notes = [], {}
    # The line of the op each open `{` belongs to.
    owners = []
    for line in body:
        spans = list(_locs(line))
        kept, prev = [], 0
        for start, end, _ in spans:
            kept.append(line[prev:start].rstrip(" "))
            prev = end
        kept.append(line[prev:])
        stripped = "".join(kept).rstrip() if spans else line
        owner = len(out)
        in_string = escaped = False
        for c in stripped:
            if in_string:
                in_string = escaped or c != '"'
                escaped = not escaped and c == "\\"
            elif c == '"':
                in_string = True
            elif c == "}" and owners:
                owner = owners.pop()
            elif c == "{":
                owners.append(owner)
        out.append(stripped)
        # A line that leaves a region open holds only its block arguments'
        # locations; the op's own follows the region.
        if spans and not (owners and owners[-1] == len(out) - 1):
            note = label(_Parser(spans[-1][2]).parse(), aliases)
            if note:
                notes[owner] = note
    out = [
        f"{line}  // {notes[i]}" if i in notes else line for i, line in enumerate(out)
    ]
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Print MLIR with each op's location as a short trailing comment."
    )
    parser.add_argument("input", help="MLIR file printed with debug info, or -")
    parser.add_argument("-o", "--output", help="Write here instead of stdout")
    args = parser.parse_args(argv)
    text = sys.stdin.read() if args.input == "-" else open(args.input).read()
    result = readable(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
    else:
        sys.stdout.write(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
