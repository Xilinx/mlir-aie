#!/usr/bin/env python3
import os
import re
import sys


def check_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove {% raw %}...{% endraw %} blocks
    content_no_raw = re.sub(
        r"\{%\s*raw\s*%\}.*?\{%\s*endraw\s*%\}", "", content, flags=re.DOTALL
    )

    # Check for remaining {%
    # In .td files, we assume ANY {% that is not in a raw block is suspicious
    # because .td files are for MLIR definitions, not Liquid templates.
    # If it's MLIR syntax, it MUST be escaped (e.g. inside raw block) to pass Jekyll.

    match = re.search(r"\{%", content_no_raw)
    if match:
        # Iterate through all {% occurrences in original content.
        # Check if each occurrence is inside a raw block.

        raw_blocks = []
        for m in re.finditer(
            r"\{%\s*raw\s*%\}.*?\{%\s*endraw\s*%\}", content, flags=re.DOTALL
        ):
            raw_blocks.append(m.span())

        for m in re.finditer(r"\{%", content):
            start, end = m.span()

            # Check if this {% is part of a raw block definition itself
            if re.match(r"\{%\s*(raw|endraw)\s*%\}", content[start:]):
                continue

            # Check if it is inside a raw block
            is_inside = False
            for rb_start, rb_end in raw_blocks:
                if start >= rb_start and end <= rb_end:
                    is_inside = True
                    break

            if not is_inside:
                # Found an unescaped {%
                line_num = content.count("\n", 0, start) + 1
                print(f"Error: Unescaped '{{%' found in {filepath} at line {line_num}")
                print(f"Context: {content[start:start+20]}...")
                return False

    return True


def main():
    root_dir = "include"
    extensions = [".td"]
    has_error = False

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                filepath = os.path.join(root, file)
                if not check_file(filepath):
                    has_error = True

    if has_error:
        print("\nFound potential Liquid syntax errors in .td files.")
        print(
            "Please escape '{%' using '{% raw %}{%{% endraw %}' or add a space '{ %' if it represents MLIR syntax."
        )
        sys.exit(1)
    else:
        print("No Liquid syntax errors found in .td files.")
        sys.exit(0)


if __name__ == "__main__":
    main()
