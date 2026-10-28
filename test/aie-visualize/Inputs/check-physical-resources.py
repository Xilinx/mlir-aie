# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import collections
import pathlib
import re
import sys


def main() -> None:
    dot = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    classes = collections.Counter()
    states = collections.Counter()
    for class_list in re.findall(r'class="([^"]+)"', dot):
        names = class_list.split()
        if not names or not names[0].startswith("physical-"):
            continue
        classes[names[0]] += 1
        if len(names) > 1 and names[-1] in {"used", "unused"}:
            states[(names[0], names[-1])] += 1

    for name in sorted(classes):
        print(
            f"{name} total={classes[name]} used={states[name, 'used']} "
            f"unused={states[name, 'unused']}"
        )


if __name__ == "__main__":
    main()
