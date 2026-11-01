# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Write a design's linked object only once aiecc waits for it, then let it go on."""

import argparse
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--aiecc", required=True)
parser.add_argument("--design", type=Path, required=True)
parser.add_argument("--built", type=Path, required=True, help="the object, built")
parser.add_argument(
    "--linked", type=Path, required=True, help="where it is linked from"
)
parser.add_argument("--device-cache", type=Path, required=True)
args = parser.parse_args()

assert not args.linked.exists(), args.linked
aiecc = subprocess.Popen(
    [
        args.aiecc,
        "-v",
        "--await-link-files",
        "--get-pdi",
        f"--device-cache={args.device_cache}",
        str(args.design),
    ],
    stdin=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)
log = []
for line in aiecc.stderr:
    log.append(line)
    if line == "aiecc: awaiting the link files\n":
        break
else:
    raise AssertionError("aiecc never waited:\n" + "".join(log))
shutil.copyfile(args.built, args.linked)
aiecc.stdin.write("\n")
aiecc.stdin.close()
log.extend(aiecc.stderr)
assert aiecc.wait(timeout=120) == 0, "".join(log)
assert "aiecc: device cache store: a\n" in log, "".join(log)
