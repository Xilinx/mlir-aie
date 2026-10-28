# AIE visualization viewer

Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Generate matching SVG and semantic JSON files from the same input and options:

```shell
aie-visualize --emit-dot --contract=port design.mlir | dot -Tsvg > design.svg
aie-visualize --emit-json --contract=port design.mlir > design.json
```

The viewer infers `design.svg` and `design.json` from `aie-visualize.html` when
the files use that base name. Query parameters override the inferred URLs:

```text
aie-visualize.html?svg=design.svg&json=design.json&source-root=../src/
```

Use `source.<file>` to override one source URL. Percent-encode the file name and
URL when either contains reserved URL characters:

```text
&source.design.py=https%3A%2F%2Fexample.test%2Fsrc%2Fdesign.py
```

Without an override, the viewer resolves each source file below `source-root`.
It rejects absolute URLs and paths that escape that directory. Use a
`source.<file>` override when a source file resides elsewhere.

The equivalent HTML metadata names are `aie-visualize-svg`,
`aie-visualize-json`, and `aie-visualize-source-root`. Query parameters take
precedence. The viewer fetches SVG, JSON, and source files, so the server must
permit those requests through its same-origin or CORS policy. A `file:` URL may
block these requests; serve the directory over HTTP when that occurs.

Semantic JSON uses `format: "aie-semantic-graph"` and `version: 2`. Every node
and edge has an `id` that equals the generated SVG group ID. Its `sources` array
contains only source file names and sorted line sets:

```json
{
  "id": "edge__tile_0_2__tile_0_3",
  "sources": [
    {"file": "design.py", "lines": [30, 32]},
    {"file": "runtime.py", "lines": [7]}
  ]
}
```

Contraction unions the source line sets from removed nodes and edges into the
replacement edge.