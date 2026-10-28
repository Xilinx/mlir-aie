# Multi-IR visualization project design

Status: research proposal

This document proposes a project format and ingestion interface for viewing
several MLIR stages from one compilation. It does not specify production code
changes.

The proposal starts from mlir-aie commit
`49e854a4ef938352bf132c0d5ff16d7c130c6ef9`. The research also inspected:

- `fifield/aie-xray` at
  `64bedd767f3a6d9dc025435a4d561d97a0e57f6f`.
- `hunhoffe/iron-design-explorer` at
  `2dfda0639e0bd07e80bd961eef64d8791f39a955`.

## Recommendation

Keep the current semantic graph as the schema for one device at one stage.
Add a versioned project document around those graphs. The project document
should own stage order, artifact metadata, stable entity identities, and
cross-stage provenance.

Add inputs in this order:

1. Accept repeated files with `--input`.
2. Accept an explicit project manifest with `--manifest`.
3. Accept a directory with `--directory` and conservative discovery.
4. Add `--watch` after project extraction has content hashes and isolated
   stage diagnostics.

Use MLIR locations for source navigation and diagnostic context. Use explicit
attributes for compiler identity and lowering provenance. Locations do not
provide complete identity because several generated operations can share one
location, and one generated operation can have several ancestors.

ObjectFIFO lowering should attach explicit origin identities before it erases
the frontend operations. Every generated pool, endpoint, route, flow, buffer,
lock, DMA program, channel start, BD, and shim allocation should retain those
identities. This instrumentation is the main upstream requirement for complete
ObjectFIFO lineage.

## Current implementation facts

### `aie-visualize`

`tools/aie-visualize/aie-visualize.cpp` declares one required positional
`fileName`. The tool parses one `ModuleOp`. It calls
`AIE::emitSemanticGraphJson` when `--emit-json` is present. It requires
`--device` when the module contains more than one `aie.device`.

`lib/Visualize/AIEVisualize.cpp` defines `SemanticGraph`,
`analyzeSemanticGraph`, `selectTopology`, and `writeSemanticGraphJson`.
`writeSemanticGraphJson` emits `format: "aie-semantic-graph"` and `version: 1`.
The root contains one device, one node array, and one edge array.

`SemanticGraph` uses presentation-oriented IDs such as tile coordinates,
ports, buffer symbols, and `edge__<source>__<dest>`. These IDs work inside one
graph. They do not identify an entity across stages or extractions.

The `sourceLocations` helper walks nested MLIR locations and collects every
`FileLineColLoc`. This supports source navigation through `FusedLoc` and
`CallSiteLoc`. The emitted JSON flattens the location structure into a set of
file, line, and column records. It therefore loses call-site order, fused-part
grouping, names, and location metadata.

### `aiecc` stages and artifacts

`tools/aiecc/CommandLineOptions.h` accepts exactly one positional MLIR input.
`getWorkDir` chooses `<input-basename>.prj` when `--tmpdir` is absent.

`buildMainGraph` in `tools/aiecc/aiecc.cpp` declares the compilation graph.
Important whole-module stages include:

| Edge name | Meaning |
| --- | --- |
| `input.mlir` | Parsed input artifact |
| `default_stack_size.mlir` | Input after default stack policy |
| `placed.mlir` | Placed design |
| `traced.mlir` | Design after trace insertion |
| `input_with_symbols.mlir` | ObjectFIFO and related symbolic lowering |
| `input_physical.mlir` | Routed physical design |
| `input_with_addresses.mlir` | Linked and address-assigned design |

The graph also contains keyed artifacts. Examples include
`perCore_{0}.mlir`, `perDeviceCompile_{0}.mlir`, `lowered_{0}.mlir`, core
objects, ELFs, and directory bundles.

`Engine::run` in `tools/aiecc/ExecutionEngine.h` computes the reachable graph
from requested outputs. It executes edges in a valid topological order. It
materializes intermediate files only when a consumer or output requires them.
Directory order is therefore not stage order.

`NodeSerializer<T>` in `tools/aiecc/Graph.h` serializes keyed node items.
`NodeSerializer<OpInModule<KeyOp>>` stores one module plus operation walk
indexes. `NodeSerializer<Directory>` copies a complete directory bundle.
These serializers show that a project may contain both files and directory
artifacts. They also show that one stage may contain several keyed instances.

`writeCheckpoint` in `tools/aiecc/ExecutionEngine.h` writes a `manifest.json`
with `argv` and `frontier`. Each frontier entry contains an edge name, a
relative directory, and a serializer descriptor. This manifest serves resume,
not visualization. It has no format tag, version, full stage list, content
digest, semantic graph, or provenance table.

`DeviceCache` in `tools/aiecc/DeviceCache.h` keys entries from routed IR,
linked files, options, and tool identities. It deliberately excludes
locations. A cache hit restores the storing build's locations. A
provenance-aware viewer must not use those locations as proof that the current
source produced the cached artifact.

### ObjectFIFO lowering

`docs/ObjectFifoLowering.md` describes these stages:

1. Frontend ObjectFIFO operations.
2. Pools, segments, core endpoints, DMA endpoints, route endpoints, and routes.
3. Buffers, locks, flows, DMA programs, BDs, and shim allocations.

`AIEObjectFifoSplitPass` in
`lib/Dialect/AIE/Transforms/AIEObjectFifoSplit.cpp` creates the intermediate
pool and endpoint operations. `createPool`, `createCoreEndpoint`,
`createDmaEndpoint`, and `createRouteEndpoint` receive the source FIFO or link
location. The pass also records `fifoName` on pools, DMA endpoints, and route
endpoints. A link pool takes the `ObjectFifoLinkOp` location and can represent
several input and output FIFOs.

`AIEObjectFifoAllocatePass` in
`lib/Dialect/AIE/Transforms/AIEObjectFifoAllocate.cpp` creates buffers and
locks with `pool.getLoc()`. It creates physical flows with the route location.
It creates shim allocations with `endpoint.getLoc()`.

`AIEObjectFifoLowerDMAsPass` in
`lib/Dialect/AIE/Transforms/AIEObjectFifoLowerDMAs.cpp` creates DMA programs,
channel starts, BDs, lock uses, and `next_bd` operations with
`endpoint.getLoc()`.

`test/objectFifo-stateful-transform/base/loc_preservation.mlir` verifies that
generated buffers and locks retain the source ObjectFIFO's
`FileLineColLoc`. The test proves source attribution. It does not prove stable
entity identity or complete many-to-many lineage.

`AIE_ObjectFifoPoolOp` and `AIE_ObjectFifoDmaEndpointOp` in
`include/aie/Dialect/AIE/IR/AIEOps.td` retain `fifoName`. This name supports a
useful inferred link. It does not identify several source FIFOs that contribute
to one link pool. It also disappears when transient operations are erased.

## Input interface options

The examples below use `aie-visualize`. A separate project extraction tool
could use the same options. The schema and behavior matter more than the
executable name.

### Repeated files

```text
aie-visualize --emit-project-json \
  --input placed.mlir \
  --input input_with_symbols.mlir \
  --input input_physical.mlir \
  --input input_with_addresses.mlir \
  -o project.json
```

The argument order defines stage order. A caller may add metadata:

```text
--input placed=build/placed.mlir
--input physical=build/input_physical.mlir
```

Advantages:

- The behavior is explicit.
- The form supports files outside one directory.
- Scripts can select a subset of stages.
- The form has no discovery ambiguity.

Costs:

- Long commands become difficult to maintain.
- The form cannot describe keyed artifacts or dependencies without more
  syntax.
- The caller must know the stage order.

Use repeated files as the first multi-stage CLI. Treat this form as shorthand
for an in-memory manifest.

### Manifest

```text
aie-visualize --emit-project-json --manifest aie-project.json -o project.json
```

The manifest should use paths relative to the manifest file. It should permit
files, directory bundles, and embedded semantic graphs. It should also permit
one artifact to contain several devices.

Recommended source manifest shape:

```json
{
  "format": "aie-visualization-input",
  "version": "1.0.0",
  "project": {
    "name": "matmul",
    "root": "."
  },
  "stages": [
    {
      "id": "placed",
      "label": "Placed",
      "artifact": "placed.mlir",
      "kind": "mlir",
      "producer": "placed.mlir"
    },
    {
      "id": "symbols",
      "label": "ObjectFIFO lowering",
      "artifact": "input_with_symbols.mlir",
      "kind": "mlir",
      "producer": "input_with_symbols.mlir",
      "dependsOn": ["placed"]
    },
    {
      "id": "physical",
      "label": "Physical routing",
      "artifact": "input_physical.mlir",
      "kind": "mlir",
      "producer": "input_physical.mlir",
      "dependsOn": ["symbols"]
    }
  ]
}
```

The manifest is the authoritative interface for automation and reproducible
captures. It should not reuse the current checkpoint manifest without a
versioned adapter. The checkpoint manifest describes a resume frontier. A
visualization manifest describes all available stages.

### Directory

```text
aie-visualize --emit-project-json --directory build/aie.mlir.prj \
  -o project.json
```

Directory ingestion should inspect these sources in priority order:

1. A visualization input manifest named `aie-visualization.json`.
2. A versioned `aiecc` artifact index, when one exists.
3. A checkpoint manifest through an explicit compatibility adapter.
4. Known `aiecc` artifact names.
5. Other MLIR files as unclassified stages.

The tool should never infer stage order from lexical filename order or file
timestamps. Known `aiecc` names should map to a versioned stage catalogue.
Unclassified files should appear after classified files and carry
`classification: "unknown"`. The tool should emit a warning for each one.

Recursive discovery should be off by default. `--recursive` may include
keyed per-core and per-device artifacts. The project should group these by
their producer edge and key. For example, all `lowered_<key>.mlir` artifacts
belong to one logical stage with several instances.

Directory ingestion must ignore output files written by the same invocation.
It should also ignore temporary files, editor swap files, and non-regular
files. The implementation should reject symlink escapes from the project root
unless the manifest explicitly allows an external path.

### CLI exclusivity and output

Exactly one input mode should be active:

- One positional file for the current single-stage behavior.
- One or more `--input` options.
- One `--manifest` option.
- One `--directory` option.

`--emit-json` should retain its current single-stage meaning.
`--emit-project-json` should always emit the project wrapper. A one-stage
project remains valid.

The tool should write data to standard output and diagnostics to standard
error. A parse or schema error should produce no partial JSON unless the user
passes `--allow-partial`.

## Stage discovery and order

### Stage identity

A stage is a logical compiler state. An artifact is one file or directory
that represents that state. A stage may have several keyed artifact
instances. One artifact may contain several devices.

Each stage needs these fields:

- `id`: unique within the project and stable under path relocation.
- `label`: display text.
- `role`: an open string such as `source`, `placed`, `symbols`, `physical`,
  `addressed`, `npu-lowered`, or `core-lowered`.
- `producer`: the `aiecc` edge name when known.
- `artifact`: a relative path or embedded data reference.
- `kind`: `mlir`, `directory`, or `semantic-graph`.
- `key`: the edge item key for a keyed artifact.
- `dependsOn`: stage IDs.
- `contentDigest`: SHA-256 over the artifact bytes or canonical directory
  inventory.
- `status`: `ok`, `missing`, `invalid`, or `unsupported`.
- `diagnostics`: stage-local diagnostics.

The stage ID should derive from the producer edge and key when those values
exist. A manifest-supplied ID takes precedence. A filename-derived ID is a
fallback and should carry `identityStrength: "filename"`.

### Ordering algorithm

Use these rules:

1. Preserve manifest array order when it satisfies all `dependsOn` edges.
2. Otherwise compute a stable topological order.
3. Use the `aiecc` stage catalogue to add dependencies for known producers.
4. Use manifest order as the tie breaker.
5. Use normalized artifact path as the last tie breaker.

A cycle is a hard project error. A missing dependency is a hard error unless
`--allow-partial` is present. In partial mode, the stage remains in the
project with `status: "missing"`.

The project should store both `ordinal` and `dependsOn`. The ordinal supports
simple viewers. The dependencies support branched compiler graphs. A linear
stage list alone cannot represent per-core lowering or alternative output
paths.

### `aiecc` artifact index

The long-term source of stage discovery should be an `aiecc` artifact index.
The index can extend graph metadata that already exists in `Graph`,
`EdgeBase`, and `NodeSerializer`. It should record every materialized edge,
item key, path, digest, dependencies, and tool version.

The artifact index should not force `--dump-intermediates`. It should list
only materialized artifacts unless a mode requests stage capture. A later
capture option may request selected semantic stages without materializing
every core compiler scratch file.

## Project JSON

### Root schema

Recommended output shape:

```json
{
  "format": "aie-semantic-project",
  "version": "1.0.0",
  "generator": {
    "name": "aie-visualize",
    "version": "<tool version>",
    "mlirAieRevision": "<revision>"
  },
  "project": {
    "id": "sha256:<manifest identity digest>",
    "name": "matmul",
    "root": "."
  },
  "stages": [],
  "entities": [],
  "provenance": [],
  "diagnostics": []
}
```

Each stage should contain one semantic graph per device and optional graph
variants per runtime sequence:

```json
{
  "id": "physical",
  "ordinal": 2,
  "role": "physical",
  "producer": "input_physical.mlir",
  "artifact": {
    "path": "input_physical.mlir",
    "kind": "mlir",
    "sha256": "..."
  },
  "graphs": [
    {
      "device": "npu",
      "runtimeSequence": "main",
      "semanticGraph": {
        "format": "aie-semantic-graph",
        "version": 1,
        "device": "npu",
        "nodes": [],
        "edges": []
      }
    }
  ]
}
```

The project schema should reference the existing graph schema by value. It
should not duplicate graph nodes in a new project-wide node table. The root
`entities` table should contain only stable entity records that connect stage
nodes. This separation permits the graph schema and project schema to evolve
at different rates.

### Version rules

Use semantic versioning for the project schema:

- A major version changes required meaning or removes data.
- A minor version adds optional fields or enum values.
- A patch version clarifies validation without changing valid documents.

Readers should reject unknown major versions. Readers may accept a newer minor
version after they validate required fields. They should preserve unknown
fields when they transform a document.

The nested semantic graph keeps its own version. A project reader must validate
each nested graph independently.

## Stable identity

### Identity levels

The design needs three identity levels:

1. `graphNodeId` identifies one node in one rendered graph.
2. `stageEntityId` identifies one semantic entity in one stage.
3. `originId` identifies a compiler concept across stages.

Do not reuse one string for all three levels.

Recommended references:

```json
{
  "stage": "physical",
  "entity": "device:npu/buffer:of_in_buff_0",
  "graphNode": "buffer_of_in_buff_0"
}
```

### Stage-local identity

The extractor should derive a `stageEntityId` from semantic keys when the IR
provides them:

- Device: symbol name.
- Tile: device plus physical coordinate, or logical tile symbol before
  placement.
- Symbol operation: operation name plus fully qualified symbol reference.
- Buffer: device, tile identity, and symbol name.
- Lock: device, tile identity, and symbol name before ID assignment; assigned
  lock ID after assignment.
- DMA channel: device, tile, direction, and channel.
- BD: DMA channel plus assigned BD ID. Before assignment, use the explicit
  origin token and descriptor ordinal.
- Flow: explicit origin token. Use source/destination ports only as a fallback.
- Runtime argument: runtime sequence symbol plus argument index.
- Call: enclosing core or runtime sequence plus callee and operation identity.

Sequential walk indexes and source order should be weak fallback identities.
They should never back persistent bookmarks.

### Origin identity

An `originId` should be an opaque string minted by the compiler frontend or
the first lowering pass that needs one. A readable prefix helps diagnostics:

```text
of:device/main/of_in
of-link:device/main/link_3
flow:device/main/17
runtime:device/main/sequence/arg/0
```

The identity must not depend on absolute paths, line numbers, temporary
directories, physical placement, assigned resource IDs, or operation order.

For hand-written MLIR without origin attributes, the extractor may synthesize
an origin identity from this tuple:

```text
module symbol path + operation name + symbol name + normalized source location
```

The project must label this identity `strength: "inferred"`. It should label
an explicit compiler identity `strength: "explicit"`.

### Many-to-many provenance

Store provenance as edges between stage entity references:

```json
{
  "id": "prov:42",
  "from": [
    {"stage": "placed", "entity": "device:npu/objectfifo:in0"},
    {"stage": "placed", "entity": "device:npu/objectfifo-link:join0"}
  ],
  "to": [
    {"stage": "symbols", "entity": "device:npu/pool:join0_pool"},
    {"stage": "symbols", "entity": "device:npu/dma-endpoint:join0_in"}
  ],
  "relation": "lowered-to",
  "pass": "aie-objectfifo-split",
  "evidence": [
    {"kind": "origin-attribute", "value": "of-link:device/npu/join0"},
    {"kind": "location"}
  ],
  "confidence": "exact"
}
```

An edge may have several sources and several targets. This directly represents
join, distribute, merge, split, clone, fold, and shared-resource cases. A list
of one-to-one links loses that structure.

Recommended relations include:

- `lowered-to`
- `split-into`
- `merged-into`
- `allocated-as`
- `routed-as`
- `implemented-by`
- `references`
- `unchanged-from`

The `pass` field should name the transformation when known. The `evidence`
array should distinguish explicit attributes, MLIR locations, symbol names,
endpoint structure, and heuristics. The `confidence` field should be `exact`,
`inferred`, or `ambiguous`.

## MLIR locations and explicit attributes

### `FileLineColLoc`

Use `FileLineColLoc` for navigation to a source text position. IRON already
captures user declarations in `python/helpers/sourceloc.py`. `SourceSite`
creates a file location and can wrap it in a name.

Do not use a file location as a persistent entity ID. Two operations may share
one source statement. Formatting changes can move the location without
changing the entity.

### `NameLoc`

Use `NameLoc` to retain a human semantic name such as an IRON ObjectFIFO name
or worker name. It improves source labels and diagnostics. It remains display
metadata, not identity, unless the symbol contract makes that name unique.

### `CallSiteLoc`

Use `CallSiteLoc` for an emitted operation and the Python or MLIR call chain
that caused the emission. Preserve the call-site structure in project JSON:

```json
{
  "callee": {"file": "library.py", "line": 80, "column": 4},
  "caller": {"file": "design.py", "line": 31, "column": 8}
}
```

Do not flatten the chain into an unordered set. The source pane needs the
callee and caller roles.

### `FusedLoc`

Use `FusedLoc` when one operation represents several source operations. Tile
placement already uses `rewriter.getFusedLoc` in
`lib/Dialect/AIE/Transforms/AIEPlaceTiles.cpp` when it merges logical tiles
into one physical tile.

Preserve each fused part as a separate location record. Preserve optional
`FusedLoc` metadata. A set of `FileLineColLoc` values cannot state which
contributors formed one merge.

### Explicit provenance attributes

Use discardable attributes so the dialect verifier and assembly format do not
need one field per provenance use. Recommended names:

```text
aie.origin.id
aie.origin.parents
aie.origin.role
aie.origin.ordinal
```

Example:

```mlir
aie.buffer(...) {
  aie.origin.id = "of:device/main/of_in/buffer/0",
  aie.origin.parents = ["of:device/main/of_in"],
  aie.origin.role = "objectfifo-buffer",
  aie.origin.ordinal = 0 : i64
}
```

The project extractor should consume the attributes. The visualizer should not
require them for single-file use. Production artifact generation may strip
them after project capture if binary size or textual noise becomes a concern.

Locations and explicit attributes should coexist. A diagnostic uses the
location. Cross-stage queries use the explicit attributes.

## ObjectFIFO provenance requirements

### Split pass

`AIEObjectFifoSplitPass` should mint or propagate identities before it erases
`ObjectFifoCreateOp` and `ObjectFifoLinkOp`.

It should annotate:

- Each pool with all contributing FIFO and link identities.
- Each segment with the contributing FIFO identity and participant ordinal.
- Each core endpoint with the FIFO identity, end role, and consumer ordinal.
- Each DMA endpoint with the FIFO identity, end role, consumer ordinal, and
  selected segment identities.
- Each route endpoint with the FIFO identity and end role.
- Each route with all FIFO and link identities that it implements.

A shared-memory FIFO has one pool and no generated flow. The provenance graph
must represent that absence as an implementation choice, not as missing data.

A join or distribute needs many-to-many ancestry. The link operation and all
participating FIFOs contribute to the shared pool, segments, endpoints, and
route.

### Allocation pass

`AIEObjectFifoAllocatePass` should propagate identities to:

- Buffers: pool identity plus object ordinal.
- Binary locks: pool identity plus object ordinal.
- Semaphore locks: segment identity plus produce or consume role.
- Assigned DMA channels: endpoint identity.
- Circuit or packet flows: route identity plus destination ordinal.
- Shim DMA allocations: shim endpoint identity.

The pass should append ancestry when it merges resources. It should not replace
an existing parent list.

### DMA lowering pass

`AIEObjectFifoLowerDMAsPass` should propagate identities to:

- DMA body: all endpoint identities implemented by that body.
- `aie.dma_start`: endpoint identity and channel identity.
- Each BD block: endpoint identity, buffer ordinal, segment ordinal, and copy
  ordinal.
- `aie.dma_bd`: BD identity and buffer identity.
- `aie.use_lock`: BD identity and lock identity.
- `aie.next_bd`: BD identity.

One endpoint can create several BDs. One BD can implement one segment of one
buffer for one repeated copy. The ordinals make those BDs distinct even before
the assign-BD-ID pass runs.

### Routing pass

The routing lowering should propagate the flow or route origin to each
switchbox connection, shim mux connection, packet rule, and master-set entry.
This enables a query from an ObjectFIFO to every physical route segment. It
also enables reverse queries from a congested switchbox resource to the
ObjectFIFOs that consume it.

### Tests needed upstream

Add tests for:

- One FIFO lowered through DMA to buffers, locks, flow, and BDs.
- One shared-memory FIFO with no flow.
- Broadcast with several physical flow destinations.
- Join and distribute with several parents on one pool.
- Packet flow lowering.
- Registered external buffers and shim allocations.
- Fused locations plus explicit parent identities.
- Identity survival through placement, routing, address assignment, and
  printing plus reparsing.

## Views and queries enabled

The project model enables these views without changing a stage graph:

- Stage timeline with available, missing, invalid, and unchanged stages.
- Side-by-side semantic graph comparison.
- Entity evolution from source declaration to physical resources.
- Reverse attribution from a buffer, lock, BD, DMA channel, or route segment
  to ObjectFIFOs and Python source.
- ObjectFIFO expansion tree for pools, endpoints, buffers, locks, flows, and
  BDs.
- Resource growth between stages.
- Placement movement from logical tiles to physical tiles.
- Route evolution and congestion attribution.
- Runtime argument to DMA channel and buffer mapping.
- Core-specific lowering views for keyed stages.
- Structural diff between two compilations when explicit origin identities
  match.
- Failed-stage display with the last valid stage and stage-local diagnostics.
- Cache attribution that distinguishes current artifacts from restored
  artifacts.

Useful queries include:

```text
show origin of <stage/entity>
show descendants of <stage/entity>
show resources implementing <ObjectFIFO>
show ObjectFIFOs using <tile/channel/lock/BD>
compare <entity> between <stage-a> and <stage-b>
show entities with ambiguous or inferred provenance
```

## Incremental and watch mode

### Transport

Use newline-delimited JSON on standard output. Each record should be a typed
event:

```json
{"type":"snapshot","sequence":1,"project":{}}
{"type":"stage-update","sequence":2,"stage":{}}
{"type":"stage-remove","sequence":3,"stageId":"physical"}
{"type":"diagnostic","sequence":4,"diagnostic":{}}
```

Start with full `snapshot` events. Add stage delta events after consumers
support them. A consumer must ignore an event with a sequence number less than
or equal to the latest applied number.

### Change detection

Watch the manifest and every resolved artifact. For directory mode, also watch
the directory entries that can create or remove a known stage.

Debounce a group of file events. Wait until each changed regular file has a
stable size and modification time across two observations. Then hash the
content. Do not emit an update when the digest remains unchanged.

Reparse only changed stages. Recompute provenance for these sets:

- Links that touch a changed stage.
- Links that touch a stage whose explicit origin index changed.
- Project diagnostics that depend on changed links.

The first implementation may rebuild the complete project after each digest
change. The schema and transport should permit later stage-level updates.

### Atomic writer behavior

`aiecc` may rewrite a file through a temporary path and rename it. The watcher
must track the target path across rename events. A temporary missing file
should not remove a stage until the debounce period expires.

If a stage fails to parse, retain the previous valid stage as
`lastValidStage`. Emit the new diagnostic and mark the live stage invalid.
The viewer should display that distinction. It must not present stale data as
the current stage.

### Cache interaction

Use content digests as the extraction cache key. Include:

- Artifact content digest.
- Semantic graph schema version.
- Extractor version.
- Relevant extraction options.
- Target model identity.

Do not use the `aiecc` device cache location data as current-source
provenance. An artifact index should mark cache restores and record the source
entry digest. The viewer can then state that a stage came from a cache entry.

## Validation and errors

### Boundary validation

Validate at these boundaries:

1. Input manifest syntax and project version.
2. Resolved paths and artifact kinds.
3. MLIR parse and verification.
4. Per-stage semantic graph schema.
5. Project schema.
6. Cross-stage references and provenance endpoints.

Each diagnostic should contain:

- Stable diagnostic code.
- Severity.
- Stage ID when applicable.
- Artifact path when applicable.
- MLIR location when applicable.
- Message.
- Related entity references.

Recommended exit codes:

| Code | Meaning |
| --- | --- |
| 0 | Complete project emitted |
| 1 | CLI usage error |
| 2 | Input or manifest error |
| 3 | Stage parse or verification error |
| 4 | Schema or cross-reference error |
| 5 | Watch transport error |

The current `aie-visualize` exit codes need not change for single-file mode.
Project mode can map its codes through a documented compatibility table.

### Partial projects

Default behavior should reject missing required stages, invalid MLIR, duplicate
stage IDs, dependency cycles, and dangling provenance references.

`--allow-partial` should emit a valid project with stage diagnostics. It should
omit invalid semantic graphs. It should retain stage metadata with an invalid
status. The project root should contain `complete: false`.

Warnings should cover unknown files, inferred stage identities, inferred
provenance, unsupported operations, and unknown minor schema versions.

## Migration from the single-file API

The current command remains valid:

```text
aie-visualize --emit-json input_physical.mlir
```

Its JSON remains `aie-semantic-graph` version 1 during the first project
release.

Add an opt-in project command:

```text
aie-visualize --emit-project-json --input input_physical.mlir
```

This emits one project stage that embeds the existing graph unchanged. A
consumer can therefore adopt the project wrapper without changing its graph
renderer.

Later releases may add:

```text
aie-visualize --emit-project-json --manifest project-input.json
aie-visualize --emit-project-json --directory aie.mlir.prj
aie-visualize --emit-project-json --directory aie.mlir.prj --watch
```

Do not make a directory positional argument. A file and a directory require
different discovery and error behavior. An explicit option avoids an
ambiguous compatibility rule.

Library migration should mirror the CLI:

1. Keep `emitSemanticGraphJson(ModuleOp, ...)`.
2. Add a project data model that contains stage descriptors and nested graph
   JSON values or typed graph objects.
3. Add an extraction entry point that accepts resolved artifacts.
4. Keep filesystem discovery outside the graph analysis library.

## Prototype lessons

### aie-xray

The inspected revision is
`64bedd767f3a6d9dc025435a4d561d97a0e57f6f`.

Useful patterns:

- `packages/schema/src/index.ts` separates `AieDesignModel` from
  `AieProject`.
- The schema validates dangling references at the boundary.
- Project extraction stores ordered stage models and a project provenance
  table.
- `packages/extractor` uses NDJSON watch output and hashes snapshots.
- The renderer consumes JSON and has no MLIR dependency.
- The CLI shares pure provenance queries with the renderer.

Limits that this proposal should address:

- `extract_project.py` recognizes a fixed set of filenames and requires
  `input_physical.mlir`.
- It discovers the source from the most frequent file location.
- Stage roles permit only source, physical, and NPU-lowered forms.
- IDs such as `op:N` and `flow:N` are stable only inside one extraction.
- `provenance.py` maps one lowered entity to one source operation by exact
  location.
- It cannot represent one generated entity with several parents.
- Some entity validation is deliberately lenient because extraction lacks
  first-class entities.

The strict schema boundary and transport model should be retained. The fixed
stage catalogue and one-to-one location provenance should not become the
production contract.

### IRON Design Explorer

The inspected revision is
`2dfda0639e0bd07e80bd961eef64d8791f39a955`.

Useful patterns:

- `src/iron_explorer/lower.py` records known aiecc semantic stages.
- `src/iron_explorer/real.py` selects the nearest stage that can supply each
  view.
- `schema/explorer.schema.json` separates stages, layers, source text, and
  lineage.
- Lineage records a reason such as location, symbol, endpoint, route,
  placement, or reads.
- `src/iron_explorer/sources.py` distinguishes fused location parts from
  call-site chains.
- Partial captures keep missing layers visible.
- The artifact browser joins `aiecc --emit-dot`, profile data, and materialized
  files.

Limits that this proposal should address:

- `lower.py` contains a fixed linear `STAGES` list.
- It reruns `aiecc` with `--dump-intermediates` and then deduplicates files by
  content.
- One layer names one stage, although a view may combine concepts from several
  stages.
- Lineage uses heuristic matching rules and local layer IDs.
- The schema version is an integer draft version.
- The capture process removes some intermediate objects after extraction.

The location parser, partial-stage presentation, and artifact-plan view provide
useful implementation references. The production format should use explicit
compiler identities and graph dependencies.

## Tradeoffs

### Wrapper versus one merged graph

A project wrapper retains exact stage graphs and permits independent schema
versions. A merged graph makes cross-stage traversal direct, but node and edge
kinds become difficult to interpret because logical and physical entities
coexist.

Recommendation: use a wrapper plus a provenance table. A viewer can derive a
merged lineage view when needed.

### Discovery versus manifest

Directory discovery reduces command length. It depends on filename policy and
cannot represent all graph branches. A manifest requires more setup. It gives
reproducible stage identity and order.

Recommendation: make the manifest authoritative. Treat directory discovery as
a manifest generator with diagnostics.

### Locations versus explicit attributes

Locations already survive many passes and support source navigation. They can
be shared, fused, absent, stale through cache reuse, or changed by source
formatting. Explicit attributes add IR text and require pass discipline. They
support exact identity and many-to-many ancestry.

Recommendation: retain both. Use locations for source context. Use explicit
attributes for identity and provenance.

### Full snapshots versus deltas

Full watch snapshots simplify consumers and recovery. Large projects cost more
to serialize and transfer. Deltas reduce transfer size. They require ordering,
recovery, and state management.

Recommendation: start with content-deduplicated full snapshots. Reserve typed
delta events in the protocol.

### Embedded graphs versus graph references

Embedded graphs produce one portable file. Large projects duplicate source and
device data. References reduce size but require a directory bundle or server.

Recommendation: embed by default. Add a bundle form with relative graph files
after project JSON use shows a size problem.

## Phased plan

### Phase 0: schema and fixtures

- Publish `aie-semantic-project` version `1.0.0`.
- Define stage descriptors, nested graphs, entity references, provenance
  hyperedges, diagnostics, and version rules.
- Add fixtures for a linear project, branched keyed stages, a partial project,
  and a provenance ambiguity.
- Add a validator that rejects dangling references and dependency cycles.

This phase changes no compiler passes.

### Phase 1: explicit multi-file extraction

- Add repeated `--input` and `--emit-project-json`.
- Preserve argument order.
- Embed the current semantic graph without schema changes.
- Derive inferred stage entities from symbols, coordinates, assigned resource
  IDs, and structured locations.
- Emit provenance evidence and confidence.
- Keep the single-file CLI and library API unchanged.

This phase provides useful comparison and navigation with incomplete lineage.

### Phase 2: manifest and directory ingestion

- Add `aie-visualization-input` version `1.0.0`.
- Add `--manifest` and `--directory`.
- Implement the known `aiecc` stage catalogue.
- Group keyed stage instances.
- Adapt checkpoint manifests explicitly.
- Emit warnings for unclassified artifacts.

This phase should not require `aiecc` changes.

### Phase 3: ObjectFIFO instrumentation

- Add explicit origin attributes in ObjectFIFO split, allocation, DMA
  lowering, and routing.
- Add many-to-many link ancestry.
- Add identity ordinals for buffers, locks, destinations, segments, copies,
  and BDs.
- Add pass tests for identity survival.
- Prefer explicit provenance over inferred links in the extractor.

This phase supplies exact ObjectFIFO lineage.

### Phase 4: `aiecc` artifact index

- Add a versioned artifact index generated from the compilation graph.
- Record producer edges, item keys, dependencies, digests, cache status, tool
  identity, and paths.
- Add a selected-stage capture option that materializes semantic stages.
- Let directory ingestion consume this index before filename discovery.

This phase supplies exact stage discovery and order.

### Phase 5: watch and incremental extraction

- Add NDJSON full snapshots with sequence numbers.
- Add content digest caches and stage-local reparsing.
- Retain invalid current stages and separate last-valid data.
- Add stage delta events after consumers implement recovery.

### Phase 6: advanced views

- Add ObjectFIFO expansion and reverse resource attribution.
- Add cross-stage semantic diff.
- Add route congestion attribution.
- Add artifact and compiler-plan integration.
- Add links from diagnostics to all contributing source entities.

## Integration instructions

An implementation branch should integrate in this order:

1. Land the project schema and validator without compiler instrumentation.
2. Land repeated-file extraction and single-stage compatibility tests.
3. Land manifest and directory ingestion.
4. Land ObjectFIFO origin attributes in a separate compiler change.
5. Update project extraction to prefer explicit attributes.
6. Land the `aiecc` artifact index in a separate driver change.
7. Add watch mode after static project extraction stabilizes.

Keep compiler instrumentation separate from UI work. This permits command-line
and schema tests to review the provenance contract before a renderer depends on
it.

Downstream consumers should:

- Validate the project root and every nested graph.
- Index stage entities by `(stageId, stageEntityId)`.
- Index provenance by every source and target reference.
- Preserve evidence and confidence in UI and CLI output.
- Treat inferred links as suggestions.
- Treat explicit links as compiler claims.
- Continue to accept `aie-semantic-graph` for single-stage files.

## Upstream instrumentation summary

The first two phases need no upstream pass changes. They can infer useful
links from locations, symbols, `fifoName`, endpoints, and physical resource
keys.

Complete lineage needs these upstream changes:

- ObjectFIFO split must mint and propagate origin identities.
- ObjectFIFO allocation must annotate generated buffers, locks, flows,
  channels, and shim allocations.
- ObjectFIFO DMA lowering must annotate DMA programs and every BD.
- Routing lowering must annotate physical route segments and arbitration
  records.
- `aiecc` should emit a versioned artifact index for exact stage discovery.
- The device cache should record provenance source and cache entry identity.

No renderer should infer exact many-to-many ObjectFIFO lineage from naming
conventions after these attributes exist.