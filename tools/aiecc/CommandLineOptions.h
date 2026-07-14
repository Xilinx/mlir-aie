//===- CommandLineOptions.h ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Command-line interface for the declarative AIE compiler driver (aiecc).
//
// All `cl::opt`/`cl::list`/`cl::alias` definitions live here, keeping the main
// driver (aiecc.cpp) focused on the compilation graph. `main` and the
// graph-builder helpers in aiecc.cpp read these globals directly; centralizing
// the definitions here keeps the driver logic uncluttered.
//
// Flags that are accepted for compatibility but not yet implemented are
// tracked in FEATURE_GAPS.md.
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_COMMANDLINEOPTIONS_H
#define AIECC_COMMANDLINEOPTIONS_H

#include "AIECCVersion.h"

#include "aie/Dialect/AIE/Transforms/AIEPlacer.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <vector>

namespace xilinx::aiecc::cli {

namespace cl = llvm::cl;

//===----------------------------------------------------------------------===//
// Command-line options
//===----------------------------------------------------------------------===//

// Positional arguments: the input `.mlir` plus any host C/C++ source files.
// The MLIR file is the one ending in `.mlir` (or the first non-host-source
// positional); every other C/C++ positional is a host source.
inline cl::list<std::string>
    positionalArgs(cl::Positional,
                   cl::desc("<input mlir> [host source files...]"),
                   cl::ZeroOrMore);
inline cl::opt<std::string>
    hostOutputName("o", cl::desc("Output filename for host compilation"),
                   cl::init("a.out"));
inline cl::opt<std::string>
    outputDir("output-dir",
              cl::desc("Output directory for generated artifacts"),
              cl::init("."));
inline cl::opt<std::string>
    workDir("tmpdir",
            cl::desc("Intermediate workdir (default: <input>.prj in cwd)"),
            cl::init(""));
inline cl::opt<bool> verbose("verbose", cl::desc("Verbose execution"));
inline cl::alias verboseAlias("v", cl::desc("Alias for --verbose"),
                              cl::aliasopt(verbose));
inline cl::opt<bool>
    keepIntermediates("dump-intermediates",
                      cl::desc("Persist intermediates to the output dir"));
inline cl::opt<bool> emitDot(
    "emit-dot",
    cl::desc("Print a GraphViz `dot` description of the compilation graph "
             "(only the edges reachable from the requested outputs) to stdout "
             "and exit, without running the pipeline. Pipe through `dot`."));
inline cl::opt<bool> keepLoc(
    "keep-loc",
    cl::desc("Emit a <bin>.locmap.json sidecar next to each NPU instruction "
             "binary, mapping each transaction word back to the MLIR Location "
             "of the op that produced it (and its regdb register name where "
             "applicable). Off by default."));
inline cl::opt<std::string>
    deviceName("device-name",
               cl::desc("Filter to a single aie.device by symbol name"));
inline cl::opt<int> coresPerCol("cores-per-col",
                                cl::desc("Max cores per column"), cl::init(-1));
inline cl::opt<xilinx::AIE::PlacerType> placerType(
    "placer", cl::desc("Placement algorithm to use"),
    cl::values(clEnumValN(xilinx::AIE::PlacerType::SequentialPlacer,
                          "sequential_placer",
                          "Sequential column-major placement"),
               clEnumValN(xilinx::AIE::PlacerType::SAPlacer, "sa_placer",
                          "Simulated annealing placement")),
    cl::init(xilinx::AIE::PlacerType::SequentialPlacer));
inline cl::opt<int>
    saSeed("sa-seed",
           cl::desc("Random seed for SA placer (0 = non-deterministic)"),
           cl::init(1));
inline cl::opt<std::string> allocScheme("alloc-scheme",
                                        cl::desc("Buffer allocation scheme"));
inline cl::opt<bool> dynamicObjFifos("dynamic-objFifos",
                                     cl::desc("Dynamic objectFIFOs"),
                                     cl::init(true));
inline cl::opt<bool> packetSwObjFifos("packet-sw-objFifos",
                                      cl::desc("Packet-switched objectFIFOs"));
inline cl::opt<bool>
    ctrlPktOverlay("generate-ctrl-pkt-overlay",
                   cl::desc("Route shim-to-tile control overlay"));
inline cl::opt<bool> bf16Emulation("bf16-emulation", cl::desc("Emulate bf16"));
inline cl::opt<std::string> peanoInstallDir("peano",
                                            cl::desc("Peano install dir"));
// Optimization level (0-3), default 2. The peano compile flow caps the
// `opt` mid-end at O1 (higher levels make the SLP vectorizer emit types that
// crash GlobalISel) while `llc` and the peano link step use the raw value;
// O>=3 additionally disables the loop-idiom memset recognizer. See
// buildObjectSubgraph in aiecc.cpp.
inline cl::opt<unsigned> optLevel("O", cl::Prefix,
                                  cl::desc("Optimization level (0-3)"),
                                  cl::init(2));
inline cl::alias optLevelAlias("opt-level", cl::desc("Alias for -O"),
                               cl::aliasopt(optLevel));
inline cl::opt<bool> expandLoadPdis(
    "expand-load-pdis",
    cl::desc("Expand `load_pdi { device_ref }` into explicit write sequences "
             "(avoids per-switch full PDI reload)"));
inline cl::opt<bool> xchesscc(
    "xchesscc",
    cl::desc("Compile cores with the Chess toolchain (xchesscc) instead of "
             "Peano"),
    cl::init(true));
inline cl::opt<bool> noXchesscc(
    "no-xchesscc",
    cl::desc("Compile cores with Peano instead of the Chess toolchain "
             "(implies --no-xbridge)"));
inline cl::opt<bool> xbridge(
    "xbridge",
    cl::desc("Link cores with the Chess toolchain (xbridge/BCF) instead of "
             "Peano lld"),
    cl::init(true));
inline cl::opt<bool> noXbridge(
    "no-xbridge",
    cl::desc("Link cores with Peano lld instead of the Chess toolchain "
             "(xbridge/BCF)"));
inline cl::opt<std::string> aietoolsDir(
    "aietools",
    cl::desc("Path to the aietools (Vitis AIE) install dir; auto-discovered "
             "from $AIETOOLS_ROOT or xchesscc on PATH when unset"));
inline cl::opt<bool> unified(
    "unified",
    cl::desc("Compile all cores of a device together into one shared object; "
             "link each core separately against it."));
inline cl::opt<bool> noUnified(
    "no-unified",
    cl::desc("Compile cores independently (negates --unified; the default)"));

//===----------------------------------------------------------------------===//
// Stage gating
//===----------------------------------------------------------------------===//
// Enable/disable the core compile and link stages. Both default on. Disabling
// either means no per-core ELFs are freshly built; the driver instead reuses
// the `elf_file` attributes already present in the input IR (pre-baked ELFs).

inline cl::opt<bool> compile(
    "compile",
    cl::desc("Compile AIE cores (default; --no-compile reuses pre-baked ELFs)"),
    cl::init(true));
inline cl::opt<bool> noCompile("no-compile",
                               cl::desc("Disable compiling of AIE cores"));
inline cl::opt<bool>
    link("link",
         cl::desc("Link AIE cores (default; --no-link reuses pre-baked ELFs)"),
         cl::init(true));
inline cl::opt<bool> noLink("no-link", cl::desc("Disable linking of AIE code"));

// Runtime sequence to compile (empty = all). Filters the per-sequence NPU
// instruction stream by RuntimeSequenceOp symbol name.
inline cl::opt<std::string>
    sequenceName("sequence-name",
                 cl::desc("Runtime sequence name to compile (default: all)"));

// Skip the aie-materialize-runtime-sequences pass in NPU lowering.
inline cl::opt<bool> noMaterialize(
    "no-materialize",
    cl::desc("Skip aie-materialize-runtime-sequences pass in NPU lowering"));

// Parallelism. Independent edges (e.g. the per-core compile/link subprocesses)
// are dispatched concurrently to a pool of this size. 1 runs fully
// sequentially; 0 auto-detects the hardware concurrency.
inline cl::opt<unsigned> numThreads(
    "j", cl::Prefix,
    cl::desc("Number of parallel compilation threads (0 = auto-detect)"),
    cl::init(0));
inline cl::alias numThreadsAlias("nthreads", cl::desc("Alias for -j"),
                                 cl::aliasopt(numThreads));

//===----------------------------------------------------------------------===//
// Host compilation options
//===----------------------------------------------------------------------===//
// Compile the user-provided host source files (positional C/C++ args) into a
// host executable that drives the array via libxaiengine. The host program
// `#include`s the generated `aie_inc.cpp` array-configuration source, so the
// workdir (where it lands) is added to the include path automatically.
//
// Any arguments following a `--` separator on the command line are forwarded
// verbatim to the host compiler (all AIE architectures). Arguments *before*
// `--` are parsed strictly: unknown options are rejected, not silently
// forwarded.

inline cl::opt<bool> compileHost(
    "compile-host",
    cl::desc("Compile the host program from the positional C/C++ source "
             "files into a host executable"));
inline cl::opt<bool> noCompileHost(
    "no-compile-host",
    cl::desc("Disable compiling of the host program (negates --compile-host)"));
inline cl::opt<std::string>
    hostTarget("host-target", cl::desc("Target triple of the host program"),
               cl::init("x86_64-linux-gnu"));
inline cl::opt<std::string>
    sysroot("sysroot", cl::desc("Sysroot for host cross-compilation"));
inline cl::opt<bool> aiesim(
    "aiesim",
    cl::desc("Generate an AIE simulator Work folder (requires --xbridge)"));
inline cl::opt<bool>
    noAiesim("no-aiesim",
             cl::desc("Do not generate an AIE simulator Work folder"));
inline cl::list<std::string>
    hostIncludeDirs("I", cl::Prefix, cl::desc("Host include directory"));
inline cl::list<std::string>
    hostLibDirs("L", cl::Prefix, cl::desc("Host library search directory"));
inline cl::list<std::string> hostLibs("l", cl::Prefix,
                                      cl::desc("Host link library"));

// Host-compiler passthrough: the arguments following a `--` separator on the
// command line, forwarded verbatim to the host compiler (all AIE
// architectures). Not a cl::opt — the `--` split happens in main before cl
// parsing sees the tail, and main populates this global from argv.
inline std::vector<std::string> hostPassthroughArgs;

//===----------------------------------------------------------------------===//
// Output selection
//===----------------------------------------------------------------------===//
// Which compilation artifacts to emit. Each `aie-generate-*` flag selects one
// output; the adjacent `*-name` options set its filename template ({0} expands
// to the device / sequence key). These are the only options that decide what
// the driver produces — the graph itself is declared unconditionally and the
// engine prunes back from whatever is selected here.

inline cl::opt<bool> generateNpuInsts(
    "aie-generate-npu-insts",
    cl::desc("Generate NPU instructions (.bin) per runtime sequence"));
inline cl::opt<std::string> npuInstsName(
    "npu-insts-name",
    cl::desc("Output NPU insts filename template (use {0} for multi-device)"),
    cl::init("insts_{0}.bin"));

inline cl::opt<bool> generateInputWithAddresses(
    "aie-generate-input-with-addresses",
    cl::desc("Emit input_with_addresses.mlir (the last MLIR form before "
             "aie-translate) into the work directory. Downstream tooling — "
             "notably the trace parser and the JIT DMA-size validator — reads "
             "it from the .prj."));

inline cl::opt<bool> generateScratchpadParams(
    "emit-scratchpad-parameters",
    cl::desc("Emit params.txt (scratchpad runtime-parameter descriptions) into "
             "the work directory (.prj), for the host ParameterScratchpad "
             "runtime."));

inline cl::opt<bool> generateElf(
    "aie-generate-elf",
    cl::desc("Emit a per-device instruction ELF (NPU instruction stream "
             "assembled via aiebu-asm)"));
inline cl::opt<std::string> elfName(
    "elf-name",
    cl::desc("Output instruction ELF filename (use {0} for multi-device)"),
    cl::init("design.elf"));

inline cl::opt<bool> generateCdo(
    "aie-generate-cdo",
    cl::desc("Emit the per-device CDO binaries (libxaie v2 configuration)"));

inline cl::opt<bool>
    generatePdi("aie-generate-pdi",
                cl::desc("Emit a per-device PDI binary (via bootgen)"));
inline cl::opt<std::string>
    pdiName("pdi-name",
            cl::desc("Output PDI filename template (use {0} for multi-device)"),
            cl::init("{0}.pdi"));

inline cl::opt<bool>
    generateTxn("aie-generate-txn",
                cl::desc("Emit the per-device transaction configuration MLIR"));
inline cl::opt<std::string> txnName(
    "txn-name",
    cl::desc("Output transaction MLIR filename template (use {0} for device)"),
    cl::init("{0}_transaction.mlir"));

inline cl::opt<bool> generateCtrlpkt(
    "aie-generate-ctrlpkt",
    cl::desc("Emit per-device control-packet configuration artifacts"));
inline cl::opt<std::string> ctrlpktName(
    "ctrlpkt-name",
    cl::desc("Output control-packet binary filename template (use {0})"),
    cl::init("{0}_ctrlpkt.bin"));
inline cl::opt<std::string> ctrlpktDmaSeqName(
    "ctrlpkt-dma-seq-name",
    cl::desc("Output control-packet DMA-sequence binary template (use {0})"),
    cl::init("{0}_ctrlpkt_dma_seq.bin"));
inline cl::opt<std::string> ctrlpktElfName(
    "ctrlpkt-elf-name",
    cl::desc("Output combined control-packet ELF filename template (use {0})"),
    cl::init("{0}_ctrlpkt.elf"));

inline cl::opt<bool> generateXclbin("aie-generate-xclbin",
                                    cl::desc("Generate an xclbin per device"));
inline cl::opt<std::string> xclbinName(
    "xclbin-name",
    cl::desc("Output xclbin filename template (use {0} for multi-device)"),
    cl::init("aie.xclbin"));
inline cl::opt<std::string> xclbinKernelName("xclbin-kernel-name",
                                             cl::desc("Kernel name in xclbin"),
                                             cl::init("MLIR_AIE"));
inline cl::opt<std::string>
    xclbinInstanceName("xclbin-instance-name",
                       cl::desc("Instance name in xclbin"),
                       cl::init("MLIRAIE"));
inline cl::opt<std::string> xclbinKernelId("xclbin-kernel-id",
                                           cl::desc("Kernel ID in xclbin"),
                                           cl::init("0x901"));
inline cl::opt<std::string> xclbinInput(
    "xclbin-input",
    cl::desc("Input xclbin to extend with this design's kernel/PDI instead of "
             "creating a new one from scratch"));

inline cl::opt<bool> generateFullElf(
    "generate-full-elf",
    cl::desc(
        "Bundle all PDIs + NPU insts into one combined ELF via aiebu-asm"));
inline cl::opt<std::string>
    fullElfName("full-elf-name", cl::desc("Output filename for combined ELF"),
                cl::init("aie.elf"));

// General-purpose output selector: request one or more graph outputs by their
// public name (repeatable, or comma-separated). Complements the dedicated
// `aie-generate-*` flags for outputs that don't have one (e.g. the core
// `objects`/`elfs`). It also doubles as the graph-cut specification: the edges
// named here are exactly the artifacts a `--checkpoint` captures (and a
// `--resume` reloads). The set of recognized names is defined where the graph
// is built; an unknown name is a hard error.
inline cl::list<std::string> getOutputs(
    "get",
    cl::desc("Request graph output(s) by name (repeatable / comma-separated); "
             "also the cut points for --checkpoint"),
    cl::CommaSeparated, cl::value_desc("name"));
inline cl::alias getOutputsAlias("g", cl::desc("Alias for --get"),
                                 cl::aliasopt(getOutputs));
// Restrict which item keys of the requested outputs are written to the output
// directory (a key identifies a device/core/sequence instance). Empty writes
// every key. Pairs with --get for surgical single-instance extraction; does not
// affect what a --checkpoint captures (a checkpoint always snapshots all keys).
inline cl::list<std::string> getKeys(
    "get-key",
    cl::desc("Restrict generated outputs to these item keys (repeatable / "
             "comma-separated); empty emits all keys"),
    cl::CommaSeparated, cl::value_desc("key"));

//===----------------------------------------------------------------------===//
// Backward-compatibility / deferred flags
//===----------------------------------------------------------------------===//
// Accepted so existing invocations and scripts don't fail with "unknown
// argument". Most are no-ops; the ones with real behavior this driver does not
// yet implement are tracked in FEATURE_GAPS.md.

// Print version info and exit.
inline cl::opt<bool> showVersion("aie-version",
                                 cl::desc("Show version information and exit"));
inline cl::opt<bool> dryRun("n", cl::desc("Dry run"));
// Deprecated/ignored.
inline cl::opt<bool> profile("profile", cl::desc("Deprecated, ignored"));
inline cl::opt<bool> progress(
    "progress",
    cl::desc("Show single-line execution progress: overwrite one status line "
             "'(x/y) <edge> (a inputs)' per executed edge, where x/y is the "
             "step out of the total reachable edges and a is the number of "
             "input items the edge consumes"));
// Graph cut / checkpoint & resume. `--checkpoint=<dir>` dumps the artifacts
// selected by `--get` (narrowed by `--get-key`) plus a `manifest.json`
// describing them into <dir> after a successful run — a "prefix" of the build.
// `--resume=<manifest.json>` rebuilds the graph from the manifest's recorded
// argv, reloads those artifacts from disk instead of recomputing them, and
// continues the "suffix" (optionally narrowed with `--get`). Not tied to any
// failure mode; the cut is wherever `--get` points.
inline cl::opt<std::string> checkpointDir(
    "checkpoint",
    cl::desc("After a successful run, write the --get artifacts + a "
             "manifest.json to this dir (a resumable graph cut)"),
    cl::value_desc("dir"), cl::init(""));
inline cl::opt<std::string> resumeManifest(
    "resume",
    cl::desc("Resume from a checkpoint manifest.json (rebuilds the graph from "
             "its argv; only execution-only flags may accompany it)"),
    cl::value_desc("manifest.json"), cl::init(""));

// On-failure reproducer ("repeater") scripts. When enabled, a failing run dumps
// a checkpoint of the failed edge's inputs (a manifest.json + the artifacts)
// and prints a `To reproduce, run: aiecc --resume=...` command, so the failure
// can be replayed without rerunning the whole prefix. Off by default; --disable
// wins over --enable.
inline cl::opt<bool> enableRepeaterScripts(
    "enable-repeater-scripts",
    cl::desc("On failure, dump a resumable reproducer checkpoint and print a "
             "'To reproduce' command"));
inline cl::opt<bool> disableRepeaterScripts(
    "disable-repeater-scripts",
    cl::desc("Disable on-failure reproducer checkpoints (the default)"));
inline cl::opt<std::string> repeaterOutputDir(
    "repeater-output-dir",
    cl::desc("Directory for the on-failure reproducer checkpoint "
             "(default: <workdir>/repeater)"),
    cl::value_desc("dir"), cl::init(""));

//===----------------------------------------------------------------------===//
// Resolved options
//===----------------------------------------------------------------------===//
// A handful of options are not independent, so main calls resolveOptions()
// once, right after command-line parsing and before building the graph. It
// settles the coupled Chess/Peano toolchain selection in place (xchesscc/
// xbridge) and folds each enable/`--no-*` flag pair into one of the globals
// below. These live alongside the command-line options so the rest of the
// driver reads them exactly like any other option.

// --aiesim is active unless explicitly negated.
inline bool wantAiesim = false;
// Fresh per-core ELFs are built only when both the compile and link stages are
// enabled; otherwise the driver reuses the pre-baked `elf_file` attributes
// already present in the input IR.
inline bool doBuildElfs = false;
// Core objects are compiled whenever the compile stage is enabled, regardless
// of linking: with --no-link the driver stops at the per-core objects.
inline bool doCompileObjects = false;
// Compile all cores of a device into one shared object (negated by
// --no-unified).
inline bool doUnified = false;
// Compile the host program (negated by --no-compile-host).
inline bool doCompileHost = false;

// Resolve inter-option coupling and populate the resolved-option globals above.
// Chess compile and xbridge link are coupled: xbridge can only link
// Chess-compiled objects (BCF expects Chess-specific sections), and Chess
// objects cannot be linked by Peano lld. Asking for either selects the full
// Chess flow; explicit --no-xchesscc/--no-xbridge force the Peano flow and win
// over the coupling (--no-xchesscc implies --no-xbridge because Peano objects
// cannot be linked by the Chess bridge). --aiesim additionally implies the
// Chess toolchain because the simulator consumes Chess-built core ELFs, and
// only errors when the user explicitly forced the Peano flow.
//
// Returns false (after a diagnostic) if the requested combination is impossible
// — currently only --aiesim together with an explicitly forced Peano flow.
inline bool resolveOptions() {
  if (noXchesscc) {
    xchesscc = false;
    xbridge = false;
  }
  if (noXbridge)
    xbridge = false;
  if ((xchesscc || xbridge) && !noXchesscc && !noXbridge) {
    xchesscc = true;
    xbridge = true;
  }

  wantAiesim = aiesim && !noAiesim;
  if (wantAiesim && !xbridge) {
    if (noXbridge || noXchesscc) {
      llvm::errs() << "aiecc: --aiesim requires --xbridge (the AIE simulator "
                      "consumes Chess-compiled cores)\n";
      return false;
    }
    xchesscc = true;
    xbridge = true;
  }

  doCompileObjects = compile && !noCompile;
  doBuildElfs = doCompileObjects && (link && !noLink);
  doUnified = unified && !noUnified;
  doCompileHost = compileHost && !noCompileHost;
  return true;
}

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

inline void printVersion(llvm::raw_ostream &os) {
  os << "aiecc (mlir-aie declarative driver)\n";
  os << "  git SHA:  " << AIECC_GIT_SHA << "\n";
  os << "  compiled: " << __DATE__ << " " << __TIME__ << "\n";
}

// A positional argument is a host source file when it has a C/C++ extension.
inline bool isHostSourceFile(llvm::StringRef name) {
  return name.ends_with(".c") || name.ends_with(".cpp") ||
         name.ends_with(".cc") || name.ends_with(".cxx") ||
         name.ends_with(".C");
}

// The MLIR input among the positionals: the first `.mlir`, else the first
// positional that isn't a host source file, else the first positional.
inline std::string getInputFilename() {
  for (const auto &arg : positionalArgs)
    if (llvm::StringRef(arg).ends_with(".mlir"))
      return arg;
  for (const auto &arg : positionalArgs)
    if (!isHostSourceFile(arg))
      return arg;
  return positionalArgs.empty() ? std::string() : positionalArgs.front();
}

// The absolutized intermediate work directory (the `.prj` folder). Defaults to
// `<input-basename>.prj` in the cwd when --tmpdir is unset. Absolutized so
// paths baked into the IR are cwd-invariant, and computed here so every caller
// (main and the graph builders) agrees on a single location.
inline std::string getWorkDir() {
  llvm::SmallString<256> absWork(
      workDir.empty()
          ? llvm::sys::path::filename(getInputFilename()).str() + ".prj"
          : workDir.getValue());
  llvm::sys::fs::make_absolute(absWork);
  return std::string(absWork);
}

// Every positional C/C++ source file (i.e. all positionals except the MLIR
// input).
inline std::vector<std::string> getHostSourceFiles() {
  std::string mlirFile = getInputFilename();
  std::vector<std::string> hostFiles;
  for (const auto &arg : positionalArgs)
    if (arg != mlirFile && isHostSourceFile(arg))
      hostFiles.push_back(arg);
  return hostFiles;
}

// Whether any host source file was provided, either positionally or in the
// `--` passthrough tail. Used to decide whether host compilation has work to
// do; the passthrough tail itself is forwarded verbatim to the host compiler.
inline bool hasHostSourceFiles() {
  if (!getHostSourceFiles().empty())
    return true;
  for (const auto &arg : hostPassthroughArgs)
    if (isHostSourceFile(arg))
      return true;
  return false;
}

//===----------------------------------------------------------------------===//
// Checkpoint / resume manifest handling
//===----------------------------------------------------------------------===//
// A checkpoint restores a graph cut from disk. Its manifest records the argv
// that built the cut (so a resume reconstructs an identical graph — same
// device, options and item keys) plus a frontier: for each cut edge output, the
// {name, key} it belongs to and the saved artifact `path`. Manifest parsing and
// the associated command-line rebuild live here with the rest of the CLI
// handling rather than in the driver.

struct CheckpointEntry {
  std::string name; // producing edge's output-name template
  std::string key;  // item key within that edge
  std::string path; // artifact path, relative to the manifest directory
};

struct ResumeState {
  bool active = false;
  std::string manifestDir;
  std::vector<CheckpointEntry> frontier;
};

// Execution-only flags that may accompany --resume: they don't reshape the
// graph, only what the suffix run targets (--get/--get-key) or captures
// (--checkpoint), plus reporting/parallelism. Returns 0 (rejected), 1
// (self-contained token), or 2 (also consumes the following token).
inline int resumePassthroughKind(llvm::StringRef a) {
  if (a == "-v" || a == "--verbose" || a == "--progress")
    return 1;
  if (a.starts_with("--get=") || a.starts_with("-g=") ||
      a.starts_with("--get-key=") || a.starts_with("-j") ||
      a.starts_with("--checkpoint="))
    return 1;
  if (a == "--get" || a == "-g" || a == "--get-key" || a == "--checkpoint")
    return 2;
  return 0;
}

// If argv contains `--resume=<manifest>`, parse the manifest and rebuild the
// effective command line (the manifest's recorded argv plus the execution-only
// flags passed alongside --resume; any graph-shaping arg is rejected).
// Otherwise return argv[0..argc) unchanged with `resume` inactive. `graphArgv`
// is set to the argv that defines the graph — recorded verbatim by any
// checkpoint this run writes — i.e. the manifest argv on resume, else the
// original argv. Returns std::nullopt after diagnosing an error.
inline std::optional<std::vector<std::string>>
resolveCommandLine(int argc, char **argv, ResumeState &resume,
                   std::vector<std::string> &graphArgv) {
  std::string resumePath;
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef a(argv[i]);
    if (a.consume_front("--resume="))
      resumePath = a.str();
    else if (a == "--resume" && i + 1 < argc)
      resumePath = argv[i + 1];
  }

  if (resumePath.empty()) {
    graphArgv.assign(argv, argv + argc);
    return graphArgv;
  }

  auto buf = llvm::MemoryBuffer::getFile(resumePath);
  if (!buf) {
    llvm::errs() << "aiecc: cannot read resume manifest '" << resumePath
                 << "'\n";
    return std::nullopt;
  }
  auto parsed = llvm::json::parse((*buf)->getBuffer());
  llvm::json::Object *obj = parsed ? parsed->getAsObject() : nullptr;
  if (!obj) {
    llvm::errs() << "aiecc: invalid resume manifest '" << resumePath << "'\n";
    return std::nullopt;
  }
  std::vector<std::string> manifestArgv;
  if (auto *arr = obj->getArray("argv"))
    for (auto &v : *arr)
      if (auto s = v.getAsString())
        manifestArgv.push_back(s->str());
  if (manifestArgv.empty()) {
    llvm::errs() << "aiecc: resume manifest has no argv\n";
    return std::nullopt;
  }
  if (auto *fr = obj->getArray("frontier"))
    for (auto &v : *fr)
      if (auto *fo = v.getAsObject()) {
        auto get = [&](llvm::StringRef k) {
          auto s = fo->getString(k);
          return s ? s->str() : std::string();
        };
        resume.frontier.push_back({get("name"), get("key"), get("path")});
      }
  resume.manifestDir = llvm::sys::path::parent_path(resumePath).str();
  resume.active = true;

  // Effective argv: program name + manifest argv (minus its own program name) +
  // the execution-only flags passed alongside --resume. Anything else is a hard
  // error so a resume can't silently diverge from the checkpointed graph.
  std::vector<std::string> eff;
  eff.push_back(argv[0]);
  eff.insert(eff.end(), manifestArgv.begin() + 1, manifestArgv.end());
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef a(argv[i]);
    if (a == "--resume") {
      ++i;
      continue;
    }
    if (a.starts_with("--resume="))
      continue;
    int kind = resumePassthroughKind(a);
    if (kind == 0) {
      llvm::errs() << "aiecc: --resume rejects other arguments: " << a << "\n";
      return std::nullopt;
    }
    eff.push_back(a.str());
    if (kind == 2 && i + 1 < argc)
      eff.push_back(argv[++i]);
  }

  graphArgv = manifestArgv;
  return eff;
}

} // namespace xilinx::aiecc::cli

#endif // AIECC_COMMANDLINEOPTIONS_H
