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

// Positional arguments: exactly one input MLIR file before a `--` separator (or
// no inputs for `--emit-dot`); positionals and flags after the `--` separator
// are forwarded to the host code compiler.
inline cl::list<std::string>
    positionalArgs(cl::Positional,
                   cl::desc("<input mlir> [-- <host cc args>...]"),
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
// Selected with --get-locmap: emit a <bin>.locmap.json sidecar next to each NPU
// instruction binary, mapping each transaction word back to the MLIR Location
// of the op that produced it (and its regdb register name where applicable).
inline bool keepLoc = false;
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
inline cl::opt<bool> loadPdiToCtrlPkt(
    "load-pdi-to-ctrl-pkt",
    cl::desc("Rewrite `load_pdi { device_ref }` ops into DMA tasks that stream "
             "the device configuration as control packets (requires a control "
             "overlay; implies --generate-ctrl-pkt-overlay; mutually exclusive "
             "with --expand-load-pdis)"));
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
// Compile the user-provided host source files into a host executable that
// drives the array via libxaiengine. The host program `#include`s the generated
// `aie_inc.cpp` array-configuration source, so the workdir (where it lands) is
// added to the include path automatically.
//
// Host source files and host-compiler flags are passed after a `--` separator
// and forwarded verbatim to the host compiler (all AIE architectures).

// Selected with --get-host: compile the host program from the host C/C++ source
// files given after `--` into a host executable.
inline bool generateHost = false;
inline cl::opt<std::string>
    hostTarget("host-target", cl::desc("Target triple of the host program"),
               cl::init("x86_64-linux-gnu"));
inline cl::opt<std::string>
    sysroot("sysroot", cl::desc("Sysroot for host cross-compilation"));
// Selected with --get-aiesim: generate an AIE simulator Work folder (requires
// --xbridge).
inline bool generateAiesim = false;
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
// Which compilation artifacts to emit. Every output is selected uniformly with
// `--get-<name>` (see outputSelectors() and applyOutputSelectorFlags() below),
// which sets one of the bools here; the adjacent `*-name` options set the
// artifact's filename template ({0} expands to the device / sequence key).

inline bool generateNpuInsts = false;
inline cl::opt<std::string> npuInstsName(
    "npu-insts-name",
    cl::desc("Output NPU insts filename template (use {0} for multi-device)"),
    cl::init("insts_{0}.bin"));

// Emit the per-core ELFs as an output. Cores are still compiled on demand for
// any artifact that embeds them (e.g. --get-xclbin); this flag additionally
// writes them out. (Request the pre-link objects instead with
// --get=objects_{0}.o.)
inline bool generateCoreElfs = false;

inline bool generateInputWithAddresses = false;

inline bool generateScratchpadParams = false;

inline bool generateElf = false;
inline cl::opt<std::string> elfName(
    "elf-name",
    cl::desc("Output instruction ELF filename (use {0} for multi-device)"),
    cl::init("design.elf"));

inline bool generateCdo = false;

inline bool generatePdi = false;
inline cl::opt<std::string>
    pdiName("pdi-name",
            cl::desc("Output PDI filename template (use {0} for multi-device)"),
            cl::init("{0}.pdi"));

inline bool generateTxn = false;
inline cl::opt<std::string> txnName(
    "txn-name",
    cl::desc("Output transaction MLIR filename template (use {0} for device)"),
    cl::init("{0}_transaction.mlir"));

inline bool generateCtrlpkt = false;
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

inline bool generateXclbin = false;
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

// Select which xclbinutil packages the .xclbin. Accepts an absolute/relative
// path or a bare program name (looked up on PATH). Also settable via the
// AIE_XCLBINUTIL environment variable (this flag takes precedence). When set
// but unusable, xclbin generation fails loudly rather than silently falling
// back to a PATH lookup -- this is what lets a "pure HRX" flow guarantee the
// bundled hrx-xclbinutil is used and a "pure XRT" flow guarantee XRT's.
inline cl::opt<std::string> xclbinutilPath(
    "xclbinutil-path",
    cl::desc("Path or name of the xclbinutil used to package the .xclbin "
             "(overrides PATH lookup; also settable via AIE_XCLBINUTIL)"));

inline bool generateFullElf = false;
inline cl::opt<std::string>
    fullElfName("full-elf-name", cl::desc("Output filename for combined ELF"),
                cl::init("aie.elf"));

// General-purpose output selector: request one or more graph outputs by their
// public edge name (repeatable, or comma-separated). Complements the named
// artifact shorthands (`--get-<name>`, see outputSelectors()) for outputs that
// don't have one (e.g. the core `objects`/`elfs`). The set of recognized names
// is defined where the graph is built; an unknown name is a hard error.
inline cl::list<std::string> getOutputs(
    "get",
    cl::desc("Request graph output(s) by edge name (repeatable / "
             "comma-separated); named artifacts also have --get-<name> "
             "shorthands (e.g. --get-xclbin)"),
    cl::CommaSeparated, cl::value_desc("name"));
inline cl::alias getOutputsAlias("g", cl::desc("Alias for --get"),
                                 cl::aliasopt(getOutputs));
// Checkpoint cut points: the edges (named like --get) whose outputs a
// `--checkpoint` captures and a `--resume` reloads. Built like any requested
// output, then snapshotted. Separate from --get so selecting an output does not
// implicitly cut the graph there.
inline cl::list<std::string> cutOutputs(
    "cut",
    cl::desc("Graph edge(s) to cut at for --checkpoint (repeatable / "
             "comma-separated); named like --get"),
    cl::CommaSeparated, cl::value_desc("name"));

//===----------------------------------------------------------------------===//
// Named artifact shorthands: --get-<name>
//===----------------------------------------------------------------------===//
// Each selector maps a nice name (exposed as `--get-<niceName>`) to the graph
// edge it selects and the driver bool it sets. The edge name is the artifact's
// default output-name template and may contain `{0}`. applyOutputSelectorFlags
// rewrites `--get-<niceName>` into the matching selection before command-line
// parsing and rejects any unknown `--get-<name>`.
struct OutputSelector {
  llvm::StringRef niceName;
  llvm::StringRef edgeName;
  bool *flag;
};

inline llvm::ArrayRef<OutputSelector> outputSelectors() {
  static const OutputSelector table[] = {
      {"input-with-addresses", "input_with_addresses.mlir",
       &generateInputWithAddresses},
      {"scratchpad-parameters", "params.txt", &generateScratchpadParams},
      {"core-elfs", "elfs_{0}.elf", &generateCoreElfs},
      {"npu-insts", "insts_{0}.bin", &generateNpuInsts},
      {"elf", "design.elf", &generateElf},
      {"cdo", "cdo_{0}", &generateCdo},
      {"pdi", "{0}.pdi", &generatePdi},
      {"txn", "{0}_transaction.mlir", &generateTxn},
      {"ctrlpkt", "{0}_ctrlpkt.bin", &generateCtrlpkt},
      {"xclbin", "aie.xclbin", &generateXclbin},
      {"full-elf", "aie.elf", &generateFullElf},
      {"locmap", "insts_{0}.bin.locmap.json", &keepLoc},
      {"host", "a.out", &generateHost},
      {"aiesim", "aiesim_{0}.stamp", &generateAiesim},
  };
  return table;
}

// Resolve the `--get-<niceName>` shorthands in `args` before cl parsing: set
// each recognized selector's bool and drop its token; a token after a `--`
// separator (host passthrough) is left untouched. Returns false after
// diagnosing an unknown `--get-<name>`.
inline bool applyOutputSelectorFlags(std::vector<std::string> &args) {
  std::vector<std::string> kept;
  kept.reserve(args.size());
  bool afterSeparator = false;
  for (std::string &arg : args) {
    llvm::StringRef a(arg);
    if (a == "--")
      afterSeparator = true;
    llvm::StringRef nice = a;
    if (!afterSeparator && nice.consume_front("--get-") && !nice.empty()) {
      const OutputSelector *sel = nullptr;
      for (const OutputSelector &s : outputSelectors())
        if (s.niceName == nice) {
          sel = &s;
          break;
        }
      if (!sel) {
        llvm::errs() << "aiecc: unknown output selector '--get-" << nice
                     << "'; available selectors are:\n";
        for (const OutputSelector &s : outputSelectors())
          llvm::errs() << "  --get-" << s.niceName << "  (" << s.edgeName
                       << ")\n";
        return false;
      }
      *sel->flag = true;
      continue;
    }
    kept.push_back(std::move(arg));
  }
  args = std::move(kept);
  return true;
}

//===----------------------------------------------------------------------===//
// Diagnostics, dry-run, progress, and checkpoint/resume
//===----------------------------------------------------------------------===//
// Version/diagnostic flags, the `-n` dry run, execution-progress reporting, and
// the checkpoint/resume + on-failure reproducer machinery.

// Print version info and exit.
inline cl::opt<bool> showVersion("aie-version",
                                 cl::desc("Show version information and exit"));
inline cl::opt<bool> dryRun("n", cl::desc("Dry run"));
// Print the wall-clock time each edge took to execute at the end of the run.
inline cl::opt<bool>
    profile("profile",
            cl::desc("Print a per-edge execution-time summary at the end"));
inline cl::opt<bool> progress(
    "progress",
    cl::desc("Show single-line execution progress: overwrite one status line "
             "per executed edge. On by default; disable with "
             "--no-progress. Suppressed under --verbose."));
inline cl::opt<bool> noProgress(
    "no-progress",
    cl::desc("Disable the default single-line execution progress output"));
// Graph cut / checkpoint & resume. `--checkpoint=<dir>` dumps the artifacts
// selected by `--cut` plus a `manifest.json` describing them into <dir> after a
// successful run — a "prefix" of the build. `--resume=<manifest.json>` rebuilds
// the graph from the manifest's recorded argv, reloads those artifacts from
// disk instead of recomputing them, and continues the "suffix" (optionally
// narrowed with `--get`). Not tied to any failure mode; the cut is wherever
// `--cut` points.
inline cl::opt<std::string> checkpointDir(
    "checkpoint",
    cl::desc("After a successful run, write the --cut artifacts + a "
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
// A handful of options are coupled, e.g. the enable/`--no-*` flags.
// main calls resolveOptions() once to resolve these.

// Whether to generate the AIE simulator Work folder (off unless --get-aiesim).
inline bool wantAiesim = false;
// Compile all cores of a device together into one shared object (negated by
// --no-unified).
inline bool doUnified = false;
// Compile the host program (selected by --get-host).
inline bool doCompileHost = false;

// Resolve inter-option coupling and populate the resolved-option globals above.
//
// Returns false (after a diagnostic) if the requested combination is
// impossible.
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

  wantAiesim = generateAiesim;
  if (wantAiesim && !xbridge) {
    if (noXbridge || noXchesscc) {
      llvm::errs()
          << "aiecc: --get-aiesim requires --xbridge (the AIE simulator "
             "consumes Chess-compiled cores)\n";
      return false;
    }
    xchesscc = true;
    xbridge = true;
  }

  doUnified = unified && !noUnified;
  doCompileHost = generateHost;
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

// The MLIR input: the single positional argument (empty when none was given,
// e.g. for --emit-dot).
inline std::string getInputFilename() {
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

// Host C/C++ source files, taken from the `--` passthrough tail.
inline std::vector<std::string> getHostSourceFiles() {
  std::vector<std::string> hostFiles;
  for (const auto &arg : hostPassthroughArgs)
    if (isHostSourceFile(arg))
      hostFiles.push_back(arg);
  return hostFiles;
}

// Whether any host source file was provided in the `--` passthrough tail. Used
// to decide whether host compilation has work to do.
inline bool hasHostSourceFiles() { return !getHostSourceFiles().empty(); }

//===----------------------------------------------------------------------===//
// Checkpoint / resume manifest handling
//===----------------------------------------------------------------------===//
// A checkpoint restores a graph cut from disk. Its manifest records the argv
// that built the cut (so a resume reconstructs an identical graph — same
// device, options and item keys) plus a frontier: for each cut edge, the
// {name} it belongs to, the {dir} its artifacts live in, and the per-node
// {descriptor} that its restoreNode consumes.

struct CheckpointEntry {
  std::string name; // producing edge's output-name template
  std::string dir;  // node subdir, relative to manifest
  llvm::json::Value descriptor = nullptr; // per-node restore descriptor
};

struct ResumeState {
  bool active = false;
  std::string manifestDir;
  std::vector<CheckpointEntry> frontier;
};

// Execution-only flags that may accompany --resume: they don't reshape the
// graph, only what the suffix run targets (--get), where it cuts (--cut) or
// captures (--checkpoint), plus reporting/parallelism. Returns 0 (rejected), 1
// (self-contained token), or 2 (also consumes the following token).
inline int resumePassthroughKind(llvm::StringRef a) {
  if (a == "-v" || a == "--verbose" || a == "--progress" ||
      a == "--no-progress")
    return 1;
  if (a.starts_with("--get=") || a.starts_with("-g=") ||
      a.starts_with("--cut=") || a.starts_with("-j") ||
      a.starts_with("--checkpoint="))
    return 1;
  if (a == "--get" || a == "-g" || a == "--cut" || a == "--checkpoint")
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
    std::vector<std::string> full(argv, argv + argc);
    // The recorded graph-argv must exclude the execution-only flags that select
    // outputs (--get/-g), the cut (--cut) and write the checkpoint
    // (--checkpoint): replaying them on resume would re-restrict the run to the
    // cut instead of continuing past it. This run itself still sees the full
    // argv.
    for (size_t i = 0; i < full.size(); ++i) {
      llvm::StringRef a(full[i]);
      if (a == "--get" || a == "-g" || a == "--cut" || a == "--checkpoint") {
        ++i; // also skip its separate value token
        continue;
      }
      if (a.starts_with("--get=") || a.starts_with("-g=") ||
          a.starts_with("--cut=") || a.starts_with("--checkpoint="))
        continue;
      graphArgv.push_back(full[i]);
    }
    return full;
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
        const llvm::json::Value *desc = fo->get("descriptor");
        resume.frontier.push_back({get("name"), get("dir"),
                                   desc ? *desc : llvm::json::Value(nullptr)});
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
