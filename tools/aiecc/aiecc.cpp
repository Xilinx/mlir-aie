//===- aiecc.cpp -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarative AIE compiler driver
//
// This is the main entry point to the MLIR-AIE toolchain; the compiler
// driver invokes all other parts of the toolchain as required to assemble
// the the requested compilation artifacts (binaries, sidecar files, etc.).
//
// This driver is an orchestrator of many tools. To keep its code maintainable,
// we express this orchestration in a declarative manner: A static graph encodes
// which inputs the generatable outputs depend on and what tool calls transform
// inputs to outputs.
//
// When adding code here, please...
// 1. ...express all dependencies, however small, EXPLICITLY as nodes/edges in
//       the graph. Bundles are useful for grouping multiple inputs.
// 2. ...use the `Item` abstraction for inputs and outputs. DO NOT WRITE CODE
//       THAT MANUALLY WRITES TO DISK. Create an `Item` and let the consumer
//       of your outputs decide if they need them in-memory or on disk!
// 3. ...keep the graph STATICALLY DECLARED. Building the graph should have no
//       side effects. We want to be able to visualize the graph.
// 4. ...do as LITTLE WORK as possible in the compiler driver. If what you're
//       doing is an involved transformation, it does not belong in this
//       orchestrator -- create an MLIR pass or a new tool instead.
//
//===----------------------------------------------------------------------===//

#include "AIECCVersion.h"
#include "Actions.h"
#include "CommandLineOptions.h"
#include "ExecutionEngine.h"
#include "Graph.h"
#include "IRTransforms.h"
#include "Items.h"
#include "SidecarFiles.h"
#include "Tools.h"
#include "Utils.h"

#include "aie/Conversion/Passes.h"
#include "aie/Dialect/AIEVec/Pipelines/Passes.h"
#include "aie/Dialect/AIEVec/TransformOps/DialectExtension.h"
#include "aie/InitialAllDialect.h"
#include "aie/Target/LLVMIR/Dialect/XLLVM/XLLVMToLLVMIRTranslation.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

#include <cstdlib>
#include <set>

using namespace xilinx::aiecc;
using namespace xilinx::aiecc::cli;

//===----------------------------------------------------------------------===//
// Shared subgraphs
//===----------------------------------------------------------------------===//

using ModRef = mlir::OwningOpRef<mlir::ModuleOp>;
using xilinx::AIE::DeviceOp;

// Produce a per-key object (.o) -- these are the core program memories, either
// as per-core objects or a single unified object (only difference is
// cardinality of the input module/arches edges). We define a chess path and a
// peano path; the `xchesscc` command-line flag selects which output edge is
// returned.
//
// `irLinkFiles` carries, per key, the merge-mode kernel artifacts (the core's
// `link_merge_files`, i.e. `link_with_mode = "merge"`) to llvm-link into that
// key's module before codegen. Keys with an empty list get the plain compile
// flow. Only the peano path can merge them; the chess path consumes the same
// edge solely to reject a non-empty list with a diagnostic.
static EdgeWithTypedOutput<Directory> &
buildObjectSubgraph(EdgeWithTypedOutput<ModRef> &lowered,
                    EdgeWithTypedOutput<std::string> &arches,
                    EdgeWithTypedOutput<std::vector<std::string>> &irLinkFiles,
                    const std::string &objName) {
  std::string installDir = getInstallDir();
  std::string aietoolsRoot = discoverAietoolsDir(aietoolsDir.getValue());

  // Shared between chess and peano: LLLVMIR lowering
  auto &llvmIR = lowered.map<std::string>("llvmIR_{0}.ll", translateToLLVMIR);

  // Chess path: downgrade -> chess-llvm-link (intrinsic wrapper) ->
  // `xchesscc_wrapper -c`.
  auto &chessCompat =
      llvmIR.map<std::string>("chess-compat_{0}.ll", downgradeIRForChess)
          .threadSafe();
  auto &chessLinked =
      bundle(chessCompat.out, arches.out, irLinkFiles.out)
          .map<File>("chesslinked_{0}.ll",
                     [aietoolsRoot,
                      installDir](const Item<std::string> &ir,
                                  const Item<std::string> &archItem,
                                  const Item<std::vector<std::string>> &irLinks,
                                  Item<File> &out) -> mlir::LogicalResult {
                       // The chess front-end cannot llvm-link, so merge-mode
                       // kernel artifacts have no route into the core on this
                       // path -- and the BCF emitter deliberately leaves
                       // `link_merge_files` out of `_include _file`. Reject
                       // them here, where the cause is still known, rather
                       // than let it surface as an undefined symbol from the
                       // chess linker.
                       if (!irLinks.get().empty()) {
                         llvm::errs()
                             << "aiecc: --xchesscc cannot consume merge-mode "
                                "link artifacts (link_with_mode = \"merge\"): "
                                "the Chess front-end cannot llvm-link them "
                                "into the core. Offending link_with entries:\n";
                         for (const auto &f : irLinks.get())
                           llvm::errs() << "  " << f << "\n";
                         llvm::errs()
                             << "Drop link_with_mode = \"merge\" and compile "
                                "these kernels to objects (.o) instead, or "
                                "build with the Peano front-end (drop "
                                "--xchesscc).\n";
                         return mlir::failure();
                       }
                       llvm::StringRef arch = archItem.get();
                       std::string linkTool =
                           getChessLLVMLinkPath(arch, aietoolsRoot);
                       if (linkTool.empty()) {
                         llvm::errs()
                             << "aiecc: --xchesscc/--xbridge require aietools; "
                                "set --aietools or put xchesscc on PATH\n";
                         return mlir::failure();
                       }
                       std::string wrapper =
                           getChessIntrinsicWrapperPath(arch, installDir);
                       auto cmd = ShellCommand{linkTool}
                                      .input()
                                      .arg(wrapper)
                                      .arg("-S")
                                      .output("-o");
                       return cmd(ir, out);
                     })
          .threadSafe();
  // Chess object: the `.o` and chess's sidecars (`<obj>.o.lst`, ...) land in
  // the output `Directory`; `+w` scratch shares it too.
  EdgeWithTypedOutput<Directory> &chessObject =
      bundle(arches.out, chessLinked.out)
          .map<Directory>(objName, ShellCommand{"xchesscc_wrapper"}
                                       .value()
                                       .arg("+w")
                                       .outputDir()
                                       .arg("-c")
                                       .arg("-d")
                                       .arg("+Wclang,-xir")
                                       .arg("-f")
                                       .input()
                                       .output("-o"))
          .threadSafe();

  // Peano path: downgrade -> (llvm-link merge-mode kernels) -> opt -> llc.
  unsigned optPassLevel = std::min<unsigned>(optLevel, 1u);
  ShellCommand optCmd{"opt"};
  if (optLevel >= 3)
    optCmd.arg("-disable-loop-idiom-memset");
  optCmd.arg("--passes=default<O" + std::to_string(optPassLevel) + ">")
      .arg("-inline-threshold=10")
      .arg("-S")
      .input()
      .output("-o");
  auto &peanoCompat =
      llvmIR.map<std::string>("peano-compat_{0}.ll", [](llvm::StringRef ir) {
        return downgradeIRForPeano(ir);
      });
  // Merge the core's merge-mode link artifacts (`link_merge_files`) into the
  // downgraded core IR before opt; with the kernel marked alwaysinline that
  // inlines its body into the core, leaving no func.call and no separate kernel
  // object. The ld-script/BCF emitters emit `link_files` only, so each symbol
  // is merged exactly once. Keys with no merge-mode files pass straight
  // through. Peano only: the chess front-end cannot llvm-link.
  //
  // Merged in-process (AIELLVMLink) rather than via `llvm-link`, which the
  // Peano wheel does not ship -- a bare-name lookup silently lands on whatever
  // other LLVM is on PATH. A linker reprints the merged module in its own IR
  // dialect, so downgradeIRForPeano runs again on the result: the pre-link pass
  // above cannot see the newer spellings the reprint introduces, and the
  // reprint also restores the `align` attributes it had stripped.
  auto &peanoLinked =
      bundle(peanoCompat.out, irLinkFiles.out)
          .map<File>(
              "peano-linked_{0}.ll",
              [](const Item<std::string> &ir,
                 const Item<std::vector<std::string>> &links,
                 Item<File> &out) -> mlir::LogicalResult {
                if (links.get().empty()) {
                  // Nothing to merge: the downgraded core IR is the
                  // object input. Copy it to this edge's own output path
                  // -- aliasing the peano-compat item's path collides
                  // with it (the engine requires each item's output path
                  // to be unique).
                  if (std::error_code ec =
                          llvm::sys::fs::copy_file(ir.asFile(), out.filePath)) {
                    llvm::errs() << "aiecc: peano-linked: cannot copy '"
                                 << ir.asFile() << "' to '" << out.filePath
                                 << "': " << ec.message() << "\n";
                    return mlir::failure();
                  }
                  out.value = File{};
                  return mlir::success();
                }
                if (dryRun) {
                  // Placeholder so path bookkeeping resolves without requiring
                  // the merge artifacts to exist, as the ShellCommand edges do.
                  std::error_code ec;
                  llvm::raw_fd_ostream placeholder(out.filePath, ec);
                  if (ec) {
                    llvm::errs()
                        << "aiecc: peano-linked: cannot write '" << out.filePath
                        << "': " << ec.message() << "\n";
                    return mlir::failure();
                  }
                  out.value = File{};
                  return mlir::success();
                }
                // AIELLVMLink takes module *contents*, not paths (its `Files`
                // parameter is a misnomer). parseIR sniffs each buffer, so a
                // `.bc` artifact works the same as a `.ll`.
                std::vector<std::string> modules{ir.asString()};
                for (const std::string &link : links.get()) {
                  auto buf = llvm::MemoryBuffer::getFile(link);
                  if (!buf) {
                    llvm::errs()
                        << "aiecc: peano-linked: cannot read merge-mode link "
                           "artifact '"
                        << link << "': " << buf.getError().message() << "\n";
                    return mlir::failure();
                  }
                  modules.push_back((*buf)->getBuffer().str());
                }
                std::string merged;
                llvm::raw_string_ostream mergedOs(merged);
                if (mlir::failed(xilinx::AIE::AIELLVMLink(mergedOs, modules))) {
                  llvm::errs() << "aiecc: peano-linked: cannot merge "
                                  "link_with_mode = \"merge\" artifacts into "
                                  "the core module\n";
                  return mlir::failure();
                }
                std::error_code ec;
                llvm::raw_fd_ostream os(out.filePath, ec);
                if (ec) {
                  llvm::errs() << "aiecc: peano-linked: cannot write '"
                               << out.filePath << "': " << ec.message() << "\n";
                  return mlir::failure();
                }
                os << downgradeIRForPeano(merged, /*stripAlign=*/false);
                out.value = File{};
                return mlir::success();
              })
          .threadSafe();
  auto &opted = peanoLinked.map<File>("opted_{0}.ll", optCmd).threadSafe();
  ShellCommand llcCmd{"llc"};
  llcCmd.input()
      .arg("-O" + std::to_string(optLevel.getValue()))
      .value("--march=")
      .arg("--function-sections")
      // Record each function's frame size in a `.stack_sizes` section, which
      // the emitted linker script already has an output rule for. Chess ships
      // the equivalent as `.stackinfo`; without this, peano builds carry no
      // stack accounting at all, so a core's `stack_size` cannot be checked
      // against what it actually needs and an overflow is only visible as
      // corruption of whatever buffer sits above the stack. The section is
      // metadata (no SHF_ALLOC), so it costs no data memory.
      .arg("-stack-size-section")
      .arg("--filetype=obj")
      .output("-o");
  EdgeWithTypedOutput<Directory> &peanoObject =
      bundle(opted.out, arches.out)
          .map<Directory>(objName, llcCmd)
          .threadSafe();

  return xchesscc ? chessObject : peanoObject;
}

// Host-compilation subgraph. Compiles the user's host sources against the
// per-device `aie_inc.cpp` array configuration source (shared with aiesim).
static EdgeWithTypedOutput<File> &
buildHostExeSubgraph(EdgeWithTypedOutput<std::string> &aieInc,
                     EdgeWithTypedOutput<std::string> &arches) {
  // clang++ edge: produce a single host executable.
  // We bundle aie_inc.cpp to capture it as a dependency (included as `-I`).
  // perDevice feeds the device symbol name for diagnostics; arches feeds the
  // architecture information in the `__AIEARCH__` define.
  HostRuntimeLibs rt = getHostRuntimeLibs(getInstallDir(), hostTarget);
  std::string outputName = hostOutputName;
  return bundle(aieInc.out, arches.out)
      .join<File>(
          std::move(outputName),
          [rt](const Node<std::string> &incs, const Node<std::string> &arches,
               Item<File> &out) -> mlir::LogicalResult {
            // Host compilation supports a single device only.
            if (incs.items.size() != 1) {
              llvm::errs()
                  << "aiecc: host compilation requires exactly one device, "
                  << "but " << incs.items.size()
                  << " were found; select one with --device-name\n";
              return mlir::failure();
            }
            assert(arches.items.size() == 1 && incs.items.size() == 1 &&
                   "host exe expects one device's arch and include dir");
            // Materialize aie_inc.cpp; its directory goes on the include path.
            std::string incDir = std::string(
                llvm::sys::path::parent_path(incs.items.front().asFile()));
            const std::string &arch = arches.items.front().get();

            // Compilation command
            ShellCommand cmd{"clang++"};
            cmd.arg("-std=c++17");
            if (!hostTarget.empty())
              cmd.arg("--target=" + hostTarget);
            if (!sysroot.empty()) {
              cmd.arg("--sysroot=" + sysroot);
              if (hostTarget == "aarch64-linux-gnu")
                cmd.arg("--gcc-toolchain=" + sysroot + "/usr");
            }
            cmd.arg(rt.memoryAllocator)
                .arg("-I" + rt.xaiengineInclude)
                .arg("-L" + rt.xaiengineLib)
                .arg("-Wl,-R" + rt.xaiengineLib)
                .arg("-I" + incDir)
                .arg("-fuse-ld=lld")
                .arg("-lm")
                .arg("-lxaienginecdo");
            cmd.arg(aieArchDefine(arch));
            for (const auto &d : hostIncludeDirs)
              cmd.arg("-I" + d);
            for (const auto &d : hostLibDirs)
              cmd.arg("-L" + d);
            for (const auto &l : hostLibs)
              cmd.arg("-l" + l);
            // Host sources and host-compiler flags arrive after the `--`
            // separator and are forwarded verbatim.
            for (const auto &a : hostPassthroughArgs)
              cmd.arg(a);
            cmd.output("-o");
            return cmd(out);
          });
}

// AIE-simulator work-folder subgraph. Emits the `sim/` folder the aiesimulator
// consumes: the graph/shim/scsim descriptors, the routed flows, the ps.so
// co-simulation model, the `.target` marker, and the `aiesim.sh` launcher.
//
// Each artifact is its own edge/Item. They are declared with work-dir-relative
// names, so as intermediates they land in the `.prj` (aiesim.sh derives
// `--pkg-dir` from its own location) rather than under `--output-dir`. A single
// aggregator edge depends on them all and forces each onto disk via asFile();
// the caller requests that aggregator as an output when `--aiesim` is set.
//
// Multiple devices would collide on the fixed `sim/` layout; the engine's
// duplicate-path guard surfaces that (aiesim targets a single device).
static EdgeWithTypedOutput<File> &
buildAiesimSubgraph(mlir::MLIRContext &context,
                    EdgeWithTypedOutput<OpInModule<DeviceOp>> &staticPerDevice,
                    EdgeWithTypedOutput<std::string> &aieInc) {
  std::string installDir = getInstallDir();
  std::string aietoolsRoot = discoverAietoolsDir(aietoolsDir.getValue());
  const std::string &devFilter = deviceName.getValue();

  // graph.xpe / aieshim_solution.aiesol / scsim_config.json: in-process
  // translations of the per-device module.
  auto &xpe = staticPerDevice.map<std::string>(
      "sim/reports/graph.xpe",
      [](const Item<OpInModule<DeviceOp>> &item,
         Item<std::string> &out) -> mlir::LogicalResult {
        DeviceOp d = item.get().op;
        llvm::raw_string_ostream os(out.value.emplace());
        return xilinx::AIE::AIETranslateGraphXPE(item.get().module.get(), os,
                                                 d.getSymName());
      });
  auto &shim = staticPerDevice.map<std::string>(
      "sim/arch/aieshim_solution.aiesol",
      [](const Item<OpInModule<DeviceOp>> &item,
         Item<std::string> &out) -> mlir::LogicalResult {
        DeviceOp d = item.get().op;
        llvm::raw_string_ostream os(out.value.emplace());
        return xilinx::AIE::AIETranslateShimSolution(item.get().module.get(),
                                                     os, d.getSymName());
      });
  auto &scsim = staticPerDevice.map<std::string>(
      "sim/config/scsim_config.json",
      [](const Item<OpInModule<DeviceOp>> &item,
         Item<std::string> &out) -> mlir::LogicalResult {
        DeviceOp d = item.get().op;
        llvm::raw_string_ostream os(out.value.emplace());
        return xilinx::AIE::AIETranslateSCSimConfig(item.get().module.get(), os,
                                                    d.getSymName());
      });

  // Routed flows: run `aie-find-flows` to annotate the module, emit it as
  // flows_physical.mlir, then serialize the flows to JSON.
  auto findFlowsPM = std::make_unique<mlir::PassManager>(&context);
  findFlowsPM->nest<DeviceOp>().addPass(xilinx::AIE::createAIEFindFlowsPass());
  auto &flows = staticPerDevice.map<ModRef>(
      "sim/flows_physical.mlir", PassPipeline{std::move(findFlowsPM)});
  auto &flowsJson = flows.map<std::string>(
      "sim/flows_physical.json",
      [devFilter](const Item<ModRef> &item,
                  Item<std::string> &out) -> mlir::LogicalResult {
        mlir::ModuleOp mod = item.get().get();
        std::string devName = devFilter;
        if (devName.empty())
          for (auto d : mod.getOps<DeviceOp>()) {
            devName = d.getSymName().str();
            break;
          }
        llvm::raw_string_ostream os(out.value.emplace());
        return xilinx::AIE::AIEFlowsToJSON(mod, os, devName);
      });

  // ps.so: the SystemC co-simulation model. clang++ links the toolchain's
  // `genwrapper_for_ps.cpp` (which #includes aie_inc.cpp from the work dir)
  // together with the user's host sources against the aiesim runtime.
  auto &ps =
      bundle(staticPerDevice.out, aieInc.out)
          .join<File>(
              "sim/ps/ps.so",
              [installDir,
               aietoolsRoot](const Node<OpInModule<DeviceOp>> &devs,
                             const Node<std::string> &incs,
                             Item<File> &out) -> mlir::LogicalResult {
                assert(!devs.items.empty() && !incs.items.empty());
                mlir::ModuleOp mod = devs.items.front().get().module.get();
                DeviceOp d = devs.items.front().get().op;
                std::string aieTarget = detectAIETarget(mod, d.getSymName());
                std::string archUpper = llvm::StringRef(aieTarget).upper();

                std::string genwrapper = installDir + "/aie_runtime_lib/" +
                                         archUpper +
                                         "/aiesim/genwrapper_for_ps.cpp";
                if (!dryRun && !llvm::sys::fs::exists(genwrapper)) {
                  llvm::errs() << "aiecc: aiesim requires " << genwrapper
                               << " (aietools/runtime lib for " << archUpper
                               << " not installed)\n";
                  return mlir::failure();
                }
                // Materialize aie_inc.cpp; genwrapper's `#include
                // "aie_inc.cpp"` resolves against its directory on the include
                // path.
                std::string incDir = std::string(
                    llvm::sys::path::parent_path(incs.items.front().asFile()));

                std::string archTag = hostTarget.getValue();
                if (auto pos = archTag.find('-'); pos != std::string::npos)
                  archTag = archTag.substr(0, pos);
                std::string rtl = installDir + "/runtime_lib/" + archTag;

                ShellCommand cmd{"clang++"};
                cmd.arg("-O2")
                    .arg("-fuse-ld=lld")
                    .arg("-shared")
                    .arg("-fPIC")
                    .arg("-flto")
                    .arg("-fpermissive")
                    .arg("-DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR")
                    .arg("-Wno-deprecated-declarations")
                    .arg("-Wno-enum-constexpr-conversion")
                    .arg("-Wno-format-security")
                    .arg("-DSC_INCLUDE_DYNAMIC_PROCESSES")
                    .arg("-D__AIESIM__")
                    .arg("-D__PS_INIT_AIE__")
                    .arg("-Og")
                    .arg("-Dmain(...)=ps_main(...)")
                    .arg(aieArchDefine(aieTarget))
                    .arg("-I" + incDir)
                    .arg("-I" + aietoolsRoot + "/include")
                    .arg("-I" + rtl + "/xaiengine/include")
                    .arg("-I" + aietoolsRoot + "/data/osci_systemc/include")
                    .arg("-I" + aietoolsRoot + "/include/xtlm/include")
                    .arg("-I" + aietoolsRoot +
                         "/include/common_cpp/common_cpp_v1_0/include")
                    .arg("-I" + rtl + "/test_lib/include");
                std::string memAlloc =
                    rtl + "/test_lib/lib/libmemory_allocator_sim_aie.a";
                if (llvm::sys::fs::exists(memAlloc))
                  cmd.arg(memAlloc);
                cmd.arg("-L" + rtl + "/xaiengine/lib")
                    .arg("-lxaienginecdo")
                    .arg("-L" + aietoolsRoot + "/lib/lnx64.o")
                    .arg("-L" + aietoolsRoot + "/lib/lnx64.o/Ubuntu")
                    .arg("-L" + aietoolsRoot + "/data/osci_systemc/lib/lnx64")
                    .arg("-Wl,--as-needed")
                    .arg("-lsystemc")
                    .arg("-lxtlm");
                for (const auto &dir : hostIncludeDirs)
                  cmd.arg("-I" + dir);
                for (const auto &dir : hostLibDirs)
                  cmd.arg("-L" + dir);
                for (const auto &lib : hostLibs)
                  cmd.arg("-l" + lib);
                for (const auto &a : hostPassthroughArgs)
                  cmd.arg(a);
                cmd.arg(genwrapper).output("-o");
                return cmd(out);
              });

  // Literal sidecars: the simulator target marker and the launcher script.
  auto &target = staticPerDevice.map<std::string>(
      "sim/.target",
      [](const Item<OpInModule<DeviceOp>> &,
         Item<std::string> &out) -> mlir::LogicalResult {
        out.value = "hw\n";
        return mlir::success();
      });
  auto &script = staticPerDevice.map<std::string>(
      "aiesim.sh",
      [](const Item<OpInModule<DeviceOp>> &,
         Item<std::string> &out) -> mlir::LogicalResult {
        out.value = R"(#!/bin/sh
prj_name=$(basename $(dirname $(realpath $0)))
root=$(dirname $(dirname $(realpath $0)))
vcd_filename=foo
if [ -n "$1" ]; then
  vcd_filename=$1
fi
cd $root
aiesimulator --pkg-dir=${prj_name}/sim --dump-vcd ${vcd_filename}
)";
        return mlir::success();
      });

  // Aggregator: depend on every sim artifact and force each onto disk. The
  // sim edges are work-dir intermediates, so their consumer (this edge) is what
  // materializes them -- via asFile(), the Item abstraction's "I need this on
  // disk" request -- into the `.prj`. Produces no file of its own.
  auto &aiesim =
      bundle(xpe.out, shim.out, scsim.out, flows.out, flowsJson.out, ps.out,
             target.out, script.out)
          .join<File>(
              "aiesim.stamp",
              [](const Node<std::string> &xpe, const Node<std::string> &shim,
                 const Node<std::string> &scsim, const Node<ModRef> &flows,
                 const Node<std::string> &flowsJson, const Node<File> &ps,
                 const Node<std::string> &target,
                 const Node<std::string> &script,
                 Item<File> &out) -> mlir::LogicalResult {
                const NodeBase *nodes[] = {&xpe,       &shim, &scsim,  &flows,
                                           &flowsJson, &ps,   &target, &script};
                for (const NodeBase *n : nodes)
                  for (const ItemBase *it : n->itemRefs())
                    (void)it->asFile();
                out.value = File{};
                return mlir::success();
              });
  aiesim.producesFiles = false;
  return aiesim;
}

// Translate each runtime sequence into its NPU program: one NpuProgram item
// (the transaction instruction binary + its source-location map) per sequence,
// keyed "<device>_<sequence>".
//
// DDR-patch ABI: XRT (and CPU) consume the folded firmware ABI; HRX consumes
// the producer-independent (unfolded) insts.bin and adds the AIE DDR aperture
// offset for every arg itself. cl::opt defaults to true, so only pass the
// flag when unfolding is requested.
static EdgeWithTypedOutput<NpuProgram> &buildNpuProgramSubgraph(
    EdgeWithTypedOutput<OpInModule<xilinx::AIE::RuntimeSequenceOp>> &perSeq,
    std::string programName, bool foldDDRAddrOffset) {
  auto &npuProgram = perSeq.map<NpuProgram>(
      std::move(programName),
      [foldDDRAddrOffset](
          const Item<OpInModule<xilinx::AIE::RuntimeSequenceOp>> &item,
          Item<NpuProgram> &out) -> mlir::LogicalResult {
        xilinx::AIE::RuntimeSequenceOp seq = item.get().op;
        DeviceOp devOp = seq->getParentOfType<DeviceOp>();
        NpuProgram prog;
        prog.deviceName = devOp.getSymName().str();
        std::vector<uint32_t> insts;
        if (mlir::failed(xilinx::AIE::AIETranslateNpuToBinary(
                item.get().module.get(), insts, devOp.getSymName(),
                seq.getSymName(), &prog.locmap, foldDDRAddrOffset)))
          return mlir::failure();
        prog.insts = wordsToBytes(insts);
        out.value = std::move(prog);
        return mlir::success();
      });
  npuProgram.producesFiles = false;
  return npuProgram;
}

//===----------------------------------------------------------------------===//
// Main compilation graph
//===----------------------------------------------------------------------===//

// Assemble the full compilation artifact graph into `g` and return the list of
// requested output edges. Edges named via `--cut` are appended to `cutEdges`
// (and built) so a `--checkpoint` can capture them as its cut points.
static std::vector<EdgeBase *>
buildMainGraph(mlir::MLIRContext &context, Graph &g,
               std::vector<EdgeBase *> &cutEdges) {

  //--------------------------------------------------------------------------//
  // Helpers
  //--------------------------------------------------------------------------//

  using xilinx::AIE::CoreOp;
  using xilinx::AIE::DeviceOp;
  using xilinx::AIE::RuntimeSequenceOp;
  using xilinx::AIE::TileOp;
  using ModRef = mlir::OwningOpRef<mlir::ModuleOp>;

  const std::string &devFilter = deviceName.getValue();
  std::string inputFile = getInputFilename();

  std::string workDirStr = getWorkDir();
  std::string lldPath = ShellCommand::resolveTool("ld.lld");

  auto matchesDeviceFilter = [devFilter](DeviceOp d) {
    // Empty reset devices synthesized by --expand-load-pdis must always be
    // included, regardless of --device-name.
    return devFilter.empty() || d.getSymName() == devFilter ||
           d.getSymName().starts_with("empty_");
  };

  // Split a whole-module edge into one item per DeviceOp (keyed by bare device
  // name), then drop devices that don't match --device-name.
  auto splitPerDevice = [&matchesDeviceFilter](EdgeWithTypedOutput<ModRef> &src,
                                               std::string nameFmt,
                                               std::string filterName)
      -> EdgeWithTypedOutput<OpInModule<DeviceOp>> & {
    return src
        .split<OpInModule<DeviceOp>>(std::move(nameFmt),
                                     SplitIRAction<DeviceOp>([](DeviceOp d) {
                                       return d.getSymName().str();
                                     }))
        .filter(std::move(filterName),
                [matchesDeviceFilter](const OpInModule<DeviceOp> &x) {
                  return matchesDeviceFilter(DeviceOp(x.op));
                });
  };

  //--------------------------------------------------------------------------//
  // Graph
  //--------------------------------------------------------------------------//

  std::vector<EdgeBase *> outputs;
  auto &input = g.fileInput(inputFile, "input.mlir");

  auto &traced =
      input
          .map<ModRef>("placed.mlir",
                       PassPipeline{getPlacementPipeline(
                           &context, coresPerCol.getValue(),
                           placerType.getValue(), saSeed.getValue())})
          .map<ModRef>("traced.mlir", PassPipeline{getTracePipeline(&context)});

  // --default-stack-size: a design-wide stand-in for the target's built-in
  // stack_size default, for any core that leaves stack_size absent. Needs
  // nothing but the parsed module, so it runs as early as possible -- before
  // link_files assignment, and so before every consumer of
  // CoreOp::getEffectiveStackSize() downstream (buffer placement, the stack
  // and reserved-data checks, and the core/BCF/ldscript emitters).
  auto &withDefaultStackSize = traced.map<ModRef>(
      "default_stack_size.mlir",
      [stackSize = defaultStackSize.getValue()](const ModRef &mod) -> ModRef {
        if (stackSize <= 0)
          return ModRef(mod.get().clone());
        return populateDefaultStackSize(mod.get(), stackSize);
      });

  // link_files must be known before reserved_data_size can be auto-measured
  // from it, so aie-assign-core-link-files runs here rather than where it
  // used to sit inside getInputWithAddressesPipeline (after buffer address
  // assignment). It's a pure call-graph analysis over func.func/func.call --
  // nothing about it depends on addresses, so moving it earlier is safe.
  auto &withLinkFiles = withDefaultStackSize.map<ModRef>(
      "with_link_files.mlir",
      PassPipeline{&context, [](mlir::MLIRContext *ctx, mlir::ModuleOp) {
                     auto pm = std::make_unique<mlir::PassManager>(ctx);
                     pm->nest<DeviceOp>().addPass(
                         xilinx::AIE::createAIEAssignCoreLinkFilesPass());
                     return pm;
                   }});

  // Measure each core's link_files objects and auto-populate
  // reserved_data_size where the user hasn't set it explicitly. File I/O
  // (opening the linked objects) happens here in the driver, not inside an
  // MLIR pass -- the same reasoning as resolveExternalPath's other callers.
  auto &withReservedData = withLinkFiles.map<ModRef>(
      "reserved_data.mlir",
      [inputFile, workDirStr,
       skip = noAutoReservedData.getValue()](const ModRef &mod) -> ModRef {
        if (skip)
          return ModRef(mod.get().clone());
        return populateReservedDataSize(mod.get(), inputFile, workDirStr);
      });

  // Validate each core's stack_size against what its call tree actually
  // needs, from the same link_files objects reserved_data_size just
  // measured. Unlike that edge, this one can fail the whole run (a cycle or
  // an unmeasurable symbol with no stack_size_override is an error, not a
  // warning) -- see checkStackSizeRequirements.
  auto &withStackSizeChecked = withReservedData.map<ModRef>(
      "stack_size_checked.mlir",
      [inputFile, workDirStr, skip = noAutoStackSize.getValue()](
          const Item<ModRef> &in, Item<ModRef> &out) -> mlir::LogicalResult {
        out.value = ModRef(in.get().get().clone());
        if (skip)
          return mlir::success();
        return checkStackSizeRequirements(out.value->get(), inputFile,
                                          workDirStr);
      });

  auto &withAddresses = withStackSizeChecked.map<ModRef>(
      "input_with_addresses.mlir",
      PassPipeline{
          &context,
          [scheme = allocScheme.getValue(), dyn = dynamicObjFifos.getValue(),
           pkt = packetSwObjFifos.getValue(),
           ctrl = ctrlPktOverlay.getValue() || loadPdiToCtrlPkt.getValue(),
           ldpdi = loadPdiToCtrlPkt.getValue(),
           bf16 = bf16Emulation.getValue()](mlir::MLIRContext *ctx,
                                            mlir::ModuleOp mod) {
            return getInputWithAddressesPipeline(ctx, mod, scheme, dyn, pkt,
                                                 ctrl, bf16, ldpdi);
          }});

  // Scratchpad run-time parameters sidecar file
  auto &paramsFile = withAddresses.map<std::string>(
      "params.txt", [](const ModRef &mod) -> std::string {
        std::string txt;
        llvm::raw_string_ostream os(txt);
        xilinx::AIEX::emitScratchpadParamsFile(mod.get(), os);
        return txt;
      });

  auto &physical = withAddresses.map<ModRef>(
      "input_physical.mlir", PassPipeline{getRoutingPipeline(&context)});

  // Split every core once, then filter into compile / pre-baked subviews.
  auto &allCores =
      physical
          .split<OpInModule<CoreOp>>(
              "perCore_{0}.mlir",
              SplitIRAction<CoreOp>([](CoreOp c) { return coreKey(c); }))
          .filter("perCoreInDevice",
                  [matchesDeviceFilter](const OpInModule<CoreOp> &x) {
                    return matchesDeviceFilter(
                        CoreOp(x.op)->getParentOfType<DeviceOp>());
                  });

  // Cores whose MLIR we must compile. A core without an `elf_file` attribute is
  // always compiled. A core that already carries an `elf_file` normally needs
  // no compilation -- its ELF is used verbatim (see `preBakedElfs`).
  // However, some external tests that manually link pre-baked cores rely on a
  // per-core BCF being emitted for every core, so if chess is enabled we
  // compile all cores regardless.
  auto &perCore =
      allCores.filter("perCoreCompile", [](const OpInModule<CoreOp> &x) {
        return !CoreOp(x.op).getElfFileAttr() || xbridge;
      });

  // Cores whose `elf_file` attribute already points to a built object.
  auto &preBakedElfs =
      allCores
          .filter("preBakedCores",
                  [](const OpInModule<CoreOp> &x) {
                    return (bool)CoreOp(x.op).getElfFileAttr();
                  })
          .map<File>("preBakedElfs_{0}.elf",
                     [](const Item<OpInModule<CoreOp>> &item,
                        Item<File> &out) -> mlir::LogicalResult {
                       CoreOp core = item.get().op;
                       out.filePath =
                           absolutePath(core.getElfFileAttr().getValue());
                       out.value = File{};
                       return mlir::success();
                     });
  preBakedElfs.producesFiles = false;

  // Per-core arch string (feeds link --target= and llc --march=).
  auto &perCoreArches = perCore.map<std::string>(
      "perCoreArches_{0}.txt", [](const OpInModule<CoreOp> &core) {
        return detectAIETarget(
            core.module.get(),
            core.op->getParentOfType<DeviceOp>().getSymName());
      });

  // Per-core .o node. Two strategies selectable:
  //   * unified: compile all cores of a device into one shared object, then
  //     re-key that device-wide object onto each of the device's cores;
  //   * per-core: compile each core's own module to its own object.

  // Unified strategy
  auto &physicalPerDevice = splitPerDevice(
      physical, "perDeviceCompile_{0}.mlir", "perDeviceCompileMatching");
  auto &perDeviceArches = physicalPerDevice.map<std::string>(
      "perDeviceArches_{0}.txt", [](const OpInModule<DeviceOp> &dev) {
        return detectAIETarget(dev.module.get(), DeviceOp(dev.op).getSymName());
      });
  auto &unifiedLowered = physicalPerDevice.map<ModRef>(
      "unifiedLowered_{0}.mlir",
      [](const Item<OpInModule<DeviceOp>> &item, Item<ModRef> &out) {
        DeviceOp d = item.get().op;
        return loweringPipeline(item.get().module.get(), d.getSymName(), -1, -1,
                                out);
      });
  // Merge-mode link artifacts for the device's whole core set, deduplicated
  // (the shared unified module is llvm-linked once). See buildObjectSubgraph.
  auto &perDeviceIRLinkFiles = physicalPerDevice.map<std::vector<std::string>>(
      "perDeviceIRLinkFiles_{0}.txt",
      [inputFile,
       workDirStr](const Item<OpInModule<DeviceOp>> &dev,
                   Item<std::vector<std::string>> &out) -> mlir::LogicalResult {
        std::vector<std::string> files;
        if (mlir::failed(collectDeviceIRLinkFiles(
                DeviceOp(dev.get().op), inputFile, workDirStr, files)))
          return mlir::failure();
        out.value = std::move(files);
        return mlir::success();
      });
  auto &unifiedObjects =
      buildObjectSubgraph(unifiedLowered, perDeviceArches, perDeviceIRLinkFiles,
                          "unifiedObjects_{0}.o");
  // Each core links against its device's shared object: re-key the device-keyed
  // unified objects onto the per-core keys.
  EdgeWithTypedOutput<Directory> &unifiedCoreObjects =
      perCore.rekeyFrom<Directory>(
          "objects_{0}.o", unifiedObjects.out,
          [](const OpInModule<CoreOp> &core) {
            return core.op->getParentOfType<DeviceOp>().getSymName().str();
          });

  // Per-core strategy
  auto &perCoreLowered = perCore.map<ModRef>(
      "lowered_{0}.mlir",
      [](const Item<OpInModule<CoreOp>> &item, Item<ModRef> &out) {
        CoreOp core = item.get().op;
        auto tile = mlir::cast<TileOp>(core.getTile().getDefiningOp());
        return loweringPipeline(item.get().module.get(),
                                core->getParentOfType<DeviceOp>().getSymName(),
                                tile.getCol(), tile.getRow(), out);
      });
  // Merge-mode link artifacts for this core, llvm-linked into its own module.
  auto &perCoreIRLinkFiles = perCore.map<std::vector<std::string>>(
      "perCoreIRLinkFiles_{0}.txt",
      [inputFile, workDirStr](const OpInModule<CoreOp> &core) {
        return collectCoreIRLinkFiles(CoreOp(core.op), inputFile, workDirStr);
      });
  EdgeWithTypedOutput<Directory> &perCoreObjects = buildObjectSubgraph(
      perCoreLowered, perCoreArches, perCoreIRLinkFiles, "objects_{0}.o");

  EdgeWithTypedOutput<Directory> &objects =
      doUnified ? unifiedCoreObjects : perCoreObjects;

  // ld scripts (with link_files absolutized so INPUT() is cwd-invariant).
  auto &ldScripts = perCore.map<std::string>(
      "ldScripts_{0}.ld.script",
      [inputFile, workDirStr](const Item<OpInModule<CoreOp>> &item,
                              Item<std::string> &out) -> mlir::LogicalResult {
        CoreOp op = item.get().op;
        auto tile = mlir::cast<TileOp>(op.getTile().getDefiningOp());
        // Peano-only guard. `link_files` entries become INPUT() directives, and
        // lld falls back to parsing a non-object INPUT() as a linker script: a
        // textual .ll dies with the useless `ld.lld: error: <file>:1: malformed
        // number`. Bitcode is fine -- lld accepts a .bc as an LTO input -- so
        // only .ll is rejected, and only here: the chess linker consumes the
        // BCF emitter's output instead and has its own rules.
        if (!xbridge) {
          auto isTextualIR =
              [&](llvm::StringRef f) {
                if (!f.ends_with(".ll"))
                  return false;
                llvm::errs()
                    << "aiecc: link file '" << f << "' on core ("
                    << tile.getCol() << ", " << tile.getRow()
                    << ") is textual LLVM IR, which ld.lld cannot link (it "
                       "reads unrecognized inputs as linker scripts and fails "
                       "with \"malformed number\"). Add link_with_mode = "
                       "\"merge\" to the kernel declaration so aiecc "
                       "llvm-links it into the core, or assemble it to "
                       "bitcode (.bc) or an object (.o).\n";
                return true;
              };
          // Mirror the emitter's precedence exactly (see
          // AIETranslateToLdScript): when link_files is present the deprecated
          // core-level link_with is not emitted, so it must not be diagnosed
          // either.
          if (auto filesAttr = op.getLinkFiles()) {
            for (auto f : filesAttr->getAsRange<mlir::StringAttr>())
              if (isTextualIR(f.getValue()))
                return mlir::failure();
          } else if (auto fileAttr = op.getLinkWith()) {
            if (isTextualIR(fileAttr.value()))
              return mlir::failure();
          }
        }
        auto rewritten =
            absolutizeLinkFiles(item.get().module.get(), tile.getCol(),
                                tile.getRow(), inputFile, workDirStr);
        llvm::raw_string_ostream os(out.value.emplace());
        return xilinx::AIE::AIETranslateToLdScript(
            rewritten.get(), os, tile.getCol(), tile.getRow(),
            op->getParentOfType<DeviceOp>().getSymName());
      });

  // Link each core's object into its .elf; user can chose between
  // chess/xbridge or peano

  // chess linking
  auto &bcfScripts = perCore.map<std::string>(
      "{0}.bcf",
      [](const Item<OpInModule<CoreOp>> &item,
         Item<std::string> &out) -> mlir::LogicalResult {
        CoreOp op = item.get().op;
        auto tile = mlir::cast<TileOp>(op.getTile().getDefiningOp());
        llvm::raw_string_ostream os(out.value.emplace());
        return xilinx::AIE::AIETranslateToBCF(
            item.get().module.get(), os, tile.getCol(), tile.getRow(),
            op->getParentOfType<DeviceOp>().getSymName());
      });
  auto &linkWithObjs = bcfScripts.map<std::vector<std::string>>(
      "linkwith_{0}.txt",
      [inputFile,
       workDirStr](const Item<std::string> &bcf,
                   Item<std::vector<std::string>> &out) -> mlir::LogicalResult {
        std::vector<std::string> resolved;
        for (const auto &f : parseBcfIncludeFiles(bcf.get()))
          resolved.push_back(resolveExternalPath(f, inputFile, workDirStr));
        out.value = std::move(resolved);
        return mlir::success();
      });
  // Chess link: the ELF and the sidecar files chess writes beside it (`.map`,
  // `.lst`, ...) land in the output `Directory`; `+w` scratch shares it too.
  EdgeWithTypedOutput<Directory> &chessElfs =
      bundle(perCoreArches.out, objects.out, linkWithObjs.out, bcfScripts.out)
          .map<Directory>("elfs_{0}.elf", ShellCommand{"xchesscc_wrapper"}
                                              .value()
                                              .arg("+w")
                                              .outputDir()
                                              .arg("-d")
                                              .arg("-f")
                                              .input()
                                              .inputs()
                                              .arg("+l")
                                              .input()
                                              .output("-o"))
          .threadSafe();

  // peano linking
  EdgeWithTypedOutput<Directory> &peanoElfs =
      bundle(perCoreArches.out, objects.out, ldScripts.out)
          .map<Directory>(
              "elfs_{0}.elf",
              ShellCommand{"clang"}
                  .arg("-O" + std::to_string(optLevel))
                  .value("--target=", "-none-unknown-elf")
                  .arg(lldPath.empty() ? "-fuse-ld=lld" : "-fuse-ld=" + lldPath)
                  .input()
                  .arg("-Wl,--gc-sections")
                  .arg("-Wl,--orphan-handling=error")
                  .input("-Wl,-T,")
                  .output("-o"))
          .threadSafe();

  // Fresh per-core ELFs (Chess/xbridge or Peano). Cores that already carry an
  // `elf_file` attribute are handled separately by `preBakedElfs` and merged
  // into `physicalWithElfs`.
  EdgeWithTypedOutput<Directory> &compiledElfs =
      xbridge ? chessElfs : peanoElfs;

  // --- Per-device configuration artifacts ---------------------------------

  // Patch ELF paths back into the physical IR
  auto &physicalWithElfs =
      bundle(compiledElfs.out, preBakedElfs.out, physical.out)
          .join<ModRef>(
              "physical_with_elfs.mlir",
              [](const Node<Directory> &compiled, const Node<File> &preBaked,
                 const Node<ModRef> &physicalN,
                 Item<ModRef> &out) -> mlir::LogicalResult {
                // ELF paths must be absolute for the aie-rt loader.
                llvm::StringMap<std::string> byKey;
                for (const auto &item : compiled.items)
                  byKey[item.key] = absolutePath(item.filePath);
                for (const auto &item : preBaked.items)
                  byKey[item.key] = absolutePath(item.filePath);
                out.value = patchCoreElfFiles(physicalN.get().get(), byKey);
                return mlir::success();
              });

  // NPU runtime-sequence lowering needs only the placed+routed `physical`
  // module, so feeding it keeps the instruction-sequence branch independent of
  // per-core compilation. Two cases reference the compiled cores and so run on
  // the ELF-patched `physicalWithElfs` module instead:
  //   * --expand-load-pdis references the compiled cores directly.
  //   * the transaction output embeds each core's compiled program:
  //     `convert-aie-to-transaction` reads each core's `elf_file` to emit a
  //     `@configure` sequence that reprograms the cores, so the cores must be
  //     lowered (a core without an `elf_file` is skipped from the transaction).
  //   * --load-pdi-to-ctrl-pkt expands the configuration into control packets
  //     (via the same expand-load-pdi machinery), which likewise needs the
  //     compiled cores.
  bool npuTransactionsNeedCoresLowered =
      expandLoadPdis.getValue() || generateTxn || loadPdiToCtrlPkt.getValue();
  EdgeWithTypedOutput<ModRef> &npuLoweringInput =
      npuTransactionsNeedCoresLowered
          ? static_cast<EdgeWithTypedOutput<ModRef> &>(physicalWithElfs)
          : static_cast<EdgeWithTypedOutput<ModRef> &>(physical);
  // NPU instruction sequence lowering. The default and --load-pdi-to-ctrl-pkt
  // flows share the materialize + expand prefix and diverge at DMA lowering.
  EdgeWithTypedOutput<ModRef> &npuMaterialized =
      noMaterialize.getValue()
          ? npuLoweringInput
          : static_cast<EdgeWithTypedOutput<ModRef> &>(
                npuLoweringInput.map<ModRef>(
                    "npu_materialized.mlir",
                    PassPipeline{getMaterializeRuntimeSeqPipeline(&context)}));

  // For --load-pdi-to-ctrl-pkt this edge holds the control-packet ops before
  // DMA lowering: the extraction point for the control-packet binary.
  bool ctrlPkt = loadPdiToCtrlPkt.getValue();
  EdgeWithTypedOutput<ModRef> &npuExpanded =
      (expandLoadPdis.getValue() || ctrlPkt)
          ? static_cast<EdgeWithTypedOutput<ModRef> &>(
                npuMaterialized.map<ModRef>(
                    "npu_expanded.mlir",
                    PassPipeline{
                        &context,
                        [ctrlPkt](mlir::MLIRContext *ctx, mlir::ModuleOp) {
                          return getExpandLoadPdiPipeline(ctx, ctrlPkt);
                        }}))
          : npuMaterialized;

  // The default tail unrolls runtime-sequence loops and pools dynamic BDs; the
  // ctrl-packet sequence is straight-line and only needs the per-device tail,
  // after its control packets are lowered to DMA.
  EdgeWithTypedOutput<ModRef> &npuDmaLowered =
      ctrlPkt
          ? npuExpanded
                .map<ModRef>("ctrlpkt_to_dma.mlir",
                             PassPipeline{getCtrlPktToDmaPipeline(&context)})
                .map<ModRef>(
                    "ctrlpkt_npu_lowered.mlir",
                    PassPipeline{getPerDeviceDmaLoweringPipeline(&context)})
          : npuExpanded.map<ModRef>(
                "npu_dma_lowered.mlir",
                PassPipeline{getNpuDmaLoweringPipeline(&context)});

  auto &npuLowered = npuDmaLowered.map<ModRef>(
      "npu_lowered.mlir",
      [](const Item<ModRef> &item, Item<ModRef> &out) -> mlir::LogicalResult {
        ModRef clone = item.get().get().clone();
        assignDevicePdiIds(*clone);
        assignLoadPdiIds(*clone);
        out.value = std::move(clone);
        return mlir::success();
      });

  // Root of the static configuration branch; contains compiled cores, etc., to
  // produce xclbins, or feed into the full ELF. Usually, this is completely
  // independent from the NPU runtime sequence compilation; however, two passes
  // synthesize new empty_0/1 reset devices (via the shared expand-load-pdi
  // machinery), for which we must also generate PDIs / control packets, so the
  // static branch must observe those devices by rooting on `npuLowered`:
  //   * --expand-load-pdis generates the empty_0/1 devices directly.
  //   * --load-pdi-to-ctrl-pkt runs the same expansion (with ctrl-pkt=true) and
  //     additionally materializes the reconfigure runtime sequence, so the
  //     control-packet flow (getControlPacketPipeline) sees a lowered module
  //     rather than un-materialized `aiex.configure`/`aiex.run` ops.
  EdgeWithTypedOutput<ModRef> &staticInput =
      (expandLoadPdis.getValue() || loadPdiToCtrlPkt.getValue())
          ? static_cast<EdgeWithTypedOutput<ModRef> &>(npuLowered)
          : static_cast<EdgeWithTypedOutput<ModRef> &>(physicalWithElfs);
  auto &staticPerDevice =
      splitPerDevice(staticInput, "perDevice_{0}.mlir", "perDeviceMatching");

  // Per-device CDO binaries. The CDO is a *directory* of `.bin` files (the
  // libxaie v2 configuration), so it is a `Directory` bundle: filePath is the
  // directory itself and its whole contents travel together.
  auto &cdo = staticPerDevice.map<Directory>(
      "cdo_{0}",
      [](const Item<OpInModule<DeviceOp>> &item,
         Item<Directory> &out) -> mlir::LogicalResult {
        DeviceOp d = item.get().op;
        // CDO (and the PDI/xclbin built from it) is NPU-only
        if (!d.getTargetModel().hasProperty(
                xilinx::AIE::AIETargetModel::IsNPU)) {
          llvm::errs() << "aiecc: --get-cdo/-pdi/-xclbin require an NPU "
                          "device, but '"
                       << d.getSymName() << "' is not NPU\n";
          return mlir::failure();
        }
        const std::string &cdoDir = out.filePath;
        out.value = Directory{cdoDir};
        if (dryRun)
          return mlir::success();
        // The CDO output path is itself a directory that the translation
        // writes its `.bin` files into, so create it here (prepareItem only
        // makes the parent)
        if (llvm::sys::fs::create_directories(cdoDir))
          return mlir::failure();
        if (mlir::failed(xilinx::AIE::AIETranslateToCDODirect(
                item.get().module.get(), cdoDir, d.getSymName(), false, false,
                false, false, false, /*enableCores=*/true)))
          return mlir::failure();
        return mlir::success();
      });

  // CDO + BIF → PDI via bootgen
  auto &bif =
      bundle(staticPerDevice.out, cdo.out)
          .map<std::string>("bif_{0}.bif",
                            [](const Item<OpInModule<DeviceOp>> &devItem,
                               const Item<Directory> &cdoItem,
                               Item<std::string> &out) -> mlir::LogicalResult {
                              DeviceOp d = devItem.get().op;
                              out.value =
                                  makeBifText(absolutePath(cdoItem.asFile()),
                                              d.getSymName());
                              return mlir::success();
                            });

  // BIF → PDI
#ifdef AIECC_HAS_BOOTGEN_LIBRARY
  auto &pdi = bif.map<File>(pdiName.getValue(),
                            [](const Item<std::string> &bifItem,
                               Item<File> &out) -> mlir::LogicalResult {
                              if (dryRun) {
                                std::error_code ec;
                                llvm::raw_fd_ostream f(out.filePath, ec);
                                out.value = File{};
                                return mlir::success();
                              }
                              return assemblePdi(bifItem, out, verbose,
                                                 ShellCommand::progress);
                            });
#else
  auto &pdi = bif.map<File>(pdiName.getValue(), ShellCommand{"bootgen"}
                                                    .arg("-arch")
                                                    .arg("versal")
                                                    .arg("-image")
                                                    .input()
                                                    .arg("-o")
                                                    .output()
                                                    .arg("-w"));
#endif // AIECC_HAS_BOOTGEN_LIBRARY

  // Per-device control-packet artifacts: the control-packet binary and the
  // DMA sequence that streams it in.
  auto &ctrlpktLowered = staticPerDevice.map<ModRef>(
      "ctrlpkt_lowered_{0}.mlir",
      [&context](const Item<OpInModule<DeviceOp>> &item,
                 Item<ModRef> &out) -> mlir::LogicalResult {
        DeviceOp d = item.get().op;
        ModRef clone = item.get().module.get().clone();
        auto pm =
            getControlPacketPipeline(&context, /*elfDir=*/"", d.getSymName());
        if (!pm || mlir::failed(pm->run(*clone)))
          return mlir::failure();
        out.value = std::move(clone);
        return mlir::success();
      });

  auto &ctrlpkt = ctrlpktLowered.map<std::vector<char>>(
      ctrlpktName.getValue(),
      emitBinary<ModRef>(
          [](const Item<ModRef> &item, std::vector<uint32_t> &words) {
            return xilinx::AIE::AIETranslateControlPacketsToUI32Vec(
                item.get().get(), words, item.key, "");
          }));

  auto &ctrlpktDmaSeq = ctrlpktLowered.map<std::vector<char>>(
      ctrlpktDmaSeqName.getValue(),
      emitBinary<ModRef>([&context](const Item<ModRef> &item,
                                    std::vector<uint32_t> &words)
                             -> mlir::LogicalResult {
        ModRef clone = item.get().get().clone();
        if (mlir::failed(getControlPacketDmaPipeline(&context)->run(*clone)))
          return mlir::failure();
        // DDR-patch ABI: XRT (and CPU) consume the folded firmware ABI; HRX
        // consumes the producer-independent (unfolded) insts.bin and adds the
        // AIE DDR aperture offset for every arg itself. cl::opt defaults to
        // true, so only pass the flag when unfolding is requested.
        return xilinx::AIE::AIETranslateNpuToBinary(
            clone.get(), words, item.key, "",
            /*locmap=*/nullptr,
            /*foldDDRAddrOffset=*/foldDDRAddrOffsetOpt.getValue());
      }));

  // Partial ELF containing the DMA sequence and the control packet data;
  // this is still used in combination with an xclbin. The
  // ctrlpkt_extbuf_{0}.json patch tells the assembler which runtime argument
  // slot carries the control-packet buffer and how large it is.
  auto &ctrlpktExtBuf =
      bundle(staticPerDevice.out, ctrlpkt.out)
          .map<llvm::json::Value>(
              "ctrlpkt_extbuf_{0}.json",
              [seqFilter = sequenceName.getValue()](
                  const Item<OpInModule<DeviceOp>> &devItem,
                  const Item<std::vector<char>> &ctrlItem,
                  Item<llvm::json::Value> &out) -> mlir::LogicalResult {
                out.value = makeCtrlpktExtBufJson(
                    devItem.get().op, ctrlItem.get().size(), seqFilter);
                return mlir::success();
              });

  // When --get-elf is also set, the combined control-packet ELF is the
  // artifact the user asked for at --elf-name (the plain instruction ELF is
  // skipped whenever control packets are generated). Otherwise it goes to
  // --ctrlpkt-elf-name.
  std::string ctrlpktElfOutName = (generateElf && generateCtrlpkt)
                                      ? elfName.getValue()
                                      : ctrlpktElfName.getValue();

#ifdef AIECC_HAS_AIEBU_LIBRARY
  auto &ctrlpktElf =
      bundle(ctrlpktDmaSeq.out, ctrlpkt.out, ctrlpktExtBuf.out)
          .map<File>(ctrlpktElfOutName,
                     [](const Item<std::vector<char>> &dmaSeqItem,
                        const Item<std::vector<char>> &ctrlItem,
                        const Item<llvm::json::Value> &patchItem,
                        Item<File> &out) -> mlir::LogicalResult {
                       std::string patch =
                           llvm::formatv("{0:2}", patchItem.get()).str();
                       return assembleElf(dmaSeqItem.get(), ctrlItem.get(),
                                          llvm::StringRef(patch), out, verbose,
                                          ShellCommand::progress);
                     });
#else
  auto &ctrlpktElf = bundle(ctrlpktDmaSeq.out, ctrlpkt.out, ctrlpktExtBuf.out)
                         .map<File>(ctrlpktElfOutName, ShellCommand{"aiebu-asm"}
                                                           .arg("-t")
                                                           .arg("aie2txn")
                                                           .arg("-c")
                                                           .input()
                                                           .arg("-p")
                                                           .input()
                                                           .arg("-j")
                                                           .input()
                                                           .arg("-o")
                                                           .output());
#endif // AIECC_HAS_AIEBU_LIBRARY

  // Per-device xclbin (memory topology + kernel metadata + PDI partition).
  auto &memTopo = staticPerDevice.map<llvm::json::Value>(
      "memTopology_{0}.json",
      [](const OpInModule<DeviceOp> &) { return makeMemTopologyJson(); });

  std::string kName = xclbinKernelName, iName = xclbinInstanceName,
              kId = xclbinKernelId;
  auto &kernels = staticPerDevice.map<llvm::json::Value>(
      "kernels_{0}.json",
      [kName, iName, kId, seqFilter = sequenceName.getValue()](
          const Item<OpInModule<DeviceOp>> &devItem,
          Item<llvm::json::Value> &out) -> mlir::LogicalResult {
        int numHostBOs = computeNumHostBOs(devItem.get().op, seqFilter);
        if (numHostBOs > kMaxHostBOs) {
          llvm::errs() << "error: device '" << devItem.key << "' has "
                       << numHostBOs
                       << " host buffer arguments, which exceeds the maximum "
                          "supported and verified count of "
                       << kMaxHostBOs
                       << ". Reduce the number of host buffer arguments.\n";
          return mlir::failure();
        }
        out.value = makeKernelsJson(kName, iName, kId, numHostBOs);
        return mlir::success();
      });

  // Partition JSON: bundle staticPerDevice with pdi to declare the dep.
  auto &partition =
      bundle(staticPerDevice.out, pdi.out)
          .map<llvm::json::Value>(
              "partition_{0}.json",
              [kId](const Item<OpInModule<DeviceOp>> &devItem,
                    const Item<File> &pdiItem, Item<llvm::json::Value> &out) {
                out.value = makePartitionJson(
                    devItem.get().op, absolutePath(pdiItem.asFile()), kId);
                return mlir::success();
              });

  // xclbin assembly. Two selectable options:
  //  * from scratch: memory topology + kernel metadata + PDI partition;
  //  * --xclbin-input: extend an existing xclbin by merging this design's PDI
  //    into its AIE_PARTITION and adding the kernel.

  // From-scratch flow
  EdgeWithTypedOutput<File> &xclbinFromScratch =
      bundle(memTopo.out, kernels.out, partition.out)
          .map<File>(xclbinName.getValue(), ShellCommand{"xclbinutil"}
                                                .arg("--add-replace-section")
                                                .input("MEM_TOPOLOGY:JSON:")
                                                .arg("--add-kernel")
                                                .input()
                                                .arg("--add-replace-section")
                                                .input("AIE_PARTITION:JSON:")
                                                .arg("--force")
                                                .arg("--output")
                                                .output());

  // --xclbin-input flow: dump the existing xclbin's AIE_PARTITION, append this
  // design's first PDI to it, then re-emit with the merged partition and our
  // kernel
  const std::string &inXclbin = xclbinInput.getValue();
  // xclbinutil can only emit the section to a file; lift it into a parsed JSON
  // payload (via the json Deserializer) so the merge below works on the
  // in-memory object.
  // TODO: Feels like the Item deserializer abstraction should handle this
  // deserialization step from a shell command, but it does not yet.
  auto &inputPartitionFile = staticPerDevice.map<File>(
      "input_aie_partition_{0}.json", ShellCommand{"xclbinutil"}
                                          .arg("--dump-section")
                                          .outputConcat("AIE_PARTITION:JSON:")
                                          .arg("--force")
                                          .arg("--quiet")
                                          .arg("--input")
                                          .arg(inXclbin));
  auto &inputPartition = inputPartitionFile.map<llvm::json::Value>(
      "input_aie_partition_parsed_{0}.json",
      deserializeFile<llvm::json::Value>());

  auto &mergedPartition =
      bundle(inputPartition.out, partition.out)
          .map<llvm::json::Value>(
              "merged_partition_{0}.json",
              [](const Item<llvm::json::Value> &inPart,
                 const Item<llvm::json::Value> &newPart,
                 Item<llvm::json::Value> &out) -> mlir::LogicalResult {
                llvm::json::Value merged = inPart.get();
                auto *inObj = merged.getAsObject();
                const auto *newObj = newPart.get().getAsObject();
                auto *inPartObj =
                    inObj ? inObj->getObject("aie_partition") : nullptr;
                const auto *newPartObj =
                    newObj ? newObj->getObject("aie_partition") : nullptr;
                auto *inPDIs =
                    inPartObj ? inPartObj->getArray("PDIs") : nullptr;
                const auto *newPDIs =
                    newPartObj ? newPartObj->getArray("PDIs") : nullptr;
                if (!inPDIs || !newPDIs || newPDIs->empty()) {
                  llvm::errs() << "aiecc: malformed AIE_PARTITION when "
                                  "merging --xclbin-input\n";
                  return mlir::failure();
                }
                // Append only this design's first PDI.
                inPDIs->push_back((*newPDIs)[0]);
                out.value = std::move(merged);
                return mlir::success();
              });

  EdgeWithTypedOutput<File> &xclbinExtended =
      bundle(kernels.out, mergedPartition.out)
          .map<File>(xclbinName.getValue(), ShellCommand{"xclbinutil"}
                                                .arg("--input")
                                                .arg(inXclbin)
                                                .arg("--add-kernel")
                                                .input()
                                                .arg("--add-replace-section")
                                                .input("AIE_PARTITION:JSON:")
                                                .arg("--force")
                                                .arg("--output")
                                                .output());

  EdgeWithTypedOutput<File> &xclbin =
      xclbinInput.empty() ? xclbinFromScratch : xclbinExtended;

  //--------------------------------------------------------------------------//
  // NPU instruction-sequence branch
  //--------------------------------------------------------------------------//
  auto &npuLoweredPerDevice =
      splitPerDevice(npuLowered, "perDeviceNPULowered_{0}.mlir",
                     "perDeviceNPULoweredMatching");

  // Per-device transaction configuration MLIR. `convert-aie-to-transaction`
  // reads each core's ELF (the patched IR carries absolute `elf_file` paths,
  // so the empty elf-dir is only a fallback) and emits a `@configure` runtime
  // sequence of write/blockwrite ops. The cores are lowered because
  // `npuLoweringInput` selects the ELF-patched module whenever the transaction
  // output is requested (see `npuTransactionsNeedCoresLowered`).
  auto &txn = npuLoweredPerDevice.map<ModRef>(
      txnName.getValue(),
      [&context](const Item<OpInModule<DeviceOp>> &item,
                 Item<ModRef> &out) -> mlir::LogicalResult {
        DeviceOp d = item.get().op;
        ModRef clone = item.get().module.get().clone();
        auto pm =
            getTransactionPipeline(&context, /*elfDir=*/"", d.getSymName());
        if (!pm || mlir::failed(pm->run(*clone)))
          return mlir::failure();
        out.value = std::move(clone);
        return mlir::success();
      });

  // One item per runtime sequence, keyed "<device>_<sequence>"
  auto &perSeq =
      npuLowered
          .split<OpInModule<RuntimeSequenceOp>>(
              "npu_seq_{0}.mlir",
              SplitIRAction<RuntimeSequenceOp>([](RuntimeSequenceOp s) {
                return npuSeqKey(s->getParentOfType<DeviceOp>().getSymName(),
                                 s.getSymName());
              }))
          .filter("perSeqMatching",
                  [matchesDeviceFilter, seqFilter = sequenceName.getValue()](
                      const OpInModule<RuntimeSequenceOp> &x) {
                    RuntimeSequenceOp seq = x.op;
                    if (!matchesDeviceFilter(seq->getParentOfType<DeviceOp>()))
                      return false;
                    // --sequence-name: keep only the named runtime sequence.
                    return seqFilter.empty() || seq.getSymName() == seqFilter;
                  });

  // Translate each sequence exactly once into its NPU program (the .bin bytes
  // and the locmap). Two variants are built from the same per-sequence input.
  // DDR-patch ABI: XRT (and CPU) consume the folded firmware ABI; HRX consumes
  // the producer-independent (unfolded) insts.bin and adds the AIE DDR aperture
  // offset for every arg itself. cl::opt defaults to true, so only pass the
  // flag when unfolding is requested.
  auto &npuProgram = buildNpuProgramSubgraph(
      perSeq, "npu_program_{0}.bin",
      /*foldDDRAddrOffset=*/foldDDRAddrOffsetOpt.getValue());

  auto &npuInsts = npuProgram.map<std::vector<char>>(
      npuInstsName.getValue(), [](const NpuProgram &p) { return p.insts; });

  auto &npuLocmap =
      bundle(npuInsts.out, npuProgram.out)
          .map<std::string>(
              npuInstsName.getValue() + ".locmap.json",
              [](const Item<std::vector<char>> &binItem,
                 const Item<NpuProgram> &progItem,
                 Item<std::string> &out) -> mlir::LogicalResult {
                const NpuProgram &prog = progItem.get();
                std::string binName =
                    llvm::sys::path::filename(binItem.filePath).str();
                llvm::raw_string_ostream os(out.value.emplace());
                xilinx::AIE::emitNpuLocmapJSON(os, prog.deviceName, binName,
                                               prog.locmap);
                return mlir::success();
              });

  // Partial ELF; This embeds the instruction sequence in an ELF format that is
  // loaded alongside an xclbin. It reuses the per-sequence instruction binary
  // already produced by `npuInsts` rather than re-translating the module.
#ifdef AIECC_HAS_AIEBU_LIBRARY
  auto &instElf =
      npuInsts.map<File>(elfName.getValue(),
                         [](const Item<std::vector<char>> &item,
                            Item<File> &out) -> mlir::LogicalResult {
                           return assembleElf(item.get(), /*buffer2=*/{},
                                              /*patchJson=*/{}, out, verbose,
                                              ShellCommand::progress);
                         });
#else
  auto &instElf =
      npuInsts.map<File>(elfName.getValue(), ShellCommand{"aiebu-asm"}
                                                 .arg("-t")
                                                 .arg("aie2txn")
                                                 .arg("-c")
                                                 .input()
                                                 .arg("-o")
                                                 .output());
#endif // AIECC_HAS_AIEBU_LIBRARY

  //--------------------------------------------------------------------------//
  // Combined full ELF (joins the static configuration + NPU branches)
  //--------------------------------------------------------------------------//
  // DDR-patch ABI: XRT (and CPU) consume the folded firmware ABI; HRX consumes
  // the producer-independent (unfolded) insts.bin and adds the AIE DDR aperture
  // offset for every arg itself. cl::opt defaults to true, so only pass the
  // flag when unfolding is requested.
  auto &npuProgramFullElf = buildNpuProgramSubgraph(
      perSeq, "npu_program_full_elf_{0}.bin", /*foldDDRAddrOffset=*/false);
  auto &npuInstsFullElf = npuProgramFullElf.map<std::vector<char>>(
      "npu_insts_full_elf_{0}.bin",
      [](const NpuProgram &p) { return p.insts; });

  // Full ELF: all PDIs + NPU insts + control packet data if applicable.
  //
  // Control-packet data and its buffer relocation (patch info) are computed
  // PER RUNTIME SEQUENCE, not per device: a device may hold several runtime
  // sequences, each with a different argument count and its own control-packet
  // data, and each must be streamed into the argument slot of the sequence it
  // belongs to. We therefore split `npuExpanded` into one item per runtime
  // sequence (keyed "<device>_<sequence>", matching `npuInstsFullElf`) so the
  // per-sequence artifacts line up with the per-sequence instruction binaries
  // in the full-ELF config.
  //
  // If the control packet lowering is not enabled, the empty `noCtrlPktSeqs`
  // edge is fed into the full ELF assembly bundle.
  auto &ctrlPktExpandedPerSeq =
      npuExpanded.split<OpInModule<RuntimeSequenceOp>>(
          "ctrlpkt_expanded_seq_{0}.mlir",
          SplitIRAction<RuntimeSequenceOp>([](RuntimeSequenceOp s) {
            return npuSeqKey(s->getParentOfType<DeviceOp>().getSymName(),
                             s.getSymName());
          }));
  auto &noCtrlPktSeqs = g.empty<OpInModule<RuntimeSequenceOp>>("noCtrlPktSeqs");

  // `ctrlPktSeqs` contains every runtime sequence that may carry control-packet
  // data, honoring --device-name / --sequence-name (as `perSeq` does).
  auto &ctrlPktSeqs =
      (loadPdiToCtrlPkt.getValue()
           ? static_cast<EdgeWithTypedOutput<OpInModule<RuntimeSequenceOp>> &>(
                 ctrlPktExpandedPerSeq)
           : static_cast<EdgeWithTypedOutput<OpInModule<RuntimeSequenceOp>> &>(
                 noCtrlPktSeqs))
          .filter("ctrlPktSeqs",
                  [matchesDeviceFilter, seqFilter = sequenceName.getValue()](
                      const OpInModule<RuntimeSequenceOp> &x) {
                    RuntimeSequenceOp seq = x.op;
                    if (!matchesDeviceFilter(seq->getParentOfType<DeviceOp>()))
                      return false;
                    return seqFilter.empty() || seq.getSymName() == seqFilter;
                  });

  // Per-sequence control-packet binary, dropping empties so the config only
  // references sequences that actually carry a control packet.
  auto &fullElfCtrlpkt =
      ctrlPktSeqs
          .map<std::vector<char>>(
              "full_elf_{0}.ctrlpkt.bin",
              emitBinary<OpInModule<RuntimeSequenceOp>>(
                  [](const Item<OpInModule<RuntimeSequenceOp>> &item,
                     std::vector<uint32_t> &words) -> mlir::LogicalResult {
                    RuntimeSequenceOp seq = item.get().op;
                    DeviceOp d = seq->getParentOfType<DeviceOp>();
                    return xilinx::AIE::AIETranslateControlPacketsToUI32Vec(
                        item.get().module.get(), words, d.getSymName(),
                        seq.getSymName());
                  }))
          .filter("fullElfCtrlpktNonEmpty",
                  [](const std::vector<char> &bin) { return !bin.empty(); });

  // When control packets are enabled, the control data is passed into the
  // runtime sequence as an argument. `fullElfPatchInfo` captures which argument
  // index contains that control data and the size of the control-data buffer,
  // computed from THIS sequence's pre-lowering argument count (ctrl-packet-to-
  // DMA appends the control buffer as the sequence's next argument).
  auto &fullElfPatchInfo =
      bundle(fullElfCtrlpkt.out, ctrlPktSeqs.out)
          .map<llvm::json::Value>(
              "full_elf_{0}.patch_info.json",
              [](const Item<std::vector<char>> &binItem,
                 const Item<OpInModule<RuntimeSequenceOp>> &seqItem,
                 Item<llvm::json::Value> &out) -> mlir::LogicalResult {
                RuntimeSequenceOp seq = seqItem.get().op;
                int argIdx = seq.getBody().empty()
                                 ? 0
                                 : seq.getBody().front().getNumArguments();
                out.value =
                    makePatchInfoJson(argIdx, (int64_t)binItem.get().size());
                return mlir::success();
              });

  // Combined ELF: all PDIs + NPU insts bundled
  // + control packet data, if any.
  auto &fullElfConfig =
      bundle(npuLoweredPerDevice.out, pdi.out, npuInstsFullElf.out,
             fullElfCtrlpkt.out, fullElfPatchInfo.out)
          .join<llvm::json::Value>(
              "full_elf_config.json",
              [](const Node<OpInModule<DeviceOp>> &devices,
                 const Node<File> &pdis,
                 const Node<std::vector<char>> &instsBins,
                 const Node<std::vector<char>> &ctrlPkts,
                 const Node<llvm::json::Value> &patchInfos,
                 Item<llvm::json::Value> &out) -> mlir::LogicalResult {
                llvm::StringMap<std::string> pdiPaths, instsPaths;
                llvm::StringMap<std::string> ctrlPktPaths, patchInfoPaths;
                for (const auto &item : pdis.items)
                  pdiPaths[item.key] = absolutePath(item.asFile());
                for (const auto &item : instsBins.items)
                  instsPaths[item.key] = absolutePath(item.asFile());
                for (const auto &item : ctrlPkts.items)
                  ctrlPktPaths[item.key] = absolutePath(item.asFile());
                for (const auto &item : patchInfos.items)
                  patchInfoPaths[item.key] = absolutePath(item.asFile());
                out.value = makeFullElfConfigJson(devices, pdiPaths, instsPaths,
                                                  ctrlPktPaths, patchInfoPaths);
                return mlir::success();
              });

  // TODO(aiebu-aie2_config): unlike the instruction and control-packet ELFs,
  // the full ELF is assembled by shelling out to `aiebu-asm -t aie2_config`
  // rather than calling the in-process aiebu library. The library's
  // `aiebu_assembler_buffer_type_aie2_config` entry point is a no-op in this
  // XRT build (it returns a 0-byte ELF), whereas the CLI tool assembles the
  // same config correctly. This is the one remaining shell-out edge in the ELF
  // path; it should move in-memory once the library's aie2_config support is
  // understood/fixed. Until then this stays a declarative ShellCommand edge so
  // the driver never grows ad-hoc subprocess or temp-file machinery.
  auto &fullElf =
      fullElfConfig.map<File>(fullElfName.getValue(), ShellCommand{"aiebu-asm"}
                                                          .arg("-t")
                                                          .arg("aie2_config")
                                                          .arg("-j")
                                                          .input()
                                                          .arg("-o")
                                                          .output());

  //--------------------------------------------------------------------------//
  // Host program
  //--------------------------------------------------------------------------//
  // Per-device libxaie array-configuration source (`aie_inc.cpp`). Shared by
  // host compilation (as an `-I` include) and the aiesim `ps.so` build below.
  auto &aieInc = staticPerDevice.map<std::string>(
      "aie_inc.cpp",
      [](const Item<OpInModule<DeviceOp>> &item,
         Item<std::string> &out) -> mlir::LogicalResult {
        DeviceOp d = item.get().op;
        llvm::raw_string_ostream os(out.value.emplace());
        return xilinx::AIE::AIETranslateToXAIEV2(item.get().module.get(), os,
                                                 d.getSymName());
      });

  auto &hostExe = buildHostExeSubgraph(aieInc, perDeviceArches);

  //--------------------------------------------------------------------------//
  // AIE simulator Work folder
  //--------------------------------------------------------------------------//
  auto &aiesim = buildAiesimSubgraph(context, staticPerDevice, aieInc);

  //--------------------------------------------------------------------------//
  // Output selection
  //--------------------------------------------------------------------------//
  if (generateScratchpadParams)
    outputs.push_back(&paramsFile);

  // Core-ELF output: emit the per-core ELFs when --get-core-elfs is
  // passed, or as the default when no other artifact was requested (so a bare
  // `aiecc design.mlir` builds every device's cores up front).
  bool anySpecificOutput =
      generateInputWithAddresses || generateScratchpadParams ||
      generateNpuInsts || keepLoc || generateElf || generateCdo ||
      generatePdi || generateTxn || generateCtrlpkt || generateXclbin ||
      generateFullElf || wantAiesim || doCompileHost || !getOutputs.empty() ||
      !cutOutputs.empty();
  if (generateCoreElfs || !anySpecificOutput)
    outputs.push_back(&compiledElfs);

  if (generateInputWithAddresses)
    outputs.push_back(&withAddresses);
  if (generateNpuInsts)
    outputs.push_back(&npuInsts);
  if (keepLoc)
    outputs.push_back(&npuLocmap);
  // The plain instruction ELF is skipped when control packets are also being
  // generated: in that case the combined control-packet ELF (produced below)
  // is the artifact written to --elf-name.
  if (generateElf && !generateCtrlpkt)
    outputs.push_back(&instElf);
  if (generateCdo)
    outputs.push_back(&cdo);
  if (generatePdi)
    outputs.push_back(&pdi);
  if (generateTxn)
    outputs.push_back(&txn);
  // Only emit standalone control-packet artifacts (partial ELF + DMA sequence)
  // in the non-full-ELF flow. In the full-ELF-flow, the control packet data is
  // contained in the ELF and patched into a runtime sequence argument.
  if (generateCtrlpkt && !(loadPdiToCtrlPkt && generateFullElf)) {
    outputs.push_back(&ctrlpkt);
    outputs.push_back(&ctrlpktDmaSeq);
    outputs.push_back(&ctrlpktElf);
  }
  if (generateXclbin)
    outputs.push_back(&xclbin);
  if (generateFullElf)
    outputs.push_back(&fullElf);
  // AIE simulator Work folder: only when explicitly requested. The aggregator
  // edge pulls in and materializes every sim/ artifact.
  if (wantAiesim)
    outputs.push_back(&aiesim);
  // Host executable: only when explicitly requested and host sources exist.
  if (doCompileHost) {
    if (!hasHostSourceFiles())
      llvm::errs() << "aiecc: --get-host given but no host source files "
                      "were provided; skipping host compilation\n";
    else
      outputs.push_back(&hostExe);
  }

  // --get=<name> / --cut=<name>: request outputs (and cut points) by the exact
  // name their edge is registered with. A few names are registered on two edges
  // by design: the toolchain / strategy variants that emit the same artifact
  // (chess vs peano "elfs_{0}.elf", per-core vs unified "objects_{0}.o").
  // Exactly one of each pair is live in any given build, so disambiguate by
  // keeping only edges reachable from the selected terminals (`compiledElfs` /
  // `objects`) plus whatever this run already produces.
  if (!getOutputs.empty() || !cutOutputs.empty()) {
    std::vector<EdgeBase *> liveRoots = outputs;
    liveRoots.push_back(&compiledElfs);
    liveRoots.push_back(&objects);
    llvm::DenseSet<EdgeBase *> live = reachableEdges(liveRoots);

    // resolveLiveEdges does the name->edge resolution (with chess/peano
    // disambiguation); the driver owns only the error-reporting policy.
    auto select = [&](llvm::ArrayRef<std::string> names,
                      llvm::StringRef flag) -> std::vector<EdgeBase *> {
      llvm::Expected<std::vector<EdgeBase *>> resolved =
          resolveLiveEdges(g, names, live);
      if (resolved)
        return std::move(*resolved);
      llvm::errs() << "aiecc: " << flag << ": "
                   << llvm::toString(resolved.takeError())
                   << "; known outputs are:\n";
      std::set<llvm::StringRef> known;
      for (const auto &e : g.edges)
        known.insert(e->name);
      for (llvm::StringRef n : known)
        llvm::errs() << "  " << n << '\n';
      std::exit(1);
    };

    // --get selects outputs (relocated to the output dir). --cut only marks a
    // checkpoint cut point: the edge is built (see Engine::run `buildAlso`) but
    // stays in the work dir as an intermediate, so downstream consumers that
    // reference it by path (e.g. the CDO step loading core ELFs) still find it.
    for (EdgeBase *e : select(getOutputs, "--get"))
      outputs.push_back(e);
    for (EdgeBase *e : select(cutOutputs, "--cut"))
      cutEdges.push_back(e);
  }

  return outputs;
}

//===----------------------------------------------------------------------===//
// Post-build stack-size sufficiency check
//===----------------------------------------------------------------------===//

// After a normal build completes, check whether any core's `stack_size` --
// explicit or the device default -- is smaller than its TRUE requirement:
// the core body's own top-level frame (only knowable now, from the
// just-compiled core object) plus the already-computed kernel-side bound
// (checkStackSizeRequirements's `aiecc.computed_stack_requirement`). If so,
// that build's placed buffers are wrong: exactly the silent-corruption gap
// this whole check exists to close (real usage exceeds what was assumed when
// buffers were placed, and nothing before this point ever validated
// `stack_size` against actual usage rather than a lower bound).
//
// This never auto-adjusts anything -- consistent with every other check in
// this analysis (and reserved_data_size before it): the compiler measures
// and reports, the user declares and rebuilds. Unlike
// checkStackSizeRequirements earlier in the pipeline -- which only warns on an
// explicit value, because its number is a lower bound that proves nothing when
// it happens to fit -- this later check has the TRUE total, so an explicit
// `stack_size` that is provably too small fails the build exactly like an
// absent one: a warning here would ship a proven overflow.
// `--no-auto-stack-size` skips this check entirely, the same escape hatch that
// skips the earlier warning.
//
// Re-derives the early, cheap pipeline stages (placement/trace/link-files
// assignment/stack-check -- ordinary MLIR passes, not a recompile) on a
// fresh parse of the input, since the full build graph doesn't expose its
// intermediate module as a reusable in-memory result once `engine.run` has
// returned. The compiled core object itself is not rebuilt here -- it is
// read back from where the just-finished build already wrote it. Returns
// true if any core's requirement was insufficient (the caller must fail the
// build); an `llvm::Error` reports a mechanical failure of this check itself.
static llvm::Expected<bool>
checkStackSizeIsSufficient(mlir::MLIRContext &context,
                           llvm::StringRef inputFile) {
  if (noAutoStackSize.getValue())
    return false;

  // This check re-derives placement from a fresh parse rather than reusing
  // the real build's module (see the comment above), so it depends on
  // placement being reproducible from the same flags. `--placer=sa_placer
  // --sa-seed=0` is explicitly non-deterministic (CommandLineOptions.h), so a
  // re-derived tile assignment can disagree with the one buffers were
  // actually placed against; `coreKey`/the object path computed below would
  // then silently name the wrong core's object, and measureFunctionFrameSize
  // would simply fail to find it -- a false negative, not a diagnosable
  // mismatch. Warn and skip rather than risk missing a real overflow.
  if (placerType.getValue() == xilinx::AIE::PlacerType::SAPlacer &&
      saSeed.getValue() == 0) {
    llvm::errs() << "aiecc: stack_size check: skipped -- "
                    "--placer=sa_placer --sa-seed=0 is non-deterministic, so "
                    "this check's re-derived placement cannot be trusted to "
                    "match the build's; pass a nonzero --sa-seed, or "
                    "--no-auto-stack-size to silence this note\n";
    return false;
  }

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(inputFile, &context);
  if (!module)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "could not reparse input file '%s' to "
                                   "check stack_size sufficiency",
                                   inputFile.str().c_str());

  // Reproduce --default-stack-size's population too: if it ran during the
  // real build, `assumed` below must reflect the same value the buffers were
  // actually placed against, not the target's built-in default.
  if (defaultStackSize.getValue() > 0)
    module = xilinx::aiecc::populateDefaultStackSize(
        module.get(), defaultStackSize.getValue());

  // Reproduce exactly the pipeline stages checkStackSizeRequirements' result
  // depends on: placement (tile assignment), trace flows, and link_files
  // assignment. All cheap MLIR passes, not a recompile.
  auto runStage = [&](llvm::StringRef stage,
                      std::unique_ptr<mlir::PassManager> pm) -> llvm::Error {
    if (mlir::failed(pm->run(*module)))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "%s failed while re-deriving "
                                     "stack-size state",
                                     stage.str().c_str());
    return llvm::Error::success();
  };

  // Every diagnostic in this block was already shown once during the real
  // build that produced the object this check reads back (or is a mechanical
  // re-derivation failure reported as the llvm::Error below, whose text never
  // depends on what an MLIR diagnostic said) -- printing it again here would
  // just be noise. The diagnostics this function exists to produce (the
  // "is insufficient"/"is absent" errors below) are emitted after this
  // handler goes out of scope.
  {
    mlir::ScopedDiagnosticHandler suppress(
        &context, [](mlir::Diagnostic &) { return mlir::success(); });

    if (auto err = runStage("placement pipeline",
                            xilinx::aiecc::getPlacementPipeline(
                                &context, coresPerCol.getValue(),
                                placerType.getValue(), saSeed.getValue())))
      return std::move(err);
    if (auto err = runStage("trace pipeline",
                            xilinx::aiecc::getTracePipeline(&context)))
      return std::move(err);
    {
      auto pm = std::make_unique<mlir::PassManager>(&context);
      pm->nest<xilinx::AIE::DeviceOp>().addPass(
          xilinx::AIE::createAIEAssignCoreLinkFilesPass());
      if (auto err = runStage("link-files assignment", std::move(pm)))
        return std::move(err);
    }
    if (mlir::failed(xilinx::aiecc::checkStackSizeRequirements(
            *module, inputFile, getWorkDir())))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "stack-size check failed while "
                                     "re-deriving stack-size state");
  }

  bool anyInsufficient = false;
  module->walk([&](xilinx::AIE::CoreOp coreOp) {
    auto boundAttr = coreOp->getAttrOfType<mlir::IntegerAttr>(
        xilinx::AIE::kComputedStackRequirementAttrName);
    if (!boundAttr)
      return; // No trustworthy kernel-side number -- don't guess.

    auto tile =
        mlir::cast<xilinx::AIE::TileOp>(coreOp.getTile().getDefiningOp());
    auto dev = coreOp->getParentOfType<xilinx::AIE::DeviceOp>();
    std::string key = xilinx::aiecc::coreKey(coreOp);
    std::string symbol =
        xilinx::AIE::coreFrameSymbolName(tile.getCol(), tile.getRow());
    // Must match the object this core's real compile actually wrote: the
    // "objects_{0}.o"/"unifiedObjects_{0}.o" edges above, keyed the same way
    // (per-core `key`, or the device's symbol name when --unified), each
    // materialized as Actions.h's stem-named-subdirectory convention
    // (objects_<key>/objects_<key>.o). If either edge's name or key ever
    // changes, this must change with it, or measureFunctionFrameSize below
    // silently finds nothing and this check silently no-ops for that core.
    std::string objPath =
        doUnified ? getWorkDir() + "/unifiedObjects_" + dev.getSymName().str() +
                        "/unifiedObjects_" + dev.getSymName().str() + ".o"
                  : getWorkDir() + "/objects_" + key + "/objects_" + key + ".o";
    auto ownFrame = xilinx::aiecc::measureFunctionFrameSize(objPath, symbol);
    if (!ownFrame)
      return; // Can't measure the core's own frame -- leave as-is; today's
              // behavior, not a regression.

    int64_t trueTotal = *ownFrame + boundAttr.getInt();
    uint32_t assumed = coreOp.getEffectiveStackSize();
    if (trueTotal <= static_cast<int64_t>(assumed))
      return;

    anyInsufficient = true;
    if (coreOp.getStackSizeAttr())
      coreOp.emitError()
          << "stack_size = " << assumed
          << " is insufficient (this core's buffers were placed assuming "
          << assumed << " bytes), but this core's real requirement is "
          << trueTotal << " bytes; increase stack_size to " << trueTotal
          << " (Worker(stack_size=...) in IRON) and rebuild, or pass "
             "--no-auto-stack-size to skip this check";
    else
      coreOp.emitError()
          << "stack_size is absent (this core's buffers were placed "
             "assuming the device default of "
          << assumed << " bytes), but this core's real requirement is "
          << trueTotal << " bytes; set stack_size = " << trueTotal
          << " explicitly on this aie.core (Worker(stack_size=...) in IRON) "
             "and rebuild, or pass --no-auto-stack-size to skip this check";
  });
  return anyInsufficient;
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {

  //--------------------------------------------------------------------------//
  // Context setup
  //--------------------------------------------------------------------------//

  llvm::InitLLVM y(argc, argv);
  mlir::registerAsmPrinterCLOptions();
  mlir::registerAllPasses();
  xilinx::registerConversionPasses();
  xilinx::AIE::registerAIEPasses();
  xilinx::AIEX::registerAIEXPasses();
  xilinx::aievec::registerAIEVecPasses();
  xilinx::aievec::registerAIEVecPipelines();

  llvm::cl::SetVersionPrinter(printVersion);

  // If --resume=<manifest> is given, rebuild the effective command line from
  // the checkpoint manifest (parsing lives in CommandLineOptions.h); otherwise
  // use argv as-is. `graphArgv` is what a checkpoint written by this run
  // records so a later resume rebuilds an identical graph.
  cli::ResumeState resume;
  std::vector<std::string> graphArgv;
  std::optional<std::vector<std::string>> effArgvStore =
      cli::resolveCommandLine(argc, argv, resume, graphArgv);
  if (!effArgvStore)
    return 1;
  // Resolve the `--get-<name>` artifact shorthands (setting the
  // output-selection bools) before cl parsing sees them.
  if (!cli::applyOutputSelectorFlags(*effArgvStore))
    return 1;
  std::vector<char *> effArgvPtrs;
  effArgvPtrs.reserve(effArgvStore->size());
  for (std::string &s : *effArgvStore)
    effArgvPtrs.push_back(s.data());
  int effArgc = static_cast<int>(effArgvPtrs.size());
  char **effArgv = effArgvPtrs.data();

  // Split host-compiler passthrough args (after a `--` separator) off before cl
  // parsing: everything before `--` is parsed strictly, everything after is
  // forwarded verbatim to the host compiler (AIE1 host-compilation flow only).
  // Truncating parseArgc keeps cl from treating the tail as positionals.
  int parseArgc = effArgc;
  for (int i = 1; i < effArgc; ++i)
    if (llvm::StringRef(effArgv[i]) == "--") {
      parseArgc = i;
      hostPassthroughArgs.assign(effArgv + i + 1, effArgv + effArgc);
      break;
    }
  llvm::cl::ParseCommandLineOptions(parseArgc, effArgv,
                                    "aiecc declarative driver\n");

  // Exactly one input MLIR file may appear before the `--` separator; host
  // source files and host-compiler flags belong after it.
  if (positionalArgs.size() > 1) {
    llvm::errs() << "aiecc: only one input MLIR file is allowed before '--'; "
                    "pass host source files and host-compiler flags after "
                    "'--'\n";
    return 1;
  }

  if (showVersion) {
    printVersion(llvm::outs());
    return 0;
  }

  // Resolve inter-option coupling once, up front: the Chess/Peano toolchain
  // selection (xchesscc/xbridge), the --get-aiesim implication, and the
  // resolved-option globals (wantAiesim, doUnified, doCompileHost). See
  // CommandLineOptions.h.
  if (!cli::resolveOptions())
    return 1;

  // --expand-load-pdis reconfigures via PDI swaps and routes the config branch
  // through the NPU-lowered module; control-packet generation currently
  // assumes it runs on the *pre*-NPU-lowering module. The two are incompatible
  // as implemented.
  if (expandLoadPdis && generateCtrlpkt) {
    llvm::errs() << "aiecc: --expand-load-pdis and --get-ctrlpkt are "
                    "mutually exclusive\n";
    return 1;
  }

  // --expand-load-pdis and --load-pdi-to-ctrl-pkt are two different reconfigure
  // strategies for the same `load_pdi` ops (explicit write sequences vs.
  // streamed control packets); at most one may apply.
  if (expandLoadPdis && loadPdiToCtrlPkt) {
    llvm::errs() << "aiecc: --expand-load-pdis and --load-pdi-to-ctrl-pkt are "
                    "mutually exclusive\n";
    return 1;
  }

  // Disambiguate the full-ELF control packet flow and the standalone
  // artifact flows.
  if (generateFullElf && generateCtrlpkt && !loadPdiToCtrlPkt) {
    llvm::errs() << "aiecc: --generate-full-elf and --aie-generate-ctrlpkt "
                    "together also requires --load-pdi-to-ctrl-pkt\n";
    return 1;
  }

  // --cut only makes sense as a checkpoint frontier: it stops the build at the
  // named edge(s) and snapshots them, so it requires --checkpoint to say where.
  if (!cutOutputs.empty() && checkpointDir.empty()) {
    llvm::errs()
        << "aiecc: --cut requires --checkpoint (it marks where to stop "
           "the build and snapshot it for a later --resume)\n";
    return 1;
  }

  // MLIR Context
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  xilinx::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  xilinx::aievec::registerTransformDialectExtension(registry);
  registerLLVMIRTranslations(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::SourceMgr sourceMgr;
  unsigned inputBufferId = 0;
  if (auto inputBuf = mlir::openInputFile(getInputFilename()))
    inputBufferId =
        sourceMgr.AddNewSourceBuffer(std::move(inputBuf), llvm::SMLoc());
  mlir::SourceMgrDiagnosticHandler diagHandler(sourceMgr, &context);
  if (!ShellCommand::addInstallPrefix("peano", peanoInstallDir))
    return 1;
  // discoverAietoolsDir has the same shape: it falls through to $AIETOOLS_ROOT
  // and then to xchesscc on PATH.
  if (!aietoolsDir.empty() && !llvm::sys::fs::is_directory(aietoolsDir)) {
    llvm::errs() << "aiecc: --aietools directory does not exist: "
                 << aietoolsDir << "\n";
    return 1;
  }
  ShellCommand::verbose = verbose;
  ShellCommand::dryRun = dryRun;

  // Honor an explicit xclbinutil override (--xclbinutil-path, else
  // AIE_XCLBINUTIL) when packaging an xclbin. A value with a path separator
  // must resolve to an executable file; a bare name is looked up on PATH. When
  // set but unusable we fail loudly instead of silently falling back to a PATH
  // lookup, so a pure-HRX / pure-XRT flow can guarantee which xclbinutil is
  // used.
  if (generateXclbin) {
    std::string ovr = xclbinutilPath;
    if (ovr.empty())
      if (const char *env = std::getenv("AIE_XCLBINUTIL"))
        ovr = env;
    if (!ovr.empty()) {
      std::string resolved;
      if (ovr.find('/') != std::string::npos ||
          ovr.find('\\') != std::string::npos) {
        if (llvm::sys::fs::can_execute(ovr))
          resolved = ovr;
      } else if (auto r = llvm::sys::findProgramByName(ovr)) {
        resolved = *r;
      }
      if (resolved.empty()) {
        llvm::errs() << "Error: requested xclbinutil '" << ovr
                     << "' not found or not executable\n";
        return 1;
      }
      ShellCommand::setToolOverride("xclbinutil", resolved);
    }
  }

  //--------------------------------------------------------------------------//
  // Compilation artifact graph
  //--------------------------------------------------------------------------//
  // All edge declarations live in buildMainGraph; main just builds the graph
  // and then either visualizes it (--emit-dot) or runs it through the engine.
  Graph g;
  std::vector<EdgeBase *>
      cutEdges; // the --cut points, captured by --checkpoint
  std::vector<EdgeBase *> outputs = buildMainGraph(context, g, cutEdges);

  // --emit-dot: visualize the (pruned) static graph and exit without running.
  // Needs no input file (the graph is static), so it runs before the input-file
  // check below. A --cut/--checkpoint cut is marked in the output.
  if (emitDot) {
    writeDotGraph(g, outputs, llvm::outs(), cutEdges);
    return 0;
  }

  // Every other mode actually runs the graph, which requires an input .mlir.
  if (getInputFilename().empty()) {
    llvm::errs() << "aiecc: no input file specified; expected an input .mlir\n";
    return 1;
  }

  // Reject an empty (or whitespace-only) input up front. A --resume is exempt:
  // the restored frontier feeds the downstream edges, so the original input
  // .mlir is usually pruned (and may be gone, e.g. wiped by a caller's failed-
  // compile cleanup). If it turns out to be needed, its fileInput edge errors
  // when executed.
  if (!resume.active) {
    if (sourceMgr.getNumBuffers() == 0) {
      llvm::errs() << "aiecc: could not open input file '" << getInputFilename()
                   << "'\n";
      return 1;
    }
    if (sourceMgr.getMemoryBuffer(inputBufferId)->getBuffer().trim().empty()) {
      llvm::errs() << "aiecc: input file '" << getInputFilename()
                   << "' is empty; expected MLIR containing an aie.device\n";
      return 1;
    }
  }

  // Resume: map each checkpoint frontier entry to its producing edge and
  // satisfy it from the saved artifacts instead of recomputing. Edge lookup and
  // its chess/peano disambiguation are shared with --get via resolveLiveEdge.
  llvm::DenseMap<EdgeBase *, RestoredNode> satisfied;
  if (resume.active) {
    // With --get, a resume targets exactly the requested edge(s) (a surgical
    // suffix) rather than adding to the manifest's full build.
    if (!getOutputs.empty()) {
      llvm::DenseSet<llvm::StringRef> want(getOutputs.begin(),
                                           getOutputs.end());
      std::vector<EdgeBase *> filtered;
      for (EdgeBase *e : outputs)
        if (want.count(e->name))
          filtered.push_back(e);
      outputs = std::move(filtered);
    }
    llvm::DenseSet<EdgeBase *> reach = reachableEdges(outputs);
    for (const cli::CheckpointEntry &fe : resume.frontier) {
      llvm::Expected<EdgeBase *> e = resolveLiveEdge(g, fe.name, reach);
      if (!e) {
        llvm::errs() << "aiecc: --resume: " << llvm::toString(e.takeError())
                     << "\n";
        return 1;
      }
      llvm::SmallString<256> p(resume.manifestDir);
      llvm::sys::path::append(p, fe.dir);
      satisfied[*e] = RestoredNode{fe.descriptor, std::string(p.str())};
    }
  }

  // Progress is on by default; --no-progress turns it off, and --verbose
  // (line-per-edge logging) takes precedence over the single-line display.
  bool showProgress = !noProgress && !verbose;
  ShellCommand::progress = showProgress;
  Engine engine({outputDir, getWorkDir(), verbose, showProgress,
                 keepIntermediates, numThreads, profile});
  // --cut stops the build at the cut point: only the prefix up to the cut
  // edges is produced (as work-dir intermediates) and snapshotted by
  // --checkpoint; the requested final artifacts are NOT built here (the
  // recorded manifest argv lets a later --resume build them). Without --cut,
  // build the requested outputs normally. `cutEdges` is empty on a --resume.
  const std::vector<EdgeBase *> noOutputs;
  const std::vector<EdgeBase *> &runOutputs =
      cutEdges.empty() ? outputs : noOutputs;
  // Captured so a later check (the stack-size sufficiency check below) that
  // fails the build can remove these again: without this, a build whose
  // stack_size turns out insufficient still leaves a complete-looking xclbin
  // in outputDir, and a caller that doesn't check aiecc's exit code -- e.g.
  // `make`, which then sees an up-to-date target on the next invocation --
  // picks up a binary with buffers placed against the wrong stack size.
  std::vector<std::string> writtenOutputPaths;
  if (mlir::failed(engine.run(g, runOutputs, satisfied,
                              DeserializeContext{&context}, cutEdges,
                              &writtenOutputPaths))) {
    // On-failure reproducer ("repeater"): dump a checkpoint of the failed
    // edge's already-computed inputs and print a command that reloads them and
    // re-runs just the failed edge. Opt-in via --enable-repeater-scripts.
    if (enableRepeaterScripts && !disableRepeaterScripts && engine.failedEdge) {
      std::string dir = repeaterOutputDir.empty()
                            ? getWorkDir() + "/repeater"
                            : repeaterOutputDir.getValue();
      std::vector<EdgeBase *> frontierEdges;
      for (NodeBase *n : engine.failedEdge->inputNodes())
        if (n && n->producer)
          frontierEdges.push_back(n->producer);
      // Record argv that rebuilds this graph, narrowed to the failed edge so a
      // resume reloads its inputs and re-runs only it.
      std::vector<std::string> reproArgv = graphArgv;
      reproArgv.push_back("--get=" + engine.failedEdge->name);
      writeCheckpoint(frontierEdges, dir, reproArgv);
      llvm::errs() << "aiecc: To reproduce, run: aiecc --resume=" << dir
                   << "/manifest.json\n";
    }
    llvm::errs() << "aiecc: pipeline failed\n";
    return 1;
  }

  // --checkpoint: dump the --cut cut (artifacts + manifest.json) so a later
  // --resume can reload it and continue.
  if (!checkpointDir.empty())
    writeCheckpoint(cutEdges, checkpointDir, graphArgv);

  // The stack_size sufficiency check needs a real, complete build (a
  // compiled core object to read back) -- skip it for a --cut early stop or
  // a --dry-run, neither of which produced one.
  if (cutEdges.empty() && !dryRun) {
    llvm::Expected<bool> insufficient =
        checkStackSizeIsSufficient(context, getInputFilename());
    if (!insufficient) {
      llvm::errs() << "aiecc: stack_size check: "
                   << llvm::toString(insufficient.takeError()) << "\n";
      return 1;
    }
    if (*insufficient) {
      // The build just placed this design's buffers against a stack_size
      // that is now proven too small -- the artifacts engine.run wrote are
      // exactly the silent-corruption case this check exists to catch, so
      // leaving them in outputDir would let a caller that doesn't check
      // aiecc's exit code (or `make`, which would see an up-to-date target
      // next time) pick up a binary that corrupts memory at runtime.
      for (const std::string &path : writtenOutputPaths) {
        std::error_code ec = llvm::sys::fs::remove(path);
        if (ec && ec != std::errc::no_such_file_or_directory)
          llvm::errs() << "aiecc: warning: could not remove invalid output '"
                       << path << "': " << ec.message() << "\n";
      }
      llvm::errs() << "aiecc: pipeline failed\n";
      return 1;
    }
  }

  // aiesim.sh is produced as a plain-text Item; make it launchable. (The Item
  // abstraction has no notion of an executable bit, so set it here on the
  // materialized artifact.)
  if (wantAiesim && !dryRun) {
    std::string script = getWorkDir() + "/aiesim.sh";
    if (llvm::sys::fs::exists(script)) {
      if (std::error_code ec = llvm::sys::fs::setPermissions(
              script, llvm::sys::fs::perms::owner_all |
                          llvm::sys::fs::perms::group_exe |
                          llvm::sys::fs::perms::others_exe))
        llvm::errs() << "aiecc: cannot make '" << script
                     << "' executable: " << ec.message() << "\n";
    }
  }

  return 0;
}
