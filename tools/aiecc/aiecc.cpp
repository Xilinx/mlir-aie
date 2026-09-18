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
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
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

namespace {

//===----------------------------------------------------------------------===//
// Shared subgraphs
//===----------------------------------------------------------------------===//

using ModRef = mlir::OwningOpRef<mlir::ModuleOp>;
using xilinx::AIE::DeviceOp;

// Produce a per-key object (.o) -- these are the core program memories. Both
// lowering strategies feed this per core; they differ only in how the modules
// arriving here were produced. We define a chess path and a peano path; the
// `xchesscc` command-line flag selects which output edge is returned.
//
// `irLinkFiles` carries, per key, the merge-mode kernel artifacts (the core's
// `link_merge_files`, i.e. `link_with_mode = "merge"`) to llvm-link into that
// key's module before codegen. Keys with an empty list get the plain compile
// flow. Only the peano path can merge them; the chess path consumes the same
// edge solely to reject a non-empty list with a diagnostic.
EdgeWithTypedOutput<Directory> &
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
EdgeWithTypedOutput<File> &
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
EdgeWithTypedOutput<File> &
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
        return xilinx::AIE::AIETranslateGraphXPE(item.get().mod(), os,
                                                 d.getSymName());
      });
  auto &shim = staticPerDevice.map<std::string>(
      "sim/arch/aieshim_solution.aiesol",
      [](const Item<OpInModule<DeviceOp>> &item,
         Item<std::string> &out) -> mlir::LogicalResult {
        DeviceOp d = item.get().op;
        llvm::raw_string_ostream os(out.value.emplace());
        return xilinx::AIE::AIETranslateShimSolution(item.get().mod(), os,
                                                     d.getSymName());
      });
  auto &scsim = staticPerDevice.map<std::string>(
      "sim/config/scsim_config.json",
      [](const Item<OpInModule<DeviceOp>> &item,
         Item<std::string> &out) -> mlir::LogicalResult {
        DeviceOp d = item.get().op;
        llvm::raw_string_ostream os(out.value.emplace());
        return xilinx::AIE::AIETranslateSCSimConfig(item.get().mod(), os,
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
                mlir::ModuleOp mod = devs.items.front().get().mod();
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
EdgeWithTypedOutput<NpuProgram> &buildNpuProgramSubgraph(
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
                item.get().mod(), insts, devOp.getSymName(), seq.getSymName(),
                &prog.locmap, foldDDRAddrOffset)))
          return mlir::failure();
        prog.insts = wordsToBytes(insts);
        out.value = std::move(prog);
        return mlir::success();
      });
  npuProgram.producesFiles = false;
  return npuProgram;
}

//===----------------------------------------------------------------------===//
// --reconfig-method: fold N single-config designs, split into init/configs
//===----------------------------------------------------------------------===//

// A device is "tile-bearing" (a config, not a tile-less host) if it declares
// any placed `aie.tile` (TileOp) OR unplaced `aie.logical_tile`
// (LogicalTileOp) -- the shape real IRON `as_mlir()` output emits, before
// AIEPlaceTiles assigns coordinates. Keying on TileOp alone misclassifies a
// real (unplaced) idiomatic design as a tile-less host, silently bypassing
// the fold below.
// Reconfiguration delivery method, resolved once from the --reconfig-method
// flag string instead of re-comparing the literal at each decision point.
enum class ReconfigMethod { None, Loadpdi, Write32, Ctrlpkt };
static ReconfigMethod parseReconfigMethod(llvm::StringRef m) {
  if (m == "loadpdi")
    return ReconfigMethod::Loadpdi;
  if (m == "write32")
    return ReconfigMethod::Write32;
  if (m == "ctrlpkt")
    return ReconfigMethod::Ctrlpkt;
  return ReconfigMethod::None;
}

static bool deviceHasTiles(xilinx::AIE::DeviceOp d) {
  return !d.getOps<xilinx::AIE::TileOp>().empty() ||
         !d.getOps<xilinx::AIE::LogicalTileOp>().empty();
}

// Synthesize the persistent-host scaffolding for idiomatic single-device
// inputs (no tile-less host), in-memory: one @main host device carrying one
// entry sequence per design (name preserved from the design; each
// `configure @config_i { run @<seq>(args) }`) plus the N config devices renamed
// @config_1..@config_N. Writes the merged module; returns the path, or "" on
// error (already diagnosed).

// Write a union/merged module to <workDir>/config_union.mlir. Returns the path,
// or "" on a filesystem error (diagnostic already emitted). Shared by the two
// --reconfig-method fold paths (conformIdiomaticInputs,
// unionConfigDesigns).
static std::string writeMergedModule(mlir::ModuleOp mod,
                                     llvm::StringRef workDir) {
  if (auto ec = llvm::sys::fs::create_directories(workDir)) {
    llvm::errs() << "aiecc: --reconfig-method: cannot create " << workDir
                 << ": " << ec.message() << "\n";
    return {};
  }
  std::string outPath = (workDir + "/config_union.mlir").str();
  std::error_code ec;
  llvm::raw_fd_ostream os(outPath, ec);
  if (ec) {
    llvm::errs() << "aiecc: --reconfig-method: cannot write " << outPath << ": "
                 << ec.message() << "\n";
    return {};
  }
  mod.print(os);
  return outPath;
}

// Label the fold-synthesized entry (top-level host) device with the
// aiex.entrypoint marker -- a dictionary carrying the reconfig_method. Its
// presence is what downstream (splitMultiConfigEntry entry select, SidecarFiles
// host-keep) uses to identify the entry device, robust to the sequence
// sym_names. See kEntrypointAttr.
static void markEntrypointDevice(xilinx::AIE::DeviceOp host,
                                 llvm::StringRef reconfigMethod) {
  mlir::MLIRContext *ctx = host.getContext();
  mlir::NamedAttribute methodAttr(
      mlir::StringAttr::get(ctx, xilinx::aiecc::kReconfigMethodKey),
      mlir::StringAttr::get(ctx, reconfigMethod));
  host->setAttr(xilinx::aiecc::kEntrypointAttr,
                mlir::DictionaryAttr::get(ctx, {methodAttr}));
}

static std::string conformIdiomaticInputs(llvm::ArrayRef<std::string> inputs,
                                          llvm::StringRef workDir,
                                          mlir::MLIRContext &context,
                                          llvm::StringRef reconfigMethod) {
  using namespace mlir;
  using xilinx::AIE::DeviceOp;
  using xilinx::AIE::EndOp;
  using xilinx::AIE::RuntimeSequenceOp;
  using xilinx::AIEX::ConfigureOp;
  using xilinx::AIEX::RunOp;

  Location loc = UnknownLoc::get(&context);
  OpBuilder b(&context);
  OwningOpRef<ModuleOp> merged = ModuleOp::create(b, loc);
  DeviceOp host;
  Block *hostBody = nullptr;
  std::optional<xilinx::AIE::AIEDevice> arch;
  unsigned i = 0;

  for (StringRef in : inputs) {
    OwningOpRef<ModuleOp> mod = parseSourceFile<ModuleOp>(in, &context);
    if (!mod) {
      llvm::errs() << "aiecc: --reconfig-method: failed to parse " << in
                   << "\n";
      return {};
    }

    // Exactly one tile-bearing device, no tile-less host.
    DeviceOp cfg = nullptr;
    bool sawHost = false;
    for (DeviceOp d : mod->getOps<DeviceOp>()) {
      if (!deviceHasTiles(d))
        sawHost = true;
      else if (!cfg)
        cfg = d;
      else {
        llvm::errs() << "aiecc: --reconfig-method: " << in
                     << ": expected exactly one tile-bearing device\n";
        return {};
      }
    }
    if (sawHost || !cfg) {
      llvm::errs() << "aiecc: --reconfig-method: " << in
                   << ": expected an idiomatic single-device design\n";
      return {};
    }

    // Arch consistency (ConfigureOp verifier requires host==config arch).
    if (!arch) {
      arch = cfg.getDevice();
      b.setInsertionPointToStart(merged->getBody());
      host = DeviceOp::create(b, loc, *arch); // model: AIEExpandLoadPdi.cpp:100
      // @main matches the device name IRON's own full-ELF path uses
      // (program.py resolve_program(device_name="main")), so jit derives the
      // same <device>:<sequence> kernel name it dispatches by -- no override.
      host.setSymName("main");
      // Label the synthesized entry device so downstream identifies it by the
      // marker's presence rather than the (migrating) sequence sym_names.
      markEntrypointDevice(host, reconfigMethod);
      hostBody = b.createBlock(&host.getRegion());
      OpBuilder endBuilder(hostBody, hostBody->end());
      EndOp::create(endBuilder, loc); // aie.end terminator
    } else if (cfg.getDevice() != *arch) {
      llvm::errs() << "aiecc: --reconfig-method: mixed device targets\n";
      return {};
    }

    // Root runtime sequence (exactly one).
    RuntimeSequenceOp seq = nullptr;
    unsigned nseq = 0;
    for (RuntimeSequenceOp s : cfg.getOps<RuntimeSequenceOp>()) {
      seq = s;
      ++nseq;
    }
    if (nseq != 1) {
      llvm::errs() << "aiecc: --reconfig-method: " << in
                   << ": expected exactly one runtime_sequence (found " << nseq
                   << ")\n";
      return {};
    }
    // The design's runtime_sequence name IS the user's entrypoint name (the jit
    // name= key, or the default "sequence"). It labels the LIFTED entrypoint
    // the fold synthesizes below and deduces the config device name; the
    // design's own sequence -- now internal (referenced only by aiex.run) --
    // reverts to the canonical "sequence". Only the entrypoint is ever
    // dispatched (main:<entrypoint>), so the config device and its inner
    // sequence are purely internal. Distinct designs must therefore carry
    // distinct names (splitMultiConfigEntry loud-fails on a duplicate
    // entrypoint name).
    std::string entryName = seq.getSymName().str();

    // Reject a pre-embedded load_pdi: the union fold and the downstream
    // ctrl-pkt materialization own PDI sequencing, so a design that already
    // carries its own aiex.npu.load_pdi would double up (or race) with that.
    if (!seq.getBody().getOps<xilinx::AIEX::NpuLoadPdiOp>().empty()) {
      llvm::errs() << "aiecc: --reconfig-method: " << in
                   << ": idiomatic input must not carry a pre-embedded "
                      "load_pdi (emit on the non-full-ELF path)\n";
      return {};
    }

    // Config device is deduced as <entrypoint>_config, so its name follows the
    // user's design name (never persists the input's own device name); the
    // final verify() is the backstop against a collision.
    std::string cfgName = entryName + "_config";

    ++i;

    // Clone the config device into the merged module, renamed <name>_config,
    // and revert its internal runtime_sequence to the canonical "sequence" (it
    // is referenced only by the entrypoint's aiex.run, never dispatched
    // directly).
    b.setInsertionPointToEnd(merged->getBody());
    auto *clone = b.clone(*cfg.getOperation()); // model: aiecc.cpp:697
    auto cfgDev = cast<DeviceOp>(clone);
    cfgDev.setSymName(cfgName);
    for (RuntimeSequenceOp s : cfgDev.getOps<RuntimeSequenceOp>())
      s.setSymName("sequence");

    // Append the LIFTED entrypoint before @main's aie.end, named with the
    // user's entrypoint name so the dispatchable kernel is main:<entrypoint>.
    // It issues `configure @<name>_config { run @sequence }`;
    // splitMultiConfigEntry keeps this name (block order = chain order) and,
    // for the init methods, synthesizes a shared main:init.
    SmallVector<Type> argTys(seq.getBody().getArgumentTypes());
    SmallVector<Location> argLocs(argTys.size(), loc);
    OpBuilder hb(hostBody->getTerminator());
    auto seqOp = RuntimeSequenceOp::create(
        hb, loc, entryName, BoolAttr{}); // model: AIEToConfiguration.cpp:976
    Block *seqEntry = hb.createBlock(&seqOp.getBody(), {}, argTys, argLocs);
    OpBuilder sb(seqEntry, seqEntry->end());
    auto conf =
        ConfigureOp::create(sb, loc, FlatSymbolRefAttr::get(&context, cfgName),
                            /*expand_mode=*/nullptr);
    Block *confBody = sb.createBlock(
        &conf.getBody()); // model: AIEMaterializeRuntimeSequences.cpp:131
    OpBuilder cb(confBody, confBody->end());
    RunOp::create(cb, loc, FlatSymbolRefAttr::get(&context, "sequence"),
                  seqEntry->getArguments());
  }
  if (i == 0) {
    llvm::errs() << "aiecc: --reconfig-method: no inputs\n";
    return {};
  }

  if (failed(verify(*merged))) {
    llvm::errs() << "aiecc: --reconfig-method: synthesized module failed "
                    "verification\n";
    return {};
  }
  return writeMergedModule(*merged, workDir);
}

// The baked --reconfig-method flow needs EACH config's control-packet
// stream materialized side by side so every config gets its own baked
// `.ctrldata` entry. It does this by folding the N designs into ONE module
// whose host device carries N runtime sequences -- `seq_1..seq_N`, one per
// config -- each still issuing that design's `aiex.configure @config_k`,
// alongside the N (distinct) config devices. Routed through the normal
// --load-pdi-to-ctrl-pkt overlay pipeline, this materializes each seq_k's full
// control packets; every downstream artifact (per-seq run-seq DMA, per-seq
// ctrlpkt bin, N-entry overlay ELF) regenerates from that merged module.
//
// Each design's config device symbol must be distinct (the reconfiguration
// examples key them by the `--consts A` constant, e.g. @add_1..@add_N); a
// collision is rejected. Writes the merged module under the work dir and
// returns its path; returns "" on error (already diagnosed).
static std::string unionConfigDesigns(mlir::MLIRContext &context,
                                      llvm::ArrayRef<std::string> inputs,
                                      llvm::StringRef workDir,
                                      llvm::StringRef reconfigMethod) {
  using xilinx::AIE::DeviceOp;
  using xilinx::AIE::RuntimeSequenceOp;

  // The host device declares no tiles and carries the runtime sequence(s); the
  // config device declares the compute tiles (placed `aie.tile` or unplaced
  // `aie.logical_tile`; see deviceHasTiles).
  auto findDevices = [](mlir::ModuleOp m, DeviceOp &host, DeviceOp &config) {
    host = nullptr;
    config = nullptr;
    for (DeviceOp d : m.getOps<DeviceOp>()) {
      if (!deviceHasTiles(d)) {
        if (!host)
          host = d;
      } else if (!config)
        config = d;
    }
  };

  if (inputs.empty()) {
    llvm::errs() << "aiecc: --reconfig-method: no input designs to fold\n";
    return {};
  }

  mlir::OwningOpRef<mlir::ModuleOp> base =
      mlir::parseSourceFile<mlir::ModuleOp>(inputs.front(), &context);
  if (!base) {
    llvm::errs() << "aiecc: --reconfig-method: failed to parse "
                 << inputs.front() << "\n";
    return {};
  }
  DeviceOp baseHost, baseConfig;
  findDevices(base.get(), baseHost, baseConfig);
  if (!baseHost) {
    // No tile-less host device: if the base input is an idiomatic
    // single-device design (tile-bearing config, own runtime_sequence, no
    // host), synthesize the host scaffolding instead of erroring.
    if (baseConfig)
      return conformIdiomaticInputs(inputs, workDir, context, reconfigMethod);
    llvm::errs() << "aiecc: --reconfig-method: no host (tile-less) or "
                    "tile-bearing device in "
                 << inputs.front() << "\n";
    return {};
  }

  // Label the base host as the entry device (its presence, not the sequence
  // names, is what identifies it downstream).
  markEntrypointDevice(baseHost, reconfigMethod);

  // Keep the base design's host runtime sequence name as-authored here: it is
  // the entrypoint (dispatch) name the design chose (main:<name>). Entry names
  // are kept VERBATIM; colliding ones are a hard error (see the collision check
  // after the fold, below) -- the toolchain does not rename them. Require
  // exactly one host runtime sequence.
  unsigned nBaseSeq = 0;
  for (RuntimeSequenceOp s : baseHost.getOps<RuntimeSequenceOp>()) {
    (void)s;
    ++nBaseSeq;
  }
  if (nBaseSeq != 1) {
    llvm::errs() << "aiecc: --reconfig-method: " << inputs.front()
                 << " must have exactly one host runtime sequence (found "
                 << nBaseSeq << ")\n";
    return {};
  }

  std::set<std::string> configNames;
  if (baseConfig)
    configNames.insert(baseConfig.getSymName().str());

  mlir::OpBuilder modBuilder =
      mlir::OpBuilder::atBlockEnd(base.get().getBody());
  // The host DeviceOp body ends in an `aie.end` terminator; new sequences must
  // go before it, not after.
  mlir::OpBuilder seqBuilder(baseHost.getBody()->getTerminator());

  for (llvm::StringRef extra : inputs.drop_front()) {
    mlir::OwningOpRef<mlir::ModuleOp> mod =
        mlir::parseSourceFile<mlir::ModuleOp>(extra, &context);
    if (!mod) {
      llvm::errs() << "aiecc: --reconfig-method: failed to parse " << extra
                   << "\n";
      return {};
    }
    DeviceOp host, config;
    findDevices(mod.get(), host, config);
    if (!host || !config) {
      llvm::errs() << "aiecc: --reconfig-method: " << extra
                   << " must have a host device and a config (tile-bearing) "
                      "device\n";
      return {};
    }
    if (!configNames.insert(config.getSymName().str()).second) {
      llvm::errs() << "aiecc: --reconfig-method: duplicate config device '"
                   << config.getSymName()
                   << "' across designs; each config must be distinct\n";
      return {};
    }
    // Fold in this design's config device verbatim and its host runtime
    // sequence, KEEPING the design's chosen entrypoint name (retargeting
    // nothing: the cloned sequence still references its own config symbol,
    // which travels with the cloned device). Distinct designs must carry
    // distinct entrypoint names -- splitMultiConfigEntry loud-fails on a
    // duplicate.
    modBuilder.clone(*config.getOperation());
    unsigned added = 0;
    for (RuntimeSequenceOp s : host.getOps<RuntimeSequenceOp>()) {
      seqBuilder.clone(*s.getOperation());
      ++added;
    }
    if (added != 1) {
      llvm::errs() << "aiecc: --reconfig-method: " << extra
                   << " must have exactly one host runtime sequence (found "
                   << added << ")\n";
      return {};
    }
  }

  // Defense-in-depth: a base host device with no config (tile-bearing)
  // device anywhere in the inputs. The idiomatic-ingest misclassification
  // the sweep hit is caught upstream by deviceHasTiles; this guards the
  // residual host-only-no-config shape.
  if (configNames.empty()) {
    llvm::errs() << "aiecc: --reconfig-method: no config (tile-bearing) "
                    "device found in inputs\n";
    return {};
  }

  // Reject colliding entrypoint names. Each folded design's host runtime
  // sequence name IS its dispatch entrypoint (main:<name>), so two designs
  // sharing a name would produce an ambiguous entrypoint and the verifier would
  // reject the redefinition. The toolchain does NOT auto-rename: a design that
  // emits N configs from one template must give each config a distinct sequence
  // name (e.g. a distinct @iron.jit(name=) per design) so the entrypoints are
  // unambiguous.
  {
    llvm::StringSet<> seen;
    for (RuntimeSequenceOp s : baseHost.getOps<RuntimeSequenceOp>()) {
      if (!seen.insert(s.getSymName()).second) {
        llvm::errs()
            << "aiecc: --reconfig-method: multiple designs share the host "
               "runtime sequence name '"
            << s.getSymName()
            << "'; each design must name its runtime sequence uniquely (e.g. a "
               "distinct @iron.jit(name=) per design) so its dispatch "
               "entrypoint main:<name> is unambiguous. The toolchain no longer "
               "auto-renames colliding entries to config_1..N.\n";
        return {};
      }
    }
  }

  return writeMergedModule(base.get(), workDir);
}

// Prepare the entry device's N host runtime sequences (the ENTRYPOINTS) for
// dispatch, and for the init methods synthesize one shared `init` entry. The
// entrypoint device is the one carrying the aiex.entrypoint marker; every entry
// is processed in BLOCK ORDER (= chain order) and its sym_name is KEPT VERBATIM
// -- the dispatch name is the entrypoint's own name (main:<name>), chosen by
// the design (single-design: its runtime_sequence; multi-design: each lifted
// entrypoint). The toolchain never renames entries or special-cases any name
// form (no seq_<n>/config_<n> magic): a design that names its sequences
// config_1..N simply dispatches main:config_1..N.
//
// `expectInit` and `loadPdiNoInit` select which of three load_pdi shapes the
// entries must already be in (passed in, never inferred):
//   * expectInit == true (--reconfig-method=ctrlpkt or =write32): each
//     entrypoint carries exactly one `aiex.npu.load_pdi` (after the
//     --load-pdi-to-ctrl-pkt / reset-free expansion). Synthesizes a shared
//     `init` whose dispatch stands up the resident overlay / resets to @empty
//     and streams nothing. The `init` shape depends on `ctrlPkt`: for ctrlpkt
//     (ctrlPkt == true) it is a fresh sequence taking only the uniform trailing
//     ctrl-pkt-stream arg that AIECtrlPacketToDma appended to every entrypoint,
//     with a body of just the overlay's own cloned load_pdi (design-
//     independent); for write32 (ctrlPkt == false, no ctrl-pkt lowering ran, so
//     no uniform arg exists) it is a clone of the first entrypoint truncated
//     right after its first load_pdi.
//     `stripRearm` (fed by selfClear, now unconditional for ctrlpkt/write32)
//     then STRIPS every entrypoint's load_pdi -- the `init` keeps the sole
//     real standup and the in-band self-clear supplies each per-config reset;
//     otherwise each entrypoint keeps its own load_pdi re-arm so a separately-
//     dispatched config re-establishes the reset.
//   * expectInit == false, loadPdiNoInit == true (--reconfig-method=loadpdi):
//     each entrypoint STILL carries exactly one un-expanded full-PDI-reload
//     load_pdi (its own self reset). A load_pdi fully resets on every apply, so
//     a shared `init` is redundant -- synthesize NONE; entrypoints keep their
//     load_pdi.
//   * expectInit == false, loadPdiNoInit == false: no current reconfig method
//     selects this (write32 now always synthesizes the @empty init above). The
//     branch synthesizes NO `init` and leaves each entrypoint as-is; kept for a
//     load_pdi-free streamed-config caller that resets out of band.
// An unexpected load_pdi count for the mode fails loud instead of silently mis-
// splitting. Mutates `mod` in place; a no-op for a module with no marked entry
// device.
static mlir::LogicalResult
splitMultiConfigEntry(mlir::ModuleOp mod, bool stripRearm, bool expectInit,
                      bool loadPdiNoInit, bool ctrlPkt) {
  using xilinx::AIE::DeviceOp;
  using xilinx::AIE::RuntimeSequenceOp;
  using xilinx::AIEX::NpuLoadPdiOp;

  // The entry device carries the aiex.entrypoint marker (kEntrypointAttr) --
  // the SOLE discriminator. Split only the marked device's entrypoint
  // sequences; every other device is a config template. No marker -> nothing to
  // split.
  for (DeviceOp dev : mod.getOps<DeviceOp>()) {
    if (!dev->hasAttr(xilinx::aiecc::kEntrypointAttr))
      continue;

    // Entrypoint sequences in block order (= chain order). An init method
    // (expectInit: --reconfig-method=ctrlpkt or =write32) carries exactly one
    // load_pdi per entrypoint post-expansion, so select the load_pdi-carrying
    // config entrypoints; every other mode takes every non-empty entrypoint.
    // The per-mode load_pdi count is validated below.
    // Entrypoint NAMES ARE KEPT VERBATIM: the sym_name IS the dispatch name
    // (main:<name>), chosen by the design (a design may name its sequences
    // config_1..N itself); the toolchain never renames or format-special-cases
    // them (no seq_<n>/config_<n> magic).
    llvm::SmallVector<RuntimeSequenceOp> seqs;
    for (RuntimeSequenceOp s : dev.getOps<RuntimeSequenceOp>()) {
      if (s.getBody().empty())
        continue;
      if (expectInit) {
        if (llvm::any_of(s.getBody().front(), [](mlir::Operation &op) {
              return llvm::isa<NpuLoadPdiOp>(op);
            }))
          seqs.push_back(s);
      } else {
        seqs.push_back(s);
      }
    }
    if (seqs.empty())
      continue;

    // Loud-fail on duplicate entrypoint names within the marked device -- the
    // multi-design fold names each entrypoint with its design's name=, and two
    // designs sharing a name would silently collide on one dispatch kernel.
    {
      llvm::StringSet<> seen;
      for (RuntimeSequenceOp s : seqs)
        if (!seen.insert(s.getSymName()).second) {
          s.emitError() << "aiecc: --reconfig-method: duplicate entry sequence "
                           "name '"
                        << s.getSymName() << "' -- set name= per design";
          return mlir::failure();
        }
    }

    // Validate the per-mode load_pdi count on every selected entrypoint.
    for (RuntimeSequenceOp s : seqs) {
      unsigned nLoadPdi = 0;
      for (mlir::Operation &op : s.getBody().front())
        if (llvm::isa<NpuLoadPdiOp>(op))
          ++nLoadPdi;
      if (expectInit) {
        if (nLoadPdi != 1) {
          s.emitError() << "aiecc: --reconfig-method: expected exactly one "
                           "load_pdi in runtime sequence '"
                        << s.getSymName() << "', found " << nLoadPdi;
          return mlir::failure();
        }
      } else if (loadPdiNoInit) {
        if (nLoadPdi != 1) {
          s.emitError() << "aiecc: --reconfig-method: expected exactly one "
                           "load_pdi in runtime sequence '"
                        << s.getSymName()
                        << "' (no-init --reconfig-method=loadpdi), found "
                        << nLoadPdi;
          return mlir::failure();
        }
      } else if (nLoadPdi != 0) {
        // expectInit == false && loadPdiNoInit == false: no current reconfig
        // method routes here (see the load_pdi-shape table above) -- a
        // defensive guard for a future no-load_pdi mode.
        s.emitError() << "aiecc: --reconfig-method: expected zero "
                         "load_pdi in runtime sequence '"
                      << s.getSymName() << "', found " << nLoadPdi;
        return mlir::failure();
      }
    }

    if (expectInit) {
      RuntimeSequenceOp first = seqs.front();
      mlir::OpBuilder builder(first);
      builder.setInsertionPoint(first);
      if (ctrlPkt) {
        // ctrlpkt: synthesize ONE shared `init` from the overlay, design-
        // independent. Its signature is only the uniform trailing ctrl-pkt-
        // stream buffer that AIECtrlPacketToDma appended as the LAST block arg
        // of every entrypoint (read off the first entrypoint rather than
        // hardcoded, in case the type ever varies); its body is nothing but
        // that entrypoint's own load_pdi (validated exactly-one above), cloned
        // so its device_ref/id/expand_mode come along verbatim.
        mlir::Type ctrlArgType =
            first.getBody().getArguments().back().getType();
        NpuLoadPdiOp firstLoadPdi;
        for (mlir::Operation &op : first.getBody().front())
          if (auto loadPdi = llvm::dyn_cast<NpuLoadPdiOp>(op)) {
            firstLoadPdi = loadPdi;
            break;
          }
        assert(
            firstLoadPdi &&
            "expectInit validated exactly one load_pdi per entrypoint above");

        auto initSeq = RuntimeSequenceOp::create(
            builder, first.getLoc(), mlir::StringAttr{}, mlir::BoolAttr{});
        initSeq.setSymName("init");
        initSeq.getBody().push_back(new mlir::Block);
        initSeq.getBody().addArgument(ctrlArgType, first.getLoc());
        builder.setInsertionPointToStart(&initSeq.getBody().front());
        builder.clone(*firstLoadPdi);
      } else {
        // write32: no ctrl-pkt lowering ran, so there is
        // no uniform trailing arg to key off of. Clone the first entrypoint and
        // truncate right after its first load_pdi -- the shared `init` keeps
        // the sole @empty reset standup and streams nothing.
        auto initSeq = mlir::cast<RuntimeSequenceOp>(builder.clone(*first));
        initSeq.setSymName("init");
        mlir::Block &b = initSeq.getBody().front();
        bool afterLoadPdi = false;
        llvm::SmallVector<mlir::Operation *> toErase;
        for (mlir::Operation &op : b) {
          if (afterLoadPdi)
            toErase.push_back(&op);
          else if (llvm::isa<NpuLoadPdiOp>(op))
            afterLoadPdi = true;
        }
        for (mlir::Operation *op : llvm::reverse(toErase))
          op->erase();
      }
    }
    // expectInit == false: no shared standup entry is synthesized, whether or
    // not the configs individually carry a load_pdi. Only loadpdi reaches here
    // now (write32 is expectInit == true): its `loadPdiNoInit` configs keep
    // their own self-reset load_pdi below. Either way, no shared `init`
    // dispatches -- just config_1..config_N.

    // `stripRearm` (fed by selfClear, now unconditional for ctrlpkt/write32)
    // STRIPS every per-entrypoint load_pdi re-arm: the @init synthesized above
    // keeps the sole real standup load_pdi and the entrypoints stay
    // load_pdi-free (the in-band self-clear supplies each per-config reset).
    // Otherwise (default load_pdi re-arm) each entrypoint keeps its own
    // load_pdi so a separately-dispatched config re-arms via a PDI reload.
    // Names are untouched
    // -- the dispatch name is the entrypoint's own sym_name (main:<name>).
    if (stripRearm)
      for (RuntimeSequenceOp s : seqs) {
        mlir::Block &b = s.getBody().front();
        for (mlir::Operation &op : llvm::make_early_inc_range(b))
          if (llvm::isa<NpuLoadPdiOp>(op))
            op.erase();
      }
  }
  return mlir::success();
}

} // namespace

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

  auto &withAddresses =
      input
          .map<ModRef>("placed.mlir",
                       PassPipeline{getPlacementPipeline(
                           &context, coresPerCol.getValue(),
                           placerType.getValue(), saSeed.getValue())})
          .map<ModRef>("traced.mlir", PassPipeline{getTracePipeline(&context)})
          .map<ModRef>(
              "input_with_addresses.mlir",
              PassPipeline{&context,
                           [scheme = allocScheme.getValue(),
                            dyn = dynamicObjFifos.getValue(),
                            pkt = packetSwObjFifos.getValue(),
                            // --reconfig-method=write32 does not feed into
                            // ctrl or ldpdi: write32 is overlay-free (no
                            // column-control-overlay pass, no
                            // @ctrl_pkt_overlay, no reserve-control-ids /
                            // auto-packetize). Only ctrlPktOverlay and
                            // loadPdiToCtrlPkt need the ctrl-overlay setup.
                            ctrl = ctrlPktOverlay.getValue() ||
                                   loadPdiToCtrlPkt.getValue(),
                            ldpdi = loadPdiToCtrlPkt.getValue(),
                            bf16 = bf16Emulation.getValue(),
                            skipVerify = skipObjectFifoVerify.getValue(),
                            // Resolved by resolveOptions() from the default-on
                            // --ctrlpkt-auto-packetize bool (negated by
                            // --ctrlpkt-auto-packetize=false).
                            autoPkt = doAutoPacketizeControlIngress,
                            xtileDma = dmaFenceSharedMem.getValue()](
                               mlir::MLIRContext *ctx, mlir::ModuleOp mod) {
                             return getInputWithAddressesPipeline(
                                 ctx, mod, scheme, dyn, pkt, ctrl, bf16, ldpdi,
                                 skipVerify, autoPkt, xtileDma);
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
      "input_physical.mlir",
      PassPipeline{getRoutingPipeline(&context, doReconfigPinControl,
                                      doReconfigPinControlDesignAware)});

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
            core.mod(), core.op->getParentOfType<DeviceOp>().getSymName());
      });

  // Per-core .o node. Two strategies selectable, differing only in how many
  // times the lowering pipeline runs:
  //   * unified: lower once per device, then carve that module into one module
  //     per core;
  //   * per-core: lower once per core, each run on a clone of the whole design.
  // Either way every core compiles its own object, so the object stage keeps
  // its per-core parallelism.

  // Unified strategy
  auto &physicalPerDevice = splitPerDevice(
      physical, "perDeviceCompile_{0}.mlir", "perDeviceCompileMatching");
  auto &perDeviceArches = physicalPerDevice.map<std::string>(
      "perDeviceArches_{0}.txt", [](const OpInModule<DeviceOp> &dev) {
        return detectAIETarget(dev.mod(), DeviceOp(dev.op).getSymName());
      });
  // Lower once per device, then carve out one module per core. Keyed like
  // `perCore`, so the per-core arches and link files below apply unchanged --
  // except that `perCore` drops cores that already carry an `elf_file`, so
  // filter the carved set to match or the object subgraph joins on a key its
  // other inputs do not have.
  auto &unifiedPerCoreLowered = physicalPerDevice.split<ModRef>(
      "lowered_{0}.mlir", [](const Item<OpInModule<DeviceOp>> &dev) {
        // Same predicate as the `perCoreCompile` filter above: a core with an
        // `elf_file` is used verbatim, so it must not appear here either.
        return splitLoweredCores(
            dev, [](CoreOp c) { return !c.getElfFileAttr() || xbridge; });
      });

  // Per-core strategy
  auto &perCoreLowered = perCore.map<ModRef>(
      "lowered_{0}.mlir",
      [](const Item<OpInModule<CoreOp>> &item, Item<ModRef> &out) {
        CoreOp core = item.get().op;
        auto tile = mlir::cast<TileOp>(core.getTile().getDefiningOp());
        return loweringPipeline(item.get().mod(),
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

  EdgeWithTypedOutput<Directory> &unifiedObjects =
      buildObjectSubgraph(unifiedPerCoreLowered, perCoreArches,
                          perCoreIRLinkFiles, "objects_{0}.o");

  EdgeWithTypedOutput<Directory> &objects =
      doUnified ? unifiedObjects : perCoreObjects;

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
            absolutizeLinkFiles(item.get().mod(), tile.getCol(), tile.getRow(),
                                inputFile, workDirStr);
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
            item.get().mod(), os, tile.getCol(), tile.getRow(),
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
  //   * --reconfig-method=write32 expands each config to direct writes (via the
  //     same expand-load-pdi machinery, reset-free mode), which reads the
  //     compiled cores just like --expand-load-pdis.
  const ReconfigMethod method = parseReconfigMethod(reconfigMethod);
  bool npuTransactionsNeedCoresLowered =
      expandLoadPdis.getValue() || generateTxn || loadPdiToCtrlPkt.getValue() ||
      method == ReconfigMethod::Write32;
  EdgeWithTypedOutput<ModRef> &npuLoweringInput =
      npuTransactionsNeedCoresLowered
          ? static_cast<EdgeWithTypedOutput<ModRef> &>(physicalWithElfs)
          : static_cast<EdgeWithTypedOutput<ModRef> &>(physical);
  // IRON's fused decode arrives as ONE `aie.runtime_sequence` holding N
  // `aiex.configure` ops on the `aiex.entrypoint`-marked host (llama: 322
  // configures over 19 config devices). splitMultiConfigEntry (below) and the
  // aie-expand-load-pdi self-clear (getExpandLoadPdiPipeline) both require
  // exactly one configure/load_pdi per runtime-sequence block, so explode the
  // monolith into N one-configure sequences HERE -- before
  // getMaterializeRuntimeSeqPipeline rewrites `aiex.configure` into `aie.run`
  // (after which nothing is left to split). Gated on the --reconfig-method fold
  // (generateMultiConfigElf); the pass itself keys on the entrypoint marker and
  // is a genuine no-op on already-split single-configure sequences (conformed
  // multi-config inputs), so those fold inputs pass through
  // unchanged.
  EdgeWithTypedOutput<ModRef> &npuConfigureSplit =
      generateMultiConfigElf
          ? static_cast<EdgeWithTypedOutput<ModRef> &>(
                npuLoweringInput.map<ModRef>(
                    "npu_split_configure.mlir",
                    PassPipeline{getSplitConfigureEntriesPipeline(&context)}))
          : npuLoweringInput;
  // NPU instruction sequence lowering. The default and --load-pdi-to-ctrl-pkt
  // flows share the materialize + expand prefix and diverge at DMA lowering.
  EdgeWithTypedOutput<ModRef> &npuMaterialized =
      noMaterialize.getValue()
          ? npuConfigureSplit
          : static_cast<EdgeWithTypedOutput<ModRef> &>(
                npuConfigureSplit.map<ModRef>(
                    "npu_materialized.mlir",
                    PassPipeline{getMaterializeRuntimeSeqPipeline(&context)}));

  // For --load-pdi-to-ctrl-pkt this edge holds the control-packet ops before
  // DMA lowering: the extraction point for the control-packet binary.
  // --reconfig-method=write32 instead expands to direct writes with no resident
  // overlay (reset-free mode): it takes the expand branch but NOT the
  // ctrl-packet DMA lowering below.
  bool ctrlPkt = loadPdiToCtrlPkt.getValue();
  bool resetFree = method == ReconfigMethod::Write32;
  // Self-clear teardown is mandatory for the resident-overlay methods; "off"
  // is incorrect behavior (the overlay accrues switch/DMA state across
  // configs), so it is applied unconditionally rather than gated behind a flag.
  bool selfClear = (ctrlPkt || resetFree);
  // write32 ALWAYS synthesizes the shared `main:init` (@empty reset); the host
  // dispatches it or not (dispatch => explicit reset; skip => firmware
  // teardown reset). So write32 is always "with reset" at emit time; ctrlpkt
  // stands up its own init; loadpdi self-resets per config.
  bool withReset = resetFree;
  // Column-parallel delivery is on by default but only engages under ctrlpkt
  // (the ctrl-packet-to-dma pass runs only there); a silent no-op otherwise.
  bool parallelColumnsFlag =
      parallelColumns.getValue() && method == ReconfigMethod::Ctrlpkt;
  EdgeWithTypedOutput<ModRef> &npuExpanded =
      (expandLoadPdis.getValue() || ctrlPkt || resetFree)
          ? static_cast<EdgeWithTypedOutput<ModRef> &>(
                npuMaterialized.map<ModRef>(
                    "npu_expanded.mlir",
                    PassPipeline{&context,
                                 [ctrlPkt, resetFree, selfClear, withReset,
                                  parallelColumnsFlag](mlir::MLIRContext *ctx,
                                                       mlir::ModuleOp) {
                                   return getExpandLoadPdiPipeline(
                                       ctx, ctrlPkt, resetFree, selfClear,
                                       withReset, parallelColumnsFlag);
                                 }}))
          : npuMaterialized;

  // The default tail unrolls runtime-sequence loops and pools dynamic BDs; the
  // ctrl-packet sequence is straight-line and only needs the per-device tail,
  // after its control packets are lowered to DMA.
  EdgeWithTypedOutput<ModRef> &npuDmaLowered =
      ctrlPkt ? npuExpanded
                    .map<ModRef>("ctrlpkt_to_dma.mlir",
                                 PassPipeline{getCtrlPktToDmaPipeline(
                                     &context, parallelColumnsFlag)})
                    .map<ModRef>(
                        "ctrlpkt_npu_lowered.mlir",
                        PassPipeline{getPerDeviceDmaLoweringPipeline(&context)})
              : npuExpanded.map<ModRef>(
                    "npu_dma_lowered.mlir",
                    PassPipeline{getNpuDmaLoweringPipeline(&context)});

  // The --reconfig-method fold folds N per-config designs into
  // seq_1..seq_N (unionConfigDesigns) and splits them here into
  // init + config_1..config_N (or just config_1..config_N, for the
  // no-init methods), so every downstream consumer of `npuLowered` sees the
  // split uniformly. Non-fold flows leave `npuLowered` untouched. This call
  // site is shared by every method (generateMultiConfigElf is true for
  // loadpdi, write32, and ctrlpkt): expectInit is true for ctrlpkt (which
  // stands up a resident overlay that must be shared) and for write32 (which
  // always synthesizes the @empty-reset `main:init`; withReset == resetFree).
  // It is false for loadpdi: a load_pdi is a full reset every time it is
  // applied, so loadpdi's per-config self-reset load_pdi already makes a shared
  // `init` standup redundant (see splitMultiConfigEntry's loadPdiNoInit case).
  bool splitMultiConfig = generateMultiConfigElf;
  bool expectInit = method == ReconfigMethod::Ctrlpkt || withReset;
  bool loadPdiNoInit = method == ReconfigMethod::Loadpdi;
  auto &npuLowered = npuDmaLowered.map<ModRef>(
      "npu_lowered.mlir",
      [splitMultiConfig, selfClear, expectInit, loadPdiNoInit, ctrlPkt](
          const Item<ModRef> &item, Item<ModRef> &out) -> mlir::LogicalResult {
        ModRef clone = item.get().get().clone();
        assignDevicePdiIds(*clone);
        assignLoadPdiIds(*clone);
        // On the self-clear arms strip each config's load_pdi re-arm: the
        // synthesized `init` keeps the sole standup load_pdi and the per-config
        // switch self-clear epilogue supplies the reset instead of a PDI
        // reload.
        if (splitMultiConfig &&
            mlir::failed(splitMultiConfigEntry(*clone,
                                               /*stripRearm=*/selfClear,
                                               /*expectInit=*/expectInit,
                                               /*loadPdiNoInit=*/
                                               loadPdiNoInit,
                                               /*ctrlPkt=*/ctrlPkt)))
          return mlir::failure();
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
  //   * --reconfig-method=write32 runs the same expansion (reset-free mode) and
  //     splits into init + configs, so the static branch must root on
  //     `npuLowered` to observe the synthesized/overlay devices too.
  EdgeWithTypedOutput<ModRef> &staticInput =
      (expandLoadPdis.getValue() || loadPdiToCtrlPkt.getValue() ||
       method == ReconfigMethod::Write32)
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
                item.get().mod(), cdoDir, d.getSymName(), false, false, false,
                false, false, /*enableCores=*/true)))
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
        ModRef clone = item.get().mod().clone();
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
        ModRef clone = item.get().mod().clone();
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
                        item.get().mod(), words, d.getSymName(),
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
  // Control-packet overlay ELF (--reconfig-method)
  //--------------------------------------------------------------------------//
  // Reuses the full-ELF assembly machinery, but with an overlay-mode config
  // (makeFullElfConfigJson `overlayOnly=true`) that keeps only the host-invoked
  // runtime-sequence device + its partition/reset (@ctrl_pkt_overlay) PDIs and
  // drops the streamed-config compute devices (and their PDIs). The overlay
  // BAKES each config: it threads the SAME per-sequence control-packet and
  // patch-info edges the embedded full-ELF branch uses (fullElfCtrlpkt /
  // fullElfPatchInfo) into the overlay config, so each `config_k` instance
  // gets its own `ctrl_packet_file` + `patch_info_file` (an internal
  // `control-packet` `.ctrldata` relocation). Entrypoint names are kept
  // VERBATIM through the split (`config_k` both as the edge key
  // `<device>_config_k` and as the overlay device instance), so the two sides
  // already line up. rekeyToConfigs is a compatibility no-op for that verbatim
  // naming: it only rewrites a legacy `_seq_`-infixed key (no current path
  // produces one, since the toolchain never renames entries), and is retained
  // as a harmless bridge rather than removed.
  auto rekeyToConfigs = [](llvm::StringRef key) -> std::string {
    size_t p = key.find("_seq_");
    if (p == llvm::StringRef::npos)
      return key.str();
    return (key.take_front(p) + "_config_" + key.drop_front(p + 5)).str();
  };
  auto &ctrlPktOverlayConfig =
      bundle(npuLoweredPerDevice.out, pdi.out, npuInstsFullElf.out,
             fullElfCtrlpkt.out, fullElfPatchInfo.out)
          .join<llvm::json::Value>(
              "ctrl_pkt_overlay_config.json",
              [rekeyToConfigs](
                  const Node<OpInModule<DeviceOp>> &devices,
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
                  ctrlPktPaths[rekeyToConfigs(item.key)] =
                      absolutePath(item.asFile());
                for (const auto &item : patchInfos.items)
                  patchInfoPaths[rekeyToConfigs(item.key)] =
                      absolutePath(item.asFile());
                out.value = makeFullElfConfigJson(devices, pdiPaths, instsPaths,
                                                  ctrlPktPaths, patchInfoPaths,
                                                  /*overlayOnly=*/true);
                return mlir::success();
              });

  auto &multiConfigElf = ctrlPktOverlayConfig.map<File>(
      fullElfName.getValue(), ShellCommand{"aiebu-asm"}
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
        return xilinx::AIE::AIETranslateToXAIEV2(item.get().mod(), os,
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
      generateFullElf || generateMultiConfigElf || wantAiesim ||
      doCompileHost || !getOutputs.empty() || !cutOutputs.empty();
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
  // --reconfig-method requires --get-full-elf and folds the N configs into ONE
  // combined ELF (multiConfigElf, written to --full-elf-name). In that flow the
  // plain single-config fullElf is redundant, so emit only the folded ELF: the
  // combined artifact IS the full ELF.
  if (generateFullElf && !generateMultiConfigElf)
    outputs.push_back(&fullElf);
  // Folded multi-config ELF (--reconfig-method): the combined ELF written to
  // --full-elf-name. For ctrlpkt the baked overlay writes each config's
  // control-packet stream straight into the ELF's `.ctrldata`, so the host
  // never binds a separate config buffer.
  if (generateMultiConfigElf)
    outputs.push_back(&multiConfigElf);
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
  xilinx::AIE::registerAIEObjectFifoPipeline();
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

  // --reconfig-method is the delivery-method selector: it folds N single-config
  // designs into one combined ELF (--full-elf-name) and drives the internal
  // lowering flags directly. Every method sets generateMultiConfigElf (the
  // fold + split); the delivery differs:
  //   * write32 -- out-of-band direct writes, no resident overlay (reset-free
  //     by default). Drives the expand-load-pdi machinery in reset-free mode
  //     (see resetFree in buildMainGraph). Does NOT set loadPdiToCtrlPkt.
  //   * ctrlpkt -- in-band control packets through a resident
  //   @ctrl_pkt_overlay.
  //     Sets loadPdiToCtrlPkt.
  //   * loadpdi -- each split `config_k` keeps its own un-expanded
  //     `load_pdi @config_k`; no expansion pipeline runs (a true
  //     full-PDI-reload baseline). Sets neither.
  // --reconfig-method REQUIRES --get-full-elf: the fold engages only in the
  // full-ELF flow, and its output IS the full ELF (written to --full-elf-name).
  if (!reconfigMethod.empty()) {
    const ReconfigMethod method = parseReconfigMethod(reconfigMethod);
    if (method == ReconfigMethod::None) {
      llvm::errs()
          << "aiecc: --reconfig-method must be loadpdi|write32|ctrlpkt\n";
      return 1;
    }
    if (!generateFullElf) {
      llvm::errs() << "aiecc: --reconfig-method requires --get-full-elf\n";
      return 1;
    }
    generateMultiConfigElf = true;
    if (method == ReconfigMethod::Ctrlpkt)
      loadPdiToCtrlPkt = true;
  }

  // Exactly one input MLIR file may appear before the `--` separator; host
  // source files and host-compiler flags belong after it. The exception is
  // --reconfig-method, which ingests several `design*.mlir` (folding them
  // into an N-sequence overlay module); it emits an ELF only and never uses the
  // `--` host-arg tail this guard disambiguates.
  if (positionalArgs.size() > 1 && !generateMultiConfigElf) {
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

  // --reconfig-method=write32 is its own config-delivery strategy
  // (overlay-free, out-of-band direct writes via reset-free mode). It drives
  // the expand-load-pdi machinery itself, so it cannot be combined with the
  // other two expansion strategies.
  if (parseReconfigMethod(reconfigMethod) == ReconfigMethod::Write32 &&
      (expandLoadPdis || loadPdiToCtrlPkt)) {
    llvm::errs()
        << "aiecc: --reconfig-method=write32 is mutually exclusive with "
           "--expand-load-pdis and --load-pdi-to-ctrl-pkt\n";
    return 1;
  }

  // Reset is a runtime dispatch decision (write32 always emits main:init), so
  // there is no --reconfig-with-reset to gate. --ctrlpkt-parallel-columns is
  // on by default and guarded to the ctrlpkt method at its read (a silent
  // no-op elsewhere). --ctrlpkt-pinned-overlay / --ctrlpkt-auto-packetize are
  // resolved in resolveOptions() and self-gate to no-ops without an overlay,
  // so none of them needs a method-mismatch guard here.

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

  // --reconfig-method may take several `design*.mlir` and route the whole
  // build as one merged module (each design's entry sequence is kept as its
  // own verbatim-named entrypoint so the multi-entry split can emit one config
  // per design). Done here (not in buildMainGraph) so parse/merge failure can
  // abort cleanly, and before the SourceMgr buffer below so diagnostics point
  // at the merged input. Runs for any N (N=1 included).
  if (generateMultiConfigElf) {
    std::vector<std::string> inputs(positionalArgs.begin(),
                                    positionalArgs.end());
    std::string merged =
        unionConfigDesigns(context, inputs, getWorkDir(), reconfigMethod);
    if (merged.empty())
      return 1;
    cli::inputFilenameOverride = merged;
    // The merged module is also written to <workDir>/config_union.mlir
    // (see unionConfigDesigns), so lit can inspect the fold under
    // --dump-intermediates --tmpdir without a bespoke emit-and-exit flag.
  }

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
  if (mlir::failed(engine.run(g, runOutputs, satisfied,
                              DeserializeContext{&context}, cutEdges))) {
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
