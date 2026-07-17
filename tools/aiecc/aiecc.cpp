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
#include "aiecc_aiesim.h"

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

namespace {

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
EdgeWithTypedOutput<Directory> &
buildObjectSubgraph(EdgeWithTypedOutput<ModRef> &lowered,
                    EdgeWithTypedOutput<std::string> &arches,
                    std::string objName) {
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
      bundle(chessCompat.out, arches.out)
          .map<File>("chesslinked_{0}.ll",
                     [aietoolsRoot,
                      installDir](const Item<std::string> &ir,
                                  const Item<std::string> &archItem,
                                  Item<File> &out) -> mlir::LogicalResult {
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

  // Peano path: downgrade -> opt -> llc.
  unsigned optPassLevel = std::min<unsigned>(optLevel, 1u);
  ShellCommand optCmd{"opt"};
  if (optLevel >= 3)
    optCmd.arg("-disable-loop-idiom-memset");
  optCmd.arg("--passes=default<O" + std::to_string(optPassLevel) + ">")
      .arg("-inline-threshold=10")
      .arg("-S")
      .input()
      .output("-o");
  auto &opted =
      llvmIR.map<std::string>("peano-compat_{0}.ll", downgradeIRForPeano)
          .map<File>("opted_{0}.ll", optCmd)
          .threadSafe();
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

// Host-compilation subgraph. Emits the per-device `aie_inc.cpp` array
// configuration source and compiles the user's host sources against it.
EdgeWithTypedOutput<File> &
buildHostExeSubgraph(EdgeWithTypedOutput<OpInModule<DeviceOp>> &perDevice,
                     EdgeWithTypedOutput<std::string> &arches) {
  auto &aieInc = perDevice.map<std::string>(
      "aie_inc.cpp",
      [](const Item<OpInModule<DeviceOp>> &item,
         Item<std::string> &out) -> mlir::LogicalResult {
        DeviceOp d = item.get().op;
        llvm::raw_string_ostream os(out.value.emplace());
        return xilinx::AIE::AIETranslateToXAIEV2(item.get().module.get(), os,
                                                 d.getSymName());
      });

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

// Translate each runtime sequence into its NPU program: one NpuProgram item
// (the transaction instruction binary + its source-location map) per sequence,
// keyed "<device>_<sequence>".
//
// `foldDDRAddrOffset` selects how host-buffer addresses are encoded in the TXN:
//   * true  -- the DDR-aperture offset is folded into the transaction, as the
//              xclbin + instruction-buffer runtime expects;
//   * false -- the offset is left out, as the full-ELF runtime (xrt.ext.kernel)
//              assigns NPU-space device addresses to every host buffer itself.
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

} // namespace

//===----------------------------------------------------------------------===//
// Main compilation graph
//===----------------------------------------------------------------------===//

// Assemble the full compilation artifact graph into `g` and return the list of
// requested output edges. Edges named via `--cut` are appended to `cutEdges`
// (and built) so a `--checkpoint` can capture them as its cut points.
std::vector<EdgeBase *> buildMainGraph(mlir::MLIRContext &context, Graph &g,
                                       std::vector<EdgeBase *> &cutEdges) {

  //--------------------------------------------------------------------------//
  // Helpers
  //--------------------------------------------------------------------------//

  using xilinx::AIE::CoreOp;
  using xilinx::AIE::DeviceOp;
  using xilinx::AIE::RuntimeSequenceOp;
  using xilinx::AIE::TileOp;
  using ModRef = mlir::OwningOpRef<mlir::ModuleOp>;

  std::string devFilter = deviceName.getValue();
  std::string inputFile = getInputFilename();

  // Chess/xbridge toolchain locations
  std::string aietoolsRoot = discoverAietoolsDir(aietoolsDir.getValue());
  std::string installDir = getInstallDir();
  std::string workDirStr = getWorkDir();
  // xchesscc_wrapper scratch dir
  std::string chessWork = workDirStr + "/chess_work";
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
                            ctrl = ctrlPktOverlay.getValue(),
                            bf16 = bf16Emulation.getValue()](
                               mlir::MLIRContext *ctx, mlir::ModuleOp mod) {
                             return getInputWithAddressesPipeline(
                                 ctx, mod, scheme, dyn, pkt, ctrl, bf16);
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
  auto &unifiedObjects = buildObjectSubgraph(unifiedLowered, perDeviceArches,
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
  EdgeWithTypedOutput<Directory> &perCoreObjects =
      buildObjectSubgraph(perCoreLowered, perCoreArches, "objects_{0}.o");

  EdgeWithTypedOutput<Directory> &objects =
      doUnified ? unifiedCoreObjects : perCoreObjects;

  // ld scripts (with link_files absolutized so INPUT() is cwd-invariant).
  auto &ldScripts = perCore.map<std::string>(
      "ldScripts_{0}.ld.script",
      [inputFile, workDirStr](const Item<OpInModule<CoreOp>> &item,
                              Item<std::string> &out) -> mlir::LogicalResult {
        CoreOp op = item.get().op;
        auto tile = mlir::cast<TileOp>(op.getTile().getDefiningOp());
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
  bool npuTransactionsNeedCoresLowered =
      expandLoadPdis.getValue() || generateTxn.getValue();
  EdgeWithTypedOutput<ModRef> &npuLoweringInput =
      npuTransactionsNeedCoresLowered
          ? static_cast<EdgeWithTypedOutput<ModRef> &>(physicalWithElfs)
          : static_cast<EdgeWithTypedOutput<ModRef> &>(physical);
  auto &npuLowered = npuLoweringInput.map<ModRef>(
      "npu_lowered.mlir",
      [expand = expandLoadPdis.getValue(),
       materialize = !noMaterialize.getValue()](
          const Item<ModRef> &item, Item<ModRef> &out) -> mlir::LogicalResult {
        ModRef clone = item.get().get().clone();
        if (mlir::failed(
                getNpuLoweringPipeline(clone->getContext(), materialize)
                    ->run(*clone)))
          return mlir::failure();
        if (expand)
          if (mlir::failed(
                  getExpandLoadPdiPipeline(clone->getContext())->run(*clone)))
            return mlir::failure();
        assignDevicePdiIds(*clone);
        assignLoadPdiIds(*clone);
        out.value = std::move(clone);
        return mlir::success();
      });

  // Root of the static configuration branch; contains compiled cores, etc., to
  // produce xclbins, or feed into the full ELF. Usually, this is completely
  // independent from the NPU runtime sequence compilation; however, the
  // --expand-load-pdis pass generates new empty_0/1 devices, for which we must
  // also generate PDIs, which is the only reason for the selection here
  EdgeWithTypedOutput<ModRef> &staticInput =
      expandLoadPdis.getValue()
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
          llvm::errs()
              << "aiecc: --aie-generate-cdo/-pdi/-xclbin require an NPU "
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
                              return assemblePdi(bifItem, out);
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
        // The control-packet DMA sequence only ever pairs with the xclbin /
        // instruction-buffer runtime (full ELF is rejected up front), which
        // folds the DDR-aperture offset into the TXN.
        return xilinx::AIE::AIETranslateNpuToBinary(
            clone.get(), words, item.key, "",
            /*locmap=*/nullptr, /*foldDDRAddrOffset=*/true);
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

  // When --aie-generate-elf is also set, the combined control-packet ELF is the
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
                                          llvm::StringRef(patch), out);
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
  std::string inXclbin = xclbinInput.getValue();
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
  // and the locmap). Two variants are built from the same per-sequence input,
  // differing only in whether the DDR-aperture offset is folded into the TXN:
  //   * npuProgram (folded) drives the xclbin / instruction-buffer artifacts
  //     (instElf, insts.bin, locmap), whose runtime needs the offset folded in;
  //   * npuProgramFullElf (not folded, built just below the full-ELF section)
  //     drives the combined full ELF, whose runtime assigns host-buffer
  //     addresses itself.
  auto &npuProgram = buildNpuProgramSubgraph(perSeq, "npu_program_{0}.bin",
                                             /*foldDDRAddrOffset=*/true);

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
  auto &instElf = npuInsts.map<File>(
      elfName.getValue(),
      [](const Item<std::vector<char>> &item,
         Item<File> &out) -> mlir::LogicalResult {
        return assembleElf(item.get(), /*buffer2=*/{}, /*patchJson=*/{}, out);
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
  // The full ELF consumes its own NPU program variant, translated WITHOUT
  // folding the DDR-aperture offset (foldDDRAddrOffset=false): the full-ELF
  // runtime assigns host-buffer addresses itself, so the offset must not be
  // baked into the TXN.
  auto &npuProgramFullElf = buildNpuProgramSubgraph(
      perSeq, "npu_program_full_elf_{0}.bin", /*foldDDRAddrOffset=*/false);
  auto &npuInstsFullElf = npuProgramFullElf.map<std::vector<char>>(
      "npu_insts_full_elf_{0}.bin",
      [](const NpuProgram &p) { return p.insts; });

  // Combined ELF: all PDIs + NPU insts bundled into one aie2_config ELF.
  auto &fullElfConfig =
      bundle(npuLoweredPerDevice.out, pdi.out, npuInstsFullElf.out)
          .join<llvm::json::Value>(
              "full_elf_config.json",
              [](const Node<OpInModule<DeviceOp>> &devices,
                 const Node<File> &pdis,
                 const Node<std::vector<char>> &instsBins,
                 Item<llvm::json::Value> &out) -> mlir::LogicalResult {
                llvm::StringMap<std::string> pdiPaths, instsPaths;
                for (const auto &item : pdis.items)
                  pdiPaths[item.key] = absolutePath(item.asFile());
                for (const auto &item : instsBins.items)
                  instsPaths[item.key] = absolutePath(item.asFile());
                out.value =
                    makeFullElfConfigJson(devices, pdiPaths, instsPaths);
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
  auto &hostExe = buildHostExeSubgraph(staticPerDevice, perDeviceArches);

  //--------------------------------------------------------------------------//
  // AIE simulator Work folder
  //--------------------------------------------------------------------------//
  // Per device, emit the `sim/` work folder (graph.xpe, shim solution, scsim
  // config, flows), build ps.so, and the `aiesim.sh` launcher.
  //
  // TODO: aiecc_aiesim.cpp is a non-declarative blackbox — it shells out to
  // aie-translate/aie-opt/clang++ itself and writes a fixed directory layout
  // (Work folder + ps.so + aiesim.sh) straight to disk, bypassing the Item
  // abstraction and the graph. It should be refactored into proper graph edges
  // (one Item per generated artifact). It hangs off `staticPerDevice`, so the
  // engine still schedules it after the core ELFs the simulator reads are
  // built.
  xilinx::aiecc::AiesimConfig aiesimCfg;
  aiesimCfg.enabled = wantAiesim;
  aiesimCfg.compileHost = doCompileHost;
  aiesimCfg.verbose = verbose;
  aiesimCfg.dryRun = dryRun;
  aiesimCfg.hostTarget = hostTarget.getValue();
  aiesimCfg.aietoolsPath = aietoolsRoot;
  aiesimCfg.installPath = installDir;
  for (const auto &dir : hostIncludeDirs)
    aiesimCfg.hostArgs.push_back("-I" + dir);
  for (const auto &dir : hostLibDirs)
    aiesimCfg.hostArgs.push_back("-L" + dir);
  for (const auto &lib : hostLibs)
    aiesimCfg.hostArgs.push_back("-l" + lib);
  for (const auto &a : hostPassthroughArgs)
    aiesimCfg.hostArgs.push_back(a);

  auto &aiesimWork = staticPerDevice.map<File>(
      "aiesim_{0}.stamp",
      [aiesimCfg, workDirStr](const Item<OpInModule<DeviceOp>> &item,
                              Item<File> &out) -> mlir::LogicalResult {
        DeviceOp d = item.get().op;
        mlir::ModuleOp mod = item.get().module.get();
        std::string devName = d.getSymName().str();
        std::string aieTarget = detectAIETarget(mod, d.getSymName());
        if (mlir::failed(xilinx::aiecc::generateAieIncCpp(mod, workDirStr,
                                                          devName, aiesimCfg)))
          return mlir::failure();
        if (mlir::failed(xilinx::aiecc::generateAiesim(mod, workDirStr, devName,
                                                       aieTarget, aiesimCfg)))
          return mlir::failure();
        out.value = File{};
        return mlir::success();
      });
  // The aiesim module performs its own disk writes (Work folder + ps.so +
  // aiesim.sh), so there is no single File for the engine to materialize.
  aiesimWork.producesFiles = false;

  //--------------------------------------------------------------------------//
  // Output selection
  //--------------------------------------------------------------------------//
  if (generateScratchpadParams)
    outputs.push_back(&paramsFile);

  // Core-ELF output: emit the per-core ELFs when --aie-generate-core-elfs is
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
  if (generateCtrlpkt) {
    outputs.push_back(&ctrlpkt);
    outputs.push_back(&ctrlpktDmaSeq);
    outputs.push_back(&ctrlpktElf);
  }
  if (generateXclbin)
    outputs.push_back(&xclbin);
  if (generateFullElf)
    outputs.push_back(&fullElf);
  // AIE simulator Work folder: only when explicitly requested.
  if (wantAiesim)
    outputs.push_back(&aiesimWork);
  // Host executable: only when explicitly requested and host sources exist.
  if (doCompileHost) {
    if (!hasHostSourceFiles())
      llvm::errs() << "aiecc: --compile-host given but no host source files "
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

    auto resolveNames = [&](llvm::ArrayRef<std::string> names,
                            llvm::StringRef flag) -> std::vector<EdgeBase *> {
      std::vector<EdgeBase *> resolved;
      for (const std::string &want : names) {
        llvm::Expected<EdgeBase *> chosen = resolveLiveEdge(g, want, live);
        if (!chosen) {
          llvm::errs() << "aiecc: " << flag << ": "
                       << llvm::toString(chosen.takeError())
                       << "; known outputs are:\n";
          std::set<llvm::StringRef> names;
          for (const auto &e : g.edges)
            names.insert(e->name);
          for (llvm::StringRef n : names)
            llvm::errs() << "  " << n << '\n';
          std::exit(1);
        }
        resolved.push_back(*chosen);
      }
      return resolved;
    };

    // --get selects outputs (relocated to the output dir). --cut only marks a
    // checkpoint cut point: the edge is built (see Engine::run `buildAlso`) but
    // stays in the work dir as an intermediate, so downstream consumers that
    // reference it by path (e.g. the CDO step loading core ELFs) still find it.
    for (EdgeBase *e : resolveNames(getOutputs, "--get"))
      outputs.push_back(e);
    for (EdgeBase *e : resolveNames(cutOutputs, "--cut"))
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
  // selection (xchesscc/xbridge), the --aiesim implication, and the
  // resolved-option globals (wantAiesim, doUnified, doCompileHost). See
  // CommandLineOptions.h.
  if (!cli::resolveOptions())
    return 1;

  // --expand-load-pdis reconfigures via PDI swaps and routes the config branch
  // through the NPU-lowered module; control-packet generation currently
  // assumes it runs on the *pre*-NPU-lowering module. The two are incompatible
  // as implemented.
  if (expandLoadPdis && generateCtrlpkt) {
    llvm::errs() << "aiecc: --expand-load-pdis and --aie-generate-ctrlpkt are "
                    "mutually exclusive\n";
    return 1;
  }

  // The control-packet DMA sequence targets the xclbin / instruction-buffer
  // runtime, never the full-ELF runtime, so the two must not be requested
  // together (see the hard-coded fold in the ctrlpktDmaSeq edge).
  if (generateFullElf && generateCtrlpkt) {
    llvm::errs() << "aiecc: --generate-full-elf and --aie-generate-ctrlpkt are "
                    "mutually exclusive\n";
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
  ShellCommand::addInstallPrefix("peano", peanoInstallDir);
  ShellCommand::verbose = verbose;
  ShellCommand::dryRun = dryRun;

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

  // Reject an empty (or whitespace-only) input up front.
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
  Engine engine({outputDir, getWorkDir(), verbose, showProgress,
                 keepIntermediates, numThreads, profile});
  if (mlir::failed(engine.run(g, outputs, satisfied,
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

  return 0;
}
