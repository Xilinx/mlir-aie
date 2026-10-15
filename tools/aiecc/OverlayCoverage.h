//===- OverlayCoverage.h ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The columns a control packet overlay covers. The router refuses a detour
// through a column the overlay misses, so on a routing failure OverlayCoverage
// routes again without the overlay, covers each column that routing
// configures, and reruns the edge. It keeps the routed result for
// CachedRouting, so a design that routes first time is routed once.
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_OVERLAY_COVERAGE_H
#define AIECC_OVERLAY_COVERAGE_H

#include "Actions.h"
#include "IRTransforms.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/xxhash.h"

#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace xilinx::aiecc {

// `loc`, rebuilt in `ctx`.
inline mlir::Location translateLocation(mlir::Location loc,
                                        mlir::MLIRContext *ctx) {
  return llvm::TypeSwitch<mlir::LocationAttr, mlir::Location>(loc)
      .Case([&](mlir::FileLineColLoc l) -> mlir::Location {
        return mlir::FileLineColLoc::get(ctx, l.getFilename().getValue(),
                                         l.getLine(), l.getColumn());
      })
      .Case([&](mlir::NameLoc l) -> mlir::Location {
        return mlir::NameLoc::get(
            mlir::StringAttr::get(ctx, l.getName().getValue()),
            translateLocation(l.getChildLoc(), ctx));
      })
      .Case([&](mlir::CallSiteLoc l) -> mlir::Location {
        return mlir::CallSiteLoc::get(translateLocation(l.getCallee(), ctx),
                                      translateLocation(l.getCaller(), ctx));
      })
      .Case([&](mlir::FusedLoc l) -> mlir::Location {
        llvm::SmallVector<mlir::Location> locs;
        for (mlir::Location sub : l.getLocations())
          locs.push_back(translateLocation(sub, ctx));
        return mlir::FusedLoc::get(ctx, locs);
      })
      .Default([&](mlir::LocationAttr) { return mlir::UnknownLoc::get(ctx); });
}

// `mod` printed and parsed into `ctx`, with its locations.
inline mlir::OwningOpRef<mlir::ModuleOp> copyInto(mlir::ModuleOp mod,
                                                  mlir::MLIRContext *ctx) {
  std::string text;
  llvm::raw_string_ostream os(text);
  mod.print(os, mlir::OpPrintingFlags().enableDebugInfo());
  return mlir::parseSourceString<mlir::ModuleOp>(text, ctx);
}

// A context for trial runs whose diagnostics are held back. aiecc runs other
// edges at the same time on the shared context, whose diagnostics a handler
// there would take too.
struct TrialContext {
  mlir::MLIRContext ctx;
  explicit TrialContext(mlir::MLIRContext *shared)
      : ctx(shared->getDialectRegistry(),
            mlir::MLIRContext::Threading::DISABLED) {
    ctx.loadAllAvailableDialects();
  }
};

// The diagnostics emitted in `source` while it is alive, held back for replay
// in `target`.
struct CapturedDiagnostics {
  struct Diag {
    mlir::Location loc;
    mlir::DiagnosticSeverity severity;
    std::string text;
    std::vector<std::pair<mlir::Location, std::string>> notes;
  };
  mlir::MLIRContext *ctx;
  std::vector<Diag> diags;
  mlir::DiagnosticEngine::HandlerID id;

  CapturedDiagnostics(mlir::MLIRContext *source, mlir::MLIRContext *target)
      : ctx(source) {
    id = ctx->getDiagEngine().registerHandler(
        [this, target](mlir::Diagnostic &d) {
          Diag out{translateLocation(d.getLocation(), target),
                   d.getSeverity(),
                   d.str(),
                   {}};
          for (mlir::Diagnostic &n : d.getNotes())
            out.notes.push_back(
                {translateLocation(n.getLocation(), target), n.str()});
          diags.push_back(std::move(out));
          return mlir::success();
        });
  }
  ~CapturedDiagnostics() { ctx->getDiagEngine().eraseHandler(id); }

  static void replay(mlir::MLIRContext *ctx, const std::vector<Diag> &diags) {
    for (const Diag &d : diags) {
      mlir::InFlightDiagnostic out =
          ctx->getDiagEngine().emit(d.loc, d.severity);
      out << d.text;
      for (const auto &[loc, text] : d.notes)
        out.attachNote(loc) << text;
    }
  }
};

// Routed modules OverlayCoverage produced, with the diagnostics their routing
// emitted, keyed by a fingerprint of the module they route.
struct RoutedCache {
  struct Entry {
    mlir::OwningOpRef<mlir::ModuleOp> routed;
    std::vector<CapturedDiagnostics::Diag> diags;
  };
  std::mutex mu;
  llvm::DenseMap<uint64_t, Entry> byFingerprint;

  static uint64_t fingerprint(mlir::ModuleOp mod) {
    std::string text;
    llvm::raw_string_ostream os(text);
    mod.print(os);
    return llvm::xxh3_64bits(text);
  }
};

struct OverlayCoverage {
  // Builds the edge's pipeline with the given cover-columns.
  using Build = std::function<std::unique_ptr<mlir::PassManager>(
      mlir::MLIRContext *, mlir::ModuleOp, llvm::ArrayRef<int>)>;
  mlir::MLIRContext *ctx;
  Build build;
  std::shared_ptr<RoutedCache> cache;

  // Per column, the device and op that make the routing configure it.
  using Configured = std::map<int, std::pair<std::string, std::string>>;

  // The columns `mod`'s devices send control packets to (TileControl
  // destinations).
  static std::set<int> coveredColumns(mlir::ModuleOp mod) {
    std::set<int> cols;
    mod.walk([&](xilinx::AIE::PacketDestOp dest) {
      if (dest.getBundle() == xilinx::AIE::WireBundle::TileControl)
        if (auto tile = dest.getTile().getDefiningOp<xilinx::AIE::TileOp>())
          cols.insert(tile.colIndex());
    });
    return cols;
  }

  // The columns whose stream switches or shim muxes a routed `mod`
  // configures.
  static Configured configuredColumns(mlir::ModuleOp mod) {
    Configured cols;
    auto note = [&](mlir::Operation *op, mlir::Value tile) {
      auto tileOp = tile.getDefiningOp<xilinx::AIE::TileOp>();
      if (!tileOp || op->getRegion(0).front().without_terminator().empty())
        return;
      auto device = op->getParentOfType<xilinx::AIE::DeviceOp>();
      cols.try_emplace(tileOp.colIndex(), device.getSymName().str(),
                       op->getName().getStringRef().str());
    };
    mod.walk([&](mlir::Operation *op) {
      if (auto box = llvm::dyn_cast<xilinx::AIE::SwitchboxOp>(op))
        note(op, box.getTile());
      else if (auto mux = llvm::dyn_cast<xilinx::AIE::ShimMuxOp>(op))
        note(op, mux.getTile());
    });
    return cols;
  }

  static mlir::Location columnLocation(mlir::ModuleOp mod, int col) {
    mlir::Location loc = mod.getLoc();
    mod.walk([&](xilinx::AIE::TileOp tile) {
      if (tile.colIndex() != col || tile.rowIndex() != 0)
        return mlir::WalkResult::advance();
      loc = tile.getLoc();
      return mlir::WalkResult::interrupt();
    });
    return loc;
  }

  // Runs the edge's pipeline on a copy of `base`; null on failure, with the
  // pipeline's diagnostics emitted.
  mlir::OwningOpRef<mlir::ModuleOp>
  runPipeline(mlir::ModuleOp base, llvm::ArrayRef<int> cover) const {
    mlir::OwningOpRef<mlir::ModuleOp> mod(base.clone());
    auto pm = build(ctx, *mod, cover);
    if (!pm) {
      llvm::errs() << "aiecc: PassPipeline builder returned null\n";
      return {};
    }
    if (mlir::failed(runPasses(*pm, *mod)))
      return {};
    return mod;
  }

  template <typename T>
  mlir::LogicalResult
  operator()(const Item<T> &in,
             Item<mlir::OwningOpRef<mlir::ModuleOp>> &out) const {
    mlir::OwningOpRef<mlir::ModuleOp> base = asModule(in, ctx);
    if (!base) {
      llvm::errs() << "aiecc: OverlayCoverage could not obtain input module\n";
      return mlir::failure();
    }
    std::vector<int> cover;
    Configured added;
    mlir::OwningOpRef<mlir::ModuleOp> first;
    for (;;) {
      mlir::OwningOpRef<mlir::ModuleOp> trial = runPipeline(*base, cover);
      if (!trial)
        return mlir::failure();

      mlir::OwningOpRef<mlir::ModuleOp> routed;
      std::vector<CapturedDiagnostics::Diag> routingDiags;
      bool routes;
      {
        TrialContext trialCtx(ctx);
        mlir::OwningOpRef<mlir::ModuleOp> mod = copyInto(*trial, &trialCtx.ctx);
        {
          CapturedDiagnostics routing(&trialCtx.ctx, ctx);
          routes = mod && mlir::succeeded(runPasses(
                              *getRoutingPipeline(&trialCtx.ctx), *mod));
          routingDiags = std::move(routing.diags);
        }
        if (routes)
          routed = copyInto(*mod, ctx);
        routes = routes && routed;
      }
      if (routes) {
        for (const auto &[col, why] : added)
          ctx->getDiagEngine().emit(columnLocation(*trial, col),
                                    mlir::DiagnosticSeverity::Remark)
              << "control overlay: column " << col << " covered: the router "
              << "configures it (" << why.second << " in @" << why.first
              << ") outside the columns the design occupies";
        uint64_t key = RoutedCache::fingerprint(*trial);
        {
          std::lock_guard<std::mutex> lock(cache->mu);
          cache->byFingerprint.insert_or_assign(
              key,
              RoutedCache::Entry{std::move(routed), std::move(routingDiags)});
        }
        out.value = std::move(trial);
        return mlir::success();
      }
      if (!first)
        first = trial->clone();

      // Where the design routes when the overlay leaves every tile free.
      bool grew = false;
      {
        TrialContext trialCtx(ctx);
        mlir::OwningOpRef<mlir::ModuleOp> free =
            copyInto(*trial, &trialCtx.ctx);
        bool freeRoutes = false;
        if (free) {
          free->walk([](xilinx::AIE::DeviceOp device) {
            device->removeAttr("has_ctrl_pkt_overlay");
          });
          CapturedDiagnostics quiet(&trialCtx.ctx, ctx);
          freeRoutes = mlir::succeeded(
              runPasses(*getRoutingPipeline(&trialCtx.ctx), *free));
        }
        if (freeRoutes) {
          std::set<int> covered = coveredColumns(*trial);
          for (const auto &[col, why] : configuredColumns(*free))
            if (!covered.count(col) && !llvm::is_contained(cover, col)) {
              cover.push_back(col);
              added.try_emplace(col, why);
              grew = true;
            }
        }
      }
      if (!grew) {
        // No column answers the failure, and the routing edge reports it.
        out.value = std::move(first);
        return mlir::success();
      }
      llvm::sort(cover);
    }
  }
};

// The `input_physical.mlir` edge under a control packet overlay. It reuses the
// routing OverlayCoverage already ran on this module, with its diagnostics,
// else runs the routing pipeline.
struct CachedRouting {
  mlir::MLIRContext *ctx;
  std::shared_ptr<RoutedCache> cache;

  template <typename T>
  mlir::LogicalResult
  operator()(const Item<T> &in,
             Item<mlir::OwningOpRef<mlir::ModuleOp>> &out) const {
    {
      uint64_t key = RoutedCache::fingerprint(in.get().get());
      std::lock_guard<std::mutex> lock(cache->mu);
      auto it = cache->byFingerprint.find(key);
      if (it != cache->byFingerprint.end()) {
        out.value = std::move(it->second.routed);
        CapturedDiagnostics::replay(ctx, it->second.diags);
        cache->byFingerprint.erase(it);
        return mlir::success();
      }
    }
    return PassPipeline{getRoutingPipeline(ctx)}(in, out);
  }
};

} // namespace xilinx::aiecc

#endif // AIECC_OVERLAY_COVERAGE_H
