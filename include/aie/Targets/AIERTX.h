//
// Created by maksim on 3/6/24.
//

#ifndef AIE_AIERTX_H
#define AIE_AIERTX_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIEEnums.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include "llvm/Support/raw_ostream.h"

extern "C" {
#include "xaiengine/xaie_core.h"
#include "xaiengine/xaie_dma.h"
#include "xaiengine/xaie_elfloader.h"
#include "xaiengine/xaie_interrupt.h"
#include "xaiengine/xaie_locks.h"
#include "xaiengine/xaie_plif.h"
#include "xaiengine/xaie_ss.h"
#include "xaiengine/xaiegbl.h"
#include "xaiengine/xaiegbl_defs.h"
}

#include <map>
#include <optional>
#include <string>

#define AIERC_STR(x) x, #x
static const std::map<AieRC, std::string> AIERCTOSTR = {
    {AIERC_STR(XAIE_OK)},
    {AIERC_STR(XAIE_ERR)},
    {AIERC_STR(XAIE_INVALID_DEVICE)},
    {AIERC_STR(XAIE_INVALID_RANGE)},
    {AIERC_STR(XAIE_INVALID_ARGS)},
    {AIERC_STR(XAIE_INVALID_TILE)},
    {AIERC_STR(XAIE_ERR_STREAM_PORT)},
    {AIERC_STR(XAIE_INVALID_DMA_TILE)},
    {AIERC_STR(XAIE_INVALID_BD_NUM)},
    {AIERC_STR(XAIE_ERR_OUTOFBOUND)},
    {AIERC_STR(XAIE_INVALID_DATA_MEM_ADDR)},
    {AIERC_STR(XAIE_INVALID_ELF)},
    {AIERC_STR(XAIE_CORE_STATUS_TIMEOUT)},
    {AIERC_STR(XAIE_INVALID_CHANNEL_NUM)},
    {AIERC_STR(XAIE_INVALID_LOCK)},
    {AIERC_STR(XAIE_INVALID_DMA_DIRECTION)},
    {AIERC_STR(XAIE_INVALID_PLIF_WIDTH)},
    {AIERC_STR(XAIE_INVALID_LOCK_ID)},
    {AIERC_STR(XAIE_INVALID_LOCK_VALUE)},
    {AIERC_STR(XAIE_LOCK_RESULT_FAILED)},
    {AIERC_STR(XAIE_INVALID_DMA_DESC)},
    {AIERC_STR(XAIE_INVALID_ADDRESS)},
    {AIERC_STR(XAIE_FEATURE_NOT_SUPPORTED)},
    {AIERC_STR(XAIE_INVALID_BURST_LENGTH)},
    {AIERC_STR(XAIE_INVALID_BACKEND)},
    {AIERC_STR(XAIE_INSUFFICIENT_BUFFER_SIZE)},
    {AIERC_STR(XAIE_ERR_MAX)}};
#undef AIERC_STR

static const std::map<xilinx::AIE::WireBundle, StrmSwPortType>
    WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE = {
        {xilinx::AIE::WireBundle::Core, StrmSwPortType::CORE},
        {xilinx::AIE::WireBundle::DMA, StrmSwPortType::DMA},
        // missing control from StrmSwPortType
        {xilinx::AIE::WireBundle::FIFO, StrmSwPortType::FIFO},
        {xilinx::AIE::WireBundle::South, StrmSwPortType::SOUTH},
        {xilinx::AIE::WireBundle::West, StrmSwPortType::WEST},
        {xilinx::AIE::WireBundle::North, StrmSwPortType::NORTH},
        {xilinx::AIE::WireBundle::East, StrmSwPortType::EAST},
        // missing PLIO from WireBundle
        // missing NOC from WireBundle
        {xilinx::AIE::WireBundle::Trace, StrmSwPortType::TRACE},
};

// https://stackoverflow.com/a/32230306
template <typename H1>
llvm::raw_ostream &showAIEXRTArgs(llvm::raw_ostream &out, const char *label,
                                  H1 &&value) {
  return out << label << "=" << std::forward<H1>(value);
}

template <typename H1, typename... T>
llvm::raw_ostream &showAIEXRTArgs(llvm::raw_ostream &out, const char *label,
                                  H1 &&value, T &&...rest) {
  const char *pcomma = strchr(label, ',');
  return showAIEXRTArgs(out.write(label, pcomma - label)
                            << "=" << std::forward<H1>(value) << ',',
                        pcomma + 1, std::forward<T>(rest)...);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const XAie_LocType &loc);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const XAie_Lock &lock);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const XAie_Packet &packet);

#define SHOW_AIERTX_ARGS(os, ...) showAIEXRTArgs(os, #__VA_ARGS__, __VA_ARGS__)

// So that we can use the pattern if(auto r = TRY_XAIE_API...) { // r is nonzero
// }
static_assert(XAIE_OK == 0);

#define TRY_XAIE_API_FATAL_ERROR(API, ...)                                     \
  do {                                                                         \
    LLVM_DEBUG(llvm::dbgs() << "trying XAIE API: " << #API << " with args: "); \
    LLVM_DEBUG(SHOW_AIERTX_ARGS(llvm::dbgs(), __VA_ARGS__));                   \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                          \
    if (auto r = API(__VA_ARGS__))                                             \
      llvm::report_fatal_error(llvm::Twine(#API " failed with ") +             \
                               AIERCTOSTR.at(r));                              \
  } while (0)

#define TRY_XAIE_API_EMIT_ERROR(OP, API, ...)                                  \
  do {                                                                         \
    LLVM_DEBUG(llvm::dbgs() << "trying XAIE API: " << #API << " with args: "); \
    LLVM_DEBUG(SHOW_AIERTX_ARGS(llvm::dbgs(), __VA_ARGS__));                   \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                          \
    if (auto r = API(__VA_ARGS__))                                             \
      return OP.emitOpError() << #API " failed with " << AIERCTOSTR.at(r);     \
  } while (0)

#define TRY_XAIE_API_LOGICAL_RESULT(API, ...)                                  \
  do {                                                                         \
    LLVM_DEBUG(llvm::dbgs() << "trying XAIE API: " << #API << " with args: "); \
    LLVM_DEBUG(SHOW_AIERTX_ARGS(llvm::dbgs(), __VA_ARGS__));                   \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                          \
    if (auto r = API(__VA_ARGS__)) {                                           \
      llvm::errs() << #API " failed with " << AIERCTOSTR.at(r);                \
      return failure();                                                        \
    }                                                                          \
  } while (0)

#define XAIE_BASE_ADDR 0x40000000
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20
#define XAIE_SHIM_ROW 0
#define XAIE_MEM_TILE_ROW_START 1
#define XAIE_PARTITION_BASE_ADDR 0x0

#define NPI_ADDR 0x0
#define NUM_LOCKS 16
#define EVEN_BD_NUM_START 0
#define ODD_BD_NUM_START 24
#define MEM_TILE_LOCK_ID_INCR 64
#define BASE_ADDR_A_INCR 0x80000

namespace xilinx::AIE {
struct AIEControl {
  XAie_Config configPtr;
  XAie_DevInst devInst;
  const AIETargetModel &targetModel;

  AIEControl(size_t partitionStartCol, size_t partitionNumCols,
             const xilinx::AIE::AIETargetModel &tm);

  mlir::LogicalResult setIOBackend(bool aieSim, bool xaieDebug);
  mlir::LogicalResult configureBdInBlock(XAie_DmaDesc &dmaTileBd,
                                         mlir::Block &block,
                                         XAie_LocType &tileLoc, int bdId,
                                         std::optional<int> nextBdId);
  mlir::LogicalResult pushToBdQueueAndEnable(mlir::Operation &op,
                                             XAie_LocType &tileLoc, int chNum,
                                             const DMAChannelDir &channelDir,
                                             int bdId, int repeatCount);
  mlir::LogicalResult configureLocksAndBd(mlir::Block &block,
                                          XAie_LocType tileLoc);
  mlir::LogicalResult initLocks(DeviceOp &targetOp);
  mlir::LogicalResult configureSwitches(DeviceOp &targetOp);
  mlir::LogicalResult enableCoresInDevice(DeviceOp &targetOp);
  mlir::LogicalResult configureLocksInBdBlock(XAie_DmaDesc &dmaTileBd,
                                              mlir::Block &block,
                                              XAie_LocType &tileLoc);
  mlir::LogicalResult dmaUpdateBdAddr(DeviceOp &targetOp, int col, int row,
                                      size_t addr, size_t bdId);
};

} // namespace xilinx::AIE

#endif // AIE_AIERTX_H
