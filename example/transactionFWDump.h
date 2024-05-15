//
// Created by mlevental on 5/15/24.
//

#ifndef AIE_TRANSACTIONFWDUMP_H
#define AIE_TRANSACTIONFWDUMP_H

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

extern "C" {
#include "xaiengine/xaie_core.h"
#include "xaiengine/xaie_elfloader.h"
#include "xaiengine/xaie_txn.h"
#include "xaiengine/xaiegbl.h"
#include "xaiengine/xaiegbl_defs.h"
}

#include <cstring>
#include <sstream>
#include <string>

#define XAIE_BASE_ADDR 0x40000000
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20
#define XAIE_SHIM_ROW 0
#define XAIE_MEM_TILE_ROW_START 1
constexpr std::uint64_t CONFIGURE_OPCODE = 2;
constexpr uint32_t SIZE_4K = 4 * 1024;
constexpr uint32_t OFFSET_3K = 3 * 1024;

static inline uint64_t getTileAddr(uint8_t c, uint8_t r) {
  return (((uint64_t)r & 0xFFU) << XAIE_ROW_SHIFT) |
         (((uint64_t)c & 0xFFU) << XAIE_COL_SHIFT);
}

typedef struct {
  uint64_t address;
} register_data_t;

typedef struct {
  uint32_t count;
  register_data_t data[1]; // variable size
} read_register_op_t;

enum op_types {
  e_TRANSACTION_OP,
  e_WAIT_OP,
  e_PENDINGBDCOUNT_OP,
  e_DBGPRINT_OP,
  e_PATCHBD_OP
};

typedef struct {
  enum op_types type;
  unsigned int size_in_bytes;
} op_base;

typedef struct {
  op_base b;
} transaction_op_t;

class transaction_op {
public:
  transaction_op() = delete;

  transaction_op(void *txn) {
    XAie_TxnHeader *Hdr = (XAie_TxnHeader *)txn;
    printf("Header version %d.%d\n", Hdr->Major, Hdr->Minor);
    printf("Device Generation: %d\n", Hdr->DevGen);
    printf("Cols, Rows, NumMemRows : (%d, %d, %d)\n", Hdr->NumCols,
           Hdr->NumRows, Hdr->NumMemTileRows);
    printf("TransactionSize: %u\n", Hdr->TxnSize);
    printf("NumOps: %u\n", Hdr->NumOps);

    transaction_op_t *tptr = new transaction_op_t();
    tptr->b.type = e_TRANSACTION_OP;
    tptr->b.size_in_bytes = sizeof(transaction_op_t) + Hdr->TxnSize;

    cmdBuf_ = new uint8_t[Hdr->TxnSize];
    memcpy(cmdBuf_, txn, Hdr->TxnSize);
    op_ptr_ = (op_base *)tptr;
    TxnSize = Hdr->TxnSize;
  }

  unsigned size() const { return op_ptr_->size_in_bytes; }

  void serialize(void *ptr) const {
    memcpy(ptr, op_ptr_, sizeof(transaction_op_t));
    ptr = (char *)ptr + sizeof(transaction_op_t);
    memcpy(ptr, cmdBuf_, TxnSize);
  }

  ~transaction_op() {
    transaction_op_t *tptr = reinterpret_cast<transaction_op_t *>(op_ptr_);
    delete tptr;
    if (cmdBuf_)
      delete[] cmdBuf_;
  }

  uint8_t *cmdBuf_;
  uint32_t TxnSize;
  op_base *op_ptr_;
};

class op_buf {
public:
  void addOP(const transaction_op &instr) {
    size_t ibuf_sz = ibuf_.size();
    ibuf_.resize(ibuf_sz + instr.size());
    instr.serialize((void *)&ibuf_[ibuf_sz]);
  }

  size_t size() const { return ibuf_.size(); }

  const void *data() const { return ibuf_.data(); }
  std::vector<uint8_t> ibuf_;
};

#define XAIE_PARTITION_BASE_ADDR 0x0

void dumpRegistersTransaction() {
  unsigned int deviceIndex = 0;
  auto device = xrt::device(deviceIndex);
  std::string xclbinFile(
      "/home/mlevental/dev_projects/mlir-aie/example/1x4.xclbin");
  auto xclbin = xrt::xclbin(xclbinFile);
  auto xkernel = xclbin.get_kernel("XDP_KERNEL");
  auto kernelName = xkernel.get_name();
  device.register_xclbin(xclbin);

  XAie_Config configPtr =
      XAie_Config{/*AieGen*/ XAIE_DEV_GEN_AIEML,
                  /*BaseAddr*/ XAIE_BASE_ADDR,
                  /*ColShift*/ XAIE_COL_SHIFT,
                  /*RowShift*/ XAIE_ROW_SHIFT,
                  /*NumRows*/ static_cast<uint8_t>(6),
                  /*NumCols*/ static_cast<uint8_t>(5),
                  /*ShimRowNum*/ XAIE_SHIM_ROW,
                  /*MemTileRowStart*/ XAIE_MEM_TILE_ROW_START,
                  /*MemTileNumRows*/ static_cast<uint8_t>(1),
                  /*AieTileRowStart*/
                  static_cast<uint8_t>(XAIE_MEM_TILE_ROW_START + 1),
                  /*AieTileNumRows*/
                  static_cast<uint8_t>(6 - 1 - 1),
                  /*PartProp*/ {0},
                  // without this you get a segfault in XAie_CfgInitialize
                  XAIE_IO_BACKEND_CDO};
  XAie_DevInst devInst;
  XAie_InstDeclare(_devInst, &configPtr);
  devInst = _devInst;
  //  XAie_SetupPartitionConfig(&devInst, XAIE_PARTITION_BASE_ADDR,
  //                            /*partitionStartCol*/ 0, /*partitionNumCols*/
  //                            1);
  XAie_CfgInitialize(&devInst, &configPtr);

  uint8_t *txnPtr;
  read_register_op_t *op;
  std::size_t opSize;
  std::vector<register_data_t> opRegisterData;
  int counterId = 0;
  std::vector<unsigned long> regs{//            0x340D0,
                                  //            0x340D4,
                                  //            0x340D8,
                                  0x00032004};

  int col = 0, row = 2;
  for (const auto &reg : regs) {
    opRegisterData.push_back(register_data_t{reg + getTileAddr(col, row)});
    counterId++;
  }

  opSize =
      sizeof(read_register_op_t) + sizeof(register_data_t) * (counterId - 1);
  op = (read_register_op_t *)malloc(opSize);
  op->count = counterId;
  for (size_t i = 0; i < opRegisterData.size(); i++)
    op->data[i] = opRegisterData[i];

  // record transaction
  XAie_StartTransaction(&devInst, XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);
  XAie_RequestCustomTxnOp(&devInst);
  XAie_RequestCustomTxnOp(&devInst);
  auto readOpCode = XAie_RequestCustomTxnOp(&devInst);
  XAie_AddCustomTxnOp(&devInst, (uint8_t)readOpCode, (void *)op, opSize);
  txnPtr = XAie_ExportSerializedTransaction(&devInst, 1, 0);
  XAie_ClearTransaction(&devInst);

  // initializeKernel
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  // submitTransaction
  op_buf instrBuf;
  instrBuf.addOP(transaction_op(txnPtr));
  xrt::bo instrBo;

  // Configuration bo
  instrBo = xrt::bo(context.get_device(), instrBuf.ibuf_.size(),
                    XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  instrBo.write(instrBuf.ibuf_.data());
  instrBo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto run = kernel(CONFIGURE_OPCODE, instrBo, instrBo.size() / sizeof(int), 0,
                    0, 0, 0, 0);
  run.wait2();

  // syncResults
  xrt::bo result_bo = xrt::bo(context.get_device(), SIZE_4K,
                              XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  //  auto boflags = XRT_BO_FLAGS_CACHEABLE;
  //  auto ext_boflags = XRT_BO_USE_DEBUG << 4;
  //  auto size = static_cast<size_t>(4096);
  //
  //  {
  //    hw_ctx hwctx(device.get_handle().get(), xclbinFile);
  //    auto bo = hwctx.get()->alloc_bo(size, get_bo_flags(boflags,
  //    ext_boflags)); auto dbg_p = static_cast<uint32_t *>(
  //        bo->map(xrt_core::buffer_handle::map_type::read));
  //    bo.get()->sync(buffer_handle::direction::device2host, 4096, 0);
  //    for (int i = 0; i < 4096; ++i)
  //      if (dbg_p[i] != 0)
  //        std::cout << dbg_p[i] << ", ";
  //    std::cout << "\n";
  //  }
}

#endif // AIE_TRANSACTIONFWDUMP_H
