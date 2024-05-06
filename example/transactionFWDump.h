//
// Created by mlevental on 5/6/24.
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

void validate() {
  std::string xclbinFile(
      "/home/mlevental/dev_projects/mlir-aie/example/1x4.xclbin");

  try {
    unsigned int deviceIndex = 0;
    auto device = xrt::device(deviceIndex);
    auto xclbin = xrt::xclbin(xclbinFile);
    auto xkernel = xclbin.get_kernel("DPU_PDI_0");
    auto kernelName = xkernel.get_name();
    device.register_xclbin(xclbin);

    xrt::hw_context context(device, xclbin.get_uuid());
    auto kernel = xrt::kernel(context, kernelName);
    const int iteration = 100;
    const int dummy_buffer_size = 4096;   /* in bytes */
    const int noop_instrction_size = 128; /* in bytes */

    auto bo_instr = xrt::bo(device, noop_instrction_size,
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(5));
    auto bo_ifm = xrt::bo(device, dummy_buffer_size, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(1));
    auto bo_param = xrt::bo(device, dummy_buffer_size, XRT_BO_FLAGS_HOST_ONLY,
                            kernel.group_id(2));
    auto bo_ofm = xrt::bo(device, dummy_buffer_size, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(3));
    auto bo_inter = xrt::bo(device, dummy_buffer_size, XRT_BO_FLAGS_HOST_ONLY,
                            kernel.group_id(4));
    auto bo_mc = xrt::bo(device, dummy_buffer_size, XRT_BO_FLAGS_HOST_ONLY,
                         kernel.group_id(7));

    // Fill no-op instrctions
    std::memset(bo_instr.map<char *>(), 0, noop_instrction_size);

    // Sync Input BOs
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_ifm.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_param.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_mc.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    uint64_t opcode = 1;
    auto start = std::chrono::high_resolution_clock::now();

    // Set kernel argument and trigger it to run. A run object will be returned.
    auto run = kernel(opcode, bo_ifm, bo_param, bo_ofm, bo_inter, bo_instr,
                      noop_instrction_size / sizeof(uint32_t), bo_mc);
    // Wait on the run object
    run.wait2();

    for (int i = 1; i < iteration; i++) {
      // Re-start the same run object with same arguments
      run.start();
      run.wait2();
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Host test microseconds: " << duration.count() << std::endl;
    std::cout << "Host test average latency: " << duration.count() / iteration
              << " us/iter" << std::endl;
  } catch (const std::exception &ex) {
    std::cout << "ERROR: Caught exception: " << ex.what() << '\n';
    std::cout << "TEST FAILED!" << std::endl;
  }

  std::cout << "TEST PASSED!" << std::endl;
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
  std::string type() const { return "transaction_op"; }

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

void dumpRegistersTransaction() {
  unsigned int deviceIndex = 0;
  auto device = xrt::device(deviceIndex);
  std::string xclbinFile(
      "/home/mlevental/dev_projects/mlir-aie/example/1x4.xclbin");
  auto xclbin = xrt::xclbin(xclbinFile);
  auto xkernel = xclbin.get_kernel("DPU_PDI_0");
  auto kernelName = xkernel.get_name();
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

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
                  /*PartProp*/ {},
                  // without this you get a segfault in XAie_CfgInitialize
                  XAIE_IO_BACKEND_CDO};
  XAie_DevInst devInst;
  XAie_InstDeclare(_devInst, &configPtr);
  devInst = _devInst;
  XAie_CfgInitialize(&devInst, &configPtr);

  uint8_t *txnPtr;
  read_register_op_t *op;
  std::size_t opSize;
  std::vector<register_data_t> opProfileData;
  int counterId = 0;
  std::vector<unsigned long> regs{// corestatus
                                  0x00032004,
                                  // Stream_Switch_Master_Config_AIE_Core0
                                  0x0003F000,
                                  // module clock control
                                  0x00060000,
                                  // Event_Group_Stream_Switch_Enable
                                  0x00034518,
                                  // core le
                                  0x00031150,
                                  // Core_CR
                                  0x00031170};

  for (int row = 0; row < 6; ++row) {
    for (int col = 0; col < 5; ++col) {
      for (const auto &reg : regs) {
        opProfileData.push_back(register_data_t{reg + getTileAddr(col, row)});
        counterId++;
      }
    }
  }

  opSize =
      sizeof(read_register_op_t) + sizeof(register_data_t) * (counterId - 1);
  op = (read_register_op_t *)malloc(opSize);
  op->count = counterId;
  for (size_t i = 0; i < opProfileData.size(); i++)
    op->data[i] = opProfileData[i];

  XAie_StartTransaction(&devInst, XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);

  auto readOpCode = XAIE_IO_CUSTOM_OP_BEGIN + 2;

  XAie_AddCustomTxnOp(&devInst, (uint8_t)readOpCode, (void *)op, opSize);
  txnPtr = XAie_ExportSerializedTransaction(&devInst, 1, 0);
  XAie_ClearTransaction(&devInst);

  op_buf instrBuf;
  instrBuf.addOP(transaction_op(txnPtr));

  // start validate copy-paste
  xrt::bo instrBo = xrt::bo(context.get_device(), instrBuf.ibuf_.size(),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(5));

  const int dummy_buffer_size = 4096; /* in bytes */
  auto bo_ifm = xrt::bo(device, dummy_buffer_size, XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(1));
  auto bo_param = xrt::bo(device, dummy_buffer_size, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(2));
  auto bo_ofm = xrt::bo(device, dummy_buffer_size, XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));
  auto bo_inter = xrt::bo(device, dummy_buffer_size, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(4));
  auto bo_mc = xrt::bo(device, dummy_buffer_size, XRT_BO_FLAGS_HOST_ONLY,
                       kernel.group_id(7));

  instrBo.write(instrBuf.ibuf_.data());
  instrBo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_ifm.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_param.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_mc.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(CONFIGURE_OPCODE, bo_ifm, bo_param, bo_ofm, bo_inter,
                    instrBo, instrBuf.ibuf_.size(), bo_mc);
  run.wait2();
  // end

  xrt::bo resultBo = xrt::bo(context.get_device(), SIZE_4K,
                             XCL_BO_FLAGS_CACHEABLE, kernel.group_id(5));
  if (!resultBo)
    throw std::runtime_error("couldn't get resultBo");

  resultBo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  uint8_t *resultBoMap = resultBo.map<uint8_t *>();
  uint32_t *output = reinterpret_cast<uint32_t *>(resultBoMap + OFFSET_3K);

  for (uint32_t i = 0; i < op->count; i++) {
    std::stringstream msg;
    int col = (op->data[i].address >> XAIE_COL_SHIFT) & 0x1F;
    int row = (op->data[i].address >> XAIE_ROW_SHIFT) & 0x1F;
    int reg = (op->data[i].address) & 0xFFFFF;
    uint32_t val = output[i];

    msg << "Debug tile (" << col << ", " << row << ") "
        << "hex address/values: " << std::hex << reg << " : " << val
        << std::dec;
    std::cout << msg.str() << "\n";
  }
}

#endif // AIE_TRANSACTIONFWDUMP_H
