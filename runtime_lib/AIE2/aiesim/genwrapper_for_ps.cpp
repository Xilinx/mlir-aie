#include <adf/wrapper/wrapper.h>
#include <xtlm.h>
#define BUSWIDTH 128

extern int main(int argc, char *argv[]);

// #ifdef _HYBRID_EMU_
// std::map<std::string, adf::graph*> _gr_inst = {
// 	{"mygraph", &mygraph}
// };
// std::map<std::string, adf::GMIO*> gmioMap = {};
// std::map<std::string, std::pair<adf::input_port*,std::string>> inputPortMap =
// {}; std::map<std::string, std::pair<adf::inout_port*,std::string>>
// inoutPortMap = {}; #include "adf/wrapper/xrt_ps_wrapper.inc" #endif

// Global flag for controlling simulator exit after PS IP execution is completed
extern int ps_main_complete;
// Global flag for aximm dumps in real NoC path
class PSIP_ps_i3 : public IPBlock {
  SC_HAS_PROCESS(PSIP_ps_i3);

public:
  xtlm::xtlm_aximm_initiator_socket PS_AxiMM_Rd;
  xtlm::xtlm_aximm_initiator_socket PS_AxiMM_Wr;
  xtlm::xtlm_aximm_initiator_rd_socket_util *PS_AxiMM_Rd_Util;
  xtlm::xtlm_aximm_initiator_wr_socket_util *PS_AxiMM_Wr_Util;
  xtlm::xtlm_aximm_mem_manager *mem_manager;
  static PSIP_ps_i3 *createInstance(sc_module_name name);
  //  Returns the instance created using createInstance()
  static PSIP_ps_i3 *getInstance();
  void write32(uint64_t Addr, uint32_t Data);
  uint32 read32(uint64_t Addr);
  void write128(uint64_t Addr, uint32_t *Data);
  void read128(uint64_t Addr, uint32_t *Data);
  void writeGM(uint64_t addr, const void *data, uint64_t size);
  void readGM(uint64_t addr, void *data, uint64_t size);
  void aximm_transaction(xtlm::xtlm_aximm_initiator_rd_socket_util &rd_util,
                         xtlm::xtlm_aximm_initiator_wr_socket_util &wr_util,
                         xtlm::xtlm_command command, unsigned long long address,
                         unsigned char *pData,
                         unsigned int trans_size_in_bytes);
  sc_event_queue toggle_AIE_array_clk;

private:
  PSIP_ps_i3(sc_module_name nm);
  static PSIP_ps_i3 *psObj;
  // Sets attributes for PS-ME, common to both R/W transaction
  void set_payload_attr(xtlm::aximm_payload *trans, size_t transBytes);
  void main_action();
  void response_process();
  sc_event transRspAvail;
};

PSIP_ps_i3 *PSIP_ps_i3::psObj = NULL;
PSIP_ps_i3 *PSIP_ps_i3::createInstance(sc_module_name name) {
  if (psObj == NULL)
    psObj = new PSIP_ps_i3(name);
  return psObj;
}

PSIP_ps_i3 *PSIP_ps_i3::getInstance() {
  if (psObj != NULL) {
    return psObj;
  } else {
    std::cout << "[WARNING]: psObj is a NULL ptr";
    return NULL;
  }
}
void PSIP_ps_i3::write32(uint64_t Addr, uint32_t Data) {
  size_t transBytes = sizeof(uint32_t);
  // Get the payload object
  xtlm::aximm_payload *trans = mem_manager->get_payload();
  trans->acquire();
  set_payload_attr(trans, transBytes);
  trans->set_command(xtlm::XTLM_WRITE_COMMAND);
  trans->set_address(Addr);
  memcpy(trans->get_data_ptr(), &(Data), transBytes);

  sc_time delay = SC_ZERO_TIME;
  PS_AxiMM_Wr_Util->b_transport(*trans, delay);

  trans->release();
}
uint32_t PSIP_ps_i3::read32(uint64_t Addr) {
  size_t transBytes = sizeof(uint32_t);
  // Get the payload object
  xtlm::aximm_payload *trans = mem_manager->get_payload();
  trans->acquire();
  set_payload_attr(trans, transBytes);
  trans->set_command(xtlm::XTLM_READ_COMMAND);
  trans->set_address(Addr);

  sc_time delay = SC_ZERO_TIME;
  PS_AxiMM_Rd_Util->b_transport(*trans, delay);

  uint32 data = 0;
  data = *(uint32 *)(trans->get_data_ptr());
  trans->release();
  return data;
}
void PSIP_ps_i3::write128(uint64_t Addr, uint32_t *Data) {
  size_t transBytes = 4 * sizeof(uint32_t);
  // Get the payload object
  xtlm::aximm_payload *trans = mem_manager->get_payload();
  trans->acquire();
  set_payload_attr(trans, transBytes);
  trans->set_command(xtlm::XTLM_WRITE_COMMAND);
  trans->set_address(Addr);
  memcpy(trans->get_data_ptr(), Data, transBytes);
  sc_time delay = SC_ZERO_TIME;
  PS_AxiMM_Wr_Util->b_transport(*trans, delay);
  trans->release();
}
void PSIP_ps_i3::read128(uint64_t Addr, uint32_t *Data) {
  size_t transBytes = 4 * sizeof(uint32_t);
  // Get the payload object
  xtlm::aximm_payload *trans = mem_manager->get_payload();
  trans->acquire();
  set_payload_attr(trans, transBytes);
  trans->set_command(xtlm::XTLM_READ_COMMAND);
  trans->set_address(Addr);
  sc_time delay = SC_ZERO_TIME;
  PS_AxiMM_Rd_Util->b_transport(*trans, delay);
  memcpy(Data, trans->get_data_ptr(), transBytes);
  trans->release();
}
void PSIP_ps_i3::writeGM(uint64_t addr, const void *data, uint64_t size) {
  toggle_AIE_array_clk.notify(1, SC_NS);
  uint64_t remaining = size;
  uint64_t currentAddr = addr;
  unsigned char *currentDataPtr = (unsigned char *)data;
  while (remaining >= 4096) // Axi maximum transaction size is 4096 byte
  {
    aximm_transaction(*PS_AxiMM_Rd_Util, *PS_AxiMM_Wr_Util,
                      xtlm::XTLM_WRITE_COMMAND, currentAddr, currentDataPtr,
                      4096);
    currentAddr += 4096;
    currentDataPtr += 4096;
    remaining -= 4096;
  }
  if (remaining > 0)
    aximm_transaction(*PS_AxiMM_Rd_Util, *PS_AxiMM_Wr_Util,
                      xtlm::XTLM_WRITE_COMMAND, currentAddr, currentDataPtr,
                      remaining);
  toggle_AIE_array_clk.notify(SC_ZERO_TIME);
}

void PSIP_ps_i3::readGM(uint64_t addr, void *data, uint64_t size) {
  toggle_AIE_array_clk.notify(1, SC_NS);
  uint64_t remaining = size;
  uint64_t currentAddr = addr;
  unsigned char *currentDataPtr = (unsigned char *)data;
  while (remaining >= 4096) // Axi maximum transaction size is 4096 byte
  {
    aximm_transaction(*PS_AxiMM_Rd_Util, *PS_AxiMM_Wr_Util,
                      xtlm::XTLM_READ_COMMAND, currentAddr, currentDataPtr,
                      4096);
    currentAddr += 4096;
    currentDataPtr += 4096;
    remaining -= 4096;
  }
  if (remaining > 0)
    aximm_transaction(*PS_AxiMM_Rd_Util, *PS_AxiMM_Wr_Util,
                      xtlm::XTLM_READ_COMMAND, currentAddr, currentDataPtr,
                      remaining);
  toggle_AIE_array_clk.notify(SC_ZERO_TIME);
}

void PSIP_ps_i3::aximm_transaction(
    xtlm::xtlm_aximm_initiator_rd_socket_util &rd_util,
    xtlm::xtlm_aximm_initiator_wr_socket_util &wr_util,
    xtlm::xtlm_command command, unsigned long long address,
    unsigned char *pData, unsigned int trans_size_in_bytes) {
  unsigned int trans_size_in_multiple_of_16bytes =
      (trans_size_in_bytes + 15) / 16;
  xtlm::aximm_payload *payload = mem_manager->get_payload();
  sc_core::sc_time time_delay = SC_ZERO_TIME;
  payload->acquire();
  payload->set_response_status(xtlm::XTLM_INCOMPLETE_RESPONSE);
  payload->set_command(command); // read or write
  payload->set_data_ptr(pData,
                        trans_size_in_bytes); // caller manages data memory
  payload->set_address(address);              // DDR address
  payload->set_burst_type(
      1); // 1:INCR to allow burst length to reach maximum 256
  payload->set_burst_length(
      trans_size_in_multiple_of_16bytes); // burst length, maximum 256
  payload->set_burst_size(16); // 128-bit (16-byte) burst size (buswidth)
  if (command == xtlm::XTLM_READ_COMMAND) {
    if (!rd_util.is_slave_ready())
      wait(rd_util.transaction_sampled);
    rd_util.send_transaction(*payload, time_delay);

    wait(rd_util.data_available);
    payload = rd_util.get_data();
  } else if (command == xtlm::XTLM_WRITE_COMMAND) {
    if (!wr_util.is_slave_ready())
      wait(wr_util.transaction_sampled);
    wr_util.send_transaction(*payload, time_delay);

    wait(wr_util.resp_available);
    payload = wr_util.get_resp();
  }
}

PSIP_ps_i3::PSIP_ps_i3(sc_module_name nm)
    : IPBlock(nm), toggle_AIE_array_clk("toggle_AIE_clk"),
      PS_AxiMM_Rd("ps_axi_rd", BUSWIDTH), PS_AxiMM_Wr("ps_axi_wr", BUSWIDTH) {
  std::cout << "IP-INFO: [" << basename() << "] IP loaded." << std::endl;
  PS_AxiMM_Rd_Util = new xtlm::xtlm_aximm_initiator_rd_socket_util(
      "PS_AxiMM_Util_rd_socket", xtlm::aximm::TRANSACTION, BUSWIDTH);
  PS_AxiMM_Wr_Util = new xtlm::xtlm_aximm_initiator_wr_socket_util(
      "PS_AxiMM_Util_wr_socket", xtlm::aximm::TRANSACTION, BUSWIDTH);
  mem_manager = new xtlm::xtlm_aximm_mem_manager(this);
  PS_AxiMM_Rd_Util->rd_socket.bind(PS_AxiMM_Rd);
  PS_AxiMM_Wr_Util->wr_socket.bind(PS_AxiMM_Wr);

  SC_THREAD(main_action);

  SC_THREAD(response_process);
  sensitive << (PS_AxiMM_Wr_Util->resp_available);
  sensitive << (PS_AxiMM_Rd_Util->data_available);
#ifdef _HYBRID_EMU_
  if (is_hybrid_emu()) {
    load_xcl_client_graph();
  }
#endif
};
void PSIP_ps_i3::main_action() {
  std::cout << "IP-INFO: [" << basename() << "] IP started." << std::endl;
#ifdef _HYBRID_EMU_
  if (is_hybrid_emu()) {
    xrt_ps_main();
  } else
#endif
  {
    ps_main();
  }
  ps_main_complete = 1;
}
void PSIP_ps_i3::set_payload_attr(xtlm::aximm_payload *trans,
                                  size_t transBytes) {
  trans->create_and_get_data_ptr(transBytes);
  unsigned char *byte_en_ptr =
      trans->create_and_get_byte_enable_ptr(transBytes);
  for (int i = 0; i < transBytes; i++) {
    byte_en_ptr[i] = 0xff;
  }
  trans->set_data_length(transBytes);
  // Data transfer for sequence of write/read transaction with same AWID/ARID
  // must be in order.
  trans->set_axi_id(0);
  // Entire payload transfer in 1-beat
  trans->set_burst_length(1);
  // No of bytes in 1 data beat
  trans->set_burst_size(transBytes);
  trans->set_burst_type(1); // INCR(by AWSIZE/ARSIZE)
}
void PSIP_ps_i3::response_process() {
  while (true) {
    wait(); // Waits for data available in read transaction, or for response
            // available in write transaction
    if (PS_AxiMM_Rd_Util->is_data_available()) {
      transRspAvail.notify(SC_ZERO_TIME);
    }
    if (PS_AxiMM_Wr_Util->is_resp_available()) {
      transRspAvail.notify(SC_ZERO_TIME);
    }
  }
}
extern "C" {
void ess_Write32(uint64 Addr, uint Data) {
  (PSIP_ps_i3::getInstance())->write32(Addr, Data);
}
uint32 ess_Read32(uint64 Addr) {
  return ((PSIP_ps_i3::getInstance())->read32(Addr));
}
void ess_Write128(uint64_t Addr, uint32_t *Data) {
  (PSIP_ps_i3::getInstance())->write128(Addr, Data);
}
void ess_Read128(uint64_t Addr, uint32_t *Data) {
  (PSIP_ps_i3::getInstance())->read128(Addr, Data);
}
void ess_WriteGM(uint64 addr, const void *data, uint64_t size) {
  (PSIP_ps_i3::getInstance())->writeGM(addr, data, size);
}
void ess_ReadGM(uint64 addr, void *data, uint64_t size) {
  (PSIP_ps_i3::getInstance())->readGM(addr, data, size);
}
IPBlock *create_ip(sc_module_name name) {
  return (PSIP_ps_i3::createInstance(name));
}
void destroy_ip(IPBlock *ip) {
  std::cout << "IP-INFO: deleting ip PSIP_ps_i3 " << std::endl;
  delete ip;
}
}
