//
// Created by mlevental on 5/15/24.
//

#ifndef AIE_HWCTX_H
#define AIE_HWCTX_H

#include "core/common/device.h"

using namespace xrt_core;
std::string program;
bool printing_on;

void handle_eptr(std::exception_ptr eptr) // passing by value is ok
{
  try {
    if (eptr)
      std::rethrow_exception(eptr);
  } catch (const std::exception &e) {
    std::cout << "Caught exception: '" << e.what() << "'\n";
  }
}

class hw_ctx {
public:
  hw_ctx(device *dev, std::string xclbinpath_) {
    hw_ctx_init(dev, xclbinpath_);
  }

  ~hw_ctx() {}

  xrt_core::hwctx_handle *get() { return m_handle.get(); }

private:
  device *m_dev;
  std::unique_ptr<xrt_core::hwctx_handle> m_handle;

  void hw_ctx_init(device *dev, std::string xclbin_path) {
    xrt::xclbin xclbin;
    std::exception_ptr eptr;
    try {
      xclbin = xrt::xclbin(xclbin_path);
    } catch (...) {
      eptr = std::current_exception();
      handle_eptr(eptr);
      throw std::runtime_error(xclbin_path +
                               " not found?\n"
                               "specify xclbin path or run \"build.sh "
                               "-xclbin_only\" to download them");
    }
    dev->record_xclbin(xclbin);
    auto xclbin_uuid = xclbin.get_uuid();
    xrt::hw_context::qos_type qos{{"gops", 100}};
    xrt::hw_context::access_mode mode = xrt::hw_context::access_mode::shared;

    m_handle = dev->create_hw_context(xclbin_uuid, qos, mode);
    m_dev = dev;
    if (printing_on)
      std::cout << "loaded " << xclbin_path << std::endl;
  }
};

uint64_t get_bo_flags(uint32_t flags, uint32_t ext_flags) {
  xcl_bo_flags f = {};

  f.flags = flags;
  f.extension = ext_flags;
  return f.all;
}

#endif // AIE_HWCTX_H
