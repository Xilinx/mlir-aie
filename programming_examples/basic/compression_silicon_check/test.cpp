// Silicon-level probe for AIE-ML BD Enable_Compression.
// Pumps an int32 vector through shim -> compute(0,2) -> shim with optional
// compression bits flipped via npu_maskwrite32 in the runtime sequence.
// Reports first-mismatch index + a short hex dump so we can see WHICH bytes
// changed across configs (the diagnostic signal lives in the structure of
// the divergence, not just pass/fail).

#include "cxxopts.hpp"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

int main(int argc, const char *argv[]) {
  cxxopts::Options options("compression_silicon_check");
  options.add_options()
    ("h,help", "produce help message")
    ("x,xclbin", "the input xclbin path", cxxopts::value<std::string>())
    ("k,kernel", "the kernel name in the XCLBIN",
        cxxopts::value<std::string>())
    ("v,verbosity", "verbosity (0..2)",
        cxxopts::value<int>()->default_value("0"))
    ("i,instr", "userspace npu instruction binary",
        cxxopts::value<std::string>())
    ("l,length", "transfer length in int32 elements",
        cxxopts::value<int>()->default_value("4096"))
    ("c,config", "label only: base|cmp_only|dcmp_only|both",
        cxxopts::value<std::string>()->default_value("base"));

  auto vm = options.parse(argc, argv);
  if (vm.count("help") || !vm.count("xclbin") || !vm.count("kernel") ||
      !vm.count("instr")) {
    std::cerr << options.help() << std::endl;
    return 1;
  }

  const int N = vm["length"].as<int>();
  const int verbosity = vm["verbosity"].as<int>();
  const std::string config = vm["config"].as<std::string>();

  // For "both" and "*_zero" suffix we use all-zero input (lossless under
  // any sparsity scheme, and maximally compressible so the compressor
  // would shrink the byte count if it's actually doing something).
  // For everything else, a recognizable ramp so corruption is visually
  // obvious in the hex dump.
  std::vector<uint32_t> srcVec(N);
  bool zero_input = (config == "both") ||
                    (config.size() >= 5 && config.compare(config.size() - 5, 5, "_zero") == 0);
  if (zero_input) {
    std::fill(srcVec.begin(), srcVec.end(), 0u);
  } else {
    for (int i = 0; i < N; i++) srcVec[i] = static_cast<uint32_t>(i);
  }

  auto instr_v = test_utils::load_instr_binary(vm["instr"].as<std::string>());

  auto device = xrt::device(0u);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  device.register_xclbin(xclbin);
  auto kernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(kernels.begin(), kernels.end(),
      [&](xrt::xclbin::kernel &k) {
        return k.get_name().rfind(vm["kernel"].as<std::string>(), 0) == 0;
      });
  xrt::hw_context ctx(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(ctx, xkernel.get_name());

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inA = xrt::bo(device, N * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_inB = xrt::bo(device, N * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out = xrt::bo(device, N * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  // Sentinel-fill the output so we can see which slots were untouched
  // (positive signal for early-truncation when compress > decompress
  // bytes, or for hangs that partially completed).
  uint32_t *bufOut = bo_out.map<uint32_t *>();
  const uint32_t SENTINEL = 0xDEADBEEFu;
  for (int i = 0; i < N; i++) bufOut[i] = SENTINEL;

  int32_t *bufInA = bo_inA.map<int32_t *>();
  std::memcpy(bufInA, srcVec.data(), N * sizeof(uint32_t));

  void *bufInstr = bo_instr.map<void *>();
  std::memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::cout << "[" << config << "] running, N=" << N << "..." << std::flush;
  auto t0 = std::chrono::steady_clock::now();
  bool tdr = false;
  try {
    auto run = kernel(3u, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_out);
    run.wait();
  } catch (const std::exception &e) {
    tdr = true;
    std::cout << "\n[" << config << "] EXCEPTION (likely TDR): " << e.what()
              << std::endl;
  }
  auto t1 = std::chrono::steady_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  if (!tdr) std::cout << " done in " << ms << "ms" << std::endl;

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Scan output and classify
  int matches = 0, mismatches = 0, sentinels = 0;
  int first_mismatch = -1;
  for (int i = 0; i < N; i++) {
    uint32_t expected = srcVec[i];
    uint32_t got = bufOut[i];
    if (got == SENTINEL) {
      sentinels++;
    } else if (got == expected) {
      matches++;
    } else {
      mismatches++;
      if (first_mismatch < 0) first_mismatch = i;
    }
  }

  std::cout << "[" << config << "] result:"
            << " matches=" << matches
            << " mismatches=" << mismatches
            << " untouched(sentinel)=" << sentinels
            << " (TDR=" << (tdr ? "yes" : "no") << ")"
            << std::endl;
  if (first_mismatch >= 0) {
    std::cout << "[" << config << "] first divergence at i=" << first_mismatch
              << ":" << std::endl;
    std::cout << "  in [i..i+8]:";
    for (int k = 0; k < 8 && first_mismatch + k < N; k++)
      std::printf(" %08x", srcVec[first_mismatch + k]);
    std::cout << std::endl;
    std::cout << "  out[i..i+8]:";
    for (int k = 0; k < 8 && first_mismatch + k < N; k++)
      std::printf(" %08x", bufOut[first_mismatch + k]);
    std::cout << std::endl;
  }
  // First slot of output (always informative)
  std::cout << "[" << config << "] out[0..8]:";
  for (int k = 0; k < 8; k++) std::printf(" %08x", bufOut[k]);
  std::cout << std::endl;

  return tdr ? 2 : (mismatches == 0 && sentinels == 0 ? 0 : 1);
}
