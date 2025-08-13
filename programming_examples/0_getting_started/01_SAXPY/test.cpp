// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 AMD Inc.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <stdfloat>

using INPUT_DATATYPE = std::bfloat16_t;
constexpr unsigned tensor_size = 4096;

void reference(INPUT_DATATYPE *A, INPUT_DATATYPE *B, INPUT_DATATYPE *C) {  
    for(int i = 0; i < tensor_size; i++) {
        C[i] = 3.141f * static_cast<float>(A[i]) + static_cast<float>(B[i]);    
    }   
}

int read_insts(std::string insts_path, std::vector<char>& into) {
	std::ifstream insts_file(insts_path, std::ios::binary);
	if (!insts_file) {
		return 1;
	}
	into.assign((std::istreambuf_iterator<char>(insts_file)), std::istreambuf_iterator<char>());
	return 0;
}

int main(int argc, const char *argv[]) {
	// Assign command line arguments
	if (argc != 4) {
		std::cerr << "Usage: " << argv[0] << " <xclbin> <insts> <kernel>" << std::endl;
		return 1;
	}
	std::string xclbin_path = argv[1];
	std::string insts_path = argv[2];
	std::string kernel_name = argv[3];

	// Read insts.bin file (instructions to the NPU's command processor)
	std::vector<char> insts = {};
	if (0 != read_insts(insts_path, insts)) {
		std::cerr << "Unable to open insts file: " << insts_path << std::endl;
		return 1;
	}

	// Initialize the NPU and load our design
    constexpr unsigned device_index = 0;
    xrt::device device = xrt::device(device_index);
    xrt::xclbin xclbin(xclbin_path);
    device.register_xclbin(xclbin);
    xrt::hw_context context(device, xclbin.get_uuid());
    xrt::kernel kernel = xrt::kernel(context, kernel_name);

	// Initialzie input/output XRT buffers
    xrt::bo bo_insts = xrt::bo(device, insts.size() * sizeof(insts[0]), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    xrt::bo bo_a = xrt::bo(device, tensor_size * sizeof(INPUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    xrt::bo bo_b = xrt::bo(device, tensor_size * sizeof(INPUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    xrt::bo bo_c = xrt::bo(device, tensor_size * sizeof(INPUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    char *buf_insts = bo_insts.map<char *>();
    std::copy(insts.begin(), insts.end(), buf_insts);
    INPUT_DATATYPE *buf_a = bo_a.map<INPUT_DATATYPE *>();
    INPUT_DATATYPE *buf_b = bo_b.map<INPUT_DATATYPE *>();
    INPUT_DATATYPE *buf_c = bo_c.map<INPUT_DATATYPE *>();

	// Prepare input data (initialize random matrices) and sync to NPU
	std::generate(buf_a, buf_a + tensor_size, []() { return rand() % 256; });
	std::generate(buf_b, buf_b + tensor_size, []() { return rand() % 256; });
	std::fill(buf_c, buf_c + tensor_size, 0);
    bo_insts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

	// Run our design
    auto t_start = std::chrono::system_clock::now();
    constexpr unsigned opcode = 3;
    auto run = kernel(opcode, bo_insts, insts.size(), bo_a, bo_b, bo_c);
    ert_cmd_state r = run.wait();
    auto t_stop = std::chrono::system_clock::now();
    if (r != ERT_CMD_STATE_COMPLETED) {
        std::cout << "Kernel did not complete. Returned status: " << r << std::endl;
        return 1;
    }
    float t_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t_stop - t_start).count();
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

	// // Print elapsed time
	// constexpr unsigned n_ops = M * K * N * 2;
  //   float throughput = n_ops / t_elapsed / 1e3;  // GOP/s
  //   std::cout << "Elapsed: " << t_elapsed << " us "
  //             << "(" << throughput << " GOP/s)" << std::endl;

	// Validate correctness of output
	INPUT_DATATYPE *ref_c = static_cast<INPUT_DATATYPE *>(std::malloc(tensor_size * sizeof(INPUT_DATATYPE))); // reference output calculated on the CPU
	reference(buf_a, buf_b, ref_c);

	if (std::equal(ref_c, ref_c + tensor_size, buf_c)) {
        std::cout << "PASS!" << std::endl;
    }   else {
        std::cout << "FAIL." << std::endl;
        return 1;
    }

    return 0;
}
