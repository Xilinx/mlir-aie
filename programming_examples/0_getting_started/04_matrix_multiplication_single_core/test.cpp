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

constexpr unsigned M = 32;
constexpr unsigned K = 32;
constexpr unsigned N = 32;

constexpr unsigned buf_alignment = 4096;

void reference(int16_t *A, int16_t *B, int16_t *C) {
    for(int row = 0; row < M; row++) {
        for(int col = 0; col < N; col++) {
            int16_t acc = 0;
            for(int i = 0; i < K; i++) {
                acc += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = acc;
        }
    }   
}

void print_matrix(unsigned n_rows, unsigned n_cols, int16_t *matrix) {
    for (unsigned row = 0; row < n_rows; row++) {
        for (unsigned col = 0; col < n_cols; col++) {
            std::cout << std::setw(6) << matrix[row * n_cols + col] << " ";
        }
        std::cout << std::endl;
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
    constexpr unsigned size_a = M * K;
    constexpr unsigned size_b = K * N;
    constexpr unsigned size_c = M * N;
    xrt::bo bo_insts = xrt::bo(device, insts.size() * sizeof(insts[0]), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    xrt::bo bo_a = xrt::bo(device, size_a * sizeof(int16_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    xrt::bo bo_b = xrt::bo(device, size_b * sizeof(int16_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    xrt::bo bo_c = xrt::bo(device, size_c * sizeof(int16_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    char *buf_insts = bo_insts.map<char *>();
    std::copy(insts.begin(), insts.end(), buf_insts);
    int16_t *buf_a = bo_a.map<int16_t *>();
    int16_t *buf_b = bo_b.map<int16_t *>();
    int16_t *buf_c = bo_c.map<int16_t *>();

	// Prepare input data (initialize random matrices) and sync to NPU
	std::generate(buf_a, buf_a + size_a, []() { return rand() % 256; });
	std::generate(buf_b, buf_b + size_b, []() { return rand() % 256; });
	std::fill(buf_c, buf_c + size_c, 0);
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

	// Print elapsed time
	constexpr unsigned n_ops = M * K * N * 2;
    float throughput = n_ops / t_elapsed / 1e3;  // GOP/s
    std::cout << "Elapsed: " << t_elapsed << " us "
              << "(" << throughput << " GOP/s)" << std::endl;

	// Validate correctness of output
	int16_t *ref_c = static_cast<int16_t *>(std::malloc(M * N * sizeof(int16_t))); // reference output calculated on the CPU
	reference(buf_a, buf_b, ref_c);

    std::cout << std::endl;
    std::cout << "Reference:" << std::endl;
    print_matrix(M, N, ref_c);
    std::cout << std::endl;
    std::cout << "Output:" << std::endl;
    print_matrix(M, N, buf_c);

	if (std::equal(ref_c, ref_c + size_c, buf_c)) {
        std::cout << "PASS!" << std::endl;
    }   else {
        std::cout << "FAIL." << std::endl;
        return 1;
    }

    return 0;
}
