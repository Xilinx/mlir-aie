//===- golden_reference_verification.h -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This file contains verification functions that use PyTorch-generated golden reference values instead of host-side computation

#ifndef GOLDEN_REFERENCE_VERIFICATION_H
#define GOLDEN_REFERENCE_VERIFICATION_H

#include "common.h"
#include "golden_reference.h"
#include <algorithm>
#include <optional>
#include <vector>

namespace golden_reference_verification {


template <typename Tin, typename Tout, typename Tacc>
int verify_against_golden(const std::vector<Tout>& actual_O, int verbosity = 0, 
                         float abs_tol = 0.05, float rel_tol = 0.05) {
    
    // Check dimensions match
    if (actual_O.size() != golden_reference::HEADS * golden_reference::S_q * golden_reference::d) {
        std::cerr << "Error: Output size mismatch. Expected " 
                  << golden_reference::HEADS * golden_reference::S_q * golden_reference::S_kv 
                  << " but got " << actual_O.size() << std::endl;
        return -1;
    }
    
    int n_errors = 0;
    float average_error = 0.0f;
    Tin max_abs_error = 0;
    Tin min_abs_error = std::numeric_limits<Tin>::max();

    std::vector<matmul_common::error<Tout>> errors;
    Tout max_rel_error = (Tout)0.0f;

    for (int head = 0; head < golden_reference::HEADS; head++) {
        for (int row = 0; row < golden_reference::S_q; row++) {
            for (int col = 0; col < golden_reference::d; col++) {

                int idx = (head * golden_reference::S_q * golden_reference::d) + (row * golden_reference::d) + col;

                Tout expected = (Tout)golden_reference::O[idx];
                Tout actual = actual_O[idx];

                average_error += std::abs(actual - expected);
                max_abs_error = std::max(max_abs_error, std::abs(actual - expected));
                min_abs_error = std::min(min_abs_error, std::abs(actual - expected));

                std::optional<matmul_common::error<Tout>> error =
                    matmul_common::verify_single(std::cout, head, row, col, expected, actual, 
                                            abs_tol, rel_tol);
                
                if (error.has_value()) {

                    if (n_errors < matmul_common::max_printable_errors) {
                        errors.push_back(*error);
                    }

                    Tout rel_error = std::abs(error->actual - error->expected) / std::max(std::abs(error->actual), std::abs(error->expected));
                    
                    if (rel_error > max_rel_error) {
                        max_rel_error = rel_error;
                    }
                    n_errors++;
                }
            }
        }
    }
    average_error /= actual_O.size();

    std::cout << "Absolute tolerence: " << abs_tol << std::endl;
    std::cout << "Relative tolerence: " << rel_tol << std::endl;
    std::cout << "\nAverage relative error: " << average_error << std::endl;
    std::cout << "Max absolute error: " << max_abs_error << std::endl;
    std::cout << "Min absolute error: " << min_abs_error << std::endl << std::endl;
    
    matmul_common::print_error_summary(std::cout, n_errors, golden_reference::HEADS * golden_reference::S_q * golden_reference::d, errors, max_rel_error);
    
    if (n_errors > -1 && verbosity >= 1) {
        std::cout << std::endl << "Golden Reference:" << std::endl;
        std::vector<Tout> golden_vec(golden_reference::O.begin(), golden_reference::O.end());
        matmul_common::print_matrix(golden_vec, golden_reference::d, 32, 16);

        std::cout << std::endl << "Actual Output:" << std::endl;
        matmul_common::print_matrix(actual_O, golden_reference::d, 32, 16);

        std::cout << std::endl << "Difference:" << std::endl;
        std::vector<Tout> diff(actual_O.size());

        for (int i = 0; i < actual_O.size(); i++) {
            diff[i] = golden_vec[i] - actual_O[i];
        }
        matmul_common::print_matrix(diff, golden_reference::d, 32, 16, std::cout, " | ", " ... ", 6);
    }
    
    return n_errors;
}

// Load input matrices from golden references
template <typename Tin>
void load_golden_inputs(std::vector<Tin>& Q, std::vector<Tin>& K, std::vector<Tin>& V) {
    
    // Copy from golden reference arrays
    for (size_t i = 0; i < golden_reference::Q.size(); i++) {
        Q[i] = (Tin)golden_reference::Q[i];
    }
    for (size_t i = 0; i < golden_reference::K.size(); i++) {
        K[i] = (Tin)golden_reference::K[i];
    }
    for (size_t i = 0; i < golden_reference::V.size(); i++) {
        V[i] = (Tin)golden_reference::V[i];
    }
}

// Print golden reference info
void print_golden_reference_info() {
    std::cout << "Golden Reference Info:" << std::endl;
    std::cout << "  MHA dimensions: " << std::endl;
    std::cout << "- Heads " << golden_reference::HEADS << std::endl; 
    std::cout << "- Sequence Length Q " << golden_reference::S_q << std::endl; 
    std::cout << "- Sequence Length KV " << golden_reference::S_kv << std::endl; 
    std::cout << "- Embedding Dimension " << golden_reference::d << std::endl; 
    std::cout << "Generated by PyTorch" << std::endl;
}

} // namespace golden_reference_verification

#endif // GOLDEN_REFERENCE_VERIFICATION_H
