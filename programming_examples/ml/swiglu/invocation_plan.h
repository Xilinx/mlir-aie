// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 AMD Inc.

// A version of the InvocationPlan API provided by this file will be made 
// available in future versions of the test_utils lib. For now, we include
// it along with this example to show how to invoke the xrt::runlist.

#ifndef INVOCATION_PLAN_H
#define INVOCATION_PLAN_H

#include <vector>
#include <map>
#include <set>
#include <stdfloat>
#include <chrono>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include "test_utils.h"

#ifndef DTYPE
#define DTYPE std::bfloat16_t
#endif

// --------------------------------------------------------------------------
// Info classes
// --------------------------------------------------------------------------
// These describe the metadata of their counterparts without concretely 
// instantiating them.

struct KernelInfo {
    std::string name; // Name to be used in InvocationPlan
    std::string insts; // Path to instruction sequence to run for invocations of this kernel; if you need different insts.bins for the same kernel, create multiple KernelInfos with different names
    std::string xclbin_name; // Name as defined in the xclbin (aiecc.py --xclbin-kernel-name=XX); can be omitted if the same as 'name'
};

struct KernelBufferInfo {
    enum Direction {
        IN = 0b01,
        OUT = 0b10,
        INOUT = 0b11
    };
    std::string name;
    size_t size;
    Direction direction = Direction::INOUT;
    const DTYPE *reference = nullptr;
};

struct KernelInvocationInfo {
    std::string name;
    std::vector<std::string> args;
};

// Details about kernels, buffers needed by those kernels, and an ordered execution of the kernels.
struct InvocationPlanInfo {
    std::string xclbin;
    std::vector<KernelInfo> kernels;
    std::vector<KernelBufferInfo> buffers;
    std::vector<KernelInvocationInfo> runlist;

    void validate() const;
};


// --------------------------------------------------------------------------
// Concrete classes
// --------------------------------------------------------------------------
// These are built from based on information from the Info classes above.

struct Kernel {
    xrt::kernel kernel;
    std::vector<uint32_t> insts;
    xrt::bo insts_bo;
    uint32_t *insts_buf;
};

// Details about a buffer that one or multiple kernels needs as input or output.
struct KernelBuffer {
    xrt::bo bo;
    size_t size;
    DTYPE *buf = nullptr;
    KernelBufferInfo::Direction direction = KernelBufferInfo::Direction::INOUT;
    const DTYPE *reference = nullptr;
};

struct KernelInvocation {
    //xrt::run run;
    Kernel *kernel;
    std::vector<KernelBuffer *> args;
};

struct InvocationResult {
    bool success;
    float elapsed_time;
};

// A concrete set up run of an InvocationPlan, with initialized buffers, kernels, etc.
struct InvocationPlan {
    xrt::device &device;
    xrt::xclbin &xclbin;
    xrt::hw_context &context;
    xrt::runlist runlist;
    std::map<std::string, Kernel> kernels;
    std::map<std::string, KernelBuffer> buffers;
    std::vector<KernelInvocation> invocations;

    static InvocationPlan fromInfo(const InvocationPlanInfo &info, xrt::device &device, xrt::xclbin &xclbin, xrt::hw_context &context);
    struct InvocationResult invoke();
    std::vector<std::pair<std::string, unsigned>> verifyOutputBuffers(float epsilon, float abs_th) const;

private:
    void setupContext(const std::string &xclbin_path);
    void setupKernels(const std::vector<KernelInfo> &kernels);
    void setupBuffers(const std::vector<KernelBufferInfo> &buffers, const std::vector<KernelInvocationInfo> &invocations);
    void setupRunlist(const std::vector<KernelInvocationInfo> &invocations);
};


// --------------------------------------------------------------------------
// InvocationPlanInfo
// --------------------------------------------------------------------------

void InvocationPlanInfo::validate() const {
    std::set<std::string> kernel_names;
    for(const KernelInfo &kernel : kernels) {
        if(kernel.name.empty()) {
            throw std::invalid_argument("Kernel name cannot be empty");
        }
        if(kernel_names.count(kernel.name) > 0) {
            throw std::invalid_argument("Duplicate kernel name: " + kernel.name);
        }
        kernel_names.insert(kernel.name);
    }

    std::set<std::string> buffer_names;
    for(const KernelBufferInfo &buffer : buffers) {
        if(buffer.name.empty()) {
            throw std::invalid_argument("Buffer name cannot be empty");
        }
        if(buffer_names.count(buffer.name) > 0) {
            throw std::invalid_argument("Duplicate buffer name: " + buffer.name);
        }
        buffer_names.insert(buffer.name);
        if(buffer.size == 0) {
            throw std::invalid_argument("Buffer size cannot be zero: " + buffer.name);
        }
    }

    for(const KernelInvocationInfo &invocation : runlist) {
        if(invocation.name.empty()) {
            throw std::invalid_argument("Invocation kernel name cannot be empty");
        }
        if(kernel_names.count(invocation.name) == 0) {
            throw std::invalid_argument("Invocation references unknown kernel: " + invocation.name);
        }
        for(const std::string &arg : invocation.args) {
            if(arg.empty()) {
                throw std::invalid_argument("Invocation argument name cannot be empty");
            }
            if(buffer_names.count(arg) == 0) {
                throw std::invalid_argument("Invocation references unknown buffer: " + arg);
            }
        }
    }
}


// --------------------------------------------------------------------------
// InvocationPlan
// --------------------------------------------------------------------------

InvocationPlan InvocationPlan::fromInfo(const InvocationPlanInfo &info, xrt::device &device, xrt::xclbin &xclbin, xrt::hw_context &context) {
    info.validate();
    InvocationPlan ret = {device, xclbin, context, {}, {}, {}, {}};
    ret.setupKernels(info.kernels);
    ret.setupBuffers(info.buffers, info.runlist);
    ret.setupRunlist(info.runlist);
    return ret;
}

void InvocationPlan::setupContext(const std::string &xclbin_path) {
  xclbin = xrt::xclbin(xclbin_path);
  device.register_xclbin(xclbin);
  context = xrt::hw_context(device, xclbin.get_uuid());
}

void InvocationPlan::setupKernels(const std::vector<KernelInfo> &kernel_infos) {
    for (const KernelInfo &info : kernel_infos) {
        Kernel &new_kernel = kernels[info.name];
        const std::string &xclbin_name = (info.xclbin_name.empty() ? info.name : info.xclbin_name);
        new_kernel.kernel = xrt::kernel(context, xclbin_name);
        new_kernel.insts = test_utils::load_instr_binary(info.insts);
        new_kernel.insts_bo = xrt::bo(device, new_kernel.insts.size() * sizeof(new_kernel.insts[0]), XCL_BO_FLAGS_CACHEABLE, new_kernel.kernel.group_id(1));
        new_kernel.insts_buf = new_kernel.insts_bo.map<uint32_t *>();
        std::copy(new_kernel.insts.begin(), new_kernel.insts.end(), new_kernel.insts_buf);
    }
}

void InvocationPlan::setupBuffers(const std::vector<KernelBufferInfo> &buffer_infos, const std::vector<KernelInvocationInfo> &invocation_infos) {
    // Map buffers to which kernels use them; this will be needed to get the correct group_id
    std::map<std::string, std::vector<std::pair<std::string, unsigned>>> buffer_users;
    for (const KernelInvocationInfo &invocation_info : invocation_infos) {
        unsigned arg_index = 0;
        for (const std::string &arg : invocation_info.args) {
            buffer_users[arg].push_back(std::make_pair(invocation_info.name, arg_index));
            arg_index++;
        }
    }

    // Allocate buffers
    for (const KernelBufferInfo &buffer_info : buffer_infos) {
        const std::string &buffer_name = buffer_info.name;
        if(0 == buffer_users[buffer_name].size()) {
            continue; // Don't allocate buffers  that aren't used
        }
        std::pair<std::string, uint32_t> first_user = buffer_users[buffer_name][0];
        const Kernel &first_user_kernel = kernels[first_user.first];
        int group_id = first_user_kernel.kernel.group_id(first_user.second + 3);  // +3 to account for insts buffer and opcode
        for(auto &[user_kernel_name, arg_index] : buffer_users[buffer_name]) {
            Kernel &user_kernel = kernels[user_kernel_name];
            int user_group_id = user_kernel.kernel.group_id(arg_index + 3);
            if(user_group_id != group_id) {
                std::stringstream ss;
                ss << "Error: buffer " << buffer_name << " used by multiple kernels with different group IDs" << std::endl;
                ss << "Kernel " << first_user.first << " requires group id " << group_id << std::endl;
                ss << "Kernel " << user_kernel_name << " requires group id " << user_group_id << std::endl;
                throw std::invalid_argument(ss.str());
            }
        }

        KernelBuffer &new_buffer = buffers[buffer_name];
        new_buffer.bo = xrt::bo(
            device, 
            buffer_info.size * sizeof(DTYPE), 
            XCL_BO_FLAGS_HOST_ONLY, 
            group_id
        );
        new_buffer.size = buffer_info.size;
        new_buffer.buf = new_buffer.bo.map<DTYPE *>();
        new_buffer.direction = buffer_info.direction;
        new_buffer.reference = buffer_info.reference;
        if(new_buffer.reference && (buffer_info.direction & KernelBufferInfo::Direction::IN)) {
            memcpy(new_buffer.buf, new_buffer.reference, new_buffer.size * sizeof(DTYPE));
        } else {
            memset(new_buffer.buf, 0xDEAD, new_buffer.size * sizeof(DTYPE));
        }
    }
}

void InvocationPlan::setupRunlist(const std::vector<KernelInvocationInfo> &invocation_infos) {
    runlist = xrt::runlist(context);
    constexpr unsigned opcode = 3;
    for(auto &[kernel_name, kernel] : kernels) {
        kernel.insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    for(auto &[buffer_name, buffer] : buffers) {
        buffer.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    for (const KernelInvocationInfo &info : invocation_infos) {
        Kernel &kernel = kernels[info.name];
        xrt::run run = xrt::run(kernel.kernel);
        run.set_arg(0, opcode);
        run.set_arg(1, kernel.insts_bo);
        run.set_arg(2, kernel.insts.size());
        unsigned arg_index = 3;
        for(const std::string &arg_name : info.args) {
            run.set_arg(arg_index, buffers[arg_name].bo);
            arg_index++;
        }
        runlist.add(run);
    }
}

InvocationResult InvocationPlan::invoke() {
    auto t_start = std::chrono::system_clock::now();
    runlist.execute();
    runlist.wait();
    auto t_stop = std::chrono::system_clock::now();
    for(auto &[buffer_name, buffer] : buffers) {
        buffer.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }
    float t_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t_stop - t_start).count();
    return {true, t_elapsed};
}

std::vector<std::pair<std::string, unsigned>> InvocationPlan::verifyOutputBuffers(float epsilon, float abs_th) const {
    std::vector<std::pair<std::string, unsigned>> errors;
    for(const auto &[buffer_name, buffer] : buffers) {
        if(!(buffer.direction & KernelBufferInfo::Direction::OUT) || !buffer.reference) {
            continue;
        }
        for(unsigned i = 0; i < buffer.size; i++) {
            if(!test_utils::nearly_equal(static_cast<float>(buffer.buf[i]), static_cast<float>(buffer.reference[i]), epsilon, abs_th)) {
                errors.emplace_back(buffer_name, i);
            }
        }
    }
    return errors;
}

#endif