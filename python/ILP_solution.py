#!/usr/bin/env python3
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.
# ===============================================================================#
# This file implements an experimental ILP solver for
# automatic tiling space exploration.
# ===============================================================================#

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time

# -------------------------------------------------------------------------------#
# Algorithmic parameters
# -------------------------------------------------------------------------------#

# The list of loop bounds
loop_bounds = [64, 64, 64]

# The constant matrix that reflects how data tensors are related with
# loop induction variables
# +--------------------+
# |     | L0 | L1 | L2 |
# +--------------------+
# | in1 | 1  | 0  | 1  |
# +--------------------+
# | in2 | 0  | 1  | 1  |
# +--------------------+
# | out | 1  | 1  | 0  |
# +--------------------+
tensor_IV = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]

# -------------------------------------------------------------------------------#
# Architectural parameters
# -------------------------------------------------------------------------------#

# In AIE, we typically have three architectural (memory/compute) hierarchy levels.
# L3->L2 copies data from L3 memory to L2 shared cache. L2->L1 copies data from
# L2 cache to L1 private cache. L2->L1 also indicates the transition from
# temporal to spatial execution. L1 indicates the transition from spatial
# to temporal task on each compute core.
mem_levels = 3

# memory capacity for L3, L2, L1
mem_capacity = [["L3", 2**20], ["L2", 2**16], ["L1", 2**11]]

# The ratios according to which the memory spaces are allocated for each data
# tensor, ignoring the L3 level. For example, [0.3, 0.3, 0.4] means 30% of
# memory space is reserved for two input tensors, and 40% of memory space is
# estimated to store the output tensor.
# L2, L1: [in1, in2, out]
mem_ratios = [[0.3, 0.3, 0.4], [0.3, 0.3, 0.4]]

# memory bandwidth for L3, L2, L1
# recalculated as log(data_size/#cycles)
mem_bandwidth = [["L3-L2", 2**30], ["L2-L1", 2 * 2**30]]

# frequency
freq = 600 * 10**6

# compute cores of which L2 is in charge
spatial_dim = [8, 8]

# -------------------------------------------------------------------------------#
# ILP formulation
# -------------------------------------------------------------------------------#


def prime_factorize(loop_bounds):
    """Factorize the original loops bounds into a list of prime factors.
    Input: a list of loop bounds
    Output: a super-list of prime factor lists
    """
    prime_factor_list = []
    for loop_bound in loop_bounds:
        prime_factors = []
        while loop_bound % 2 == 0:
            prime_factors.append(2)
            loop_bound /= 2
        if loop_bound > 3:
            for i in range(3, loop_bound, 2):
                while loop_bound % i == 0:
                    prime_factors.append(i)
                    loop_bound /= i
        if loop_bound > 2:
            prime_factors.append(loop_bound)
        prime_factor_list.append(prime_factors)
    return prime_factor_list


def ILP_formulation(util_factor=0.5, compute_factor=1, traffic_factor=0.2):
    # Create a new model
    m = gp.Model("loopnest")

    # Initialize inputs
    prime_factor_list = prime_factorize(loop_bounds)
    prime_factor_len = sum([len(x) for x in prime_factor_list])

    # Create decision variables
    ## Binary decision variables - from factorized subloops to a specific
    ## loop order
    x = {}
    for f1, pf_list in enumerate(prime_factor_list):
        for f2, prime_factors in enumerate(pf_list):
            for p in range(prime_factor_len):
                var_name = "X_{}_{}_{}".format(f1, f2, p)
                x[(f1, f2, p)] = m.addVar(vtype=GRB.BINARY, name=var_name)

    ## Integer decision variables
    ## Y[(0, p)]: L3 mem, Y[(1, p)]: L2 mem, Y[(2, p)]: L1 mem
    y = {}
    for l in range(mem_levels):
        for p in range(prime_factor_len):
            var_name = "Y_{}_{}".format(l, p)
            y[(l, p)] = m.addVar(lb=0, ub=1, vtype=GRB.INTEGER, name=var_name)

    # Add Constraints
    ## One prime factor subloop has one assignment
    for f1, pf_list in enumerate(prime_factor_list):
        for f2, prime_factors in enumerate(pf_list):
            col_sum = 0
            for p in range(prime_factor_len):
                col_sum += x[(f1, f2, p)]
            m.addConstr(col_sum == 1, "col_sum_{}_{}".format(f1, f2))

    ## One ordering slot has only one subloop
    for p in range(prime_factor_len):
        row_sum = 0
        for f1, pf_list in enumerate(prime_factor_list):
            for f2, prime_factors in enumerate(pf_list):
                row_sum += x[(f1, f2, p)]
        m.addConstr(row_sum == 1, "row_sum_{}".format(p))

    ## monotone non-decreasing
    for l in range(mem_levels):
        for p in range(prime_factor_len - 1):
            m.addConstr(y[(l, p)] <= y[(l, p + 1)], "y_leq_{}_{}".format(l, p))

    ## L3 region > L2 region > L1 region
    L2_region = 0
    L1_region = 0
    for p in range(prime_factor_len):
        L2_region += y[(0, p)] - y[(1, p)]
        L1_region += y[(1, p)] - y[(2, p)]
    m.addConstr(L2_region >= 1, "y_l2_region")
    m.addConstr(L1_region >= 2, "y_l1_region")

    ## memory capacity
    ### L2: accommodates all tensors within L2 region
    L2_util = {}
    for v, iv_map in enumerate(tensor_IV):
        L2_util[v] = 0
        for f1, pf_list in enumerate(prime_factor_list):
            for f2, prime_factors in enumerate(pf_list):
                for p in range(prime_factor_len):
                    L2_util[v] += (
                        tensor_IV[v][f1]
                        * np.log2(prime_factor_list[f1][f2])
                        * x[(f1, f2, p)]
                        * y[(0, p)]
                    )
        v_available = mem_capacity[1][1] * mem_ratios[0][v]
        m.addConstr(L2_util[v] <= np.log2(v_available), "mem_capacity_L2_{}".format(v))

    ### L1: accommodates all tensors mapped to temporal dimension within L1 region
    L1_util = {}
    for v, iv_map in enumerate(tensor_IV):
        L1_util[v] = 0
        for f1, pf_list in enumerate(prime_factor_list):
            for f2, prime_factors in enumerate(pf_list):
                for p in range(prime_factor_len):
                    L1_util[v] += (
                        tensor_IV[v][f1]
                        * np.log2(prime_factor_list[f1][f2])
                        * x[(f1, f2, p)]
                        * y[(2, p)]
                    )
        v_available = mem_capacity[2][1] * mem_ratios[1][v]
        m.addConstr(L1_util[v] <= np.log2(v_available), "mem_capacity_L1_{}".format(v))

    ## memory bandwidth - the amount of data copy size in a unit of time
    ## #Data_size_that_requires_moving / #compute_cycles
    ### L3->L2 traffic
    L3_L2_tensor_traffic = {}
    data_L2 = L2_util
    cycles_L2 = 0
    for f1, pf_list in enumerate(prime_factor_list):
        for f2, prime_factors in enumerate(pf_list):
            for p in range(prime_factor_len):
                cycles_L2 += (
                    np.log2(prime_factor_list[f1][f2])
                    * x[(f1, f2, p)]
                    * (y[(0, p)] - y[(1, p)] + y[(2, p)])
                )
    for v, iv_map in enumerate(tensor_IV):
        bw_log = (
            np.log2(mem_bandwidth[0][1]) + np.log2(mem_ratios[0][v]) - np.log2(freq)
        )
        L3_L2_tensor_traffic[v] = data_L2[v] - cycles_L2
        m.addConstr(L3_L2_tensor_traffic[v] <= bw_log, "L2_bandwidth_{}".format(v))
    L3_L2_traffic = sum(data_L2.values()) - cycles_L2

    ### L2->L1 traffic
    L2_L1_tensor_traffic = {}
    data_L1 = {}
    cycles_L1 = 0
    for v, iv_map in enumerate(tensor_IV):
        data_L1[v] = 0
        for f1, pf_list in enumerate(prime_factor_list):
            for f2, prime_factors in enumerate(pf_list):
                for p in range(prime_factor_len):
                    data_L1[v] += (
                        tensor_IV[v][f1]
                        * np.log2(prime_factor_list[f1][f2])
                        * x[(f1, f2, p)]
                        * y[(1, p)]
                    )
    for f1, pf_list in enumerate(prime_factor_list):
        for f2, prime_factors in enumerate(pf_list):
            for p in range(prime_factor_len):
                cycles_L1 += (
                    np.log2(prime_factor_list[f1][f2]) * x[(f1, f2, p)] * y[(2, p)]
                )
    for v, iv_map in enumerate(tensor_IV):
        bw_log = (
            np.log2(mem_bandwidth[1][1]) + np.log2(mem_ratios[1][v]) - np.log2(freq)
        )
        L2_L1_tensor_traffic[v] = data_L1[v] - cycles_L1
        m.addConstr(L2_L1_tensor_traffic[v] <= bw_log, "L1_bandwidth_{}".format(v))
    L2_L1_traffic = sum(data_L1.values()) - cycles_L1

    ## spatial resource limitation
    spatial_tile = 0
    for f1, pf_list in enumerate(prime_factor_list):
        for f2, prime_factors in enumerate(pf_list):
            for p in range(prime_factor_len):
                spatial_tile += (
                    np.log2(prime_factor_list[f1][f2])
                    * x[(f1, f2, p)]
                    * (y[(1, p)] - y[(2, p)])
                )
    m.addConstr(spatial_tile <= sum(np.log2(spatial_dim)), "spatial_tile_limit")

    # Set objective function
    ## utilization
    total_util = 0
    for v, iv_map in enumerate(tensor_IV):
        total_util += L2_util[v] + L1_util[v]
    total_util += spatial_tile

    ## compute latency
    ## The product of all dimensions that map to temporal
    total_cycles = 0
    for f1, pf_list in enumerate(prime_factor_list):
        for f2, prime_factors in enumerate(pf_list):
            for p in range(prime_factor_len):
                total_cycles += (
                    np.log2(prime_factor_list[f1][f2])
                    * x[(f1, f2, p)]
                    * (1 - y[(1, p)] + y[(2, p)])
                )

    ## traffic
    total_traffic = L3_L2_traffic + L2_L1_traffic

    loopnest_obj = -util_factor * total_util + compute_factor * total_cycles

    m.setObjective(loopnest_obj, GRB.MINIMIZE)

    begin_time = time.time()
    m.optimize()
    end_time = time.time()
    runtime = end_time - begin_time

    # Logging to a file
    m.write("debug.lp")

    # print results
    print("---runtime--- ", runtime)
    m.printAttr("X")

    ## L2 utilization
    print("---L2_utilization---")
    for key, val in L2_util.items():
        print(key, "-", val.getValue())

    ## L1 utilization
    print("---L1_utilization---")
    for key, val in L1_util.items():
        print(key, "-", val.getValue())

    ## spatial tile limit
    print("---spatial_tile_limit---")
    print(spatial_tile.getValue())

    ## objective - utilization
    print("objective: total_utilization = ", total_util.getValue())
    print("objective: total_cycles = ", total_cycles.getValue())
    print("objective: L3_L2_traffic")
    for key, val in L3_L2_tensor_traffic.items():
        print(key, "-", val.getValue())
    print("objective: L2_L1_traffic")
    for key, val in L2_L1_tensor_traffic.items():
        print(key, "-", val.getValue())


if __name__ == "__main__":
    try:
        ILP_formulation()

    except gp.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))

    except AttributeError:
        print("Encountered an attribute error")
