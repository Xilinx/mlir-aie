# Copyright (C) 2019-2022 Xilinx, Inc.
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: LicenseRef-AMD-Proprietary

all: test_groups.x run_groups

groups: test_groups.x run_groups

.PHONY: clean run

test.x: test_magika_v3p3.c magika_v3p3.c
	@g++ -DMAGIKA_TOP test_magika_v3p3.c magika_v3p3.c -o test.x ;\
	chmod a+x test.x

test_groups.x: test_groups.c magika_v3p3.c
	@g++ test_groups.c magika_v3p3.c -o test_groups.x ;\
	chmod a+x test_groups.x
	
	
get_sqnr.x: get_sqnr.cpp
	@g++ get_sqnr.cpp -o get_sqnr.x ;\
	chmod a+x get_sqnr.x

run:
	./test.x

run_groups: test_groups.x
	./test_groups.x

clean:
	@rm -rf ./*.x
