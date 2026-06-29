// Copyright (C) 2018-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
 * Copyright (C) 2013-2016 Xilinx, Inc.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

void usage(char *prog)
{
	printf("usage: %s ADDR\n",prog);
	printf("\n");
	printf("ADDR may be specified as hex values\n");
}


int main(int argc, char *argv[])
{
	int fd;
	void *ptr;
	unsigned addr, page_addr, page_offset;
	unsigned page_size=sysconf(_SC_PAGESIZE);

	if(argc!=2) {
		usage(argv[0]);
		exit(-1);
	}

	fd=open("/dev/mem",O_RDONLY);
	if(fd<1) {
		perror(argv[0]);
		exit(-1);
	}

	addr=strtoul(argv[1],NULL,0);
	page_addr=(addr & ~(page_size-1));
	page_offset=addr-page_addr;

	ptr=mmap(NULL,page_size,PROT_READ,MAP_SHARED,fd,(addr & ~(page_size-1)));
	if((int)ptr==-1) {
		perror(argv[0]);
		exit(-1);
	}

	printf("0x%08x\n",*((unsigned *)(ptr+page_offset)));
	return 0;
}


