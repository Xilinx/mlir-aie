# Copyright (C) 2021 Xilinx, Inc.
# Copyright (C) 2022 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

PEEK = peek
POKE = poke

# Add any other object files to this list below
PEEK_OBJS = peek.o
POKE_OBJS = poke.o

all: $(PEEK) $(POKE)

$(POKE): $(POKE_OBJS)
	$(CC) $(LDFLAGS) -o $@ $(POKE_OBJS) $(LDLIBS)

$(PEEK): $(PEEK_OBJS)
	$(CC) $(LDFLAGS) -o $@ $(PEEK_OBJS) $(LDLIBS)

clean:
	-rm -f $(POKE) $(PEEK) *.elf *.gdb *.o


