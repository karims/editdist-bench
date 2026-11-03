# Builds the C kernels as a shared library loaded by bench/cmyers.py.
CC      ?= cc
CFLAGS  ?= -O3 -fPIC -Wall -Wextra -std=c11
# Tune for the build machine: -mcpu=native (ARM) or -march=native (x86).
ARCHFLAGS := $(shell $(CC) -mcpu=native -x c -c /dev/null -o /dev/null 2>/dev/null && echo -mcpu=native || \
               ($(CC) -march=native -x c -c /dev/null -o /dev/null 2>/dev/null && echo -march=native))

LIB := build/libmyers.so

all: $(LIB)

$(LIB): csrc/myers.c Makefile
	@mkdir -p build
	$(CC) $(CFLAGS) $(ARCHFLAGS) -shared -o $@ csrc/myers.c
	@printf '%s\n' "cc=$$($(CC) --version | head -1)" "flags=$(CFLAGS) $(ARCHFLAGS)" > build/BUILDINFO

clean:
	rm -rf build

.PHONY: all clean
