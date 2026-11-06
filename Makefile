# Builds the C kernels as a shared library loaded by bench/cmyers.py.
CC      ?= cc
CFLAGS  ?= -O3 -fPIC -Wall -Wextra -std=c11
# Tune for the build machine: -mcpu=native (ARM) or -march=native (x86).
# Choose by CPU type: on x86, gcc accepts -mcpu=native but treats it as
# -mtune only (no AVX etc.), so that flag must not be used there.
UNAME_M := $(shell uname -m)
ifneq ($(filter arm64 aarch64,$(UNAME_M)),)
ARCHFLAGS := -mcpu=native
else
ARCHFLAGS := -march=native
endif

# One build directory per OS/arch: the Mac and its Linux VM share this folder.
BUILD := build/$(shell uname -s)-$(shell uname -m)
LIB := $(BUILD)/libmyers.so
GPUCHECK := $(BUILD)/libgpucheck.so

# CPython extension; build with the venv's interpreter, e.g. make PY=.venv/bin/python
PY      ?= python3
PYINC   := $(shell $(PY) -c "import sysconfig; print(sysconfig.get_paths()['include'])" 2>/dev/null)
EXTSUF  := $(shell $(PY) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" 2>/dev/null)
EXT     := $(BUILD)/_edpy$(EXTSUF)
EXTLDFLAGS := $(if $(filter Darwin,$(shell uname -s)),-undefined dynamic_lookup,)

all: $(LIB) $(EXT) $(GPUCHECK)

$(EXT): csrc/edpy.c csrc/myers.c Makefile
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(ARCHFLAGS) -I$(PYINC) -shared $(EXTLDFLAGS) -o $@ csrc/edpy.c

$(LIB): csrc/myers.c Makefile
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(ARCHFLAGS) -shared -o $@ csrc/myers.c
	@printf '%s\n' "cc=$$($(CC) --version | head -1)" "flags=$(CFLAGS) $(ARCHFLAGS)" > $(BUILD)/BUILDINFO

clean:
	rm -rf build

.PHONY: all clean gpucheck

# Host build of the GPU kernels' device functions (tested on the CPU).
gpucheck: $(GPUCHECK)
$(GPUCHECK): csrc/gpu_hostcheck.cpp csrc/gpu_myers.cuh Makefile
	@mkdir -p $(BUILD)
	$(CXX) -O2 -fPIC -Wall -Wextra -Wno-unknown-pragmas -std=c++17 -shared -o $@ csrc/gpu_hostcheck.cpp
