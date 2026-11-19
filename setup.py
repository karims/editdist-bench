"""Build the C extension (shared with the benchmark: csrc/edpy.c includes
csrc/myers.c) and ship the CUDA sources inside the package.

Portable by default. BITLEV_NATIVE=1 tunes for the build machine
(-mcpu=native on ARM, -march=native on x86); the published benchmark
numbers used native builds."""
import os
import platform
import shutil

from setuptools import Extension, setup
from setuptools.command.build_py import build_py

HERE = os.path.dirname(os.path.abspath(__file__))
CUDA_SOURCES = ["gpu_kernels.cu", "gpu_myers.cuh"]

args = ["-O3", "-std=c11"]
if os.environ.get("BITLEV_NATIVE") == "1":
    args.append("-mcpu=native" if platform.machine().lower() in ("arm64", "aarch64") else "-march=native")


class BuildPy(build_py):
    """Copy csrc/gpu_kernels.cu and gpu_myers.cuh into bitlev/cuda/ in the build."""

    def run(self):
        super().run()
        dst = os.path.join(self.build_lib, "bitlev", "cuda")
        os.makedirs(dst, exist_ok=True)
        for f in CUDA_SOURCES:
            shutil.copy(os.path.join(HERE, "csrc", f), dst)


setup(
    ext_modules=[Extension("bitlev._edpy", sources=["csrc/edpy.c"], include_dirs=["csrc"],
                           extra_compile_args=args)],
    cmdclass={"build_py": BuildPy},
)
