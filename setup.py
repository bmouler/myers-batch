import sys

from setuptools import Extension, setup

# NEON is baseline on aarch64, so no architecture flag is needed and none is
# passed: -march=native would produce wheels that do not run on other machines.
if sys.platform == "win32":
    cflags = ["/O2"]
else:
    cflags = ["-O3", "-std=c11"]

setup(
    ext_modules=[
        Extension(
            "myers_batch._kernel",
            sources=["src/myers_batch/module.c", "src/myers_batch/kernel.c"],
            extra_compile_args=cflags,
        )
    ]
)
