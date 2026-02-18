#!/usr/bin/env python
# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved

import os
import re
import setuptools
import glob
from os import path

def get_extensions():
    # Skip C++ extensions if YOLOX_NO_EXTENSIONS environment variable is set
    # or if on Windows without MSVC compiler
    if os.environ.get("YOLOX_NO_EXTENSIONS") == "1":
        return []
    
    try:
        import torch
        from torch.utils.cpp_extension import CppExtension, BuildExtension
        
        torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
        assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"
        
        this_dir = path.dirname(path.abspath(__file__))
        extensions_dir = path.join(this_dir, "yolox", "layers", "csrc")

        main_source = path.join(extensions_dir, "vision.cpp")
        sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

        sources = [main_source] + sources
        extension = CppExtension

        extra_compile_args = {"cxx": ["-O3"]}
        define_macros = []

        include_dirs = [extensions_dir]

        ext_modules = [
            extension(
                "yolox._C",
                sources,
                include_dirs=include_dirs,
                define_macros=define_macros,
                extra_compile_args=extra_compile_args,
            )
        ]

        return ext_modules
    except Exception as e:
        print(f"Warning: Could not configure C++ extensions: {e}")
        print("Installing without C++ extensions. Some features may be slower.")
        return []


with open("yolox/__init__.py", "r") as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        f.read(), re.MULTILINE
    ).group(1)


with open("README.md", "r") as f:
    long_description = f.read()


# Try to get extensions, fallback to empty list if compilation not possible
ext_modules = get_extensions()

# Only set cmdclass if torch is available and extensions were built
try:
    from torch.utils.cpp_extension import BuildExtension
    cmdclass = {"build_ext": BuildExtension} if ext_modules else {}
except ImportError:
    cmdclass = {}
    ext_modules = []

setuptools.setup(
    name="yolox",
    version=version,
    author="basedet team",
    python_requires=">=3.10",
    long_description=long_description,
    ext_modules=ext_modules,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent"
    ],
    cmdclass=cmdclass,
    packages=setuptools.find_namespace_packages(),
)
