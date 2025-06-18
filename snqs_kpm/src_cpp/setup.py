from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

import sys

import os

# Build with : python src_cpp/setup.py build_ext --inplace --force

# linux: CC=clang CXX=clang++ python src_cpp/setup.py build_ext --inplace --force

if sys.platform == 'linux':
    # On linux (RQC or Colab) --> try to compile with OpenMP.
    # 获取 OpenMP 库的路径

    def make_extension(name, source):
        return Extension(
            name, 
            [source],
            extra_compile_args = ['-fopenmp'],
            extra_link_args = ['-fopenmp'])
    
            #extra_compile_args = ['-fopenmp'], 
            #extra_link_args = ['-fopenmp']
    
else: # On my mac, the platform in 'darwin'.
    # OpenMP not available, falling back.
    def make_extension(name, source):
        return Extension(name, [source])

# if sys.platform == 'linux':
#     # On linux (RQC or Colab) --> try to compile with OpenMP.
#     ext_modules=[
#         Extension("src.utils.sparse_math",    # location of the resulting .so
#                  ["src_cpp/sparse_math.pyx"],
#                   extra_compile_args = ['-fopenmp'],
#                   extra_link_args = ['-fopenmp'],
#                   )
#     ]
#
# else: # On my mac, the platform in 'darwin'.
#     # OpenMP not available, falling back.
#     ext_modules = [
#         Extension("src.utils.sparse_math",  # location of the resulting .so
#                   ["src_cpp/sparse_math.pyx"],
#                   )
#     ]

ext_modules = [make_extension("snqs_kpm.nqs.utils.sparse_math", "src_cpp/sparse_math.pyx"),
               make_extension("snqs_kpm.nqs.utils.hilbert_math", "src_cpp/hilbert_math.pyx"),
               make_extension("snqs_kpm.nqs.utils.hamiltonian_math", "src_cpp/hamiltonian_math.pyx")]

# run    # run with >>> python src_cpp/setup.py build_ext --inplace (--force)
setup(
    ext_modules = cythonize(ext_modules),
    include_dirs=[np.get_include()],
    setup_requires=['Cython'],
    zip_safe=False
)
