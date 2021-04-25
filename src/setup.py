import numpy as np
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize('./src/aff_prop_c.pyx'),
    include_dirs=[np.get_include()],
    zip_safe=False
)