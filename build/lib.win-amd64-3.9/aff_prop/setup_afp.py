from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext = Extension(
    "aff_prop.aff_prop_c", ['aff_prop/aff_prop_c.pyx'],
    include_dirs=[np.get_include()],
)

setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=[ext]
)