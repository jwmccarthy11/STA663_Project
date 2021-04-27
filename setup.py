from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext = Extension(
    "aff_prop.aff_prop_c", ['aff_prop/aff_prop_c.pyx'],
    include_dirs=[np.get_include()],
)

setup(
    name="aff_prop",
    version="0.1.7",
    description="Fast implementation of affinity propagation",
    author="Michael Sarkis, Jack McCarthy",
    author_email="michael.sarkis@duke.edu, jack.mccarthy@duke.edu",
    url="https://github.com/jwmccarthy11/STA663_Project",
    packages=["aff_prop"],
    install_requires=[
        "numpy",
        "cython",
        "sklearn",
        "matplotlib"
    ],
    cmdclass={"build_ext": build_ext},
    ext_modules=[ext],
    setup_requires=['wheel']
)