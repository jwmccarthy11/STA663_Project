from setuptools import find_packages, setup
from Cython.Build import cythonize
import numpy as np

ext_modules = cythonize("./src/aff_prop/*.pyx")

setup(
    name="AffinityPropagation",
    version="0.0.4",
    author="Michael Sarkis, Jack McCarthy",
    author_email="jack.mccarthy@duke.edu",
    description="Fast implementation of affinity propagation",
    url="https://github.com/jwmccarthy11/STA663_Project",
    ext_modules=ext_modules,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    include_dirs=[np.get_include()]
)