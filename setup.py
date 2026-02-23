from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import sysconfig
python_include = sysconfig.get_paths()['include']

extensions = [
    Extension(
        'classic_trees.module',
        sources=['src/classic_trees/module.pyx'],
        language='c',
        extra_compile_args=['-std=c99'],
        include_dirs=[np.get_include(),python_include]
    )
]

setup(
    name='classic_trees',
    version='0.0.1',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={'classic_trees': ['src/classic_trees/Data/*','src/classic_trees/Data/*.txt']},
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False
        }
    ),
    zip_safe=False,
    include_package_data=True
)