from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sysconfig
python_include = sysconfig.get_paths()['include']

extensions = [
    Extension(
        'CLASSIC_trees.classic_trees',
        sources=['src/classic_trees/classic_trees.pyx'],
        language='c',
        extra_compile_args=['-std=c99'],
        include_dirs=[np.get_include(),python_include]
    )
]

setup(
    name='classic_trees',
    version='0.0.1',
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False
        }
    ),
    packages=['CLASSIC_trees'],
    package_dir={'':'src'},
    zip_safe = False,
)