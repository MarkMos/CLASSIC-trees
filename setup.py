from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        'classic_trees',
        sources=['classic_trees.pyx'],
        language='c',
        include_dirs=[np.get_include()]
    )
]

setup(
    name='classic_trees_module',
    version='0.1',
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False
        }
    ),
)