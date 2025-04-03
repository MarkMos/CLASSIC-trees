from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        'walk_the_tree',
        sources=['walk_the_tree.pyx'],
        language='c'
    )
]

setup(
    name='walk_the_tree_module',
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