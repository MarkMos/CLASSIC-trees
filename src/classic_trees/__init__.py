from . import module  # Makes 'module' visible in dir()
from . import trees_class
from .module import get_tree_vals, functions, random_masses, set_trees
from .trees_class import Forest

__all__ = ['Forest']
