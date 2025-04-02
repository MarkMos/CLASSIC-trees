import numpy as np

class Tree_Node:
    def __init__(self):
        self.mhalo = 0.0
        self.jlevel = 0
        self.nchild = 0
        self.index = 0


        self.child = None
        self.parent = None
        self.formation = None
        self.sibling = None

class tree_memory_arrays:
    def __init__(self):
        self.merger_tree_aux = []

class tree_memory_arrays_passable:
    def __init__(self):
        self.merger_tree = []
def tree_index(node,merger_tree):
    if node is None:
        raise ValueError('Node is not associated with any tree')
    index = merger_tree.index(node)
    return index

def tree_formation_index(self, i_halo):
    if i_halo < 0 or i_halo >= len(self.merger_tree):
        raise ValueError('i_Halo is out of bounds')
    node = self.merger_tree[i_halo]
    if node.formation is None:
        raise ValueError('tree_formation_index: Node has no formation')
    return node.formation.index