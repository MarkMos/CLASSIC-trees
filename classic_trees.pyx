import numpy as np
cimport numpy as np

cdef class Tree_Node:
    cdef public Tree_Node child, sibling, parent
    cdef int jlevel, nchild
    cdef float64

    def __init__(self):
        self.mhalo = 0.0
        self.jlevel = 0
        self.nchild = 0
    
        self.child = None
        self.parent = None
        self.sibling = None

cdef class Tree_Values:
    def tree_index(self,node: Tree_Node,merger_tree: Tree_Node[:]):
        if node is None:
            raise ValueError('Node is not associated with any tree')
        index = merger_tree.index(node)
        return index

