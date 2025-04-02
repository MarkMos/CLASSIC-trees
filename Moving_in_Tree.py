from Tree_Node_and_Memory import *


def walk_tree_to_level(this_node, i_level_max):
    '''
    Walk through the merger tree, going no higher than a given level i_level_max.
    '''
    next_node = this_node
    while next_node is not None:
        if next_node.child is not None and next_node.jlevel <= i_level_max:
            next_node = next_node.child # Walk up the tree to the next child
        else:
            if next_node.sibling is not None:
                next_node = next_node.sibling # Walk to sibling, if exists and no children
            else:
                while next_node.sibling is None and next_node.parent is not None:
                    next_node = next_node.parent # Walk back to parents until we find more siblings, no siglings either
                if next_node.sibling is not None:
                    next_node = next_node.sibling # Next sibling
                else:
                    next_node = None # Back at base of tree - node has no parent
        if next_node is None or next_node.jlevel > i_level_max:
            break
    return next_node

def walk_tree(this_node):
    '''
    Walk through the entire merger tree.
    '''
    next_node = this_node
    if next_node.child is not None:
        next_node = next_node.child
        # print('stuck in child')
    else:
        if next_node.sibling is not None:
            next_node = next_node.sibling
            # print('stuck in sibling')
        else:
            while next_node.sibling is None and next_node.parent is not None:
                # print('stuck in parent')
                next_node = next_node.parent
            if next_node.sibling is not None:
                # print('stuck in next sibling')
                next_node = next_node.sibling
            else:
                next_node = None
    return next_node

def tree_hierarchy_level(this_node, i_level):
    '''
    Get the hierarchy level of halo in the merger tree.
    '''
    Tree_Hierarchy_Level = 0 # Initialize the hierarchy level
    parent_node = this_node
    while parent_node.j_level > i_level:
        if parent_node is None and parent_node.parent.child == parent_node:
            Tree_Hierarchy_Level += 1
            parent_node = parent_node.parent
    return Tree_Hierarchy_Level