# walk_the_tree.pyx
from Tree_Node_and_Memory import Tree_Node

def walk_tree(this_node: Tree_Node):
    '''
    Walk through the entire merger tree.
    '''
    next_node = this_node
    if next_node.child is not None:
        next_node = next_node.child
    else:
        if next_node.sibling is not None:
            next_node = next_node.sibling
        else:
            while next_node.sibling is None and next_node.parent is not None:
                next_node = next_node.parent
            if next_node.sibling is not None:
                next_node = next_node.sibling
            else:
                next_node = None
    return next_node