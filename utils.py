from tree import *

class TreeDigester:
    '''
    Takes a file full of parse trees such as those in trees/*.txt and returns a new text file full of parse trees
    made for this format (i.e. the labels are the next word in the sequence for each node in the tree.

    E.x.
    (0 (0 the) (0 (0 cats) (0 meow) ) ) -> input
    (EOS (cats the) (EOS (meow cats) (EOS meow) ) ) -> output
    '''
    def __init__(self, tree_file):
        with open(tree_file, 'r') as fid:
            self.trees = [Tree(l) for l in fid.readlines()]

        for t in self.trees:
            leftTraverse(t.root, self.node_digest)

    def node_digest(self, node, _):
        if node.parent == None:
            node.label = 'EOS'
        elif node.parent.right == node:
            node.label = 'EOS'
        elif node.parent.left == node:
            if node.isLeaf:
                node.label = getLeaves(node.parent)[1].word
            else:
                node.label = getLeaves(node.parent)[len(getLeaves(node))].word

    def print_node(self, node):
        if node.isLeaf:
            string = '( {} {} )'.format(node.label, node.word)
        else:
            string = '( {} {} {} )'.format(node.label, self.print_node(node.left), self.print_node(node.right))

        return string

    def output_trees(self, output_name):
        with open(output_name, 'w') as fid:
            for new_tree in self.trees:
                fid.write(self.print_node(new_tree.root))
                fid.write('\n')
