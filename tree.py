import random
import nltk
import math

class Node:  # a node in the tree
    def __init__(self, label, word=None):
        self.label = label
        self.word = word
        self.parent = None  # reference to parent
        self.left = None  # reference to left child
        self.right = None  # reference to right child
        # true if I am a leaf (could have probably derived this from if I have
        # a word)
        self.isLeaf = False
        # true if we have finished performing fowardprop on this node (note,
        # there are many ways to implement the recursion.. some might not
        # require this flag)

    def __str__(self):
        if self.isLeaf:
            return '[{0}:{1}]'.format(self.word, self.label)
        return '({0} <- [{1}:{2}] -> {3})'.format(self.left, self.word, self.label, self.right)


class Tree:
    def __init__(self, treeString, openChar='(', closeChar=')'):
        self.open = '('
        self.close = ')'
        self.root = self.parse(treeString)
        # get list of labels as obtained through a post-order traversal
        self.labels = get_labels(self.root)
        self.num_words = len(self.labels)

    def parse(self, tokens, parent=None):
        assert tokens[0] == self.open, "Malformed tree"
        assert tokens[-1] == self.close, "Malformed tree"

        if type(tokens) == str:
            tokens = tokens.split()

        split = 2  # position after open and label
        countOpen = countClose = 0

        if tokens[split] == self.open:
            countOpen += 1
            split += 1
        # Find where left child and right child split
        while countOpen != countClose:
            if tokens[split] == self.open:
                countOpen += 1
            if tokens[split] == self.close:
                countClose += 1
            split += 1

        # New node
        # node = Node(int(tokens[1]))  # zero index labels
        node = Node(tokens[1])

        node.parent = parent

        # leaf Node
        if countOpen == 0:
            node.word = ''.join(tokens[2]).lower()  # lower case?
            node.isLeaf = True
            return node

        node.left = self.parse(tokens[2:split], parent=node)
        node.right = self.parse(tokens[split:-1], parent=node)

        return node

    def get_words(self):
        leaves = getLeaves(self.root)
        words = [node.word for node in leaves]
        return words


def loadTrees(dataSet='train'):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    file = 'trees/%s.txt' % dataSet
    print("Loading %s trees.." % dataSet)
    with open(file, 'r') as fid:
        trees = [Tree(l) for l in fid.readlines()]

    return trees


def get_labels(node):
    if node is None:
        return []
    return get_labels(node.left) + get_labels(node.right) + [node.label]


def leftTraverse(node, nodeFn=None, args=None):
    """
    Recursive function traverses tree
    from left to right.
    Calls nodeFn at each node
    """
    if node is None:
        return
    leftTraverse(node.left, nodeFn, args)
    leftTraverse(node.right, nodeFn, args)
    nodeFn(node, args)


def getLeaves(node):
    if node is None:
        return []
    if node.isLeaf:
        return [node]
    else:
        return getLeaves(node.left) + getLeaves(node.right)


def getNonLeaves(node):
    if node is None:
        return []
    if node.isLeaf:
        return [node.parent]
    else:
        return getLeaves(node.left) + getLeaves(node.right)


def topDown(node):
    if node.word == None:
        nextNode = [topDown(node.left), topDown(node.right)]
        return nextNode
    else:
        return node.word


def getLeftmostNode(node):
    if node.isLeaf:
        return node.word
    elif node.left.isLeaf:
        return node.left.word
    else:
        return getLeftmostNode(node.left)


def getRightmostNode(node):
    if node.isLeaf:
        return node.word
    elif node.right.isLeaf:
        return node.right.word
    else:
        return getRightmostNode(node.right)


def getRandomLabel(l):
    """
    l is a node
    """
    lbl = None

    if l.parent is None:
        lbl = '<EOS>'
        return lbl
    else:
        if l == l.parent.left and l.parent is not None:
            if l.parent.right.word is None:
                lbl = getLeftmostNode(l.parent.right)
            else:
                lbl = l.parent.right.word
        elif l == l.parent.right and l.parent is not None:
            if l.parent.left.word is None:
                lbl = getRightmostNode(l.parent.left)
            else:
                lbl = l.parent.left.word

        return lbl


def composeList(l):
    if type(l[0]) == list and type(l[1]) == list:
        l = composeList(l[0]) + composeList(l[1])
    elif type(l[0]) == list and type(l[1]) == str:
        l = composeList(l[0]) + l[1]
    elif type(l[0]) == str and type(l[1]) == list:
        l = l[0] + composeList(l[1])
    elif type(l[0]) == str and type(l[1]) == str:
        l = l[0] + l[1]
    else:
        print("Fuck somethings wrong")

    return l


def calculate_probabilities(corpus):
    """
    Calculates the probabilities according to: P(w_i) = 1 - sqrt(t / f(w_i)) where t is some threshold (reported as 10e-5)
    :param corpus: list of all words in the corpus
    :return: dictionary mapping words to probabilities
    """

    fdist = nltk.FreqDist(corpus)
    t = 10e-5

    probs = {}
    for k in fdist.keys():
        probs[k] = 1 - (math.sqrt(t / fdist[k]))

    return probs