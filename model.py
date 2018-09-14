import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable
from tree import *

"""
The main class for the Recursive Neural Network model. Uses the API from PyTorch, found here: https://pytorch.org/
"""
class ReNN(nn.Module):
    def __init__(self, d, vocab, n_neg, pretrained=None, recursive=False):
        """
        Initialization function for all variables/model configurations
        Args:
            d: int, dimension of word-vectors
            vocab: dictionary, key-value pairs of unique tokens and indices assigned to those tokens
            n_neg: int, number of samples from the noise distribution
            pretrained: string, path to pretrained word vectors
            recursive: boolean, whether model configuration uses a recursive composition function
        """
        super(ReNN, self).__init__()

        self.V = vocab
        self.embed_dim = d

        self.embed = nn.Embedding(len(vocab), d)

        # Noise distribution to draw vectors from (not updated during training)
        self.noise = nn.Embedding(len(vocab), d)
        #self.noise.weight.requires_grad = False

        self.hidden = None
        self.recursive = recursive
        self.pretrained = pretrained  # to access later with model saving
        if recursive:
            self.l1 = nn.LSTM(d, d)
            self.init_hidden()
        else:
            self.l1 = nn.Linear(2 * d, d)
        self.n_neg = n_neg

        if pretrained is not None:
            self.load_pretrained(pretrained)

    def forward(self, l, loss=None, return_loss=False):
        """
        Forward composition over a node in a binary tree
        Args:
            l: node object from tree.py, current node being composed
            loss: float, current loss, if None then it will be computed for a given node
            return_loss: boolean, parameter to control whether a call to forward returns the current loss

        Returns: Tensor, composed node

        """
        assert type(l) is not None, "Error: l has become none somehow"
        label = getRandomLabel(l)  # label is randomly assigned from the neighboring constituent

        if not l.left.isLeaf and not l.right.isLeaf:  # if both children are nodes, recursively call forward on them
            v1, loss1 = self.forward(l.left, return_loss=True)
            v2, loss2 = self.forward(l.right, return_loss=True)
            # compose the two results
            if self.recursive:
                for v in [v1, v2]:
                    l, self.hidden = self.l1(v.view(1, 1, -1), self.hidden)
            else:
                l = self.l1(torch.cat([v1, v2], dim=1))
            # if there is no current loss (i.e. both children were leaves, compute current loss)
            if loss is not None:
                loss = loss + loss1 + loss2
            else:
                loss = loss1 + loss2
        elif not l.left.isLeaf and l.right.isLeaf:  # if the right child is a leaf and the left isn't
            v1, loss = self.forward(l.left, return_loss=True)  # recursively compute the vector for the left child
            if l.right.word in self.V:  # check if right word is in vocabulary
                v2 = self.embed(Variable(torch.LongTensor([self.V[l.right.word]])))
            else:  # otherwise use the unknown token
                v2 = self.embed(Variable(torch.LongTensor([self.V['UNK']])))
            # compose the vectors
            if self.recursive:
                for v in [v1, v2]:
                    l, self.hidden = self.l1(v.view(1, 1, -1), self.hidden)
            else:
                l = self.l1(torch.cat([v1, v2], dim=1))
        elif l.left.isLeaf and not l.right.isLeaf:  # inverse of the case above, where left is a leaf and right isn't
            if l.left.word in self.V:
                v1 = self.embed(Variable(torch.LongTensor([self.V[l.left.word]])))
            else:
                v1 = self.embed(Variable(torch.LongTensor([self.V['UNK']])))
            v2, loss = self.forward(l.right, return_loss=True)
            if self.recursive:
                for v in [v1, v2]:
                    l, self.hidden = self.l1(v.view(1, 1, -1), self.hidden)
            else:
                l = self.l1(torch.cat([v1, v2], dim=1))
        elif l.left.isLeaf and l.right.isLeaf: # final case where both are leaves
            if l.left.word in self.V:
                v1 = self.embed(Variable(torch.LongTensor([self.V[l.left.word]])))
            else:
                v1 = self.embed(Variable(torch.LongTensor([self.V['UNK']])))
            if l.right.word in self.V:
                v2 = self.embed(Variable(torch.LongTensor([self.V[l.right.word]])))
            else:
                v2 = self.embed(Variable(torch.LongTensor([self.V['UNK']])))
            if self.recursive:
                for v in [v1, v2]:
                    l, self.hidden = self.l1(v.view(1, 1, -1), self.hidden)
            else:
                l = self.l1(torch.cat([v1, v2], dim=1))

        if loss is not None:  # if loss has already been computed, add new loss
            loss = loss + self.loss(l.view(1, -1), label)
        else:  # otherwise compute loss
            loss = self.loss(l.view(1, -1), label)

        if return_loss:
            return l.view(1, -1), loss
        else:
            return l.view(1, -1)

    def loss(self, input, label):
        """
        NCE loss function
        Args:
            input: tensor, vector representing either a leaf or a constituent
            label: string, label word randomly assigned from a neighboring constituent

        Returns: float, computed loss

        """
        label = self.embed(Variable(torch.LongTensor([self.V[label]])))  # embed the label as a tensor
        # draw k random vectors from the noise distribution
        rand_vecs = self.noise.weight[torch.LongTensor(np.random.choice(self.embed.num_embeddings,
                                                                        size=self.n_neg))]

        loss = (input * label).sum(1).sigmoid()  # logistic regression with the true label

        for i in range(rand_vecs.data.shape[0]):  # for each randomly sampled noise vector
            loss = loss + (input * rand_vecs[i:i + 1]).sum(1).sigmoid()  # logistic regression with random vector

        return loss

    def load_pretrained(self, path):
        """
        Function to load pretrained word vectors into the embedding matrix
        Args:
            path: string, path to the file containing the word vectors

        Returns:

        """
        print("Loading pretrained vectors...")
        word_vecs = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

        W = np.zeros((len(self.V), self.embed_dim))  # initialize the embedding matrix

        count = 0

        for (i, w) in enumerate(self.V):
            if w in word_vecs.vocab:
                W[i, :] = word_vecs[w]
                count += 1
            else:
                W[i, :] = np.random.uniform(-0.25, 0.25, self.embed_dim)

        self.embed.weight.data.copy_(torch.from_numpy(W).float())
        print("{} out of {} vectors loaded!".format(count, len(self.V)))
        del W  # save memory
        del word_vecs

    def init_hidden(self):
        """
        Function to initialize the hidden state of the recurrent layer
        """
        self.hidden = (autograd.Variable(torch.zeros(1, 1, self.embed_dim)),
                       autograd.Variable(torch.zeros(1, 1, self.embed_dim)))
