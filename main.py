import torch
import argparse
import os
import tqdm
import datetime
import sys
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from tree import *
from model import *


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trees', dest='trees', required=True)
    parser.add_argument('-e', '--epochs', dest='epochs', default=1, required=True)
    parser.add_argument('-d', '--dim', dest='dim', default=300, required=True)
    parser.add_argument('-n', '--neg', dest='neg', default=20, required=True)
    parser.add_argument('-c', '--clip', dest='clip', default=0.25, required=True)
    parser.add_argument('-p', '--pretrained', dest='pretrained')
    parser.add_argument('-r', '--recursive', dest='recursive', default=False)
    parser.add_argument('-s', '--save_path', dest='save_path', default='model_saves/')
    parser.add_argument('-l', '--model_load', dest='model_load', default=False)

    return parser.parse_args()


def save_model(model, savepath):
    pretrained = '+' if model.pretrained is not None else '-'
    recursive = '+' if model.recursive else '-'
    string_convert = '{}{}{}__{}_dim_{}_pretrained_{}_recursive'.format(datetime.datetime.now().time().hour,
                                                                        datetime.datetime.now().time().minute,
                                                                        datetime.datetime.now().time().second,
                                                                        model.embed_dim, pretrained, recursive)
    torch.save(model.state_dict(), savepath + string_convert)


def load_model(model, loadpath):
    print("Loading model from {}".format(loadpath))
    model.load_state_dict(torch.load(loadpath))


if __name__ == '__main__':
    # TODO: Come up with evaluation mode where the representations are used on a different task
    args = arg_parse()

    trees = []
    raw_words = []
    vocab = []

    print("Loading trees...")
    for file in os.listdir(args.trees):
        trees.append([Tree(line) for line in open(str(args.trees) + '/' + str(file), 'r').read().splitlines()])

    trees = [s for l in trees for s in l]  # chain all trees into one array

    print("{} trees loaded!".format(len(trees)))

    print("Building vocab...")
    for tree in tqdm.tqdm(trees):
        words = tree.get_words()
        raw_words.append(words)
        for word in words:
            if word not in vocab:
                vocab.append(word)

    raw_words = [s for l in raw_words for s in l]  # chain the sublists in raw_words
    vocab = {w: i for (i, w) in enumerate(vocab)}
    vocab['<EOS>'] = max(vocab.values()) + 1  # Set <EOS> token
    if 'UNK' not in vocab:
        vocab['UNK'] = max(vocab.values()) + 1  # Set UNK token

    print("{} words found with a vocabulary size of {}".format(len(raw_words), len(vocab)))

    if bool(args.model_load):
        rnn = ReNN(d=int(args.dim), vocab=vocab, n_neg=int(args.neg),
                   recursive=bool(args.recursive))
        load_model(rnn, args.model_load)
        print("Model loaded!")
    else:
        rnn = ReNN(d=int(args.dim), vocab=vocab, n_neg=int(args.neg),
                   pretrained=args.pretrained, recursive=bool(args.recursive))

    print(bool(args.recursive))
    print(rnn.recursive)

    total_losses = []

    optim = torch.optim.Adam(params=filter(lambda p: p.requires_grad, rnn.parameters()), lr=0.01)

    for i in range(np.clip(int(args.epochs), 1, None)):  # Make sure running for at least 1 epoch
        losses = []
        print("Epoch {} training...".format(i))
        for j in range(len(trees)):
            tree = trees[j]
            optim.zero_grad()
            if args.recursive:
                rnn.init_hidden()
            #l = topDown(tree.root)
            vec, loss = rnn.forward(tree.root, return_loss=True)
            losses.append(loss.data.numpy())
            total_losses.append(loss.data.numpy())
            sys.stdout.write("\r{}/{}\t{}".format(j, len(trees), loss.data.numpy()))
            sys.stdout.flush()
            loss.backward()
            torch.nn.utils.clip_grad_norm(rnn.parameters(), args.clip)
            optim.step()
            del loss
            del vec

        print("Epoch {} loss: {}".format(i, sum(losses)/len(losses)))
        with open("logs/experiment_logs.txt", "a") as writefile:
            pretrained = '+' if args.pretrained is not None else '-'
            recursive = '+' if args.recursive else '-'
            string_convert = '{}{}{}__{}_dim_{}_pretrained_{}_recursive_{}_neg_{}_clip'.format(datetime.datetime.now().time().hour,
                                                                                datetime.datetime.now().time().minute,
                                                                                datetime.datetime.now().time().second,
                                                                                args.dim, pretrained, recursive, args.neg, args.clip)
            writefile.write("model:{}\tepoch:{}:\tloss:{}\n".format(string_convert, i, sum(losses)/len(losses)))

    print("All trained! Saving model to: {}".format(args.save_path))
    save_model(rnn, args.save_path)
