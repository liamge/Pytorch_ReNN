import pickle
import itertools
import matplotlib.pyplot as plt
import gensim
import numpy as np
import argparse
from main import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', dest='test', default=True)
    parser.add_argument('-d', '--dev', dest='dev', default=False)
    parser.add_argument('-l', '--load', dest='load')
    parser.add_argument('-a', '--average', dest='average', default=False)  # if true, take average of wvs

    return parser.parse_args()


def get_labels(loadpath):
    f = open(loadpath, 'r').readlines()

    labels = []

    for l in f:
        labels.append(l[1])

    return np.array(labels)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def load_trees(loadpath):
    f = open(loadpath, 'r').read().splitlines()

    trees = [Tree(l) for l in f]

    return trees

if __name__ == "__main__":
    args = arg_parse()
    print(bool(args.average))

    trees = []
    raw_words = []
    vocab = []

    print("Loading trees...")
    for file in os.listdir('trees'):
        trees.append([Tree(line) for line in open('trees' + '/' + str(file), 'r').read().splitlines()])

    trees = [s for l in trees for s in l]  # chain all trees into one array

    print("{} trees loaded!".format(len(trees)))

    print("Building vocab...")
    for tree in tqdm.tqdm(trees):
        words = tree.get_words()
        raw_words.append(words)

        for word in words:
            if word not in vocab:
                vocab.append(word)

    sents, raw_words = raw_words, [s for l in raw_words for s in l]  # chain the sublists in raw_words
    vocab = {w: i for (i, w) in enumerate(vocab)}
    vocab['<EOS>'] = max(vocab.values()) + 1
    if 'UNK' not in vocab:
        vocab['UNK'] = max(vocab.values()) + 1  # Set UNK token

    print("{} words found with a vocabulary size of {}".format(len(raw_words), len(vocab)))

    print("Calculating word probabilities...")
    probs = calculate_probabilities(raw_words)
    probs['UNK'] = 1 - (math.sqrt(10e-5))
    print("Probabilities calculated!")

    print("Training word2vec...")
    wv_model = gensim.models.Word2Vec(sentences=sents, size=300, negative=25)
    print("Word vectors trained!")

    print("Training doc2vec...")
    doc_model = gensim.models.Doc2Vec(documents=[TaggedDocument(sent, [i]) for i, sent in enumerate(sents)],
                                      vector_size=300, negative=25)
    print("Doc vectors trained")

    print("Training FastText...")
    ft_model = gensim.models.FastText(sentences=sents, size=300, negative=25)
    print("FastText trained!")


    X_train = load_trees('trees/digested_train.txt')
    X_dev = load_trees('trees/digested_dev.txt')
    X_test = load_trees('trees/digested_test.txt')

    rnn = ReNN(d=300, vocab=vocab, n_neg=5, recursive=True)
    load_model(rnn, args.load)


    print("Structuring word vector dataset...")
    X_wv_train = []
    X_wv_dev = []
    X_wv_test = []
    X_doc_train = []
    X_doc_dev = []
    X_doc_test = []
    X_ft_train = []
    X_ft_dev = []
    X_ft_test = []
    for tree in X_train:
        words = tree.get_words()
        vec = []
        for w in words:
            if w in wv_model.vocabulary.raw_vocab:
                vec.append(wv_model[w])

        if len(vec) > 0:
            if bool(args.average):
                X_wv_train.append(np.array(vec).mean(axis=0).reshape(1, 300))
            else:
                X_wv_train.append(np.array(vec).sum(axis=0).reshape(1, 300))
        else:
            X_wv_train.append(np.zeros((1, 300)))

        X_doc_train.append(doc_model.infer_vector(words))

        ft_vec = []
        for w in words:
            try:
                ft_vec.append(ft_model.wv[w])
            except KeyError:
                ft_vec.append(np.zeros((1, 300)))
        if len(ft_vec) > 0:
            if bool(args.average):
                X_ft_train.append(np.array(ft_vec).mean(axis=0).reshape(1, 300))
            else:
                X_ft_train.append(np.array(ft_vec).sum(axis=0).reshape(1, 300))
        else:
            X_ft_train.append(np.zeros((1, 300)))

    for tree in X_dev:
        words = tree.get_words()
        vec = []
        for w in words:
            if w in wv_model.vocabulary.raw_vocab:
                vec.append(wv_model[w])

        if len(vec) > 0:
            if bool(args.average):
                X_wv_dev.append(np.array(vec).mean(axis=0).reshape(1, 300))
            else:
                X_wv_dev.append(np.array(vec).sum(axis=0).reshape(1, 300))
        else:
            X_wv_dev.append(np.zeros((1, 300)))

        X_doc_dev.append(doc_model.infer_vector(words))

        ft_vec = []
        for w in words:
            try:
                ft_vec.append(ft_model.wv[w])
            except KeyError:
                ft_vec.append(np.zeros((1, 300)))
        if len(ft_vec) > 0:
            if bool(args.average):
                X_ft_dev.append(np.array(ft_vec).mean(axis=0).reshape(1, 300))
            else:
                X_ft_dev.append(np.array(ft_vec).sum(axis=0).reshape(1, 300))
        else:
            X_ft_dev.append(np.zeros((1, 300)))

    for tree in X_test:
        words = tree.get_words()
        vec = []
        for w in words:
            if w in wv_model.vocabulary.raw_vocab:
                vec.append(wv_model[w])

        if len(vec) > 0:
            if bool(args.average):
                X_wv_test.append(np.array(vec).mean(axis=0).reshape(1, 300))
            else:
                X_wv_test.append(np.array(vec).sum(axis=0).reshape(1, 300))
        else:
            X_wv_test.append(np.zeros((1, 300)))

        X_doc_test.append(doc_model.infer_vector(words))

        ft_vec = []
        for w in words:
            try:
                ft_vec.append(ft_model.wv[w])
            except KeyError:
                ft_vec.append(np.zeros((1, 300)))
        if len(ft_vec) > 0:
            if bool(args.average):
                X_ft_test.append(np.array(ft_vec).mean(axis=0).reshape(1, 300))
            else:
                X_ft_test.append(np.array(ft_vec).sum(axis=0).reshape(1, 300))
        else:
            X_ft_test.append(np.zeros((1, 300)))

    X_wv_train = np.array(X_wv_train).reshape(len(X_wv_train), 300)
    X_wv_dev = np.array(X_wv_dev).reshape(len(X_wv_dev), 300)
    X_wv_test = np.array(X_wv_test).reshape(len(X_wv_test), 300)

    X_doc_train = np.array(X_doc_train).reshape(len(X_doc_train), 300)
    X_doc_dev = np.array(X_doc_dev).reshape(len(X_doc_dev), 300)
    X_doc_test = np.array(X_doc_test).reshape(len(X_doc_test), 300)

    X_ft_train = np.array(X_ft_train).reshape(len(X_ft_train), 300)
    X_ft_dev = np.array(X_ft_dev).reshape(len(X_ft_dev), 300)
    X_ft_test = np.array(X_ft_test).reshape(len(X_ft_test), 300)

    print("Dataset structured!")

    print("Calculating Matrices...")
    if os.path.exists('X_train_neg.p'):
        X_train = pickle.load(open('X_train_neg.p', 'rb'))
        y_train = get_labels('sstb/train.txt')
    else:
        X_train = np.array([rnn.forward(t.root).data.numpy() for t in X_train]).reshape(len(X_train), 300)
        pickle.dump(X_train, open('X_train_neg.p', 'wb'))
        y_train = get_labels('sstb/train.txt')

    if args.dev and os.path.exists('X_dev_neg.p'):
        X_dev = pickle.load(open('X_dev_neg.p', 'rb'))
        y_dev = get_labels('sstb/dev.txt')
    elif args.dev and not os.path.exists('X_dev_neg.p'):
        X_dev = np.array([rnn.forward(t.root).data.numpy() for t in X_dev]).reshape(len(X_dev), 300)
        pickle.dump(X_dev, open('X_dev_neg.p', 'wb'))
        y_dev = get_labels('sstb/dev.txt')

    if args.test and os.path.exists('X_test_neg.p'):
        X_test = pickle.load(open('X_test_neg.p', 'rb'))
        y_test = get_labels('sstb/test.txt')
    elif args.test and not os.path.exists('X_test_neg.p'):
        X_test = np.array([rnn.forward(t.root).data.numpy() for t in X_test]).reshape(len(X_test), 300)
        pickle.dump(X_test, open('X_test_neg.p', 'wb'))
        y_test = get_labels('sstb/test.txt')

    assert X_train.shape[0] == y_train.shape[0], "Error: somethings gone terribly wrong"

    clf1 = LogisticRegression(class_weight='balanced')
    clf1.fit(X_train, y_train)

    clf2 = LogisticRegression(class_weight='balanced')
    clf2.fit(X_wv_train, y_train)

    #clf3 = LogisticRegression(class_weight='balanced')
    #clf3.fit(X_doc_train, y_train)

    #clf4 = LogisticRegression(class_weight='balanced')
    #clf4.fit(X_ft_train, y_train)

    pickle.dump(y_dev, open("y_dev.p", "wb"))

    if args.dev:
        preds = clf1.predict(X_dev)
        print("ReNN Accuracy: {}".format(accuracy_score(y_dev, preds)))
        print("ReNN F1: {}".format(f1_score(y_dev, preds, average='macro')))

        preds = clf2.predict(X_wv_dev)
        print("WV Accuracy: {}".format(accuracy_score(y_dev, preds)))
        print("WV F1: {}".format(f1_score(y_dev, preds, average='macro')))

        pickle.dump(preds, open("wv_preds.p", "wb"))

        #preds = clf3.predict(X_doc_dev)
        #print("Doc Accuracy: {}".format(accuracy_score(y_dev, preds)))
        #print("Doc F1: {}".format(f1_score(y_dev, preds, average='macro')))

        #preds = clf4.predict(X_ft_dev)
        #print("FT Accuracy: {}".format(accuracy_score(y_dev, preds)))
        #print("FT F1: {}".format(f1_score(y_dev, preds, average='macro')))
    elif args.test:
        preds = clf1.predict(X_test)
        print("ReNN Accuracy: {}".format(accuracy_score(y_test, preds)))
        print("ReNN F1: {}".format(f1_score(y_test, preds, average='macro')))