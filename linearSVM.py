__author__ = 'Ariel'

import util
import numpy as np

def process(X, Y, file):
    # X is csr_matrix
    with open(file, 'w') as f:
        for x in xrange(X.shape[0]):
            s = ""
            s += str(Y[x])
            s += " "
            tmp = X[x].toarray().flatten()
            col = tmp.nonzero()[0]
            data = tmp[col]
            for y, z in zip(col, data):
                s += str(y+1)
                s += ":"
                s += str(z)
                s += " "
            s += "\n"
            f.write(s)
    return

def generate():
    ifHash = False
    trainfile = 'yelp_reviews_train.json'
    X, y, top = util.preprocess(trainfile, ifTrain=True, ifHash=ifHash, trainTop=[])

    predfile = 'yelp_reviews_dev.json'
    x, _, _ = util.preprocess(predfile, ifTrain=False, ifHash=ifHash, trainTop=top)

    process(X, y, 'libtrain.txt')
    process(x, np.zeros(x.shape[0]), 'libdev.txt')
    return

if __name__ == '__main__':
    generate()