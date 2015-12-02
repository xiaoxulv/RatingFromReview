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
    ifHash = True
    trainfile = 'yelp_reviews_train.json'
    X, y, top = util.preprocess(trainfile, ifTrain=True, ifHash=ifHash, trainTop=[])

    predfile = 'yelp_reviews_dev.json'
    x, _, _ = util.preprocess(predfile, ifTrain=False, ifHash=ifHash, trainTop=top)

    process(X, y, 'libtrainHash.txt')
    process(x, np.zeros(x.shape[0]), 'libdevHash.txt')
    return



def duplicate():
    t = []
    with open('liblinear-2.1/libdevHashpred', 'r') as f:
        for line in f:
            t.append(line.strip())
    with open('liblinear-2.1/devHashpred1.txt', 'w') as f1:
        for item in t:
            f1.write("%s %s \n" % (item,item))
    return

if __name__ == '__main__':
    #generate()
    duplicate()