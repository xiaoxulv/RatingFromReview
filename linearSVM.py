__author__ = 'Ariel'

import itertools
from scipy.sparse import coo_matrix

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
                s += str(y)
                s += ":"
                s += str(z)
                s += " "
            s += "\n"
            f.write(s)
    return
