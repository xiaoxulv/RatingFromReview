__author__ = 'Ariel'

import numpy as np
import pickle


def storeModel(m, star_list, ifHash, options):
    # write matrix x and y to file
    if options == 2:
        output = 'dev'
    if options == 3:
        output = 'test'
    if options == 1:# train
        output = 'train'
        #outfile = open(output + 'Y.pickle', 'wb')
        #pickle.dump(star_list, outfile)

    if ifHash:
        output += 'Hash'

    print 'storing model', output
    outfile = open(output + '.pickle', 'w')
    pickle.dump(m, outfile)
    return

def loadTrainModel(ifHash):
    # load in training model
    input = 'train'
    if ifHash:
        input += 'Hash'
    input += '.pickle'
    with open(input, 'r') as f1:
        X = pickle.loads(f1.read())
    print input
    with open('trainY.pickle', 'rb') as f2:
        y = pickle.loads(f2.read())
    y = np.array(y)
    Y = np.zeros((y.shape[0], 5))
    for i,x in enumerate(y):
        Y[i][x-1] = 1
    return X, Y

def customStoreModel(m):
    outfile = open('custom.pickle', 'w')
    pickle.dump(m, outfile)
    return

def customLoadModel():
    with open('custom.pickle', 'r') as f1:
        X = pickle.loads(f1.read())

    with open('trainY.pickle', 'rb') as f2:
        y = pickle.loads(f2.read())
    y = np.array(y)
    Y = np.zeros((y.shape[0], 5))
    for i,x in enumerate(y):
        Y[i][x-1] = 1
    return X, Y
