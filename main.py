__author__ = 'Ariel'

import pickle
import numpy as np

import cProfile
import time

import multiLR
import util
import IOhelper

def main():
    # file = 'small.json'
    # ifTrain = False
    ifHash = False
    # X, Y = util.preprocess(file,False,False)


    trainfile = 'yelp_reviews_train.json'
    X, y, top = util.preprocess(trainfile, ifTrain=True, ifHash=ifHash, trainTop=[])

    W = multiLR.BSGD(X, y)

    predfile = 'yelp_reviews_dev.json'
    x, _ = util.preprocess(predfile, ifTrain=False, ifHash=ifHash, trainTop=top)

    t, s = multiLR.predict(W, x)

    util.writePred(t, s, 'v2.txt')


    return
if __name__ == '__main__':
    main()

