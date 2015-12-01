__author__ = 'Ariel'

import pickle
import numpy as np

import cProfile
import time

import multiLR
import util
import eval
import IOhelper

def main():
    #testfile = "trainsmall.json"

    ifHash = False

    trainfile = 'yelp_reviews_train.json'
    X, y, top = util.preprocess(trainfile, ifTrain=True, ifHash=ifHash, trainTop=[])

    W = multiLR.BSGD(X, y)
    t, s = multiLR.predict(W, X)
    print eval.eval(t, s, y)

    predfile = 'yelp_reviews_dev.json'
    x, _, _ = util.preprocess(predfile, ifTrain=False, ifHash=ifHash, trainTop=top)

    t, s = multiLR.predict(W, x)

    util.writePred(t, s, 'v4.txt')

    return

if __name__ == '__main__':
    main()

