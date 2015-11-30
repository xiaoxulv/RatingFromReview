__author__ = 'Ariel'

import pickle
import numpy as np

import cProfile
import time

import multiLR
import util
import IOhelper

def main():

    # file = 'yelp_reviews_train.json'
    # #file = 'yelp_reviews_dev.json'
    # #file = 'yelp_reviews_test.json'
    # ifTrain = False
    # if file[-8:-5] == 'ain':
    #     options = 1
    #     ifTrain = True
    # elif file[-8:-5] == 'dev':
    #     options = 2
    # else:
    #     options = 3
    #
    # ifHash = False
    # X, Y = util.preprocess(file, ifTrain, ifHash)
    # IOhelper.storeModel(X, Y, ifHash, options)

    file = 'yelp_reviews_train.json'
    X, _ = util.custom_preprocess(file)
    IOhelper.customStoreModel(X)


    # ifHash = False
    # X, Y = IOhelper.loadTrainModel(ifHash)
    #
    # W = multiLR.SGD(X, Y)
    # multiLR.predict(W, True, ifHash)


    return
if __name__ == '__main__':
    main()

