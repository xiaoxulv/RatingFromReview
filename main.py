__author__ = 'Ariel'

import pickle
import numpy as np
import preprocess
import multiLR
import predict
import IOhelper


def main():
    train_file = 'yelp_reviews_train.json'
    dev_file = 'yelp_reviews_dev.json'
    test_file = 'yelp_reviews_test.json'
    preprocess.generateModel(train_file, 2, False)# train 1, dev 2, test 3

    #X, Y = IOhelper.loadModel()
    #W = multiLR.BSGD2(X, Y)
    #multiLR.predict(W, True)


    return
if __name__ == '__main__':
    main()