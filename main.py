__author__ = 'Ariel'

import pickle
import numpy as np
import preprocess
import multiLR
import predict


def main():
    train_file = 'yelp_reviews_train.json'
    dev_file = 'yelp_reviews_dev.json'
    test_file = 'yelp_review_test.json'
    #preprocess.generateModel(dev_file, 2)# train 1, dev 2, test 3

    X, Y = preprocess.loadModel()
    W = multiLR.BSGD(X, Y)
    multiLR.predict(W, True)


    return
if __name__ == '__main__':
    main()