__author__ = 'Ariel'
import numpy as np


def eval(hardPredict, softPredict, real):
    hard = np.sum(hardPredict, real)/float(hardPredict.shape[0])
    soft = np.sqrt(np.sum(np.square(softPredict-real)/softPredict.shape[0]))
    return hard,soft

def accuracy(predict, real):
    return (np.sum(predict == real)+0.0)/len(predict)

