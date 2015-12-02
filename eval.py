__author__ = 'Ariel'

import numpy as np
import math

def eval(hardPredict, softPredict, real):
    hard = (np.sum(hardPredict == real)+0.0)/len(hardPredict)
    softPredict = softPredict.reshape(len(softPredict),)
    delta = (softPredict - real)
    soft = math.sqrt(np.sum(delta*delta/float(len(softPredict))))
    return hard, soft

def accuracy(predict, real):
    return (np.sum(predict == real)+0.0)/len(predict)

