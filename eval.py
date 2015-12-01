__author__ = 'Ariel'

import numpy as np
import math

def eval(hardPredict, softPredict, real):
    delta = (hardPredict - real)
    hard = np.sum(delta/float(len(softPredict)))
    deltata = (softPredict - real)
    soft = math.sqrt(np.sum(deltata*deltata/float(len(softPredict))))
    return hard, soft

def accuracy(predict, real):
    return (np.sum(predict - real)+0.0)/len(predict)

