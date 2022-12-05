import scipy as sp
import numpy as np
from scipy.spatial import distance
from scipy import stats


def knn_classifier(x_train, y_train, x_test, weight = False, k = 1, p = 1):
    dist = distance.cdist(x_train, x_test, 'minkowski', p = p)
    if (weight == False):
        indices = np.argsort(dist,axis = 0)
        indices_cut = indices[0:k]
        return stats.mode(y_train[indices_cut]).mode
    else:
        w = sp.special.softmax(-dist,axis = 0)
        maybe_0 = np.sum(((1-y_train).T * w.T).T, axis = 0)
        maybe_1 = np.sum((y_train.T * w.T).T,axis = 0)
        return (np.sign(maybe_1 - maybe_0)+1) // 2