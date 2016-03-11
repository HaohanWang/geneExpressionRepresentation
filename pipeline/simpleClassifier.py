__author__ = 'Haohan Wang'

import numpy as np

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.cross_validation import cross_val_score

labels = np.loadtxt('../data/labels_final.txt', delimiter=',')

data1 = np.loadtxt('../data/data_final_a.txt', delimiter=',')
data2 = np.loadtxt('../data/data_final_b.txt', delimiter=',')

data = np.append(data1, data2, 1)

clf = SVC()

scores = cross_val_score(clf, data, labels, cv=5)
print np.mean(scores)