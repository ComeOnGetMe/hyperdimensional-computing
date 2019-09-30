import numpy as np
# import time
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
# from xgboost import XGBClassifier

from hdc.hdclassifier import HDClassifier2
from hdc.dataloader import *

data_dir = 'data/ISOLET/'
num_level = 10
dimension = 1000
X_train, y_train = load_isolet(data_dir + 'isolet1+2+3+4.data')
X_test, y_test = load_isolet(data_dir + 'isolet5.data')

# print 'XGBoost:'
# clf = XGBClassifier().fit(X_train, y_train)
# print 'training: {}, testing: {}'.format(clf.score(X_train, y_train), clf.score(X_test, y_test))

print('HDC:')
intervals = np.linspace(0, 1, num_level)
disc_train = np.digitize(X_train, intervals)-1
disc_test = np.digitize(X_test, intervals)-1
hdc = HDClassifier2(dimension, num_level=num_level).fit(disc_train, y_train)
print("Training done.")
preds_train = hdc.predict(disc_train)
preds_test = hdc.predict(disc_test)
print('scores: training: {}, testing: {}'.format(
    accuracy_score(y_train, preds_train),
    accuracy_score(y_test, preds_test)))
