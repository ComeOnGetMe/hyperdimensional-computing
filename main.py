# import numpy as np
# import time
import HD_classifier
from dataloader import *
# from sklearn.linear_model import LogisticRegression as LR
# from xgboost import XGBClassifier

data_dir = 'data/ISOLET/'
sample_per_class = 1000
num_level = 10
dimension = 10000
num_class_hv = 1
X_train, y_train = load_isolet(data_dir + 'isolet1+2+3+4.data')
X_test, y_test = load_isolet(data_dir + 'isolet5.data')

# print 'XGBoost:'
# clf = XGBClassifier().fit(X_train, y_train)
# print 'training: {}, testing: {}'.format(clf.score(X_train, y_train), clf.score(X_test, y_test))

print 'HDC:'
intervals = np.linspace(0, 1, num_level)
disc_train = np.digitize(X_train, intervals)-1
disc_test = np.digitize(X_test, intervals)-1
hdc = HD_classifier.HDClassifier(dimension, num_class_hv=num_class_hv, level_type='rotation').fit(disc_train, y_train)
preds_train = hdc.predict(disc_train)
preds_test = hdc.predict(disc_test)
print 'training: {}, testing: {}'.format((y_train == preds_train).mean(), (y_test == preds_test).mean())
