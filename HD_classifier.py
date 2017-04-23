import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mode


class HDClassifier:
    def __init__(self, d, verbose=False, num_class_hv=1, num_level=10,
                 level_type='random', evaluation='nn'):
        """
        :param d: dimension of HD vector
        """
        assert d % 2 == 0, "[Error] dimension is odd"
        self.channel_im = {}
        self.level_im = {}
        self.d = d
        self.verbose = verbose
        self.num_class_hv = num_class_hv
        self.num_label = 0
        self.level_type = level_type
        self.num_level = num_level
        self.evaluation = evaluation
        self.hv_label = None
        self.hv = None

    def compute_sample_hv(self, sample):
        """
        compute hv for single sample
        :param sample:
            shape: (channel,)
            value: discrete indices
        :return: sample hv
        """
        hv = np.multiply(self.channel_im, self.level_im[sample]).sum(axis=0)
        return hv

    def generate_channel_im(self, c):
        self.channel_im = np.ones((c, self.d))
        for i in xrange(c):
            randidx = np.random.permutation(self.d)
            self.channel_im[i, randidx[:self.d / 2]] = -1

    def generate_level_im(self, l):
        self.level_im = np.ones((l, self.d))
        if self.level_type == 'random':
            for i in xrange(l):
                randidx = np.random.permutation(self.d)
                self.level_im[i, randidx[:self.d / 2]] = -1

        elif self.level_type == 'rotation':
            randidx = np.random.permutation(self.d)
            for i in xrange(l):
                self.level_im[i, randidx[:self.d / 2]] = -1
                randidx = np.roll(randidx, self.d / 2 / l)

    def fit(self, X, y):
        num_channel = X.shape[1]
        self.generate_channel_im(num_channel)
        self.generate_level_im(self.num_level)

        labels = np.unique(y)
        self.num_label = labels.size
        disc_degree = np.unique(X[:, 0]).size

        # Warnings
        if self.d < 2 * X.shape[0] / self.num_label and self.verbose:
            print '[Warning] Too many training samples, model might overfit'
        if disc_degree * num_channel > 1.2 * self.d and self.verbose:
            print '[Warning] small dimension for this data, HVs might not be orthogonal'

        # Training
        if self.evaluation == 'average':
            self.hv = np.zeros((self.num_label * self.num_class_hv, self.d))  # HV for single class
            for i, l in enumerate(labels):
                samples = X[y == l]
                for j, sample in enumerate(samples):
                    sample_hv = self.compute_sample_hv(sample)
                    self.hv[i*self.num_class_hv + j % self.num_class_hv] += sample_hv

        elif self.evaluation == 'nn':
            self.hv = np.zeros((X.shape[0], self.d))
            self.hv_label = y
            for i, sample in enumerate(X):
                sample_hv = self.compute_sample_hv(sample)
                self.hv[i] = sample_hv

        return self

    def predict(self, X):
        X_hv = np.zeros((X.shape[0], self.d))
        for i in xrange(X.shape[0]):
            X_hv[i] = self.compute_sample_hv(X[i])
        dists = cdist(X_hv, self.hv, 'cosine')
        rank = dists.argsort(axis=1)[:, :self.num_class_hv]
        rank %= self.num_label
        top, _ = mode(rank, axis=1)
        return np.squeeze(top)


def test():
    pass


if __name__ == '__main__':
    d = 10000
    num_sample = 100
    num_level = 10
    X = np.random.randint(0, num_level, (num_sample, 13))
    y = np.random.randint(0, 10, num_sample)
    hdc = HDClassifier(d, num_class_hv=1, level_type='rotation', evaluation='average').fit(X, y)

    preds = hdc.predict(X)
    print (preds == y).mean()
