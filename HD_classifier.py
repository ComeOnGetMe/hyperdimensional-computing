import numpy as np
from scipy.spatial.distance import cdist


class HDClassifier:
    def __init__(self, d, verbose=False):
        """
        :param d: dimension of HD vector
        """
        assert d % 2 == 0, "[Error] dimension is odd"
        self.class_hv = None
        self.channel_im = {}
        self.level_im = {}
        self.d = d
        self.verbose = verbose

    # def generate_random_hv(self):
    #     return np.random.randint(0, 2, self.d) * 2 - 1
    #
    # def lookup_item_memory(self, key):
    #     if key not in self.im:
    #         self.im[key] = self.generate_random_hv()
    #     return self.im[key]

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
        self.channel_im = np.random.randint(0, 2, (c, self.d)) * 2 - 1

    def generate_level_im(self, l):
        self.level_im = np.random.randint(0, 2, (l, self.d)) * 2 - 1

    def fit(self, X, y, num_level):
        num_channel = X.shape[1]
        self.generate_channel_im(num_channel)
        self.generate_level_im(num_level)

        labels = np.unique(y)
        disc_degree = np.unique(X[:, 0]).shape[0]

        # Warnings
        if self.d < 2 * X.shape[0] / labels.shape[0] and self.verbose:
            print '[Warning] Too many training samples, model might overfit'
        if disc_degree * num_channel > 1.2 * self.d and self.verbose:
            print '[Warning] small dimension for this data, HVs might not be orthogonal'

        # Training
        self.class_hv = np.zeros((labels.shape[0], self.d))  # HV for single class
        for i, l in enumerate(labels):
            samples = X[y == l]
            for sample in samples:
                hv = self.compute_sample_hv(sample)
                self.class_hv[i] += hv  # tunable
        return self

    def predict(self, X):
        X_hv = np.zeros((X.shape[0], self.d))
        for i in xrange(X.shape[0]):
            X_hv[i] = self.compute_sample_hv(X[i])
        preds = cdist(X_hv, self.class_hv, 'cosine').argmin(axis=1)
        return preds


def test():
    pass


if __name__ == '__main__':
    d = 10000
    num_sample = 100
    num_level = 100
    X = np.random.randint(0, num_level, (num_sample, 13))
    y = np.random.randint(0, 10, num_sample)
    hdc = HDClassifier(d).fit(X, y, 100)

    preds = hdc.predict(X)
    print (preds - y).mean()
