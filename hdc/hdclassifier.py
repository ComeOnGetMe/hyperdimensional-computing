import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mode


class HDClassifier2:
    def __init__(self,
                 hd_dim,
                 verbose=False,
                 num_level=10,
                 level_type='random'):
        assert hd_dim % 2 == 0, "[Error] dimension is odd"
        self.channel_im = {}  # c-by-h
        self.level_im = {}  # l-by-h
        self.hd_dim = hd_dim
        self.verbose = verbose
        self.num_level = num_level
        self.level_type = level_type

        self.num_channel = 0
        self.num_label = 0
        self._cls_hv = None  # y-by-c
        self._cls_map = {}
        self.generate_level_im()

    def compute_sample_hv(self, x):
        """Compute hyper vector for one input

        Args:
            sample (vector): input

        Returns:
            np.ndarray: dimension: (num of channels,)
        """
        return np.multiply(self.level_im[x], self.channel_im).sum(axis=1)

    def generate_channel_im(self):
        self.channel_im = np.ones((self.num_channel, self.hd_dim))
        for i in range(self.num_channel):
            randidx = np.random.permutation(self.hd_dim)
            self.channel_im[i, randidx[:self.hd_dim // 2]] = -1

    def generate_level_im(self):
        self.level_im = np.ones((self.num_level, self.hd_dim))
        if self.level_type == 'random':
            for i in range(self.num_level):
                randidx = np.random.permutation(self.hd_dim)
                self.level_im[i, randidx[:self.hd_dim // 2]] = -1

        elif self.level_type == 'rotation':
            randidx = np.random.permutation(self.hd_dim)
            for i in range(self.num_level):
                self.level_im[i, randidx[:self.hd_dim // 2]] = -1
                randidx = np.roll(randidx, self.hd_dim // 2 / self.num_level)

    def fit(self, X, y):
        self.num_channel = X.shape[1]
        self.generate_channel_im()

        labels = np.unique(y)
        self.num_label = labels.size
        disc_degree = np.unique(X[:, 0]).size

        # Warnings
        if self.hd_dim < 2 * X.shape[0] / self.num_label and self.verbose:
            print('[Warning] Too many training samples, model might overfit')
        if disc_degree * self.num_channel > 1.2 * self.hd_dim and self.verbose:
            print('[Warning] small dimension for this data, HVs might not be orthogonal')

        # Training
        self._cls_hv = np.zeros((self.num_label, self.num_channel))
        self._cls_map = {i: l for i, l in enumerate(labels)}
        for i, l in enumerate(labels):
            samples = X[y == l]
            for sample in samples:
                self._cls_hv[i] += self.compute_sample_hv(sample)
        return self

    def predict(self, X):
        X_hv = np.zeros((X.shape[0], self.num_channel))
        for i in range(X.shape[0]):
            X_hv[i] = self.compute_sample_hv(X[i])
        dists = cdist(X_hv, self._cls_hv, 'cosine')
        rank = dists.argmax(axis=1)
        return [self._cls_map[idx] for idx in rank]
