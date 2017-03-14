import numpy as np


def load_arabic(file_path, block_per_class, sample_per_class):
    raw = []
    num_block = 0
    with open(file_path, 'r') as f:
        for line in f:
            if line == '\t\n':
                raw.append([])
                num_block += 1
            else:
                raw[num_block - 1].append(map(float, line.strip().split()))
    num_class = num_block / block_per_class

    train = []
    label = []
    for i in xrange(num_class):
        class_train = []
        class_label = []
        for j in xrange(i * block_per_class, (i + 1) * block_per_class):
            for sample in raw[j]:
                class_train.append(sample)
                class_label.append(i)
        train += class_train[:sample_per_class]
        label += class_label[:sample_per_class]
    return np.array(train), np.array(label)


def load_isolet(file_path):
    raw = []
    label = []
    with open(file_path, 'r') as f:
        for line in f:
            datum = line.strip().split(', ')
            raw.append(datum[:-1])
            label.append(float(datum[-1]))
    return np.array(raw, dtype=np.float32), np.array(label, dtype=np.uint8)


if __name__ == '__main__':
    x, y = load_isolet('data/ISOLET/isolet5.data')
    print x.shape, y.shape, x.dtype, y.dtype
