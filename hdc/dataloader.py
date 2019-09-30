import numpy as np
import os

ARABIC_DIGIT_NUM_CLASSES = 10
ARABIC_DIGIT_TRAIN_NUM_BLOCKS_PER_CLASS = 660
ARABIC_DIGIT_TEST_NUM_BLOCKS_PER_CLASS = 220


def _load_arabic(file_name, num_blocks):
    blocks = []
    with open(file_name, 'r') as f:
        for line in f:
            if line.startswith(' '):  # new block
                blocks.append([])
            else:
                blocks[-1].append([float(x) for x in line.strip().split()])

    data = []
    label = []
    for i in range(ARABIC_DIGIT_NUM_CLASSES):
        for block in blocks[i * num_blocks:(i + 1) * num_blocks]:
            data.extend(block)
            label.extend([i] * len(block))
    return np.array(data), np.array(label)


def load_arabic(data_dir):
    """Load spoken arabic digit training and testing set.
    dataset page: https://archive.ics.uci.edu/ml/datasets/Spoken+Arabic+Digit

    Args:
        data_dir (str): the path that contains the original data files
            Train_Arabic_Digit.txt and Test_Arabic_Test.txt

    .. todo::
        include male/female info

    Returns:
        tuple: train_data, train_label, test_data, test_label
    """
    train_x, train_y = _load_arabic(
        os.path.join(data_dir, 'Train_Arabic_Digit.txt'),
        ARABIC_DIGIT_TRAIN_NUM_BLOCKS_PER_CLASS)
    test_x, test_y = _load_arabic(
        os.path.join(data_dir, 'Test_Arabic_Digit.txt'),
        ARABIC_DIGIT_TEST_NUM_BLOCKS_PER_CLASS)
    return train_x, train_y, test_x, test_y


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
    # x, y = load_isolet('data/ISOLET/isolet1+2+3+4.data')
    # print(x.shape, y.shape, x.dtype, y.dtype)

    X, y, X_test, y_test = load_arabic('../data/arabic_spoken_digits')
    print(X.shape, y.shape)
