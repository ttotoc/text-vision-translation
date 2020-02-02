# saves the training parameters of the currently trained network and vocabulary data extracted from the dataset

import pickle

# fix for cli run
import sys

sys.path.append('translation')


def save(path, **kwargs):
    f = open(path, 'wb')
    pickle.dump(kwargs, f)
    f.close()


def load(path):
    f = open(path, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj
