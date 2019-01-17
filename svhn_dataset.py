import torchvision.datasets.utils as utils
import os
import numpy as np


def load_dataset(root):
    url = ""
    filename = ""
    file_md5 = ""
    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}
    utils.download_url(split_list['train'][0], root, split_list['train'][1], split_list['train'][2])
    utils.download_url(split_list['test'][0], root, split_list['test'][1], split_list['test'][2])

    import scipy.io as sio

    loaded_train_mat = sio.loadmat(os.path.join(root, split_list['train'][1]))
    loaded_test_mat = sio.loadmat(os.path.join(root, split_list['test'][1]))

    train_data = loaded_train_mat['X']
    test_data = loaded_test_mat['X']
    train_labels = loaded_train_mat['y'].astype(np.int32).squeeze()
    test_labels = loaded_test_mat['y'].astype(np.int32).squeeze()

    np.place(train_labels, train_labels == 10, 0)
    np.place(test_labels, test_labels == 10, 0)

    train_data = np.transpose(train_data, (3, 2, 0, 1))
    test_data = np.transpose(test_data, (3, 2, 0, 1))

    return (train_data, train_labels), (test_data, test_labels)