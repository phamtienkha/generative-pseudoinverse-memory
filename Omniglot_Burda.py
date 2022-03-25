from scipy import io
import torch
from torch.utils.data import TensorDataset
from collections import Counter
import torch.utils.data as data
import numpy as np

class Omniglot_Burda(data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.train = train  # training set or test set

        mat = io.loadmat(root)
        train_set = mat['data']  # use the key for data here
        train_len = train_set.shape[-1]
        train_target = np.concatenate((np.argmax(mat['target'], axis=0).reshape(train_len, 1),
                                      mat['targetchar'].transpose()),
                                      axis=-1)

        test_set = mat['testdata']  # use the key for target here
        test_len = test_set.shape[-1]
        test_target = np.concatenate((np.argmax(mat['testtarget'], axis=0).reshape(test_len, 1),
                                      mat['testtargetchar'].transpose()),
                                      axis=-1)

        train_data = torch.from_numpy(train_set).float()
        train_target = torch.from_numpy(train_target)
        test_data = torch.from_numpy(test_set).float()
        test_target = torch.from_numpy(test_target)

        if self.train:
            self.train_data = train_data.transpose(0, 1)
            self.train_data = self.train_data.reshape(self.train_data.shape[0], 1, 28, 28)
            self.train_target = train_target
        else:
            self.test_data = test_data.transpose(0, 1)
            self.test_data = self.test_data.reshape(self.test_data.shape[0], 1, 28, 28)
            self.test_target = test_target

    def __getitem__(self, index):
        if self.train:
            data = self.train_data[index]
            target = self.train_target[index]
            if self.transform:
                data = self.transform(data)
        else:
            data = self.test_data[index]
            target = self.test_target[index]
            if self.transform:
                data = self.transform(data)
        return data, target

    def __len__(self):
        if self.train:
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]
