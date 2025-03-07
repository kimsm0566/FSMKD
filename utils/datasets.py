import torchvision
import numpy as np
from torch.utils.data import Dataset


# PyTorch Dataset 클래스 정의
class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            # print("Applying transform...")  # Transform 적용 여부 확인
            image = self.transform(image)

        return image, label


def get_data(args, data_path):

    if args.dataset == 'mnist' or args.dataset == 'fashion-mnist':

        data_file = f"{data_path}/{args.dataset}.npz"
        dataset = np.load(data_file)
        train_X, train_y = dataset['x_train'], dataset['y_train'].astype(np.int64)
        test_X, test_y = dataset['x_test'], dataset['y_test'].astype(np.int64)

        if args.dataset == 'fashion-mnist':
            train_X = np.reshape(train_X, (-1, 1, 28, 28))
            test_X = np.reshape(test_X, (-1, 1, 28, 28))
        else:
            train_X = np.expand_dims(train_X, 1)
            test_X = np.expand_dims(test_X, 1)

    elif args.dataset == 'cifar10':

        # Only load data, transformation done later

        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True)
                                                # download = True,
        train_X = trainset.data.transpose([0, 3, 1, 2])
        train_y = np.array(trainset.targets)

        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True)
        test_X = testset.data.transpose([0, 3, 1, 2])
        test_y = np.array(testset.targets)

    else:

        raise ValueError("Unknown dataset")

    return train_X, train_y, test_X, test_y

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        image, label = self.dataset[self.idxs[item]]

        return image, label
    
