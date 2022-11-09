import torch.utils.data
from torch.utils.data import Dataset
import torchvision.datasets as datasets


class Cifar10:
    def __init__(self,
                 train,
                 val,
                 transform_train=None,
                 transform_val=None,
                 root='./data',
                 download=True,
                 ):
        self.trainset = datasets.CIFAR10(root=root,
                                         train=True,
                                         download=download,
                                         transform=transform_train,
                                         )
        if val:
            self.testset = datasets.CIFAR10(root=root,
                                            train=False,
                                            download=download,
                                            transform=transform_val,
                                            )
        self.val = val

    def get_dataloader(self,
                       batch_size,
                       shuffle=True,
                       drop_last=True,
                       num_workers=2,
                       pin_memory=True,
                       ):
        train_loader = torch.utils.data.DataLoader(self.trainset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers,
                                                   drop_last=drop_last,
                                                   pin_memory=pin_memory,
                                                   )
        if self.val:
            test_loader = torch.utils.data.DataLoader(self.testset,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle,
                                                      num_workers=num_workers,
                                                      drop_last=drop_last,
                                                      pin_memory=pin_memory,
                                                      )
            return train_loader, test_loader

        return train_loader
