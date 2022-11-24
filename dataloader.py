import torch.utils.data
# import torchvision.datasets as datasets
import datasets

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
                                         train=train,
                                         download=download,
                                         transform=transform_train,
                                         )
        if val:
            self.testset = datasets.CIFAR10(root=root,
                                            train=val,
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


# class Mnist:
#     def __init__(self,
#                  train,
#                  val,
#                  transform_train=None,
#                  transform_val=None,
#                  root='./data',
#                  download=True,
#                  ):
#         self.trainset = datasets.MNIST(root=root,
#                                        train=True,
#                                        download=download,
#                                        transform=transform_train,
#                                        )
#         if val:
#             self.testset = datasets.MNIST(root=root,
#                                           train=False,
#                                           download=download,
#                                           transform=transform_val,
#                                           )
#         self.val = val
#
#     def get_dataloader(self,
#                        batch_size,
#                        shuffle=True,
#                        drop_last=True,
#                        num_workers=2,
#                        pin_memory=True,
#                        ):
#         train_loader = torch.utils.data.DataLoader(self.trainset,
#                                                    batch_size=batch_size,
#                                                    shuffle=shuffle,
#                                                    num_workers=num_workers,
#                                                    drop_last=drop_last,
#                                                    pin_memory=pin_memory,
#                                                    )
#         if self.val:
#             test_loader = torch.utils.data.DataLoader(self.testset,
#                                                       batch_size=batch_size,
#                                                       shuffle=shuffle,
#                                                       num_workers=num_workers,
#                                                       drop_last=drop_last,
#                                                       pin_memory=pin_memory,
#                                                       )
#             return train_loader, test_loader
#
#         return train_loader


class CelebA:
    def __init__(self,
                 split_train,
                 split_val,
                 subsample_train,
                 subsample_val,
                 transform_train=None,
                 transform_val=None,
                 root='./data',
                 download=True,
                 ):
        self.trainset = datasets.CelebA(root=root,
                                        subsample_size=subsample_train,
                                        split=split_train,
                                        download=download,
                                        transform=transform_train,
                                        )
        if split_train is not "all":
            self.testset = datasets.CelebA(root=root,
                                           subsample_size=subsample_val,
                                           split=split_val,
                                           download=download,
                                           transform=transform_val,
                                           )
        self.val = True if split_val is not None else False

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


