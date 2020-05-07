import torchlib
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class OnlyImage(Dataset):

    def __init__(self, img_label_dataset):
        self.img_label_dataset = img_label_dataset

    def __len__(self):
        return len(self.img_label_dataset)

    def __getitem__(self, i):
        return self.img_label_dataset[i][0]


def make_32x32_dataset(dataset, batch_size, imb_index=-1, imb_ratio=1, drop_remainder=True, shuffle=True, num_workers=0, pin_memory=False):

    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        dataset = datasets.MNIST(
            'data/MNIST', transform=transform, download=True)
        img_shape = [32, 32, 1]

    elif dataset == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        dataset = datasets.FashionMNIST(
            'data/FashionMNIST', transform=transform, download=True)
        img_shape = [32, 32, 1]

    elif dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = datasets.CIFAR10(
            'data/CIFAR10', transform=transform, download=True)
        img_shape = [32, 32, 3]

    else:
        raise NotImplementedError

    # imb ratio
    # create imbalance
    if imb_index > -1:
        targets = np.array(dataset.targets)
        classes, class_counts = np.unique(targets, return_counts=True)
        nb_classes = len(classes)
        print(class_counts)

        imbal_class_counts = [int(idx * imb_ratio) for idx in class_counts]
        print(imbal_class_counts)

        # Get class indices
        class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]

        # Get imbalanced number of instances
        imbal_class_indices = [class_idx[:imb_class_count] if class_index == imb_index else class_idx[:real_class_counts] for class_idx,
                               imb_class_count, real_class_counts, class_index in zip(class_indices, imbal_class_counts, class_counts, classes)]
        imbal_class_indices = np.hstack(imbal_class_indices)

        # Set target and data to dataset
        dataset.targets = np.array(dataset.targets)[imbal_class_indices]
        dataset.data = dataset.data[imbal_class_indices]

        targets = np.array(dataset.targets)
        classes, class_counts = np.unique(targets, return_counts=True)
        nb_classes = len(classes)
        print(class_counts)

    # dataset = OnlyImage(dataset)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, drop_last=drop_remainder, pin_memory=pin_memory)

    return data_loader, img_shape


def make_celeba_dataset(img_paths, batch_size, resize=64, drop_remainder=True, shuffle=True, num_workers=4, pin_memory=False):
    crop_size = 108

    offset_height = (218 - crop_size) // 2
    offset_width = (178 - crop_size) // 2
    def crop(x): return x[:, offset_height:offset_height +
                          crop_size, offset_width:offset_width + crop_size]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(crop),
        transforms.ToPILImage(),
        transforms.Resize(size=(resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = torchlib.DiskImageDataset(img_paths, map_fn=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, drop_last=drop_remainder, pin_memory=pin_memory)

    img_shape = (resize, resize, 3)

    return data_loader, img_shape


def make_anime_dataset(img_paths, batch_size, resize=64, drop_remainder=True, shuffle=True, num_workers=4, pin_memory=False):
    transform = transforms.Compose([
        transforms.Resize(size=(resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = torchlib.DiskImageDataset(img_paths, map_fn=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, drop_last=drop_remainder, pin_memory=pin_memory)

    img_shape = (resize, resize, 3)

    return data_loader, img_shape


def make_imagenet_datset(batch_size, drop_remainder=True, shuffle=True, num_workers=0, pin_memory=False):

    transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageNet(
        './data', split='train', download=True, transform=transform)
    img_shape = [64, 64, 1]

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, drop_last=drop_remainder, pin_memory=pin_memory)
    return data_loader, img_shape


# ==============================================================================
# =                               custom dataset                               =
# ==============================================================================

def make_custom_dataset(img_paths, batch_size, resize=64, drop_remainder=True, shuffle=True, num_workers=4, pin_memory=False):
    transform = transforms.Compose([
        transforms.Resize(size=(resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(img_paths, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, drop_last=drop_remainder, pin_memory=pin_memory)

    img_shape = (resize, resize, 3)

    return data_loader, img_shape


