import pylib as py
import imlib as im
import torch
import numpy as np
import torchlib
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset,  sampler
import torchvision.utils as vutils
import os

import module
import data


def get_indices(dataset, class_name):
    indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(i)
    return indices


py.arg('--dataset', default='fashion_mnist',
       choices=['cifar10', 'fashion_mnist', 'mnist', 'celeba', 'anime', 'custom'])
py.arg('--out_dir', required=True)
py.arg('--num_samples_per_class', type=int, required=True)
py.arg('--batch_size', type=int, default=1)
py.arg('--num', type=int, default=-1)
py.arg('--output_dir',  type=str, default='generated_imgs')

args = py.args()

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

py.mkdir(args.out_dir)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
dataset = datasets.CIFAR10(
    'data/CIFAR10', transform=transform, train=False, download=True)
if args.num == -1:
    for i in range(10):
        idx = get_indices(dataset, i)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler.SubsetRandomSampler(idx),
                                 num_workers=0, drop_last=True, pin_memory=False)
        num_samples = args.num_samples_per_class
        out_path = py.join(
            args.out_dir, args.output_dir)
        os.makedirs(out_path, exist_ok=True)
        while(num_samples > 0):
            for (x_real, labels) in iter(data_loader):
                if num_samples > 0:
                    vutils.save_image(x_real.detach(), py.join(
                        out_path, 'img-%d-%d.jpg' % (num_samples, i)),
                        normalize=True)
                    num_samples -= 1
                    print('saving ', num_samples)
else:
    idx = get_indices(dataset, args.num)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler.SubsetRandomSampler(idx),
                             num_workers=0, drop_last=True, pin_memory=False)
    num_samples = args.num_samples_per_class
    out_path = py.join(
        args.out_dir, args.output_dir)
    os.makedirs(out_path, exist_ok=True)

    while(num_samples > 0):
        for (x_real, labels) in iter(data_loader):
            if num_samples > 0:
                x_real = np.transpose(x_real.data.cpu().numpy(), (0, 2, 3, 1))
                img = im.immerge(x_real, n_rows=1).squeeze()
                im.imwrite(img, py.join(
                    out_path, 'img-%d-%d.jpg' % (num_samples, args.num)))
                num_samples -= 1
                print('saving ', num_samples)
