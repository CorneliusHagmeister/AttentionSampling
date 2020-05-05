import pylib as py
import imlib as im
import torch
import numpy as np
import torchlib

import module
import data

py.arg('--dataset', default='fashion_mnist',
       choices=['cifar10', 'fashion_mnist', 'mnist', 'celeba', 'anime', 'custom'])
# py.arg('--experiment_name', required=True)
py.arg('--checkpoint_name',  default="Epoch_inter(50).ckpt")
py.arg('--z_dim', type=int, default=128)
py.arg('--num_samples', type=int, required=True)
py.arg('--batch_size', type=int, default=1)
py.arg('--experiment_names', nargs='+', type=str)
py.arg('--output_dir',  type=str, default='generated_imgs')

args = py.args()
print(args.experiment_names)
experiment_names = args.experiment_names

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
torch.manual_seed(0)

if args.dataset in ['cifar10', 'fashion_mnist', 'mnist','imagenet']:  # 32x32
    output_channels = 3
    n_G_upsamplings = n_D_downsamplings = 3


for experiment in experiment_names:
    output_dir = py.join('output_new', 'output', experiment)

    G = module.ConvGenerator(args.z_dim, output_channels,
                             n_upsamplings=n_G_upsamplings).to(device)

    # load checkpoint if exists
    ckpt_dir = py.join(output_dir, 'checkpoints', args.checkpoint_name)
    out_dir = py.join(output_dir, args.output_dir)
    py.mkdir(ckpt_dir)
    py.mkdir(out_dir)
    ckpt = torchlib.load_checkpoint(ckpt_dir)
    G.load_state_dict(ckpt['G'])

    for i in range(args.num_samples):
        z = torch.randn(args.batch_size, args.z_dim, 1, 1).to(device)
        x_fake = G(z).detach()
        x_fake = np.transpose(x_fake.data.cpu().numpy(), (0, 2, 3, 1))
        img = im.immerge(x_fake, n_rows=1).squeeze()
        im.imwrite(img, py.join(out_dir, 'img-%d.jpg' % i))
        print(py.join(out_dir, 'img-%d.jpg' % i))
