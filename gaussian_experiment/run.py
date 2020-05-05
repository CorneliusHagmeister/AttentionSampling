import torch
import functools
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from models import Discriminator, Generator
import tensorboardX
import numpy as np
import pylib as py
import random

from gaussian_dataset import GaussianDataset

py.arg('--experiment_name', required=True, default='none')
py.arg('--top_k', default=False, type=bool)
py.arg('--largest', default=True, type=bool)
py.arg('--k', type=int)
py.arg('--batch_size', default=256, type=int)
py.arg('--k_decay', default=.99, type=float)
py.arg('--min_k', default=192, type=int)
py.arg('--dataset_name', required=True)
py.arg('--z_dim', type=int, default=2)
py.arg('--wrs', type=bool, default=False)
args = py.args()
out_dir = './%s' % args.experiment_name
os.makedirs(py.join(out_dir, 'checkpoints'), exist_ok=True)
py.args_to_yaml(py.join(out_dir, 'settings.yml'), args)

max_iterations = 100000
iteration_counter = 0
batch_size = 256
k = args.k
k_decay = .99
v = 192

topk_training = False

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")


sample_classes = []
G = Generator(args.z_dim)
D = Discriminator()
torch.manual_seed(0)

G.to(device)
D.to(device)


G_optimizer = torch.optim.Adam(G.parameters(), lr=.0001)
D_optimizer = torch.optim.Adam(D.parameters(), lr=.0001)

dataloader = DataLoader(dataset=GaussianDataset('%s.npy' % args.dataset_name), batch_size=256,
                        shuffle=True, num_workers=0)

writer = tensorboardX.SummaryWriter(py.join(out_dir, 'summaries'))


def save_model(iter_c, generator, discriminator, outpath, z_dim):
    torch.save({
        'iterations': iter_c,
        'G': generator.state_dict(),
        'D': discriminator.state_dict(),
        'z_dim': z_dim
    }, outpath)


def calc_wrs(weights, k):
    random_weight_list = []
    for i in range(len(weights)):
        rnd = random.random()
        random_weight_list.append(rnd ** (1/weights[i]))
    npa = np.asarray(random_weight_list, dtype=np.float32)
    ind = npa.argsort(axis=0)[-k:]
    return ind

# some unfinished code
def eval_samples(samples):
    eval_dict = {}
    modes_root = 25
    for sample in samples:
        rounded = np.rint(sample)
        diff = sample-rounded
        if int((rounded[0]*modes_root+rounded[1])/2) not in eval_dict:
            eval_dict[int((rounded[0]*modes_root+rounded[1])/2)] = 0
        eval_dict[int((rounded[0]*modes_root+rounded[1])/2)] += 1
    print(eval_dict)
    sample_classes.append(eval_dict)


while(iteration_counter < max_iterations):
    for i, samples in enumerate(dataloader):
        if iteration_counter > max_iterations:
            break
        iteration_counter += 1

        if(len(samples) < batch_size):
            continue

        samples = samples.to(device)

        # create noise
        disc_train_noise = (torch.randn(batch_size, args.z_dim)).to(device)

        # train disc
        D_optimizer.zero_grad()
        G_out = G(disc_train_noise)

        DX_score = D(samples.float())
        DG_score = D(G_out)
        D_loss = torch.sum(-torch.mean(torch.log(DX_score + 1e-8)
                                       + torch.log(1 - DG_score + 1e-8)))

        D_loss.backward()
        D_optimizer.step()

        # train generator

        G_optimizer.zero_grad()
        generator_train_noise = (torch.randn(
            batch_size, args.z_dim)).to(device)

        G_out = G(generator_train_noise)

        eval_samples(G_out.detach().cpu().numpy())

        DG_score = D(G_out)

        if args.top_k is True:
            DG_score, DG_indices = torch.topk(
                DG_score, max(k, v), dim=0, largest=args.largest)

        if args.wrs is True:
            score_clone = DG_score.detach().cpu().numpy()
            wrs_indices = calc_wrs(score_clone, k)
            DG_score = DG_score[wrs_indices]

        G_loss = -torch.mean(torch.log(DG_score + 1e-8))

        G_loss.backward()
        G_optimizer.step()
        if iteration_counter % 2000 == 0:
            k *= args.k_decay
            k = int(k)
        if iteration_counter % 1000 == 0:
            save_model(iteration_counter, G, D, './%s/checkpoints/checkpoint.pt' %
                       args.experiment_name, args.z_dim)

        if iteration_counter % 10000 == 0:
            save_model(iteration_counter, G, D, './%s/checkpoints/checkpoint(%d).pt' %
                       (args.experiment_name, iteration_counter), args.z_dim)

        # print/save loss
        writer.add_scalar('D_loss', D_loss, global_step=iteration_counter)
        writer.add_scalar('G_loss', G_loss, global_step=iteration_counter)
        print('Iter: ', iteration_counter, ' --- D_loss: ',
              D_loss.item(), '  ---  G_loss: ', G_loss.item())

save_model(iteration_counter, G, D, './%s/checkpoints/checkpoint.pt' %
           args.experiment_name, args.z_dim)
