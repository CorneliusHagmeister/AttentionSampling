
import functools

import imlib as im
import numpy as np
import pylib as py
import tensorboardX
import torch
import torchlib
import torchprob as gan
from torchvision.utils import make_grid
import tqdm
from torch import nn
import os
import random

import data
import module
import time
from scipy.stats import norm
import math


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('ConvTranspose') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# ==============================================================================
# =                                   param                                    =
# ==============================================================================
# command line
py.arg('--dataset', default='fashion_mnist',
       choices=['cifar10', 'fashion_mnist', 'mnist', 'celeba', 'anime', 'custom', 'imagenet'])
py.arg('--batch_size', type=int, default=256)
py.arg('--epochs', type=int, default=100)
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--n_d', type=int, default=1)  # # d updates per g update
py.arg('--z_dim', type=int, default=128)
py.arg('--adversarial_loss_mode', default='gan',
       choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none',
       choices=['none', '1-gp', '0-gp', 'lp'])
py.arg('--gradient_penalty_sample_mode', default='line',
       choices=['line', 'real', 'fake', 'dragan'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--experiment_name', default='none')
py.arg('--gradient_penalty_d_norm', default='layer_norm',
       choices=['instance_norm', 'layer_norm'])  # !!!
py.arg('--top_k', default=False, type=bool)
py.arg('--wrs', default=False, type=bool)
py.arg('--normal', default=False, type=bool)
py.arg('--k', default=256, type=int)
py.arg('--min_k', default=192, type=int)
py.arg('--v', default=1.0, type=float)
py.arg('--largest', default=True, type=bool)
py.arg('--imb_index', type=int, default=-1)
py.arg('--imb_ratio', type=float, default=1.0)
py.arg('--save', type=bool, default=True)
args = py.args()

# output_dir
if args.experiment_name == 'none':
    args.experiment_name = '%s_%s' % (args.dataset, args.adversarial_loss_mode)
    if args.gradient_penalty_mode != 'none':
        args.experiment_name += '_%s_%s' % (
            args.gradient_penalty_mode, args.gradient_penalty_sample_mode)
output_dir = py.join('output', args.experiment_name)
os.makedirs(output_dir, exist_ok=True)
# py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# others
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

# setup dataset
if args.dataset in ['cifar10', 'fashion_mnist', 'mnist']:  # 32x32
    data_loader, shape = data.make_32x32_dataset(
        args.dataset, args.batch_size, args.imb_index, args.imb_ratio, pin_memory=use_gpu)
    n_G_upsamplings = n_D_downsamplings = 3
elif args.dataset == 'imagenet':
    # ======================================
    # =               custom               =
    # ======================================
    img_paths = 'data/imagenet_small/train'
    data_loader, shape = data.make_custom_dataset(
        img_paths, args.batch_size, resize=32, pin_memory=use_gpu)
    n_G_upsamplings = n_D_downsamplings = 3  # 3 for 32x32 and 4 for 64x64
    # ======================================
    # =               custom               =
    # ======================================


# ==============================================================================
# =                                   model                                    =
# ==============================================================================

# setup the normalization function for discriminator
if args.gradient_penalty_mode == 'none':
    d_norm = 'batch_norm'
else:  # cannot use batch normalization with gradient penalty
    d_norm = args.gradient_penalty_d_norm

# networks
G = module.ConvGenerator(
    args.z_dim, shape[-1], n_upsamplings=n_G_upsamplings).to(device)
D = module.ConvDiscriminator(
    shape[-1], n_downsamplings=n_D_downsamplings, norm=d_norm).to(device)
print(G)
G.apply(weights_init)
D.apply(weights_init)
print(G)
print(D)
elapsed_time_dict ={'wrs':[],'topk':[],'gen_train_compl':[],'gen_disc_compl':[],'gen_train_backward':[],'gen_train_step':[]}

# adversarial_loss_functions
d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(
    args.adversarial_loss_mode)

# optimizer
G_optimizer = torch.optim.Adam(
    G.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))
D_optimizer = torch.optim.Adam(
    D.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))

sigmoid_func = nn.Sigmoid()


def calc_wrs(weights, k):
    start_time = time.time()
    random_weight_list = []
    for i in range(len(weights)):
        rnd = random.random()
        random_weight_list.append(rnd ** (1/weights[i]))
    npa = np.asarray(random_weight_list, dtype=np.float32)
    ind = npa.argsort(axis=0)[-k:]
    elapsed_time = time.time() - start_time
    elapsed_time_dict['wrs'].append(elapsed_time)
    return ind

# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

#create normal distribution weights
mu = 1
variance = .0025
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, args.batch_size)
normal_tensor = torch.FloatTensor(norm.pdf(x, mu, sigma)).cuda()

def train_G():
    start_time_overall = time.time()
    G.train()
    D.train()

    z = torch.randn(args.batch_size, args.z_dim, 1, 1).to(device)
    x_fake = G(z)

    x_fake_d_logit = D(x_fake)
    sigmoid_x_fake_d_logits = sigmoid_func(x_fake_d_logit)

    # extract top/bot k features
    # --------------------------
    if args.wrs == True:
        idx = calc_wrs(sigmoid_x_fake_d_logits.squeeze(), args.k)
    elif args.normal==True:
        sorted_weights, idx = torch.topk(
            sigmoid_x_fake_d_logits.squeeze(), k=args.batch_size, dim=0, largest=args.largest, sorted=True)
        sorted_weights*=normal_tensor
        idx = calc_wrs(sorted_weights, args.k)
        
    else:
        start_time = time.time()
        _, idx = torch.topk(
            sigmoid_x_fake_d_logits.squeeze(), k=args.k, dim=0, largest=args.largest)
        elapsed_time = time.time() - start_time
        elapsed_time_dict['topk'].append(elapsed_time)
    x_fake_d_logit = x_fake_d_logit[idx]
    # --------------------------

    G_loss = g_loss_fn(x_fake_d_logit)

    G.zero_grad()
    start_time = time.time()
    G_loss.backward()
    elapsed_time = time.time() - start_time
    elapsed_time_dict['gen_train_backward'].append(elapsed_time)
    start_time = time.time()
    G_optimizer.step()
    elapsed_time = time.time() - start_time
    elapsed_time_dict['gen_train_step'].append(elapsed_time)
    elapsed_time = time.time() - start_time_overall
    elapsed_time_dict['gen_train_compl'].append(elapsed_time)
    return {'g_loss': G_loss}


def train_D(x_real, labels):
    start_time = time.time()
    # sets training mode
    G.train()
    D.train()

    z = torch.randn(args.batch_size, args.z_dim, 1, 1).to(device)
    x_fake = G(z).detach()

    x_real_d_logit = D(x_real)
    x_fake_d_logit = D(x_fake)
    sigmoid_x_real_d_logits = sigmoid_func(x_real_d_logit)
    histogram = []
    avg_validities = []
    if args.top_k == True:
        _, idx = torch.topk(
            sigmoid_x_real_d_logits.squeeze(), k=args.batch_size, dim=0, largest=args.largest)
        x_real_d_logit = x_real_d_logit[idx]

    x_real_d_loss, x_fake_d_loss = d_loss_fn(x_real_d_logit, x_fake_d_logit)
    gp = gan.gradient_penalty(functools.partial(
        D), x_real, x_fake, gp_mode=args.gradient_penalty_mode, sample_mode=args.gradient_penalty_sample_mode)

    D_loss = (x_real_d_loss + x_fake_d_loss) + \
        gp * args.gradient_penalty_weight

    D.zero_grad()
    D_loss.backward()
    D_optimizer.step()
    elapsed_time = time.time() - start_time
    elapsed_time_dict['gen_disc_compl'].append(elapsed_time)

    return {'d_loss': (x_real_d_loss + x_fake_d_loss).data.cpu().numpy(),
            'gp': gp.data.cpu().numpy()}


@torch.no_grad()
def sample(z):
    G.eval()
    return G(z)

# ==============================================================================
# =                                    run                                     =
# ==============================================================================


# load checkpoint if exists
ckpt_dir = py.join(output_dir, 'checkpoints')
py.mkdir(ckpt_dir)
try:
    ckpt = torchlib.load_checkpoint(py.join(ckpt_dir, 'Last.ckpt'))
    ep, it_d, it_g = ckpt['ep'], ckpt['it_d'], ckpt['it_g']
    D.load_state_dict(ckpt['D'])
    G.load_state_dict(ckpt['G'])
    D_optimizer.load_state_dict(ckpt['D_optimizer'])
    G_optimizer.load_state_dict(ckpt['G_optimizer'])
    print('loading was successful. Starting at epoch: ', ep)
except Exception as e:
    print(e)
    ep, it_d, it_g = 0, 0, 0

# sample
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

# main loop
writer = tensorboardX.SummaryWriter(py.join(output_dir, 'summaries'))
z = torch.randn(100, args.z_dim, 1, 1).to(device)  # a fixed noise for sampling
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

iter_counter=0

for ep_ in tqdm.trange(ep, args.epochs, desc='Epoch Loop'):
    if ep_ < ep:
        continue
    ep += 1
    # if iter_counter >=1000:
    #         break
    # train for an epoch
    for (x_real, labels) in tqdm.tqdm(data_loader, desc='Inner Epoch Loop'):
        iter_counter+=1
        # if iter_counter >=1000:
        #     break
        x_real = x_real.to(device)
        labels = labels.to(device)

        D_loss_dict = train_D(x_real, labels)
        it_d += 1
        for k, v in D_loss_dict.items():
            if args.save==True:
                writer.add_scalar('D/%s' % k, v, global_step=it_d)

        if it_d % args.n_d == 0:
            G_loss_dict = train_G()
            it_g += 1
            for k, v in G_loss_dict.items():
                if args.save ==True:
                    writer.add_scalar('G/%s' %
                                    k, v.data.cpu().numpy(), global_step=it_g)

        if it_d % 2000:
            args.k = max(int(args.k*args.v), args.min_k)

        # sample
        if it_g % 1000 == 0:
            x_fake = sample(z)
            img = make_grid(x_fake, normalize=True)
            writer.add_image("samples/generated", img, ep)
            x_fake = np.transpose(x_fake.data.cpu().numpy(), (0, 2, 3, 1))
            img = im.immerge(x_fake, n_rows=10).squeeze()
            im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % it_g))
    if args.save == True:
        if ep % 10 == 0:
            torchlib.save_checkpoint({'ep': ep, 'it_d': it_d, 'it_g': it_g,
                                    'D': D.state_dict(),
                                    'G': G.state_dict(),
                                    'D_optimizer': D_optimizer.state_dict(),
                                    'G_optimizer': G_optimizer.state_dict()},
                                    py.join(ckpt_dir, 'Epoch_inter(%d).ckpt' % ep))

        # save checkpoint
        torchlib.save_checkpoint({'ep': ep, 'it_d': it_d, 'it_g': it_g,
                                'D': D.state_dict(),
                                'G': G.state_dict(),
                                'D_optimizer': D_optimizer.state_dict(),
                                'G_optimizer': G_optimizer.state_dict()},
                                py.join(ckpt_dir, 'Last.ckpt'),)

# Printing the elapsed training times
# for key, value in elapsed_time_dict.items():
#     if len(value)>0:
#         print('Avg. time for: ',key ,' ::', sum(value) / len(value)  )