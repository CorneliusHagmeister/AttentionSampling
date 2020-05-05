import torch
from models import Discriminator, Generator
import numpy as np
import os


device = 'cuda'

num_samples = 10000
experiment_name = '100_modes_gaussian_topk'
output_dir = './%s/generated_samples/samples' % experiment_name

model = torch.load('./%s/checkpoints/checkpoint.pt' % experiment_name)
print('iterations ', model['iterations'])
if 'z_dim' in model:
    z_dim = model['z_dim']
else:
    z_dim = 2
G = Generator(z_dim)
G.to(device)
G.load_state_dict(model['G'])
G.eval()
torch.manual_seed(2)

result = np.empty((0, 2))
for i in range(num_samples):
    noise = (torch.randn(1, z_dim)).to(device)
    sample = G(noise).detach().cpu().numpy()
    result = np.concatenate((result, sample), axis=0)


os.makedirs(output_dir, exist_ok=True)
np.save(output_dir, result)
