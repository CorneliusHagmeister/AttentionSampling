import numpy as np
import math
import matplotlib.pyplot as plt
import os


modes_root = 5
gan_type = ''
if gan_type == '':
    experiment_name = '%d_modes_gaussian%s' % (modes_root**2, gan_type)
else:
    experiment_name = '%d_modes_gaussian_%s' % (modes_root**2, gan_type)
load_path = './%s/generated_samples/samples.npy' % experiment_name
# load_path = './100_modes_gaussian.npy'

samples = np.load(load_path)
ptp = np.load('./%d_modes_gaussian_ptp.npy' % (modes_root**2))
s_min = np.load('./%d_modes_gaussian_min.npy' % (modes_root**2))
samples = (samples/2+.5)*ptp+s_min

stdv = .05

high_quality_samples_per_class = lst = [0] * modes_root**2
high_quality_samples = 0
print("num of samples: ", len(samples))
for sample in samples:
    rounded = np.rint(sample)
    diff = sample-rounded
    distance = math.sqrt(diff[0]**2+diff[1]**2)
    if distance <= stdv*4 and int((rounded[0]*modes_root+rounded[1])/2) < len(high_quality_samples_per_class) and int((rounded[0]*modes_root+rounded[1])/2) >= 0:
        high_quality_samples_per_class[int(
            (rounded[0]*modes_root+rounded[1])/2)] += 1
        high_quality_samples += 1
hqs_per_mode = {}
modes_recovered = 0
for samples_per_class in high_quality_samples_per_class:
    if samples_per_class > 0:
        modes_recovered += 1
    if samples_per_class not in hqs_per_mode:
        hqs_per_mode[samples_per_class] = 0
    hqs_per_mode[samples_per_class] += 1
plt.bar(list(hqs_per_mode.keys()), hqs_per_mode.values())
plt.xlabel('High quality samples per mode', fontsize=18)
plt.ylabel('Number of occurences', fontsize=18)
if not os.path.exists('./results/%d_modes' % (modes_root**2)):
    os.makedirs('./results/%d_modes' % (modes_root**2))
if gan_type == '':
    gan_type = 'plain'
plt.savefig('./results/%d_modes/%s' % (modes_root**2, gan_type))
print('high quality samples: %f' % (float(high_quality_samples)/len(samples)))
print('Modes recovered: %d in percent %f ' %
      (modes_recovered, (float(modes_recovered)/float(modes_root**2)*100)))
