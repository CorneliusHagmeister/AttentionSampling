

# 100p
python ./compute_activations.py --data_dir ../DCGAN/output_new/output/100p/DCGAN/cifar10-topk/generated_imgs_ep200 \
../DCGAN/output_new/output/100p/DCGAN/cifar10-wrs/generated_imgs_ep200 \
../DCGAN/output_new/output/100p/DCGAN/cifar10/generated_imgs_ep200 \
../DCGAN/output_new/output/100p/DCGAN/cifar10-botk/generated_imgs_ep200 --out_file_name=inception-v3_ep200.npy

# LSGAN
python ./compute_activations.py --data_dir ../DCGAN/output_new/output/100p/LSGAN/cifar10-topk/generated_imgs_ep200 \
../DCGAN/output_new/output/100p/LSGAN/cifar10-wrs/generated_imgs_ep200 \
../DCGAN/output_new/output/100p/LSGAN/cifar10/generated_imgs_ep200 \
../DCGAN/output_new/output/100p/LSGAN/cifar10-botk/generated_imgs_ep200 --out_file_name=inception-v3_ep200.npy

# DRAGAN
python ./compute_activations.py --data_dir ../DCGAN/output_new/output/100p/DRAGAN/cifar10-topk/generated_imgs_ep200 \
../DCGAN/output_new/output/100p/DRAGAN/cifar10-wrs/generated_imgs_ep200 \
../DCGAN/output_new/output/100p/DRAGAN/cifar10/generated_imgs_ep200 \
../DCGAN/output_new/output/100p/DRAGAN/cifar10-botk/generated_imgs_ep200 --out_file_name=inception-v3_ep200.npy

# WGANGP
python ./compute_activations.py --data_dir ../DCGAN/output_new/output/100p/WGANGP/cifar10-topk/generated_imgs_ep200 \
../DCGAN/output_new/output/100p/WGANGP/cifar10-wrs/generated_imgs_ep200 \
../DCGAN/output_new/output/100p/WGANGP/cifar10/generated_imgs_ep200 \
../DCGAN/output_new/output/100p/WGANGP/cifar10-botk/generated_imgs_ep200 --out_file_name=inception-v3_ep200.npy
