# Evaluation

The evaluation scripts have been taken from ... and are largely based on the original implementation of these metrics. The scripts used can be run directly on folders of images instead of the activations. Activation weights are computed using the inception-v4 model from google.

## Compute activations

Activations are computed on folders of images.

```
python ./compute_activations.py --data_dir ../DCGAN/output_new/output/100p/DCGAN/cifar10-topk/generated_imgs_ep200 \
../DCGAN/output_new/output/100p/DCGAN/cifar10-wrs/generated_imgs_ep200 \
../DCGAN/output_new/output/100p/DCGAN/cifar10/generated_imgs_ep200 \
../DCGAN/output_new/output/100p/DCGAN/cifar10-botk/generated_imgs_ep200 --out_file_name=inception-v3_ep200.npy
```

## Compute metrics

Run for example

```
python ./compute_eval_scores.py --base="${base_path}DCGAN/cifar10" --save_folder=eval_ep100 --inception_filename_base=inception-v3_ep100.npy --compare \
../DCGAN/output/cifar_base/full ../DCGAN/output/cifar_base/0  ../DCGAN/output/cifar_base/1 ../DCGAN/output/cifar_base/2 ../DCGAN/output/cifar_base/3 ../DCGAN/output/cifar_base/4 ../DCGAN/output/cifar_base/5 ../DCGAN/output/cifar_base/6 ../DCGAN/output/cifar_base/7 ../DCGAN/output/cifar_base/8 ../DCGAN/output/cifar_base/9 \
 --labels cifar_full cifar_base_0 cifar_base_1 cifar_base_2 cifar_base_3 cifar_base_4 cifar_base_5 cifar_base_6 cifar_base_7 cifar_base_8 cifar_base_9
```

--base is the folder containing the activations for the model that eveything is compared against. The name of the activation file can be configured for both the base as well as the compare folders. The script will compute the FID, PRD as well as IMPR metrics. If not configured otherwise the metric results will be saved in the folders where the respective activations of the compare models are saved.
