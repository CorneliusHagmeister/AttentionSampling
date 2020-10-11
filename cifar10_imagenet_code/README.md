# Experiments on image datasets

There are 2 main experiments conducted on the image datasets imagenet and cifar10.

## Datapreparation

This experiment setup supports CIFAR10, MNIST, FashionMNIST out of the box. If you want to use these datasets, simply pass the name to the train script and it will automatically download. For the imagenet datatset you need to manually download the 2012 imagenet where the samples are scaled down to 32x32 px.

## Training

To train a network run

```
python train.py
```

For more information on how to train DCGAN, DRAGAN, LSGAN or WGAN see the scripts under example/scripts.

For top-k/bot-k set --top_k=true and --largest=true/false.
Set --k and --min_k to adjust the number of samples that are used each iteration. --v sets the rate of decay for k, which is applied every 2000 iterations.

If you want to train on an imbalanced dataset, set --imb_index to the class you want to reduce and --imb_ratio to the percentage of samples you want to reduce the class to.

## Inference/ baseline image creation

Run

```
python infer.py
```

Make sure to set the dataset and checkpoint you want to use. The number of samles used in the thesis was 5000. The more samples are used the more accurate the resulting number will be however.

To create the baseline images (if you dont want to use the whole dataset) run

```
python infer_base.py
```

for all datasets apart from imagenet. Makre sure that the datasets are download before hand.
--number_samples_per_class controls the total number of samples per class.

If you want to create the baseline images for imagenet run

```
python infer_base_imagenet.py
```

This script takes the same parameters as the infer_baseline.py script.
