# Experiments on Mixture of Gaussians

## Creating the dataset

Run

```
python generate_dataset.py
```

Make sure to adjust the num_samples variable according to your preferences. Per default the datasets are created with 100k samples each. This will generate datasets for 25, 36, 49, ..., 121 modes. Each dataset is normalized between -1 and 1. For each dataset we get 3 files that allow to recreate the unnormalized datasets.

## Training the models

Run

```
python run.py --experiment_name=$experiment_name --dataset_name=$name_of_dataset
```

### Top-k/ Bot-k

Set --top-k=true and --largest=true/false depending on if you want the top (largest=true) or bot (largest=false) sampling. Set --min_k according to your setup.

### WRS

Set --wrs=true.

## Inference

```
python infer.py
```

In the file change the num_samples and experiment_name.

## eval

Run

```
python eval.py
```

Change modes_root and gan_type in the file. This will print number of high quality samples as well as the percentage of recovered modes.
