
from skimage.util import img_as_ubyte
from importlib import import_module
from functools import partial
from tensorflow.python.platform import tf_logging as logging
import tensorflow_datasets as tfds
import numpy as np
from models.frozen import inception
import argparse
import time
import os
import tensorflow as tf
import glob
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
import_path = os.path.join(os.path.dirname(dir_path), 'models')
sys.path.insert(0, import_path)


# logging.set_verbosity(logging.ERROR)


IMG_WIDTH = 32
IMG_HEIGHT = 32


def import_frozen(frozen_network):
    return import_module(f"models.frozen.{frozen_network}")


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000, batch_size=10):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=500)

    return ds


def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, '1'


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])



def to_activation(
    out_dir,
    file_name,
    num_samples,
    batch_size,
    frozen,
    frozen_network='inception',
    frozen_network_version='v3',
    dataset=None,
):
    activation_file = os.path.join(
        out_dir,
        file_name,
    )
    if num_samples % batch_size != 0:
        print('number of samples is not divisible by batchsize. Some samples may ignored.')
    if not os.path.exists(activation_file):
        # frozen = import_frozen(frozen_network)
        it = iter(dataset)
        activations = []
        for iteration in range(int(num_samples/batch_size)):
            batch = next(it)
            image_batch = batch
            activations.append(frozen.run(
                frozen_network_version, image_batch[0], len(image_batch[0])))

        activations = np.concatenate(activations, axis=0)
        np.save(activation_file, activations)
        return activations
    else:
        print("Activation file already exists")


def compute_inception_from_folder(out_dir, out_file_name, data_dir, frozen_network, batch_size=50):
    num_files = len(os.listdir(data_dir))
    print('num files: %d' % num_files)
    list_ds = tf.data.Dataset.list_files(str(data_dir+'/*'))

    labeled_ds = list_ds.map(process_path, num_parallel_calls=2)
    ds = prepare_for_training(labeled_ds, batch_size=batch_size)
    if out_dir is None:
        out_dir = os.path.dirname(data_dir)
    to_activation(out_dir, out_file_name, num_files,
                  batch_size, frozen_network, dataset=ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_dir", nargs='+', type=str, help="Image dirs."
    )
    parser.add_argument("--out_dir", type=str,
                        help="Directory to write data to.", default=None)
    parser.add_argument("--out_file_name", type=str,
                        help="Output file name.", default='inception-v3.npy')

    args = parser.parse_args()
    frozen = import_frozen('inception')
    for data_dir in args.data_dir:
        compute_inception_from_folder(
            args.out_dir, args.out_file_name, data_dir, frozen)
