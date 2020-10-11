from metrics.frechet_inception_distance import frechet_inception_distance
from metrics.precision_recall_for_distributions import (
    compute_prd_from_embedding,
    prd_to_max_f_beta_pair,
)
from metrics.improved_precision_and_recall import knn_precision_recall_features
import matplotlib.pyplot as plt
import argparse
import os
from misc import images, util
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=util.HelpFormatter)

    parser.add_argument(
        "--base",
        type=str,
        default=None,
        required=True,
        help="Path to a .npy file with train dataset.",
    )

    parser.add_argument(
        "--compare",
        type=str,
        nargs='+',
        default=None,
        required=True,
        help="Path to a .npy file with test dataset.",
    )

    parser.add_argument(
        "--labels",
        type=str,
        nargs='+',
        default=None,
        required=True,
        help="Labels for the compare paths provided. Must be supplied in the same order as compare paths.",
    )

    parser.add_argument(
        "--inception_filename",
        type=str,
        default='inception-v3.npy',
        help="Name of  inception file that is used for the  reference dataset",
    )

    parser.add_argument(
        "--inception_filename_base",
        type=str,
        default='inception-v3.npy',
        help="Name of inception file that is used for the  reference dataset",
    )

    parser.add_argument(
        "--save_folder",
        type=str,
        default='eval',
        help="Path of output folder.",
    )
    args = parser.parse_args()
    path2labels = {}
    if len(args.labels) != len(args.compare):
        raise ValueError(
            'Number of labels has to be equal to number of compare paths')
    for label, path in zip(args.labels, args.compare):
        path2labels[path] = label
    base = np.load(os.path.join(args.base, args.inception_filename_base))
    compare = {}
    for path in args.compare:
        compare[path] = np.load(os.path.join(path, args.inception_filename))
    fids = {}
    for path, value in compare.items():

        os.makedirs(os.path.join(
            args.base, args.save_folder, path2labels[path]), exist_ok=True)

        if not os.path.exists(os.path.join(args.base, args.save_folder, path2labels[path], 'fid.txt')):
            fid = frechet_inception_distance(base, value)
            with open(os.path.join(args.base, args.save_folder, path2labels[path], 'fid.txt'), "w") as fid_file:
                fid_file.write(str(fid))
            print('fid: ', fid)
        else:
            print('FID already exists.')

        if not os.path.exists(os.path.join(args.base, args.save_folder, path2labels[path], 'prd.npy.npz')):
            prd = compute_prd_from_embedding(
                eval_data=value,
                ref_data=base,
                num_clusters=20,
                num_angles=1001,
                num_runs=10,
                enforce_balance=True,
            )
            f_beta_data = prd_to_max_f_beta_pair(prd[0], prd[1], beta=8)
            np.savez_compressed(
                os.path.join(args.base, args.save_folder,
                             path2labels[path], 'prd.npy'),
                precision=prd[0],
                recall=prd[1],
                f_beta_data=f_beta_data,
            )
            print('Computed PRD')
        else:
            print('PRD already exists')

        if not os.path.exists(os.path.join(args.base, args.save_folder, path2labels[path], 'impar.npy.npz')):
            impar = knn_precision_recall_features(base, value, num_gpus=1)
            np.savez_compressed(
                os.path.join(args.base, args.save_folder,
                             path2labels[path], 'impar.npy'),
                precision=impar["precision"],
                recall=impar["recall"],
            )
            print('Computed IMPAR')
        else:
            print('IMPAR already exists.')
