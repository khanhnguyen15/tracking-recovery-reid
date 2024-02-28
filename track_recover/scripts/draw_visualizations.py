import argparse
import glob
from this import d
import time
import json

from ..show import *

def cli():
    parser = argparse.ArgumentParser(description="Drawing the annotations")

    parser.add_argument('-d', '--dataset', required=True, type=str, help="Directory of dataset")
    parser.add_argument('-g', '--groundtruth-folder', required=True, type=str, help="Directory of ground truth")
    parser.add_argument('-p', '--prediction-folder', required=True, type=str, help="Directory of prediction")
    parser.add_argument('-s', '--save-folder', required=True, type=str, help="Directory of annotations")

    args = parser.parse_args()
    return args

def main():
    args = cli()

    data_folder = args.dataset
    gt_folder = args.groundtruth_folder
    pr_folder = args.prediction_folder
    save_folder = args.save_folder

    draw_gt = False

    sequences = glob.glob(gt_folder + '*.json')
    sequences = [p.split('/')[-1].split('.json')[0] for p in sequences]

    for sequence in sequences:

        # check if the annotations is gt or prediction
        draw_sequence_old(sequence=sequence,
                          gtDir=gt_folder,
                          prDir=pr_folder,
                          save_folder=save_folder,
                          image_folder=data_folder,
                          draw_gt=draw_gt
                        )


if __name__ == '__main__':
    main()

