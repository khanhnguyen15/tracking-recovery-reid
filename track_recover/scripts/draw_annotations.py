import argparse
import glob
import time
import json

from ..show import *

def cli():
    parser = argparse.ArgumentParser(description="Drawing the annotations")

    parser.add_argument('-d', '--dataset', required=True, type=str, help="Directory of dataset")
    parser.add_argument('-a', '--annotations-folder', required=True, type=str, help="Directory of annotations")
    parser.add_argument('-s', '--save-folder', required=True, type=str, help="Directory of annotations")

    args = parser.parse_args()
    return args

def main():
    args = cli()

    data_folder = args.dataset
    anno_folder = args.annotations_folder
    save_folder = args.save_folder

    gt_known = False
    gt = False

    sequences = glob.glob(anno_folder + '*.json')
    sequences = [p.split('/')[-1].split('.json')[0] for p in sequences]

    for sequence in sequences:

        # check if the annotations is gt or prediction
        if not gt_known:
            anno_file = anno_folder + sequence + '.json'

            with open(anno_file, 'rb') as af:
                anno = json.load(af)

            if 'version' not in anno.keys():
                gt = True

            gt_known = True

        draw_sequence_new(sequence=sequence,
                      pred_folder=anno_folder,
                      save_folder=save_folder,
                      image_folder=data_folder,
                      gt=gt)


if __name__ == '__main__':
    main()

