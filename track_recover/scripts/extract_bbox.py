import argparse
import json
import glob
from tqdm import tqdm
import os
import shutil

from ..bbox_extraction import extract_bbox
from ..bbox_extraction import save_bbox, save_pose_meta

def cli():
    parser = argparse.ArgumentParser(description="Extraction of bboxes drom OpenPifPaf predictions")
    parser.add_argument('-i', '--input', required=True, type=str, help='Input directory of predictions')
    parser.add_argument('-d', '--dataset', default='data/data-posetrack2018/', type=str, help="Directory of the dataset")
    parser.add_argument('-o', '--output', default='bbox_extraction/', type=str, help="Output folder to stor bbox")
    parser.add_argument('-g', '--ground-truth', default=False, action='store_true', help='Flag if this is the ground truth')
    
    args = parser.parse_args()
    return args

def main():
    args = cli()
    dataset = args.dataset
    output = args.output
    gt = args.ground_truth

    if gt:
        print("Setting up for ground truth files")
    else:
        print("Setting up for prediction files")

    pred_path = args.input
    if gt:
        pred_path = dataset + 'annotations/val'

    print(pred_path)
    paths = glob.glob(pred_path + '/*.json')
    print('Found {} sequences!'.format(len(paths)))

    print('Image dataset folder: {}'.format(dataset))
    print('Reading prediction...')

    # path_sample = paths[0:2]
    seq_count = 0

    for p in paths:
        seq_count += 1
        sequence = p.split('/')[-1].split('.json')[0]
        
        with open(p) as f:
            data = json.load(f)
        
        print('[{}/{}] Extracting bboxes from {}'.format(seq_count, len(paths), sequence))
        bbox_dict, pose_dict = extract_bbox(dataset, data, gt)
        print('{} bboxes extracted'.format(len(bbox_dict)))

        save_path = output + sequence
        
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        
        os.makedirs(save_path + '/bbox')
        os.makedirs(save_path + '/pose_meta')

        print('Saving bboxes to {}'.format(save_path))
        for file_name, bbox in bbox_dict.items():
            save_bbox(save_path, file_name, bbox)

        for file_name, pose_meta in pose_dict.items():
            save_pose_meta(save_path, file_name, pose_meta)

if __name__ == '__main__':
    main()