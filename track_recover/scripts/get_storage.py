import torch
import torchvision
from torch import nn
import numpy as np
import os
import glob
import pickle
import time
import shutil
import argparse

from ..reid_inference import FeatureExtractor
from ..apps import Storage

def cli():
    parser = argparse.ArgumentParser(description="Building storages for the visual features")
    parser.add_argument('-g', '--gallery', required=True, type=str, help='Gallery folder containing bboxes')
    parser.add_argument('-s', '--storage', default='data/data-posetrack2018/', type=str, help="Storage folder to save the storage files")
    parser.add_argument('--reid-model', default='reid_model/best_model.pth.tar', type=str, help="Path to deep re-ID model")
    parser.add_argument('-d', '--device', default='cpu', type=str, help='Device type of tensor to store within the storage. Default to cpu for smaller memory usage.')
    
    args = parser.parse_args()
    return args

def main():

    args = cli()
    
    gallery_folder = args.gallery
    storage_folder = args.storage
    model_path = args.reid_model
    device = args.device

    if os.path.exists(storage_folder):
        shutil.rmtree(storage_folder)
    
    os.makedirs(storage_folder)

    print('##################################################################')
    print('Start building storages files')

    start = time.time()
    print('Loading extractor...')
    extractor = FeatureExtractor(model_path)
    print('Extractor loaded successfully in {:.4f}'.format(time.time() - start))
    
    seq_count = 0
    sequences = glob.glob(gallery_folder + '*//')

    for sequence in sequences:
        seq_count += 1

        storage_name = sequence.split('/')[-2]
        storage_file = storage_name + '.pkl'
        
        sequence_path = sequence + 'bbox/'
        storage_path = storage_folder + storage_file

        print('[{}/{}] Building storage for {}...'.format(seq_count, len(sequences), storage_name))

        storage = Storage(extractor=extractor, device=device)

        start = time.time()
        storage.build_from_files(sequence_path=sequence_path)

        print('\tBuilt storage of {} entities in {:.4f}s'.format(storage.seq_len, time.time() - start))
    
        print('\tSaving storage to {}\n\n'.format(storage_file))
        with open(storage_path, 'wb') as wf:
            pickle.dump(storage, wf)
        
        del storage


if __name__ == '__main__':
    main()