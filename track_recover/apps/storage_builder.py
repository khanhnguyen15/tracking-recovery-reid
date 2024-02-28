import glob
from pathlib import Path
from PIL import Image
import numpy as np
import torch

from ..bbox_extraction import crop


class Storage(object):

    def __init__(self, extractor, device='cpu'):
        self.frame_dict = {}
        self.id_dict = {}
        self.extractor = extractor
        self.seq_len = 0
        self.device = torch.device(device)
        self.images = {}

    def build_from_files(self, sequence_path=None):
        '''
        Only used for Posetrack2018
        Build the storage from bbox images
        '''
        sequence = glob.glob(sequence_path + '/*.jpg')
        self.seq_len = len(sequence)

        features = self.extractor(sequence)
        for i, filename in enumerate(sequence):
            meta = Path(filename).stem
            pid = int(meta.split('_')[0])
            fid = int(meta.split('_')[1])

            if fid not in self.frame_dict.keys():
                self.frame_dict[fid] = {}

            if pid not in self.id_dict.keys():
                self.id_dict[pid] = {}

            self.frame_dict[fid][pid] = features[i].unsqueeze(0).to(self.device)
    

    def update_frame(self, image, annotations, gt=True):

        curr_fid = self.seq_len

        self.frame_dict[curr_fid] = {}
        self.images[curr_fid] = {}
        
        bboxes = []
        pids = []

        for ann in annotations:
            bbox = crop(ann['bbox'], image, gt=gt)
            bboxes.append(bbox)
            pids.append(ann['id_'])

            self.images[curr_fid][ann['id_']] = bbox

        features = self.extractor(bboxes)

        for i, pid in enumerate(pids):
            self.frame_dict[curr_fid][pid] = features[i].unsqueeze(0).to(self.device)

        self.seq_len += 1


    def get_frame(self, frame_id):
        return self.frame_dict[frame_id]


    def get_frame_keys(self):
        return sorted(list(self.frame_dict.keys()))


    def get_last_frame(self):
        last_fid = self.seq_len - 1
        return last_fid, self.frame_dict[last_fid]

