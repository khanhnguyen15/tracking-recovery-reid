import pickle
import io
import glob
import os
import json

from IPython.display import clear_output

import tqdm
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np

IMG_PATH = 'data-posetrack2018/images/val/'

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def save_video(sequence, save_folder='posetrack2018-video/val/'):

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    image_folder = IMG_PATH + sequence
    video_file = save_folder + sequence + '.mp4'
    
    images = sorted(glob.glob(image_folder + '/*.jpg'))
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_file, fourcc, 20.0, (width, height))
    
    for image in tqdm.tqdm(images, desc='Processing sequence {}'.format(sequence)):
        frame = cv2.imread(image)
        video.write(frame)
    
    video.release()

def show_video(video_files):
    if len(video_files) == 1:
        # only mp4 file
        vid = cv2.VideoCapture(video_files)
        try:
            while(True):
                plt.figure(figsize = (12,8))
                # Capture frame-by-frame
                ret, frame = vid.read()
                if not ret:
                    vid.release()
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                plt.axis('off')

                plt.imshow(frame)
                plt.show()
                clear_output(wait=True)
        except KeyboardInterrupt:
            vid.release()
    else:
        # list of frames
        for image in video_files:
            plt.figure(figsize = (12,8))
            frame = cv2.imread(image)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.axis('off')

            plt.imshow(frame)
            plt.show()
            clear_output(wait=True)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)