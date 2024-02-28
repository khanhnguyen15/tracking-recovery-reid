import os

import cv2
import numpy as np
import PIL
import matplotlib
import imageio
import glob
import tqdm

from IPython.display import clear_output

import matplotlib.pyplot as plt

IMG_PATH = 'data-posetrack2018/images/val/'

def preprocess_bbox(image, bbox, gt=False):
    image_h, image_w, _ = image.shape
    
    x1, y1, x2, y2 = np.int_(bbox)
    if gt: 
        w, h = x2, y2
        x2 = x1 + w + 1
        y2 = y1 + h + 1
    
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, image_w)
    y2 = min(y2, image_h)
    
    return [x1, y1, x2, y2]


def trackings_by_frames(prediction):
    
    images = prediction['images']
    annotations = prediction['annotations']
    
    id_to_file = {}
    image_to_trackings = {}
    
    for image in images:
        id_to_file[image['frame_id']] = image['file_name']
        image_to_trackings[image['file_name']] = []

    for ann in annotations:
        image_id = ann['image_id']
        file_name = id_to_file[image_id]
        image_to_trackings[file_name].append(ann)
    
    return image_to_trackings


def save_image(image, sequence, frame_str, save_folder='posetrack2018-preds-new/images/'):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    save_path = save_folder + sequence + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    cv2.imwrite(save_path + frame_str, img)


def save_sequence_image(image, sequence, pr_type, frame_str, save_folder='posetrack2018-preds-new/images/'):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    save_path = save_folder + sequence + '/' + pr_type + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    cv2.imwrite(save_path + frame_str, img)

def create_gif(sequence, images_folder):
    filenames = sorted(glob.glob(images_folder + '*.jpg'))
    save_path = images_folder + sequence + '.gif'
    
    images = []
    for f in filenames:
        images.append(imageio.imread(f))
    
    imageio.mimsave(save_path, images)

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