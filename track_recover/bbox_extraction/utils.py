import random
import os
import shutil
import json

import PIL
import matplotlib.pyplot as plt


def visualize(bbox_files, num_bboxes=3):
    for meta, bbox in random.sample(list(bbox_files.items()), num_bboxes):
        print(meta)
        plt.imshow(bbox)
        plt.show()


def save_bbox(save_path, file_name, bbox):
    file_path = save_path + '/bbox/' + file_name + '.jpg'
    bbox_img = PIL.Image.fromarray(bbox)
    bbox_img.save(file_path)


def save_pose_meta(save_path, file_name, pose_meta):
    file_path = save_path + '/pose_meta/' + file_name + '.json'

    with open(file_path, 'w') as f:
        json.dump(pose_meta, f, ensure_ascii=False, indent=4)