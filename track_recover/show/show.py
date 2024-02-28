import json
import tqdm

import cv2
import numpy as np
import PIL
import matplotlib
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

from .utils import preprocess_bbox, save_image, save_sequence_image, trackings_by_frames
from ..trackval import load_gt_pr_pair, getPrIdxDict, getMatchGT


def visualize_query_candidates(retrieve_dict, query_paths, gallery_paths):
    for img_idx, retrieval_idx in retrieve_dict.items():
        query = query_paths[img_idx]
        query_img = np.array(PIL.Image.open(query))
        print('Query {} image path: {}'.format(img_idx, query.split('/')[-1]))
        
        retrieval = [gallery_paths[r_idx] for r_idx in retrieval_idx]

        # draw images
        ncols = retrieval_idx.size + 1
        fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))

        ax[0].imshow(query_img)
        ax[0].set_title('Query image {}'.format(img_idx))

        print('Retrieval image paths:')
        for idx, r in enumerate(retrieval):
            print('\t{}'.format(r.split('/')[-1]))
            r_img = np.array(PIL.Image.open(r))
            ax[idx + 1].imshow(r_img)
            ax[idx + 1].set_title('Retrieval  #{}'.format(idx))


# display tracking using pytorch
def display_trackings(image_file, anns, data_folder='data-posetrack2018/', gt=False):
    
    image_file = data_folder + image_file
    image = read_image(image_file)
    
    bboxes = []
    labels = []
    colors = []
    
    for ann in anns:
        bbox = preprocess_bbox(ann['bbox'], gt)
        label = 'ID: ' + str(ann['track_id'])
        
        color = matplotlib.cm.get_cmap('tab20')((ann['track_id'] % 20 + 0.05) / 20)
        color = tuple(int(i * 255) for i in color[:3])
        
        bboxes.append(bbox)
        labels.append(label)
        colors.append(color)
        
    bboxes = torch.tensor(bboxes, dtype=torch.int)
    image = draw_bounding_boxes(image, bboxes, width=5, labels=labels, colors=colors, font_size=2000)
    image = torchvision.transforms.ToPILImage()(image)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(image)


# using cv2 to save images
def draw_boxes(image, anns, gt=False, old=True):
    
    image_h, image_w, _ = image.shape
    
    for ann in anns:

        if 'bbox' not in ann.keys():
            continue

        bbox = preprocess_bbox(image, ann['bbox'], gt)
        xmin, ymin, xmax, ymax = bbox
        
        track_id = ann['track_id']
        if old:
            track_id = track_id[0]
        
        color = matplotlib.cm.get_cmap('tab20')((track_id % 20 + 0.05) / 20)
        color = tuple(int(i * 255) for i in color[:3])
        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

        # label box
        lxmin, lymin, lxmax, lymax = xmin - 1, ymin - 30, xmin + 60, ymin
        if lymin < 0:
            lymin, lymax = ymax, ymax + 30

        if lymax > image_h:
            lxmin, lxmax = xmax + 1, xmax + 60
            lymin, lymax = ymin, ymin + 30
        
        label_coord = (lxmin + 5, lymax - 10)
        
        label = 'ID: ' + str(track_id)

        if not gt:
            score = ann['score']
            if old:
                score = score[0]
                
            lxmax = lxmax + 60
            label = label + ' - ' + str(score)

        cv2.rectangle(image, (lxmin, lymin), (lxmax, lymax), color, -1)
        
        cv2.putText(image, 
                    label, 
                    label_coord, 
                    cv2.FONT_HERSHEY_PLAIN, 
                    1, 
                    (0, 0, 0), 
                    1)
    
    return image


def draw_boxes_light(image, anns, idxDict, 
                     clGroundTruth, cumuGroundTruth,
                     gt=False, old=True, draw_cl=True):
    
    image_h, image_w, _ = image.shape
    
    COLORS = {
        'BLACK': (0, 0, 0),
        'RED': (255, 0, 0),
        'GREEN': (0, 255, 0),
        'BLUE': (0, 0, 255),
        'GREY': (128,128,128),
        'DARK_GREEN': (0,95,0),
        'ORANGE': (215,95,0)
    }
    
    
    for ann in anns:

        if 'bbox' not in ann.keys():
            continue

        bbox = preprocess_bbox(image, ann['bbox'], gt)
        xmin, ymin, xmax, ymax = bbox
        
        track_id = ann['track_id']
        if old:
            track_id = track_id[0]
            
        if track_id not in idxDict:
            continue
        
        color = COLORS['BLUE']
        
        if 'old_id' in ann.keys():
            
            if idxDict[track_id]['type'] == 'FP':
                color = COLORS['GREY']
            else:
                matched_id = idxDict[track_id]['match']

                if draw_cl:
                    if track_id == clGroundTruth[matched_id]:
                        color = COLORS['DARK_GREEN']
                    else:
                        color = COLORS['ORANGE']

                else:
                    if track_id in cumuGroundTruth[matched_id]:
                        color = COLORS['DARK_GREEN']
                    else:
                        color = COLORS['ORANGE']
        
        else:
            if idxDict[track_id]['type'] == 'FP':
                color = COLORS['GREY']
            else:
                matched_id = idxDict[track_id]['match']

                if track_id == clGroundTruth[matched_id]:
                    color = COLORS['GREEN']
                else:
                    color = COLORS['RED']
        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

        # label box
        lxmin, lymin, lxmax, lymax = xmin - 1, ymin - 30, xmin + 60, ymin
        if lymin < 0:
            lymin, lymax = ymax, ymax + 30

        if lymax > image_h:
            lxmin, lxmax = xmax + 1, xmax + 60
            lymin, lymax = ymin, ymin + 30
        
        label_coord = (lxmin + 5, lymax - 10)
        
        label = 'ID: ' + str(track_id)

        if not gt:
            score = ann['score']
            if old:
                score = score[0]
                
            lxmax = lxmax + 60
            label = label + ' - ' + str(score)

        cv2.rectangle(image, (lxmin, lymin), (lxmax, lymax), color, -1)
        
        cv2.putText(image, 
                    label, 
                    label_coord, 
                    cv2.FONT_HERSHEY_PLAIN, 
                    1, 
                    (0, 0, 0), 
                    1)
    
    return image


def draw_sequence_new(sequence, pred_folder, save_folder=None, image_folder='data-posetrack2018/', gt=False):
    pred_file = pred_folder + sequence + '.json'

    with open(pred_file) as pf:
        pred = json.load(pf)

    image_dict = trackings_by_frames(pred)

    for image_file, anns in tqdm.tqdm(image_dict.items(), desc='Drawing annotations for {}'.format(sequence)):
        image_file = image_folder + image_file
        image = np.asarray(PIL.Image.open(image_file))

        if len(anns) != 0:
            image = draw_boxes(image, anns, gt)
        
        if save_folder:
            frame_str = image_file.split('/')[-1]
            save_image(image, sequence, frame_str, save_folder)


def draw_sequence_old(sequence, gtDir, prDir, save_folder=None, 
                      image_folder='data-posetrack2018/',
                      draw_gt=False
                     ):

    gtFrames, prFrames = load_gt_pr_pair(sequence, gtDir, prDir)

    if draw_gt:
        for idxGT, gtf in enumerate(gtFrames):
            image_file = image_folder + gtf['image'][0]['name']
            image = np.asarray(PIL.Image.open(image_file))
            anns = gtf['annorect']

            image = draw_boxes(image, anns, gt=True, old=True)

            if save_folder:
                save_gt_folder = save_folder + 'gt/images/'
                frame_str = str(idxGT) + '_' + image_file.split('/')[-1]
                save_image(image, sequence, frame_str, save_gt_folder)

    for idxPr, prf in enumerate(prFrames):
        image_file = image_folder + prf['image'][0]['name']
        image = np.asarray(PIL.Image.open(image_file))
        anns = prf['annorect']

        image = draw_boxes(image, anns, gt=False, old=True)

        if save_folder:
            pr_type = prDir.split('/')[-3]
            if pr_type == 'posetrack2018-preds':
                pr_type = 'original'

            save_pr_folder = save_folder + 'pr/' + pr_type + '/images/'
            frame_str = str(idxPr).zfill(2) + '_' + image_file.split('/')[-1]
            save_image(image, sequence, frame_str, save_pr_folder)


def draw_sequence_light(sequence, gtDir, prDir, originalPrDir,
                        save_folder=None, draw_cl=True,
                        image_folder='data-posetrack2018/'):
    
    gtFrames, prFrames = load_gt_pr_pair(sequence, gtDir, prDir)
    prIdxDict = getPrIdxDict(sequence, gtDir, prDir)
    
    clGT, cumuGT = getMatchGT(sequence, gtDir, originalPrDir)
    
    for idxPr, prf in enumerate(prFrames):
        
        image_file = image_folder + prf['image'][0]['name']
        image = np.asarray(PIL.Image.open(image_file))
        anns = prf['annorect']
        
        if idxPr in prIdxDict:
            idxDict = prIdxDict[idxPr]
            frGT = cumuGT[idxPr]
            image = draw_boxes_light(image, anns, idxDict, clGT, frGT,
                                     gt=False, old=True, draw_cl=draw_cl)
        
        if save_folder:
            pr_type = prDir.split('/')[-3]
            if pr_type == 'posetrack2018-preds':
                pr_type = 'original'
            else:
                if draw_cl:
                    pr_type = 'true'
                else:
                    pr_type = 'cumulative'
                
            frame_str = str(idxPr).zfill(2) + '_' + image_file.split('/')[-1]
            save_sequence_image(image, sequence, pr_type, frame_str, save_folder)
            
        plt.imshow(image)
        plt.figure(idxPr + 1)

    plt.show()