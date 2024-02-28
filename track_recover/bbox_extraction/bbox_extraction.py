import PIL
import numpy as np
from tqdm import tqdm

def crop(bbox, image, gt=False):
    x1, y1, x2, y2 = np.int_(bbox)
    if gt: 
        w, h = x2, y2
        x2 = x1 + w + 1
        y2 = y1 + h + 1

    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, 1920)
    y2 = min(y2, 1080)
    
    cropped = image[y1:y2, x1:x2]
    return cropped


def extract_bbox_meta(dataset, anno, images, gt=False):
    # dataset = './data-posetrack2018/'
    if not gt:
        image_path = dataset + anno['image_file']
    else:
        it = next((img for img in images if img['frame_id'] == anno['image_id']), None)
        image_path = dataset + it['file_name']
    
    track_id = str(anno['track_id']).zfill(4)
    frame_id = str(anno['image_id'])[-4:]
    
    image = np.asarray(PIL.Image.open(image_path))
    bbox = crop(anno['bbox'], image, gt)    
    
    return track_id, frame_id, bbox, image_path


def extract_pose_meta(anno, gt=False):

    pose_meta = {}

    pose_meta['keypoints'] = anno['keypoints']
    pose_meta['score'] = anno['score'] if not gt else None

    return pose_meta


def extract_bbox(dataset, annotations_file, gt=False):
    images = annotations_file['images']
    annotations = annotations_file['annotations']
    
    bbox_dict = {}
    pose_dict = {}
    
    for anno in tqdm(annotations):
        if 'bbox' not in anno.keys():
            continue

        track_id, frame_id, bbox, image_path = extract_bbox_meta(dataset, anno, images, gt)
        pose_meta = extract_pose_meta(anno, gt)

        file_name = track_id + '_' + frame_id
        
        if file_name in bbox_dict:
            tqdm.write('Ooops')
        
        bbox_dict[file_name] = bbox
        pose_dict[file_name] = pose_meta
    
    return bbox_dict, pose_dict
