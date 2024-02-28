import json
import copy
import os
from statistics import mode

from tqdm import tqdm

from .id_tracker import IDTracker
from ..utils import CPU_Unpickler, NpEncoder


def modify_annotations(change_log, annotations):
    for i, anno in enumerate(annotations):
        frame = int(str(anno['image_id'])[-4:])
        if frame not in change_log or len(change_log[frame]) == 0:
            continue
        
        track_id = anno['track_id']
        if track_id not in change_log[frame]:
            continue
        
        anno['old_id'] = track_id
        anno['track_id'] = change_log[frame][track_id].item() # casting to int
        
        print('\tAnnotation {}: Modified ID#{} into ID#{} at frame {}'
              .format(i, track_id, change_log[frame][track_id], frame))


def get_new_prediction(sequence, 
                       pred_path='./posetrack2018-openpifpaf/', 
                       storage_path=None,
                       threshold=70.0,
                       max_rank=10,
                       memory_len=5,
                       mode='recent'
                      ):
                      
    if storage_path != None:
        # already have stored bbox storage
        pred_file = pred_path + sequence + '.json'
        storage_file = storage_path + sequence +'.pkl'
        with open(storage_file, 'rb') as sf:
            storage = CPU_Unpickler(sf).load()
        with open(pred_file, 'rb') as pf:
            pred = json.load(pf)
            
        # building the id_tracker
        id_tracker = IDTracker(sequence, 
                               threshold=threshold, 
                               max_rank=max_rank, 
                               memory_len=memory_len, 
                               verbose=False
                              )
        
        id_tracker.set_mode(mode)

        for i in storage.get_frame_keys():
            id_tracker.update(input_IDs=storage.get_frame(i), frame=i)
            
        change_log = id_tracker.get_change_log()
        
        new_pred = copy.deepcopy(pred)
        annotations = new_pred['annotations']
        modify_annotations(change_log, annotations)
        
        return new_pred
        
    else:
        raise NotImplementedError("Integrated ID recovery not yet implemented")


def save_prediction(pred, sequence, config, save_folder='./posetrack2018-modified/'):
    
    method, th, mr, ml = config

    sub_folder = '{}-th-{}-mr-{}-ml-{}/'.format(method, th, mr, ml)

    save_folder = save_folder + sub_folder

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    save_file = save_folder + sequence + '.json'
    with open(save_file, 'w') as f:
        json.dump(pred, f, ensure_ascii=False, indent=4)
