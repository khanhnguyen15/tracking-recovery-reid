from collections import deque
from ..reid_inference import retrieve_candidates, get_best_candidates
import torch
import numpy as np

class IDTracker(object):

    def __init__(self, 
                 sequence='video', 
                 threshold=70.0, 
                 max_rank=5,
                 memory_len=5,
                 verbose=False):

        self.sequence = sequence
        self.threshold = threshold
        self.max_rank = max_rank
        self.memory_len = memory_len

        self.uIDs = set()
        self.IDs_counts = {}
        self.IDs_memory = {}
        self.history = {}
        self.change_log = {}
        self.last_frame_IDs = []
        self.mode = 'recent'

        self.verboseprint = print if verbose else lambda *args, **kwargs: None


    def set_mode(self, mode):
        self.mode = mode


    def register(self, pid, feature):
        if not pid in self.uIDs:
            self.uIDs.add(pid)
            self.IDs_counts[pid] = 0
            self.IDs_memory[pid] = deque(maxlen=self.memory_len)

        if self.mode == 'recent':
            self.register_recent(pid, feature)
        elif self.mode == 'sparse':
            self.register_sparse(pid, feature)
        elif self.mode == 'first':
            self.register_first(pid, feature)

        self.IDs_counts[pid] += 1


    def register_recent(self, pid, feature):
        self.IDs_memory[pid].append(feature)


    def register_sparse(self, pid, feature):
        if self.IDs_counts[pid] % 5 == 0:
            self.IDs_memory[pid].append(feature)


    def register_first(self, pid, feature):
        if self.IDs_counts[pid] < self.memory_len:
            self.IDs_memory[pid].append(feature)


    def get_gallery_IDs(self, IDs_to_compare):
        return np.array([ID 
                         for ID, feature in self.IDs_memory.items() 
                         if ID in IDs_to_compare 
                         for _ in range(len(feature))
                        ])


    def get_gallery_features(self, IDs_to_compare):

        gallery_from_memory = {ID: feature_list
                               for ID, feature_list in self.IDs_memory.items()
                               if ID in IDs_to_compare
                              }
        
        gallery_values = list(gallery_from_memory.values()) # list of deque
        values_as_tensor_list = list(map(lambda x: torch.cat(tuple(x), dim=0), gallery_values))
        return torch.cat(values_as_tensor_list, dim=0)


    def get_query(self, input_dict):
        query_IDs = np.array(list(input_dict.keys()))
        query_features = torch.cat(list(input_dict.values()), dim=0)

        return query_IDs, query_features


    def get_gallery(self, IDs_to_compare):
        gallery_IDs = self.get_gallery_IDs(IDs_to_compare)
        gallery_features = self.get_gallery_features(IDs_to_compare)

        return gallery_IDs, gallery_features


    def reassign_input(self, input_IDs, frame):
        for old_ID, new_ID in self.history.items():
            if old_ID in input_IDs.keys():
                input_IDs[new_ID] = input_IDs[old_ID]
                self.change_log[frame][old_ID] = new_ID
                del input_IDs[old_ID]


    def update(self, input_IDs, frame):

        input_IDs = input_IDs.copy()

        self.verboseprint('\n\nProcessing frame {}'.format(frame))
        self.change_log[frame] = {}

        if len(self.IDs_memory) == 0:
            for pid, feature in input_IDs.items():
                self.register(pid, feature)

            self.last_frame_IDs = list(input_IDs.keys())

        else:

            self.reassign_input(input_IDs, frame)

            new = {ID: feature 
                for ID, feature in input_IDs.items() 
                if ID not in self.last_frame_IDs
                }
            
            propagated = {ID: feature 
                        for ID, feature in input_IDs.items() 
                        if ID in self.last_frame_IDs
                        }
            
            lost_tracked_IDs = [ID for ID in self.IDs_memory.keys() if ID not in propagated]

            self.verboseprint('\tNew IDs tracked: {}'.format(new.keys()))
            self.verboseprint('\tIDs from previous frame: {}'.format(propagated.keys()))
            self.verboseprint('\tLost IDs tracked previously: {}\n'.format(lost_tracked_IDs))

            if len(new) == 0:
                self.verboseprint('\tThere is no new tracking indentity')
                for pid, feature in propagated.items():
                    self.register(pid, feature)

                self.last_frame_IDs = list(input_IDs.keys())
                return
            
            if len(lost_tracked_IDs) == 0:
                self.verboseprint('\tAll the IDs in history are presented in current frame')
                for pid, feature in propagated.items():
                    self.register(pid, feature)

                self.last_frame_IDs = list(input_IDs.keys())
                return
                
            self.verboseprint('\tIDs before reid: {}'.format(input_IDs.keys()))
            input_IDs_ = list(input_IDs.keys())
            input_features = torch.cat(list(input_IDs.values()), dim=0)

            qids, qf = self.get_query(new)
            gids, gf = self.get_gallery(lost_tracked_IDs)

            # print('\tQuery for deep-reid is {}'.format(qids))
            # print('\tGallery for deep-reid is {}'.format(gids))

            candmat, distmat = retrieve_candidates(
                query_features=qf,
                gallery_features=gf,
                query_IDs=qids,
                gallery_IDs=gids,
                max_rank=self.max_rank,
                return_distmat=True,
                reduce_avg=True
            )

            self.verboseprint('\tCandidates matrix {}'.format(candmat))

            candidates = get_best_candidates(distmat, candmat, query_IDs=qids)

            self.verboseprint('\tList candidates for query: {}'.format(candidates))

            for ID, (cand, dist) in candidates.items():
                # if the distance to the candidate is less than the threshold
                if dist < self.threshold:
                    self.verboseprint("\t\tDetected reid identity from {} to {}".format(ID, cand))
                    self.change_log[frame][ID] = cand
                    self.history[ID] = cand
                    input_IDs[cand] = input_IDs[ID]
                    del input_IDs[ID]

            self.verboseprint('\tNew IDs after reid: {}'.format(input_IDs.keys()))

            for pid, feature in input_IDs.items():
                self.register(pid, feature)

            self.last_frame_IDs = list(input_IDs.keys())


    def get_change_log(self):
        return self.change_log    


    def modify_annotations(self, annotations, frame):

        changes = self.change_log[frame]

        if not len(annotations) or not len(changes):
            return

        for ann in annotations:
            if type(ann) == dict:
                curr_id = ann['id_']
                if curr_id not in changes:
                    continue

                ann['id_'] = changes[curr_id].item() 

            else:
                # Annotation object
                curr_id = ann.id_
                if curr_id not in changes:
                    continue

                ann.id_ = changes[curr_id].item() 
        