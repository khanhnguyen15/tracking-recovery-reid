import glob
import os
import tqdm

import motmetrics as mm
from .trackval_helper import *

def evaluateTracking(gtDir, prDir):
    
    sequences = glob.glob(gtDir + '/*.json')
    sequences = [os.path.basename(seq).split('.')[0] for seq in sequences]

    prName = prDir.split('/')[-2]
    
    motMetricsAll = {}
    
    totalMetrics = {}
    totalMetrics['num_misses'] = 0
    totalMetrics['num_switches'] = 0
    totalMetrics['num_false_positives'] = 0
    totalMetrics['num_objects'] = 0
    
    mh = mm.metrics.create()
    
    METRICS = ['num_misses', 'num_switches', 'num_false_positives', 'num_objects', 'num_detections', 'mota']
    
    for i in tqdm.tqdm(range(len(sequences)), desc='Evaluating trackings {}'.format(prName), leave=True, position=0):
    # for i in range(len(sequences)):
        sequence = sequences[i]
        
        acc = build_acc(sequence, gtDir, prDir)
        
        metrics = mh.compute(acc, 
                     metrics=METRICS,
                     return_dataframe=False,
                     name='acc'
                    )
        
        motMetricsAll[i] = {}
        motMetricsAll[i]['seq_name'] = sequence
        motMetricsAll[i]['num_misses'] = metrics['num_misses']
        motMetricsAll[i]['num_switches'] = metrics['num_switches']
        motMetricsAll[i]['num_false_positives'] = metrics['num_false_positives']
        motMetricsAll[i]['num_objects'] = metrics['num_objects']
        motMetricsAll[i]['mota'] = metrics['mota']
        
        totalMetrics['num_misses'] += metrics['num_misses']
        totalMetrics['num_switches'] += metrics['num_switches']
        totalMetrics['num_false_positives'] += metrics['num_false_positives']
        totalMetrics['num_objects'] += metrics['num_objects']
    
    totalMetrics['mota'] = 1.0 - (totalMetrics['num_misses'] + totalMetrics['num_switches'] +
                                  totalMetrics['num_false_positives']) / totalMetrics['num_objects']
    
    return motMetricsAll, totalMetrics


def evaluateClassificationSeq(sequence, gtDir, prDir, clGroundTruths, cumulative=False):
    
    acc = build_acc(sequence, gtDir, prDir)
    
    frameIds = acc.mot_events.reset_index()['FrameId'].unique()
    
    frameMatches = {}
    
    for fidx in frameIds:
        frameMatches[fidx] = {}
        
        frame = acc.mot_events.loc[fidx]
        frame = frame[(frame['Type'] == 'MATCH') | (frame['Type'] == 'SWITCH')]
        
        for i in range(len(frame)):
            OId = int(frame.iloc[i]['OId'])
            HId = int(frame.iloc[i]['HId'])
            
            frameMatches[fidx][OId] = HId
        
    accuracy = []
    
    for fidx in frameMatches.keys():
        frameMatch = frameMatches[fidx]
        
        correct = 0
        total = 0
        
        if not cumulative:
            for gidx in frameMatch.keys():
                if frameMatch[gidx] == clGroundTruths[gidx]:
                    correct += 1
                total += 1
        else:
            for gidx in frameMatch.keys():
                if frameMatch[gidx] in clGroundTruths[fidx][gidx]:
                    correct += 1
                total += 1
            
        if total == 0:
            continue
        
        accuracy.append({"Correct": correct, 
                         "Total": total, 
                         "Accuracy": 1.0 * correct / total
                        })
        
    return frameMatches, accuracy
    

def evaluateAccuracy(gtDir, prDir, originalPrDir, original=False):
    
    sequences = glob.glob(gtDir + '/*.json')
    sequences = [os.path.basename(seq).split('.')[0] for seq in sequences]

    prName = prDir.split('/')[-2]
    
    trueAccuracies = []
    cumuAccuracies = []
    
    for i in tqdm.tqdm(range(len(sequences)), desc='Evaluating accuracy {}'.format(prName), leave=True, position=0):
    # for i in range(len(sequences)):
        sequence = sequences[i]
        
        clGT, cumuGT = getMatchGT(sequence, gtDir, originalPrDir)
        
        _, trueAccuracy = evaluateClassificationSeq(sequence, gtDir, prDir, clGT, cumulative=False)
        trueAccuracies += trueAccuracy
        
        if not original:
            _, cumuAccuracy = evaluateClassificationSeq(sequence, gtDir, prDir, cumuGT, cumulative=True)
            cumuAccuracies += cumuAccuracy


    return calAccuracy(trueAccuracies), calAccuracy(cumuAccuracies)