import numpy as np
from shapely import geometry
import sys
import os
import json
import glob
import copy

import motmetrics as mm

from .poseval.convert import convert_videos
from .poseval.eval_helpers import Joint, cleanupData, removeIgnoredPoints, getPointGTbyID, getHeadSize


def calAccuracy(accuracy):
    if len(accuracy) == 0:
        return None

    correct = 0
    total = 0
    
    for i in range(len(accuracy)):
        correct += accuracy[i]['Correct']
        total += accuracy[i]['Total']
        
    return correct / total


def load_gt_pr_pair(sequence, gt_dir, pr_dir):

    if not os.path.exists(gt_dir):
        help('Given GT directory ' + gt_dir + ' does not exist!\n')
    if not os.path.exists(pr_dir):
        help('Given prediction directory ' + pred_dir + ' does not exist!\n')

    gt_filename = gt_dir + sequence + '.json'
    pr_filename = pr_dir + sequence + '.json'
    if not os.path.exists(gt_filename):
        help('The sequence ' + sequence + ' does not exist!\n')

    # load the gt
    with open(gt_filename) as gt_file:
        gt_data = json.load(gt_file)
    if (not 'annolist' in gt_data):
        gt_data = convert_videos(gt_data)[0]
               
    gt = gt_data['annolist']
    for imgidx in range(len(gt)):
        gt[imgidx]['seq_name'] = os.path.basename(gt_filename).split('.')[0]
        gt[imgidx]['img_name'] = os.path.basename(gt[imgidx]['image'][0]['name']).split('.')[0]

    gtFrames = gt


    #load predictions
    with open(pr_filename) as pr_file:
        pr_data = json.load(pr_file)
    if (not 'annolist' in pr_data):
        pr_data = convert_videos(pr_data)[0]
    
    pr = pr_data['annolist']
    for imgidx in range(len(pr)):
        pr[imgidx]['seq_name'] = os.path.basename(pr_filename).split('.')[0]
        pr[imgidx]['img_name'] = os.path.basename(pr[imgidx]['image'][0]['name']).split('.')[0]

    prFrames = pr

    # remove non annotated frames and ingnored regions
    gtFrames, prFrames = cleanupData(gtFrames, prFrames)
    gtFrames, prFrames = removeIgnoredPoints(gtFrames, prFrames)

    assert len(gtFrames) == len (prFrames), 'The ground truths and predictions have different size'

    return gtFrames, prFrames


def getJointStats(gtFrames, prFrames):
    assert len(gtFrames) == len(prFrames)

    nJoints = Joint().count

    stats = {}

    for imgidx in range(len(gtFrames)):
        # populate distances
        dist = np.full((len(prFrames[imgidx]['annorect']), len(gtFrames[imgidx]['annorect']), nJoints), np.inf)

        # if predictions has joints
        hasPr = np.zeros((len(prFrames[imgidx]['annorect']), nJoints), dtype=bool)

        # if ground truths has joints
        hasGT = np.zeros((len(gtFrames[imgidx]['annorect']), nJoints), dtype=bool)

        # tracking IDs from gt and pr
        trackidxGT = []
        trackidxPr = []

        # remove prediction poses without points
        # likely wont happen
        idxsPr = []
        for ridxPr in range(len(prFrames[imgidx]['annorect'])):
            if (('annopoints' in prFrames[imgidx]['annorect'][ridxPr].keys()) and
                ('point' in prFrames[imgidx]['annorect'][ridxPr]['annopoints'][0].keys())):
                idxsPr += [ridxPr];
        prFrames[imgidx]['annorect'] = [prFrames[imgidx]['annorect'][ridx] for ridx in idxsPr]

        # populating hasGT and trackidxGT
        # iterating over poses (rectangles)
        for ridxGT in range(len(gtFrames[imgidx]['annorect'])):
            
            rectGT = gtFrames[imgidx]['annorect'][ridxGT]
            if ('track_id' in rectGT.keys()):
                trackidxGT += [rectGT['track_id'][0]]

            
            pointsGT = []
            if len(rectGT['annopoints']) > 0:
                pointsGT = rectGT['annopoints'][0]['point']
            # iterate over all possible body joints
            for i in range(nJoints):
                
                ppGT = getPointGTbyID(pointsGT, i)
                if len(ppGT) > 0:
                    hasGT[ridxGT, i] = True


        # populating hasPr and trackidxPr
        for ridxPr in range(len(prFrames[imgidx]['annorect'])):
            
            rectPr = prFrames[imgidx]['annorect'][ridxPr]
            if ('track_id' in rectPr.keys()):
                trackidxPr += [rectPr['track_id'][0]]


            pointsPr = rectPr['annopoints'][0]['point']
            for i in range(nJoints):
                
                ppPr = getPointGTbyID(pointsPr, i)
                if len(ppPr) > 0:
                    hasPr[ridxPr, i] = True

        
        # populating dist matrix for individual joints
        # dist[pr, gt, jointA] is distance between jointAs of pr and gt
        if len(prFrames[imgidx]['annorect']) and len(gtFrames[imgidx]['annorect']):

            # iterate over GT poses
            for ridxGT in range(len(gtFrames[imgidx]['annorect'])):

                rectGT = gtFrames[imgidx]['annorect'][ridxGT]

                headSize = getHeadSize(rectGT['x1'][0], rectGT['y1'][0],
                                                    rectGT['x2'][0], rectGT['y2'][0])

                pointsGT = []
                if len(rectGT['annopoints']) > 0:
                    pointsGT = rectGT['annopoints'][0]['point']

                # iterate over prediction poses
                for ridxPr in range(len(prFrames[imgidx]['annorect'])):

                    rectPr = prFrames[imgidx]['annorect'][ridxPr]
                    pointsPr = rectPr['annopoints'][0]['point']

                    for i in range(nJoints):
                        # get corresponding joints in GT and Pr
                        ppGT = getPointGTbyID(pointsGT, i)
                        ppPr = getPointGTbyID(pointsPr, i)

                        # compute distance between two joints
                        if hasPr[ridxPr, i] and hasGT[ridxGT, i]:

                            pointPr = [ppPr['x'][0], ppPr['y'][0]]
                            pointGT = [ppGT['x'][0], ppGT['y'][0]]

                            dist[ridxPr, ridxGT, i] = np.linalg.norm(np.subtract(pointGT, pointPr)) / headSize

                # dist = np.array(dist)
                # hasGT = np.array(hasGT)
        
        imgStat = {}

        imgStat['trackidxGT'] = trackidxGT
        imgStat['trackidxPr'] = trackidxPr
        imgStat['img_name'] = gtFrames[imgidx]['img_name']
        imgStat['distMat'] = dist
        imgStat['hasPr'] = hasPr
        imgStat['hasGT'] = hasGT

        stats[imgidx] = imgStat

    return stats


def matchHypotheses(stats, jointThresh=0.5, poseThresh=0.5):
    
    motAll = {}
    
    for imgidx in stats:
        img_name = stats[imgidx]['img_name']
        trackidxGT = stats[imgidx]['trackidxGT']
        trackidxPr = stats[imgidx]['trackidxPr']
        jDist = stats[imgidx]['distMat']
        hasPr = stats[imgidx]['hasPr']
        hasGT = stats[imgidx]['hasGT']

        jDist = np.array(jDist)
        hasGT = np.array(hasGT)

        # total annotated joints for each poses
        nGTjoints = np.sum(hasGT, axis=1)

        # only keep small distance as matchings
        jMatch = jDist <= jointThresh

        # calculate the number of matchings between poses Pr and GT
        pDist = 1.0 * np.sum(jMatch, axis=2)
        # normalize over total of GT joints
        for i in range(hasPr.shape[0]):
            for j in range(hasGT.shape[0]):
                if nGTjoints[j] > 0:
                    pDist[i, j] = 1.0 - pDist[i, j] / nGTjoints[j]

        # keep only best match?
        pMatch = pDist <= poseThresh
        
        
        mot = {}
        mot['img_name'] = img_name
        mot['trackidxGT'] = trackidxGT
        mot['trackidxPr'] = trackidxPr
        mot['dist'] = np.full((len(trackidxGT), len(trackidxPr)), np.nan)
        for pPr in range(len(trackidxPr)):
            for pGT in range(len(trackidxGT)):
                if pMatch[pPr, pGT]:
                    mot['dist'][pGT, pPr] = pDist[pPr, pGT]
                    
        motAll[imgidx] = mot
        
    return motAll

def build_acc(sequence, gtDir, prDir):

    gtFrames, prFrames = load_gt_pr_pair(sequence, gtDir, prDir)
    
    jointStats = getJointStats(gtFrames, prFrames)
    motAll = matchHypotheses(jointStats)
    
    acc = mm.MOTAccumulator(auto_id=True)
    for idx in motAll.keys():
        trackidxGT = motAll[idx]['trackidxGT']
        trackidxPr = motAll[idx]['trackidxPr']
        dist = motAll[idx]['dist']

        acc.update(
            trackidxGT,
            trackidxPr,
            dist
        )

    return acc


def getMatchGT(sequence, gtDir, originalPrDir):
    
    acc = build_acc(sequence, gtDir, originalPrDir)
    
    # acc must be original predictions
    matches = acc.mot_events[acc.mot_events['Type'] == 'MATCH'].reset_index()
    groundTruth = {}
    cGroundTruth = {}
    
    for midx in range(len(matches)):
        gtId = int(matches.iloc[midx]['OId'])
        prId = int(matches.iloc[midx]['HId'])
        
        if gtId not in groundTruth:
            groundTruth[gtId] = prId
            
    frameIds = acc.mot_events.reset_index()['FrameId'].unique()
    
    for fidx in frameIds:
        if (fidx >= 1) and (fidx - 1 in cGroundTruth):
            cGroundTruth[fidx] = copy.deepcopy(cGroundTruth[fidx - 1])
        else:
            cGroundTruth[fidx] = {}
        
        frame = acc.mot_events.loc[fidx]
        frame = frame[(frame['Type'] == 'MATCH') | (frame['Type'] == 'SWITCH')]
        
        
        for i in range(len(frame)):
            OId = int(frame.iloc[i]['OId'])
            HId = int(frame.iloc[i]['HId'])
            
            if OId in cGroundTruth[fidx]:
                if HId not in cGroundTruth[fidx][OId]:
                    cGroundTruth[fidx][OId].append(HId)
            else:
                cGroundTruth[fidx][OId] = [HId]
            
    return groundTruth, cGroundTruth


def getPrIdxDict(sequence, gtDir, prDir):
    
    idxDict = {}
    
    acc = build_acc(sequence, gtDir, prDir)
    frameIds = acc.mot_events.reset_index()['FrameId'].unique()
    
    for fidx in frameIds:
        idxDict[fidx] = {}
        
        frame = acc.mot_events.loc[fidx]
        frame = frame[(frame['Type'] == 'MATCH') | 
                      (frame['Type'] == 'SWITCH') |
                      (frame['Type'] == 'FP')
                     ]
        
        for i in range(len(frame)):
            HId = int(frame.iloc[i]['HId'])
            idxDict[fidx][HId] = {}
            
            if (frame.iloc[i]['Type'] == 'MATCH') or (frame.iloc[i]['Type'] == 'SWITCH'):
                idxDict[fidx][HId]['type'] = 'MATCH'
                idxDict[fidx][HId]['match'] = int(frame.iloc[i]['OId'])
            else:
                idxDict[fidx][HId]['type'] = 'FP'
                idxDict[fidx][HId]['match'] = -1
            
    return idxDict