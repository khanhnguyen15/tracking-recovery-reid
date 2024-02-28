import argparse
import glob
from hashlib import new
import time
import json

from ..trackval import *
from ..utils import NpEncoder

def cli():
    parser = argparse.ArgumentParser(description="Processing tracking recovery on predictions")
    parser.add_argument('--ground-truth', required=True, type=str, help="Input directory of ground truth")
    parser.add_argument('--openpifpaf-pred', default='posetrack2018_openpifpaf/', type=str, help="Directory of OpenPifPaf original predictions")
    parser.add_argument('--modified-preds', default='posetrack2018_modified/', type=str, help="Directory modified predictions")
    parser.add_argument('-o', '--output', required=True, type=str, help="Output directory to save evaluation")

    
    args = parser.parse_args()
    return args

def main():
    args = cli()

    gtDir = args.ground_truth
    openpifpaf_pred = args.openpifpaf_pred
    modified_preds_folder = args.modified_preds
    
    output_folder = args.output

    # gtDir = 'data-posetrack2018/annotations/val/'
    # originalPrDir = 'posetrack2018-preds/annotations/'

    print('Evaluating the original predictions from OpenPifPaf')
    
    _, openpifpaf_tracking = evaluateTracking(gtDir, openpifpaf_pred)
    openpifpaf_true_accuracy, _ = evaluateAccuracy(gtDir, openpifpaf_pred, openpifpaf_pred, original=True)

    print('Metrics for original: ')
    print('MOTA:')
    print(json.dumps(openpifpaf_tracking, indent=4, sort_keys=True, cls=NpEncoder))
    print('True Accuracy: {}'.format(openpifpaf_true_accuracy))
    print('\n\n')

    openpifpaf_tracking['true_accuracy'] = openpifpaf_true_accuracy

    openpifpaf_eval_folder = output_folder + 'openpifpaf/'
    if not os.path.exists(openpifpaf_eval_folder):
        os.makedirs(openpifpaf_eval_folder)

    openpifpaf_eval_file = openpifpaf_eval_folder + 'metrics.json'
    with open(openpifpaf_eval_file, 'w') as f:
        json.dump(openpifpaf_tracking, f, indent=4, sort_keys=True, cls=NpEncoder)


    for modified_pred in glob.glob(modified_preds_folder + '/*//'):
        pred_name = modified_pred.split('/')[-2]

        print('Evaluating the modified predictions from {}'.format(pred_name))

        _, tracking = evaluateTracking(gtDir, modified_pred)
        trueAccuracy, cumuAccuracy = evaluateAccuracy(gtDir, modified_pred, openpifpaf_pred)

        print('Metrics for {}:'.format(pred_name))
        print('MOTA:')
        print(json.dumps(tracking, indent=4, sort_keys=True, cls=NpEncoder))
        print('True Accuracy: {}'.format(trueAccuracy))
        print('Recovered Accuracy: {}'.format(cumuAccuracy))
        print('\n\n')

        tracking['true_accuracy'] = trueAccuracy
        tracking['recovered_accuracy'] = cumuAccuracy


        eval_folder = output_folder + pred_name + '/'
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)

        eval_file = eval_folder + 'metrics.json'

        with open(eval_file, 'w') as f:
            json.dump(tracking, f, indent=4, sort_keys=True, cls=NpEncoder)

if __name__ == '__main__':
    main()

