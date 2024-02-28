import argparse
import glob
import time

from tqdm import tqdm

from ..apps import *

def cli():
    parser = argparse.ArgumentParser(description="Processing tracking recovery on predictions")
    parser.add_argument('-i', '--input', required=True, type=str, help="Input directory of openpifpaf predictions")
    parser.add_argument('-s', '--storage', default='storage/', type=str, help="Directory features storage")
    parser.add_argument('-o', '--output', required=True, type=str, help="Output directory for predictions with recovered tracke IDs")
    parser.add_argument('--threshold', default=50.0, type=float, help="Maximum distance threshold to recover IDs")
    parser.add_argument('--max-rank', default=5, type=int, help="Maximum number of recovery candidates to select the best from")
    parser.add_argument('--memory-len', default=5, type=int, help="Length of short-term memory")
    parser.add_argument('--mode', default='recent', type=str, help="Mode of short-term memory")

    
    args = parser.parse_args()
    return args

def main():
    args = cli()
    pred_path = args.input
    save_path = args.output
    storage_path = args.storage

    threshold = args.threshold
    max_rank = args.max_rank
    memory_len = args.memory_len

    method = args.mode

    print('Method: {}'.format(method))
    print('Parameters:\n\tThreshold: {}\n\tMax rank: {}\n\tMemory length: {}\n'
                        .format(threshold, max_rank, memory_len))

    config = [method, threshold, max_rank, memory_len]

    sequences = [p.split('/')[-1].split('.json')[0] 
                 for p in glob.glob(pred_path + '*.json')
                ]

    # for sequence in tqdm(sequences, desc='Processing sequences'):
    for sequence in sequences:
        tqdm.write('#############################################')
        tqdm.write('SEQUENCE {}\n'.format(sequence))


        start = time.time()
        new_pred = get_new_prediction(sequence,
                                      pred_path=pred_path,
                                      storage_path=storage_path,
                                      threshold=threshold,
                                      max_rank=max_rank,
                                      memory_len=memory_len,
                                      mode=method
                                     )
        
        save_prediction(new_pred, sequence, config, save_folder=save_path)
        tqdm.write('\nMODIFIED PREDICTION FOR SEQUENCE {} DONE!'.format(sequence))
        tqdm.write('TIME TAKEN: {}'.format(time.time() - start))
        tqdm.write('#############################################\n\n')

if __name__ == '__main__':
    main()