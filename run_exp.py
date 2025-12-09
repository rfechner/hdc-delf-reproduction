from embed import embedding as preprocess_embedding
from evaluate import evaluate as compute_average_prec
from datetime import datetime

import os
import pickle
import numpy as np
import json

SEEDS = np.arange(10)
def evaluate(pickle_dir : str, preprocess_kwargs : dict = {}):

    # Use a stable, sorted order for the files so seeds map to files
    # deterministically across runs (os.listdir order is not guaranteed).
    pickle_filenames = sorted([x for x in os.listdir(pickle_dir) if "embedded" not in x and x.endswith('.pickle')])
    pickle_files = [os.path.join(pickle_dir, x) for x in pickle_filenames]
    print("Evaluating pickle files:\n", '\n'.join(pickle_files))

    results = {os.path.basename(key) : [] for key in pickle_files}
    for seed in SEEDS:
        np.random.seed(seed)
        preprocess_embedding(pickle_files, **preprocess_kwargs)
        processed_pickle_files = [
            os.path.join(pickle_dir, "embedded", os.path.basename(f)) for f in pickle_files
        ]

        for processed_pickle_file in processed_pickle_files:
            with open(processed_pickle_file, 'rb') as io:
                tmp = pickle.load(io)
                db, query, gts = tmp.values()
            gt_hard, gt_soft = gts['hard'].T, gts['soft'].T
            avg_prec = compute_average_prec(db, query, gthard=gt_hard, gtsoft=gt_soft).item()
            print(f'Seed: {seed}, File: {processed_pickle_file}, Avg Prec: {avg_prec:.2f}')
            key = os.path.basename(processed_pickle_file)
            results[key].append(avg_prec)

    date = datetime.now().isoformat()
    results_dir = os.path.join(pickle_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, f'{date}-results.json'), 'w') as file:
        json.dump(fp=file, obj=results)
    with open(os.path.join(results_dir, f'{date}-results.meta.json'), 'w') as file:
        json.dump(fp=file, obj={'pickle_dir' : pickle_dir, 'preprocess_kwargs' : preprocess_kwargs})
        
    for k, vs in results.items():
        print(k, "\t", "mean: ", np.mean(vs), " median: ", np.median(vs))

if __name__=='__main__':

    experiment_kwargs = {
        'normalization' : 'dataset'
    }
    evaluate('pickles/OxfordRobotCar', preprocess_kwargs=experiment_kwargs)
