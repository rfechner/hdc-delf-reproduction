from embed import embedding as preprocess_embedding
from evaluate import evaluate as compute_average_prec
from datetime import datetime

import os
import pickle
import numpy as np
import json

SEEDS = np.arange(10)
def evaluate(pickle_dir : str):

    pickle_files = [os.path.join(pickle_dir, x) for x in os.listdir(pickle_dir)]
    print("Evaluating pickle files:\n", '\n'.join(pickle_files))

    results = {key : [] for key in pickle_files}
    for seed in SEEDS:
        np.random.seed(seed)
        preprocess_embedding(pickle_files)
        processed_pickle_files = [
            os.path.join(pickle_dir, f"embedded-{os.path.basename(f)}") for f in pickle_files
        ]

        for processed_pickle_file in processed_pickle_files:
            with open(processed_pickle_file, 'rb') as io:
                tmp = pickle.load(io)
                db, query, gts = tmp.values()
            gt_hard, gt_soft = gts['hard'].T, gts['soft'].T
            avg_prec = compute_average_prec(db, query, gthard=gt_hard, gtsoft=gt_soft).item()
            print(f'Seed: {seed}, File: {processed_pickle_file}, Avg Prec: {avg_prec:.2f}')
            results[processed_pickle_file].append(avg_prec)

    date = datetime.now().isoformat()
    with open(os.path.join(pickle_dir, f'{date}-results.json'), 'w') as file:
        json.dump(fp=file, obj=results)

    for k, vs in results.items():
        print(k, "\t", "mean: ", np.mean(vs), " median: ", np.median(vs))

if __name__=='__main__':
    evaluate('pickles/OxfordRobotCar')