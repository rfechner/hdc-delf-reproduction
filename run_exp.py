from embed_like_paper import main as prepare_data
from evaluate import evaluate_new
import pickle
import numpy as np
import json
from datetime import datetime

seeds = np.arange(10)
pickle_files = ["exp2-preprocessed-DB_2014-12__Q_2015-05.pickle",
                'exp2-preprocessed-DB_2014-12__Q_2015-08.pickle']
results = {key : [] for key in pickle_files}
for seed in seeds:
    np.random.seed(seed)
    prepare_data()

    for pickle_file in pickle_files:
        with open(pickle_file, 'rb') as io:
            tmp = pickle.load(io)
            db, query, gts = tmp.values()
        gt_hard, gt_soft = gts['hard'].T, gts['soft'].T
        avg_prec = evaluate_new(db, query, gthard=gt_hard, gtsoft=gt_soft).item()
        print(f'Seed: {seed}, File: {pickle_file}, Avg Prec: {avg_prec:.2f}')
        results[pickle_file].append(avg_prec)

date = datetime.now().isoformat()
with open(f'{date}-results.json', 'w') as file:
    json.dump(fp=file, obj=results)

for k, vs in results.items():
    print(k, "\t", "mean: ", np.mean(vs), " median: ", np.median(vs))
