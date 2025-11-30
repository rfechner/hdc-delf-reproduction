from embed_like_paper import main as prepare_data
from evaluate import evaluate_new
import pickle
import numpy as np

seeds = np.arange(10)
results = []
for seed in seeds:
    np.random.seed(seed)
    prepare_data()
        

    with open('exp2-preprocessed-DB_2014-12__Q_2015-05.pickle', 'rb') as file:
        tmp = pickle.load(file)
        db, query, gts = tmp.values()
    gt_hard, gt_soft = gts['hard'].T, gts['soft'].T
    res1 = evaluate_new(db, query, gthard=gt_hard, gtsoft=gt_soft)

    with open('exp2-preprocessed-DB_2014-12__Q_2015-08.pickle', 'rb') as file:
        tmp = pickle.load(file)
        db, query, gts = tmp.values()

    gt_hard, gt_soft = gts['hard'].T, gts['soft'].T
    res2 = evaluate_new(db, query, gthard=gt_hard, gtsoft=gt_soft)
    results.append((res1, res2))
    print((res1, res2))
print(results)
