import pickle
import numpy as np

from evaluate import evaluate, evaluate_new, evaluate_new_hard

with open('exp2-preprocessed-DB_2014-12__Q_2015-05.pickle', 'rb') as file:
    tmp = pickle.load(file)
    db, query, gts = tmp.values()
gt_hard, gt_soft = gts['hard'].T, gts['soft'].T
# print(evaluate(db, query, gt_hard), evaluate(db, query, gt_soft))
print(evaluate_new(db, query, gthard=gt_hard, gtsoft=gt_soft))

with open('exp2-preprocessed-DB_2014-12__Q_2015-08.pickle', 'rb') as file:
    tmp = pickle.load(file)
    db, query, gts = tmp.values()

gt_hard, gt_soft = gts['hard'].T, gts['soft'].T
# print(evaluate(db, query, gt_hard), evaluate(db, query, gt_soft))
print(evaluate_new(db, query, gthard=gt_hard, gtsoft=gt_soft))
