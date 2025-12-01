import numpy as np
from scipy.integrate import trapezoid as trapz

def evaluate(db, query, gtsoft, gthard):
    """
        Instead of taking single best match, we're computing
        `Average Precision` over a range of thresholds \theta \in [0, 1].

        Matches in GTHard have to be found -> towards recall
        Entries other than GTSoft shouldn't be made -> precision

        Recall: TP / (TP + FN)
        Precision : TP / (TP + FP) 

        For each t in [0, 1] calculate all matches in the similarity matrix m for gtsoft and gthard.
        For recall, we calculate fn from the difference of the similarity matrix and gthard
        For precision, we calculate fp from the difference of similarity matrix and gtsoft.
    """
    # Normalize db and query along the feature dimension
    db_norm = db / np.linalg.norm(db, axis=1, keepdims=True)
    query_norm = query / np.linalg.norm(query, axis=1, keepdims=True)

    # Normalized dot product = cosine similarity
    similarity = db_norm @ query_norm.T  # [N, M]
    similarity = (similarity + 1) / 2 # map to range [0, 1]
    recalls, precisions = [0], [1]
    startV, endV = similarity.min(), similarity.max()

    # remove soft, but not hard entries
    similarity[~gthard & gtsoft] = startV

    for t in np.linspace(endV, startV, 100): # reverse iteration -> Low recall first, then high precision.
        matches = similarity >= t

        fn = (~matches & gthard).sum()
        tp = (matches & gthard).sum()
        recall = tp / (fn + tp) if (fn + tp) > 0 else 1

        tp = (matches & gtsoft).sum()
        fp = (~gtsoft & matches).sum()
        precision = tp / (fp + tp) if (fp + tp) > 0 else 1
        recalls.append(recall); precisions.append(precision)
    recalls.append(1); precisions.append(0);

    average_precision = trapz(x=recalls, y=precisions)
    return average_precision