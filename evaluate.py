import pickle
import numpy as np
from embed_exp05 import feature_indices
from scipy.integrate import trapezoid as trapz
def evaluate(db, query, gts):
    """
        Computes accuracy for database and query using *normalized* dot products.
        db : [N, d]
        query: [M, d]
        gts : [N, M]
    """

    # Normalize db and query along the feature dimension
    db_norm = db / np.linalg.norm(db, axis=1, keepdims=True)
    query_norm = query / np.linalg.norm(query, axis=1, keepdims=True)

    # Normalized dot product = cosine similarity
    inners = db_norm @ query_norm.T  # [N, M]
    winners = np.argmax(inners, axis=0)  # [M]
    truepos = gts[winners, np.arange(query.shape[0])].sum()
    acc = truepos / query.shape[0]

    return acc

def evaluate_new(db, query, gtsoft, gthard):
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


def evaluate_new_hard(db, query, gtsoft, gthard):
    """
        Exact copy of peers code
    """
    # Normalize db and query along the feature dimension
    db_norm = db / np.linalg.norm(db, axis=1, keepdims=True)
    query_norm = query / np.linalg.norm(query, axis=1, keepdims=True)

    # Normalized dot product = cosine similarity
    similarity = db_norm @ query_norm.T  # [N, M]
    similarity = (similarity + 1) / 2 # map to range [0, 1]
    recalls, precisions = [0], [1]
    startV, endV = similarity.min(), similarity.max()

    gt = gthard.astype(bool)
    
    # remove soft, but not hard entries
    similarity[~gthard & gtsoft] = startV

    for t in np.linspace(endV, startV, 100): # reverse iteration -> Low recall first, then high precision.
        matches = similarity >= t

        tp = (matches & gt).astype(int).sum()
        fn = (~matches & gt).astype(int).sum()
        fp = (matches & ~gt).astype(int).sum()

        recall = tp / (fn + tp) if (fn + tp) > 0 else 1
        precision = tp / (fp + tp) if (fp + tp) > 0 else 1
        recalls.append(recall); precisions.append(precision);
    
    recalls.append(1); precisions.append(0);
    average_precision = trapz(x=recalls, y=precisions)
    return average_precision

# LLM generated vectorized, batched version of the evaluation function with feature un-binding.
def evaluate_with_unbind(db, query, gts, topm=10, feature_indices=feature_indices, topf=200, batch_size=64, eps=1e-12):
    """
    Batched, memory-efficient version of evaluate_with_unbind.

    Args:
        db: [N, d]         - bound+bundled DB vectors
        query: [M, d]      - bound+bundled query vectors
        gts: [N, M]        - ground-truth (one-hot per column)
        topm: int          - initial top-k candidates per query
        feature_indices: [topf, d] - unbinding vectors
        topf: int          - number of feature unbind vectors to use
        batch_size: int    - number of queries to process at a time
        eps: float         - small value to avoid divide-by-zero

    Returns:
        acc: float, accuracy
        winners: [M,] int array of winning DB indices
    """
    if feature_indices is None:
        raise ValueError("feature_indices must be provided")

    Fi = feature_indices[:topf, :]  # (topf, d)
    N, d = db.shape
    M = query.shape[0]
    winners = np.zeros(M, dtype=np.int32)

    db_normed = db / np.linalg.norm(db, axis=-1, keepdims=True)
    query_normed = query / np.linalg.norm(query, axis=-1, keepdims=True)

    # 1) initial top-k retrieval
    sims = db_normed @ query_normed.T               # (N, M)
    topm_idx = np.argpartition(-sims, topm, axis=0)[:topm, :]
    col_sorted = np.argsort(-sims[topm_idx, np.arange(M)], axis=0)
    topm_idx = topm_idx[col_sorted, np.arange(M)]   # (topm, M)

    # 2) process queries in batches
    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        q_batch = query[start:end]            # (B, d)
        B = q_batch.shape[0]

        # --- query unbinding ---
        # Fi[:,None,:] * q_batch[None,:,:] -> (topf, B, d)
        q_unbound = Fi[:, None, :] * q_batch[None, :, :]
        q_norm = np.linalg.norm(q_unbound, axis=-1, keepdims=True)
        q_unbound /= (q_norm + eps)
        s_q = q_unbound.sum(axis=0)           # (B, d)

        # --- candidate unbinding ---
        topm_batch_idx = topm_idx[:, start:end]      # (topm, B)
        db_cands = db[topm_batch_idx]               # (topm, B, d)

        # compute normalized unbound rows: (topf, topm, B, d)
        numer = Fi[:, None, None, :] * db_cands[None, :, :, :]
        denom = np.linalg.norm(numer, axis=-1, keepdims=True)
        numer /= (denom + eps)
        s_db = numer.sum(axis=0)                     # (topm, B, d)
        s_db = np.transpose(s_db, (1, 0, 2))        # (B, topm, d)

        # --- compute scores ---
        scores = np.einsum('bkd,bd->bk', s_db, s_q) # (B, topm)
        best_k = np.argmax(scores, axis=1)          # (B,)
        winners[start:end] = topm_batch_idx[best_k, np.arange(B)]

    # --- compute accuracy ---
    truepos = gts[winners, np.arange(M)].sum()
    acc = truepos / float(M)
    return acc



if __name__=='__main__':
    exp_name = input("Expname: ")
    
    with open(f'{exp_name}-preprocessed-DB_2014-12__Q_2015-05.pickle', 'rb') as file:
        tmp = pickle.load(file)
        db, query, gts = tmp.values()
    gt_hard, gt_soft = gts['hard'].T, gts['soft'].T
    print(evaluate(db, query, gt_hard), evaluate(db, query, gt_soft))

    with open(f'{exp_name}-preprocessed-DB_2014-12__Q_2015-08.pickle', 'rb') as file:
        tmp = pickle.load(file)
        db, query, gts = tmp.values()

    gt_hard, gt_soft = gts['hard'].T, gts['soft'].T
    print(evaluate(db, query, gt_hard), evaluate(db, query, gt_soft))
