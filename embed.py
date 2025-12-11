import numpy as np
import pickle
import gc
import os
from tqdm import tqdm
from typing import *

nx, ny = 4, 6
w, h = 960, 1280
xb, yb = np.linspace(0, w, nx), np.linspace(0, h, ny)
binsizex, binsizey = xb[1], yb[1]
d = 1024 * 4

# Use a local Generator for positional encoding sign matrices so we don't
# perturb the global RNG state used elsewhere (e.g. by the caller that
# seeds np.random). These are intended to be fixed/deterministic.
_local_rng = np.random.default_rng(0)
bxs = _local_rng.choice([-1, 1], size=(nx + 1, d))
bys = _local_rng.choice([-1, 1], size=(ny + 1, d))

def positional_encoding_vec(points, xb=xb, yb=yb, bxs=bxs, bys=bys,
                            binsizex=binsizex, binsizey=binsizey, d=d, nx=nx, ny=ny):

    y, x = points[..., 0], points[..., 1]

    xi = np.clip(np.digitize(x, xb) - 1, 0, nx - 1)
    yi = np.clip(np.digitize(y, yb) - 1, 0, ny - 1)

    Bxl = bxs[xi]
    Bxr = bxs[xi + 1]
    Byl = bys[yi]
    Byr = bys[yi + 1]

    delta_x_r = binsizex - (x % binsizex)
    delta_y_r = binsizey - (y % binsizey)

    alpha_x = np.round(d * delta_x_r / binsizex).astype(int)
    alpha_y = np.round(d * delta_y_r / binsizey).astype(int)

    def merge_slices(left, right, alpha):
        left = np.ascontiguousarray(left)
        right = np.ascontiguousarray(right)

        d = left.shape[-1]
        alpha_flat = alpha.reshape(-1)
        n = alpha_flat.shape[0]

        L = left.reshape(n, d)
        R = right.reshape(n, d)
        P = np.empty_like(L)

        idx = np.arange(d)[None, :]
        mask = idx < alpha_flat[:, None]

        P[mask] = L[mask]
        P[~mask] = R[~mask]

        return P.reshape(*left.shape)

    X = merge_slices(Bxl, Bxr, alpha_x)
    Y = merge_slices(Byl, Byr, alpha_y)

    return X * Y


def positional_encoding_vec_batched(points, batchsize=128):
    for i in range(0, len(points), batchsize):
        yield positional_encoding_vec(points[i:i+batchsize])

def feature_batched_iterator(features, proj, batchsize=128):
    sqrt_n = np.sqrt(features.shape[-1])
    for i in range(0, len(features), batchsize):
        
        fs = features[i:i+batchsize]
        fs = np.matmul(fs, proj)

        # re-normalize and clip to range [-1, 1] 
        fs *= np.linalg.norm(fs, ord=2, axis=-1, keepdims=True) / sqrt_n
        fs = np.clip(fs, -1, 1)
        yield fs
        

def bind_bundle_batched(kps, fs, proj):
    """
    Produces a (nsamples, d) vector per sample.
    Both pos_enc and projected fs are normalized row-wise after binding.
    """
    posit = iter(positional_encoding_vec_batched(kps))
    fsit  = iter(feature_batched_iterator(fs, proj))

    outs = []
    for pos, f in zip(posit, fsit):
        tmp = pos * f            # bind
        tmp = tmp.sum(axis=1)    # bundle
        tmp = np.clip(tmp, -1, 1)
        outs.append(tmp)

    return np.concatenate(outs, axis=0)

def l2norm(features):
    return features / np.linalg.norm(features, ord=2, axis=-1, keepdims=True)

def stats(features, axis):
    mean = np.mean(features, axis=axis, keepdims=True)
    std = np.std(features, axis=axis, keepdims=True)
    return mean, std

def standardize(features, mean, std):
    return  (features - mean) / (std + 1e-12)

def embedding(pickles : list[str], 
                normalization : Literal['per-image', 'dataset'] = 'per-image', 
                use_orth_proj=False,
                row_normalize=True,
                **kwargs) -> None:

    assert all([x.endswith('.pickle') for x in pickles])
    stats_axis = {
        'per-image' : (1,),
        'dataset' : (0, 1)
    }[normalization]

    for pickle_name in tqdm(pickles, desc='Pickle Files'):
        print(f"Embedding: {pickle_name}...")

        with open(pickle_name, 'rb') as file:
            tmp = pickle.load(file)
            kps_db, _, features_db = tmp['db']

        # Paper: "We use l2-normalization to standardize descriptor magnitudes"
        features_db = l2norm(features_db)

        if use_orth_proj:
            from scipy.stats import ortho_group

            din = features_db.shape[-1]
            # generate an orthogonal basis of size [din x din]
            Q = ortho_group.rvs(din)
            
            # expand Q to a [din x d] projection matrix by repeating blocks
            # this keeps each 1024-d block orthogonal
            proj = np.concatenate([Q] * (d // din), axis=1)
            
            if proj.shape[1] < d: # add padding to meet d
                proj = np.concatenate([proj, Q[:, :d - proj.shape[1]]], axis=1)
        else:
            # define projection matrix
            proj = np.random.normal(size=(features_db.shape[-1], d))

        normalization_axis = 1 if row_normalize else 0
        proj = proj / np.linalg.norm(proj, axis=normalization_axis, keepdims=True)

        # followed by dimension-wise standardization to standard normal distributions.
        # The standardization is done using all descriptors from the current image.
        db_mean, db_std = stats(features_db, axis=stats_axis)
        features_db = standardize(features_db, db_mean, db_std)
        
        # bind and bundle
        db = bind_bundle_batched(kps_db, features_db, proj=proj)
        
        del tmp, kps_db, features_db
        gc.collect()

        # Query
        with open(pickle_name, 'rb') as file:
            tmp = pickle.load(file)
            kps_q, _, features_q = tmp['query']
            gts = tmp['gt']
            
        # bind and bundle
        features_q = l2norm(features_q)

        # Discussion: This is how it is done in the VSA-toolbox. Train and Test are normalized seperately.
        q_mean, q_std = stats(features_q, axis=stats_axis)
        features_q = standardize(features_q, q_mean, q_std)
        query = bind_bundle_batched(kps_q, features_q, proj=proj)

        # Paper: "We mean-center all holistic descriptors with the database mean"
        holistic_db_mean, holistic_db_std = stats(db, axis=(0,))
        db = standardize(db, holistic_db_mean, holistic_db_std)
        query = standardize(query, holistic_db_mean, holistic_db_std)

        del tmp, kps_q, features_q
        gc.collect()

        outdir, basename = os.path.split(pickle_name)
        outdir = os.path.join(outdir, 'embedded')
        outfile = os.path.join(outdir, basename)
        os.makedirs(outdir, exist_ok=True)
        
        print(f"Writing to: {outfile}.")
        with open(outfile, 'wb') as file:
            pickle.dump({
                'db': db,
                'query': query,
                'gt': gts
            }, file=file)
