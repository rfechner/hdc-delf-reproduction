import numpy as np
import pickle

EXP2_FLAG = True
nx, ny = 4, 6 # granularity of separation
w, h = 1280, 960 # width, height
xb, yb = np.linspace(0, w, nx), np.linspace(0, h, ny) # bins
binsizex, binsizey = xb[1], yb[1] # bin sizes
d = 1024 * 4 # dimension of vectorspace.

if EXP2_FLAG:
    bxs, bys = np.random.uniform(*[-1, 1], size=(nx + 1, d)), np.random.uniform(*[-1, 1], size=(ny + 1, d)) # borders    
else:
    bxs, bys = np.random.choice([-1, 1], size=(nx + 1, d)), np.random.choice([-1, 1], size=(ny + 1, d)) # borders

# LLM generated vectorized version of positional_encoding
def positional_encoding_vec(points, xb=xb, yb=yb, bxs=bxs, bys=bys, binsizex=binsizex, binsizey=binsizey, d=d, nx=nx, ny=ny):
    """
    points: shape (samples, batch, 2), last dim = (x, y)
    xb, yb: bin edges
    bxs, bys: arrays of shape (nx+1, d) and (ny+1, d) or whatever shape your B arrays have
    binsizex, binsizey: bin sizes
    d: feature dimension
    nx, ny: number of bins in x and y
    """
    x, y = points[..., 0], points[..., 1]

    # Bin indices
    xi = np.clip(np.digitize(x, xb) - 1, 0, nx - 1)
    yi = np.clip(np.digitize(y, yb) - 1, 0, ny - 1)

    # Left/right bin values
    Bxl = bxs[xi]  # shape: (samples, batch, d)
    Bxr = bxs[xi + 1]
    Byl = bys[yi]
    Byr = bys[yi + 1]

    # Delta ratios
    delta_x_r = binsizex - (x % binsizex)
    delta_y_r = binsizey - (y % binsizey)

    alpha_x = np.round(d * delta_x_r / binsizex).astype(int)
    alpha_y = np.round(d * delta_y_r / binsizey).astype(int)

    # We need a helper function to merge slices along last axis
    def merge_slices(left, right, alpha):
        """
        Fully vectorized version of the merge_slices operation.

        left, right : (..., d)
        alpha       : (...,) integer cut positions 0 ≤ alpha ≤ d

        Returns:
            P : same shape as left, where
                P[..., :alpha]  = left[..., :alpha]
                P[..., alpha:]  = right[..., alpha:]
        """
        # Ensure contiguous and get shapes
        left  = np.ascontiguousarray(left)
        right = np.ascontiguousarray(right)
        
        d = left.shape[-1]
        alpha_flat = alpha.reshape(-1)
        n = alpha_flat.shape[0]

        # Reshape left/right into (n, d)
        L = left.reshape(n, d)
        R = right.reshape(n, d)

        # Output array
        P = np.empty_like(L)

        # A boolean mask selecting L vs R
        # mask[i, j] = True  if j < alpha[i]
        # mask[i, j] = False if j >= alpha[i]
        idx = np.arange(d)[None, :]      # (1, d)
        mask = idx < alpha_flat[:, None] # (n, d)

        # Vectorized merge
        P[mask] = L[mask]
        P[~mask] = R[~mask]

        # Return with original shape
        return P.reshape(*left.shape)

    X = merge_slices(Bxl, Bxr, alpha_x)
    Y = merge_slices(Byl, Byr, alpha_y)

    P = X * Y
    return P

def positional_encoding_vec_batched(points, batchsize=128):
    for i in range(0, len(points), batchsize):
        out = positional_encoding_vec(points[i:i+batchsize])
        yield out

def encoding_features_batched(features, grp, batchsize=128):
    for i in range(0, len(features), batchsize):
        fs = features[i:i+batchsize]
        shape_before = fs.shape[:-1] # all but last dimension
        fs = fs.reshape(-1, fs.shape[-1])
        fs = grp.transform(fs)
        fs = fs.reshape(*shape_before, fs.shape[-1])
        yield fs

def bind_bundle_batched(grp, kps, fs):

    posit, fsit = iter(positional_encoding_vec_batched(kps)), iter(encoding_features_batched(fs, grp))
    outs = []
    for pos, f in zip(posit, fsit):
        # bind to position, bundle over feature axis
        tmp = pos * f
        tmp = tmp.sum(axis=1)
        outs.append(tmp)
    outs = np.concat(outs, axis=0)
    return outs 

def compute_train_stats(features, eps=1e-8):
    """
    Compute normalization statistics on the training (database) features.
    features: [nsamples, nfeatures, d]
    Returns: (mean, std)
    """
    # Mean/std over all database samples and feature dims
    mean = features.mean(axis=(0, 1), keepdims=True)
    std  = features.std(axis=(0, 1), keepdims=True) + eps

    # After standardizing using mean/std, compute global L2 norm
    feat_std = (features - mean) / std
    return mean, std


def apply_train_stats(features, mean, std):
    """
    Apply training-derived statistics to ANY dataset (db or query).
    """
    F = (features - mean) / std
    return F

def main():
    import gc
    from sklearn.random_projection import GaussianRandomProjection

    pickle_names = [
        'DB_2014-12__Q_2015-08.pickle',
        'DB_2014-12__Q_2015-05.pickle'
    ]
    OUT_PREFIX = 'exp2-' if EXP2_FLAG else ''

    for pickle_name in pickle_names:
        
        # --- DATABASE (TRAINING) PASS ---
        with open(pickle_name, 'rb') as file:
            tmp = pickle.load(file)
            kps_db, _, features_db = tmp['db']

        if EXP2_FLAG:
            # Compute training statistics
            mean_db, std_db = compute_train_stats(features_db)
            print(mean_db.shape, std_db.shape)
            # Normalize database
            features_db = apply_train_stats(features_db, mean_db, std_db)

        # Fit projection with TRAIN features only
        grp = GaussianRandomProjection(n_components=d).fit(
                features_db.reshape(-1, features_db.shape[-1])
        )

        db = bind_bundle_batched(grp, kps_db, features_db)

        del tmp, kps_db, features_db
        gc.collect()

        # --- QUERY (TEST) PASS ---
        with open(pickle_name, 'rb') as file:
            tmp = pickle.load(file)
            kps_q, _, features_q = tmp['query']
            gts = tmp['gt']

        if EXP2_FLAG:
            features_q = apply_train_stats(features_q, mean_db, std_db)

        query = bind_bundle_batched(grp, kps_q, features_q)

        del tmp, kps_q, features_q
        gc.collect()

        with open(f'{OUT_PREFIX}preprocessed-{pickle_name}', 'wb') as file:
            pickle.dump({
                'db' : db,
                'query' : query,
                'gt' : gts
            }, file=file)

if __name__ == '__main__':
    main()
