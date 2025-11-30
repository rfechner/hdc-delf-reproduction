import numpy as np
import pickle

nx, ny = 4, 6
w, h = 960, 1280
xb, yb = np.linspace(0, w, nx), np.linspace(0, h, ny)
binsizex, binsizey = xb[1], yb[1]
d = 1024 * 4
#bxs, bys = np.random.uniform(*[-1, 1], size=(nx + 1, d)), np.random.uniform(*[-1, 1], size=(ny + 1, d))
bxs, bys = np.random.choice([-1, 1], size=(nx + 1, d)), np.random.choice([-1, 1], size=(ny + 1, d))
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
    for i in range(0, len(features), batchsize):
        fs = features[i:i+batchsize]
        fs = fs @ proj
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

def compute_train_stats(features):
    mean = np.mean(features, axis=(0, 1), keepdims=True)
    std = np.std(features, axis=(0, 1), keepdims=True)
    return mean, std

def compute_train_stats_imagewise(features):
    """
        Returns:
            mean: [nsamples, 1, din]
            std: [nsamples, 1, din]
    """
    mean = np.mean(features, axis=(1,), keepdims=True)
    std = np.std(features, axis=(1,), keepdims=True)

    return mean, std

def standardize(features, mean, std):
    return  (features - mean) / (std + 1e-12)

# ------------------------------ main -----------------------------------

def main():
    import gc

    pickle_names = [
        'DB_2014-12__Q_2015-08.pickle',
        'DB_2014-12__Q_2015-05.pickle'
    ]

    for pickle_name in pickle_names:

        with open(pickle_name, 'rb') as file:
            tmp = pickle.load(file)
            kps_db, _, features_db = tmp['db']

        proj = np.random.normal(size=(features_db.shape[-1], d))
        # column-normalize random matrix. In github we row-normalize and mmul with transpose -> equivalent
        proj = proj / np.linalg.norm(proj, axis=0, keepdims=True) 

        print('Computing database mean')

        # Paper: "We use l2-normalization to standardize descriptor magnitudes"
        features_db = l2norm(features_db)

        # followed by dimension-wise standardization to standard normal distributions.
        # The standardization is done using all descriptors from the current image.
        db_mean, db_std = compute_train_stats_imagewise(features_db)
        features_db = standardize(features_db, db_mean, db_std)
        
        print('Binding and Bundling training set')

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
        q_mean, q_std = compute_train_stats_imagewise(features_q)
        features_q = standardize(features_q, q_mean, q_std)
        query = bind_bundle_batched(kps_q, features_q, proj=proj)

        # TODO: Paper: "We mean-center all holistic descriptors with the database mean"
        # -> I've tried this, however this drops performance quite drastically.
        holistic_db_mean, holistic_db_std = compute_train_stats(db)
        db = standardize(db, holistic_db_mean, holistic_db_std)
        query = standardize(query, holistic_db_mean, holistic_db_std)

        del tmp, kps_q, features_q
        gc.collect()

        with open(f'exp2-preprocessed-{pickle_name}', 'wb') as file:
            pickle.dump({
                'db': db,
                'query': query,
                'gt': gts
            }, file=file)


if __name__ == '__main__':
    main()
