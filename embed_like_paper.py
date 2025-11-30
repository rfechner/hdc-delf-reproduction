from tqdm import tqdm
import numpy as np
import pickle

nx, ny = 4, 6
w, h = 960, 1280
xb, yb = np.linspace(0, w, nx), np.linspace(0, h, ny)
binsizex, binsizey = xb[1], yb[1]
d = 1024 * 4
bxs, bys = np.random.choice([-1, 1], size=(nx + 1, d)), np.random.choice([-1, 1], size=(ny + 1, d))

def positional_encoding_vec(points, xb=xb, yb=yb, bxs=bxs, bys=bys,
                            binsizex=binsizex, binsizey=binsizey, d=d, nx=nx, ny=ny):
    """
        Positional Encoding Scheme. See paper https://openaccess.thecvf.com/content/CVPR2021/papers/Neubert_Hyperdimensional_Computing_as_a_Framework_for_Systematic_Aggregation_of_Image_CVPR_2021_paper.pdf
        Section 3.1.5. for details.

        Args:
            points: [nsamples, 2]

        Returns:
            positional encodings: [nsamples, d] Positional encodings of position in VSA-Vectorspace.
    """
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
        
def bind_bundle_batched(kps, fs, proj, batchsize=128):
    """
        Args:
            kps: [nsamples, 2], l2-normalized, standartized
            fs: [nsamples, nfeatures, din] Features, l2-normalized, standartized
            proj: [din, dout] Column-normalized projection matrix
        
        Returns:
            outs: [nsamples, dout] Features are bound to keypoints and bundled across feature dimension.
    """
    posit = iter(positional_encoding_vec_batched(kps, batchsize=batchsize))
    fsit  = iter(feature_batched_iterator(fs, proj, batchsize=batchsize))

    outs = []
    for pos, f in tqdm(zip(posit, fsit), desc='Bundling: ', total=int(np.ceil(fs.shape[0] / batchsize))):
        tmp = pos * f            # bind
        tmp = tmp.sum(axis=1)    # bundle
        tmp = np.clip(tmp, -1, 1)
        outs.append(tmp)

    return np.concatenate(outs, axis=0)

def l2norm(features):
    """
        Returns
            features: [nsamples, nfeatures, din], last axis l2-normalized, magnitude is one.
    """
    return features / np.linalg.norm(features, ord=2, axis=-1, keepdims=True)

def compute_train_stats(features):
    """
        Returns:
            mean: [1, 1, din]
            std: [1, 1, din]
    """
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

def normalize(features, mean, std):
    return  (features - mean) / (std + 1e-12)

def main():
    import gc

    pickle_names = [
        'DB_2014-12__Q_2015-08.pickle',
        'DB_2014-12__Q_2015-05.pickle'
    ]

    for pickle_name in pickle_names:
        print(f"Working on: {pickle_name}")
        with open(pickle_name, 'rb') as file:
            tmp = pickle.load(file)
            kps_db, _, features_db = tmp['db']

        proj = np.random.normal(size=(features_db.shape[-1], d))

        # column-normalize random matrix. In VSA-Toolbox we row-normalize and mmul with transpose -> equivalent
        proj = proj / np.linalg.norm(proj, axis=0, keepdims=True) 
        # Paper: "We use l2-normalization to standartize descriptor magnitudes"
        features_db = l2norm(features_db)

        # Paper: "followed by dimension-wise standardization to standard normal distributions.
        # The standardization is done using all descriptors from the current image."
        db_mean, db_std = compute_train_stats_imagewise(features_db)
        features_db = normalize(features_db, db_mean, db_std)

        print("Binding and bundling database vectors...")
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
        features_q = normalize(features_q, q_mean, q_std)

        print("Binding and bundling query vectors...")
        query = bind_bundle_batched(kps_q, features_q, proj=proj)

        # Paper: "We mean-center all holistic descriptors with the database mean"
        holistic_db_mean, holistic_db_std = compute_train_stats(db)
        db = normalize(db, holistic_db_mean, holistic_db_std)
        query = normalize(query, holistic_db_mean, holistic_db_std)

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
