import gc
import os
import numpy as np
import scipy.io
import pickle
from typing import *
import h5py

root = os.path.dirname(__file__)
def _read_h5_dataset(obj):
    """Convert an h5py object (dataset or group) into a Python object."""
    
    # Case 1: HDF5 Dataset → NumPy array
    if isinstance(obj, h5py.Dataset):
        data = obj[()]  # read entire dataset
        
        # Decode byte strings
        if isinstance(data, bytes):
            return data.decode("utf-8")
        if isinstance(data, np.ndarray) and data.dtype.kind == 'S':
            return data.astype(str)
        
        return data

    # Case 2: HDF5 Group → dict (MATLAB struct)
    elif isinstance(obj, h5py.Group):
        result = {}
        for key in obj.keys():
            result[key] = _read_h5_dataset(obj[key])
        return result

    # Fallback
    return obj


def load_gtfile(filepath):
    """Load MATLAB v7.3 .mat file into a nested Python dictionary."""
    result = {}
    with h5py.File(filepath, "r") as f:
        for key in f.keys():
            result[key] = _read_h5_dataset(f[key])
    return result

def order(matlab_array : np.ndarray) -> Tuple[np.ndarray, ...]:
    data = matlab_array['Y'].squeeze()
    keypoints, scores, descriptors = \
        np.stack(arrays=data['keypoints'], dtype=np.float32), \
        np.stack(arrays=data['scores'], dtype=np.float32), \
        np.stack(arrays=data['descriptors'], dtype=np.float32)
    return keypoints, scores, descriptors

def order_and_serialize(dbfile : str, queryfile : str, gtfile : str, outdir : str = os.path.dirname(__file__)) -> None:
    db, query = scipy.io.loadmat(dbfile), scipy.io.loadmat(queryfile)
    gt = load_gtfile(gtfile)

    db, query = order(db), order(query)
    dbdesc, qdesc = \
        os.path.basename(os.path.dirname(dbfile)).removesuffix('/delf.mat'), \
        os.path.basename(os.path.dirname(queryfile)).removesuffix('/delf.mat')
    filename = f'{dbdesc}--{qdesc}.pickle'
    with open(os.path.join(outdir, filename), 'wb') as file:
        pickle.dump({
            'db' : db,
            'query' : query,
            'gt' : {
                'hard' : gt['GT']['GThard'],
                'soft' : gt['GT']['GTsoft']
            }
        }, file)
    print(f"Serialized: {filename}")


def OxfordRobotCar():

    # define output directory:
    outdir = os.path.join('pickles/OxfordRobotCar/')
    os.makedirs(outdir, exist_ok=True)

    # list all .mat files in descriptor and ground truth directories
    descriptor_path = os.path.join(root, 'vpr_descriptors/descriptors/OxfordRobotCar/')
    gt_path = os.path.join(root, 'vpr_descriptors/ground_truth/OxfordRobotCar/')
    descriptors = {
        "2014-11-25-09-18-32" : os.path.join(descriptor_path, "2014-11-25-09-18-32/delf.mat"),
        "2014-12-09-13-21-02" : os.path.join(descriptor_path, "2014-12-09-13-21-02/delf.mat"),
        "2014-12-16-18-44-24" : os.path.join(descriptor_path, "2014-12-16-18-44-24/delf.mat"),
        "2015-02-03-08-45-10" : os.path.join(descriptor_path, "2015-02-03-08-45-10/delf.mat"),
        "2015-05-19-14-06-38" : os.path.join(descriptor_path, "2015-05-19-14-06-38/delf.mat"),
        "2015-08-28-09-50-22" : os.path.join(descriptor_path, "2015-08-28-09-50-22/delf.mat")
    }
    gts = [
        "2014-12-09-13-21-02--2014-11-25-09-18-32/gt.mat",
        "2014-12-09-13-21-02--2014-12-16-18-44-24/gt.mat",
        "2014-12-09-13-21-02--2015-05-19-14-06-38/gt.mat",
        "2014-12-09-13-21-02--2015-08-28-09-50-22/gt.mat",
        "2015-05-19-14-06-38--2015-02-03-08-45-10/gt.mat",
        "2015-08-28-09-50-22--2014-11-25-09-18-32/gt.mat"
    ]

    for gt in gts:
        db_key, query_key = gt.removesuffix('/gt.mat').split('--')
        db_file, query_file = descriptors[db_key], descriptors[query_key]
        order_and_serialize(db_file, query_file, os.path.join(gt_path, gt), outdir=outdir)
        gc.collect()

if __name__=='__main__':
    OxfordRobotCar()