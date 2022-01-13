''' Python script to save clearance scores for low-res data'''

import os
import numpy as np
import glob
import skimage.io as io
import argparse
from tqdm import tqdm

def getImageSetDirectories(data_dir):
    """
    Returns a list of paths to directories, one for every imageset in `data_dir`.
    Args:
        data_dir: str, path/dir of the dataset
    Returns:
        imageset_dirs: list of str, imageset directories
    """

    imageset_dirs = []
    for channel_dir in ['RED', 'NIR']:
        path = os.path.join(data_dir, channel_dir)
        for imageset_name in os.listdir(path):
            imageset_dirs.append(os.path.join(path, imageset_name))
    return imageset_dirs

def save_clearance_scores(dataset_directories):
    '''
    Saves low-resolution clearance scores as .npy under imageset dir
    Args:
        dataset_directories: list of imageset directories
    '''

    for imset_dir in tqdm(dataset_directories):
        idx_names = np.array([os.path.basename(path)[2:-4] for path in glob.glob(os.path.join(imset_dir, 'QM*.png'))])
        idx_names = np.sort(idx_names)
        lr_maps = np.array([io.imread(os.path.join(imset_dir, f'QM{i}.png')) for i in idx_names], dtype=np.uint16)
        scores = lr_maps.sum(axis=(1, 2))
        np.save(os.path.join(imset_dir, "clearance.npy"), scores)

def main():
    '''
    Calls save_clearance on train and test set.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", help="root dir of the dataset", default='./split_data/')
    args = parser.parse_args()

    prefix = args.prefix
    assert os.path.isdir(prefix)
    if os.path.exists(os.path.join(prefix, "train")):
        train_set_directories = getImageSetDirectories(os.path.join(prefix, "train"))
        save_clearance_scores(train_set_directories) # train data

    if os.path.exists(os.path.join(prefix, "val")):
        test_set_directories = getImageSetDirectories(os.path.join(prefix, "test"))
        save_clearance_scores(test_set_directories) # val data

    if os.path.exists(os.path.join(prefix, "test")):
        test_set_directories = getImageSetDirectories(os.path.join(prefix, "test"))
        save_clearance_scores(test_set_directories) # test data


if __name__ == '__main__':
    main()
