import os
import numpy as np
from os.path import join, exists, basename, isfile
import glob
from skimage import io
import warnings
warnings.filterwarnings("ignore")

imset_dir = './probav_data/train/NIR'
imset_save = './split_data/train/NIR'
if not os.path.exists(imset_save):
    os.mkdir(imset_save)

for i in os.listdir(imset_dir):
    # i -> imgsetXXXX
    idx_names = np.array([basename(path)[2:-4] for path in glob.glob(join(imset_dir, i, r'QM*.png'))])
    idx_names = np.sort(idx_names)
    # No. -> idx_names
    lr_images = np.array([io.imread(join(imset_dir, i, f'LR{j}.png')) for j in idx_names], dtype=np.uint16)
    lr_maps = np.array([io.imread(join(imset_dir, i, f'QM{j}.png')) for j in idx_names], dtype=np.uint8)
    hr_image = np.array([io.imread(join(imset_dir, i, 'HR.png'))], dtype=np.uint16)
    hr_map = np.array([io.imread(join(imset_dir, i, 'SM.png'))], dtype=np.uint8)

    n = 0
    for q,k in [[0,0],[0,32],[0,64],[32,0],[32,32],[32,64],[64,0],[64,32],[64,64]]:
        print('name:%s, shift=(%2d,%2d), mean=%.3f' % (i,q,k,np.mean(lr_maps[:, q:q + 64, k:k + 64])), end=' ')
        if np.mean(lr_maps[:, q:q + 64, k:k + 64]) >= 255 * 0.85:
            if not os.path.exists(join(imset_save, i + str(n))):
                os.mkdir(join(imset_save, i + str(n)))
            for v in range(lr_images.shape[0]):
                io.imsave(join(imset_save, i + str(n), 'LR' + idx_names[v] + '.png'), lr_images[v, q:q + 64, k:k + 64])
                io.imsave(join(imset_save, i + str(n), 'QM' + idx_names[v] + '.png'), lr_maps[v, q:q + 64, k:k + 64])
            io.imsave(join(imset_save, i + str(n), 'HR.png'), hr_image[0, 3*q:3*(q + 64), 3*k:3*(k + 64)])
            io.imsave(join(imset_save, i + str(n), 'SM.png'), hr_map[0, 3*q:3*(q + 64), 3*k:3*(k + 64)])
            print('')
        else:
            print('not good!')
        n += 1

imset_dir = './probav_data/train/RED'
imset_save = './split_data/train/RED'
if not os.path.exists(imset_save):
    os.mkdir(imset_save)

for i in os.listdir(imset_dir):
    # i -> imgsetXXXX
    idx_names = np.array([basename(path)[2:-4] for path in glob.glob(join(imset_dir, i, r'QM*.png'))])
    idx_names = np.sort(idx_names)
    # No. -> idx_names
    lr_images = np.array([io.imread(join(imset_dir, i, f'LR{j}.png')) for j in idx_names], dtype=np.uint16)
    lr_maps = np.array([io.imread(join(imset_dir, i, f'QM{j}.png')) for j in idx_names], dtype=np.uint8)
    hr_image = np.array([io.imread(join(imset_dir, i, 'HR.png'))], dtype=np.uint16)
    hr_map = np.array([io.imread(join(imset_dir, i, 'SM.png'))], dtype=np.uint8)

    n = 0
    for q,k in [[0,0],[0,32],[0,64],[32,0],[32,32],[32,64],[64,0],[64,32],[64,64]]:
        print('name:%s, shift=(%2d,%2d), mean=%.3f' % (i,q,k,np.mean(lr_maps[:, q:q + 64, k:k + 64])), end=' ')
        if np.mean(lr_maps[:, q:q + 64, k:k + 64]) >= 255 * 0.85:
            if not os.path.exists(join(imset_save, i + str(n))):
                os.mkdir(join(imset_save, i + str(n)))
            for v in range(lr_images.shape[0]):
                io.imsave(join(imset_save, i + str(n), 'LR' + idx_names[v] + '.png'), lr_images[v, q:q + 64, k:k + 64])
                io.imsave(join(imset_save, i + str(n), 'QM' + idx_names[v] + '.png'), lr_maps[v, q:q + 64, k:k + 64])
            io.imsave(join(imset_save, i + str(n), 'HR.png'), hr_image[0, 3*q:3*(q + 64), 3*k:3*(k + 64)])
            io.imsave(join(imset_save, i + str(n), 'SM.png'), hr_map[0, 3*q:3*(q + 64), 3*k:3*(k + 64)])
            print('')
        else:
            print('not good!')
        n += 1