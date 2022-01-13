import torch
import warnings
import sys
from glob import glob

import skimage
from zipfile import ZipFile

import heapq
import json
import os
import numpy as np

from skimage import io
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

from DeepNetworks.TRNet import TRNet

def main(path, out, config):
    # name of submission archive
    sub_archive = out + '/submission.zip'
    if config["testing"]["use_gpu"]:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config["testing"]["gpu_num"])
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print('generate sample solutions: ', end='', flush='True')

    for subpath in [path + '/test/RED', path + '/test/NIR']:
        for folder in os.listdir(subpath):
            temp = glob(subpath + '/' + folder + '/QM*.png')
            temp = np.sort(temp)
            idx_names = np.array([t[-7:-4] for t in temp])
            lrc = np.zeros([len(idx_names), 128, 128])

            for i, lrc_fn in zip(range(len(temp)), temp):
                lrc[i] = io.imread(lrc_fn)
                
            top_k = min(config['training']['min_L'], len(temp))
            clearance = np.sum(lrc, axis=(1,2)) # MAX: 4177920
            i_samples = heapq.nlargest(top_k, range(len(clearance)), clearance.take)
            idx_names = idx_names[i_samples]

            lr_images = np.array([io.imread(subpath + '/' + folder + '/' + f'LR{i}.png') for i in idx_names], dtype=np.uint16)
            lr_maps = np.array([io.imread(subpath + '/' + folder + '/' +  f'QM{i}.png') for i in idx_names], dtype=np.bool)

            for i in range(1, lr_images.shape[0]):
                s = phase_cross_correlation(reference_image=lr_images[0], 
                                            moving_image=lr_images[i], 
                                            return_error=False, 
                                            reference_mask=lr_maps[0], 
                                            moving_mask=lr_maps[i],
                                            overlap_ratio=0.99)
                lr_images[i] = shift(lr_images[i], s, mode='constant', cval=0)
                lr_maps[i] = shift(lr_maps[i], s, mode='constant', cval=0)

            if config["training"]["map_depend"]:
                lr_images = lr_images * lr_maps
            if config["training"]["std_depend"]:
                mean_value = np.sum(lr_images, axis=(1, 2)) / (np.sum(lr_maps, axis=(1, 2)) + 0.000000001)
                lr_images = np.where(lr_images == 0., np.expand_dims(mean_value, (1, 2)), lr_images)
                std_value = np.std(lr_images, axis=(1, 2)) * patch_size / (np.sqrt(np.sum(lr_maps, axis=(1, 2))) + 0.000000001)
                lr_images = np.where(lr_images != 0., ((lr_images - np.expand_dims(mean_value, (1, 2))) / (np.expand_dims(std_value, (1, 2)) + 0.000000001)), 0.)

            if lr_images.shape[0] < config['training']['min_L']:  # pad input to top_k
                pad = torch.zeros(config['training']['min_L'] - lr_images.shape[0], 128, 128)
                lrs = torch.cat([torch.from_numpy(skimage.img_as_float(lr_images)), pad], dim=0)
                alphas = torch.cat([torch.ones(lr_images.shape[0]), torch.zeros(config['training']['min_L'] - lr_images.shape[0])], dim=0)
            else:
                assert lr_images.shape[0] == config['training']['min_L']
                lrs = torch.from_numpy(skimage.img_as_float(lr_images))
                alphas = torch.ones(lr_images.shape[0])

            lrs = lrs.unsqueeze(0).float().to(device)
            alphas = alphas.unsqueeze(0).to(device)

            fusion_model = TRNet(config["network"])

            if subpath.split('/')[-1] == 'RED':
                pth_path = config["testing"]["model_path_band_RED"]
                if config["testing"]["pth_epoch_num_RED"]:
                    fusion_model.load_state_dict(torch.load(os.path.join(pth_path,
                                                           'TRNet'+str(config["testing"]["pth_epoch_num_RED"])+'.pth')))
                else:
                    fusion_model.load_state_dict(torch.load(os.path.join(pth_path,'TRNet.pth')))
            elif subpath.split('/')[-1] == 'NIR':
                pth_path = config["testing"]["model_path_band_NIR"]
                if config["testing"]["pth_epoch_num_NIR"]:
                    fusion_model.load_state_dict(torch.load(os.path.join(pth_path,
                                                           'TRNet'+str(config["testing"]["pth_epoch_num_NIR"])+'.pth')))
                else:
                    fusion_model.load_state_dict(torch.load(os.path.join(pth_path,'TRNet.pth')))                
            fusion_model.to(device)

            # Eval
            fusion_model.eval()
            srs = fusion_model(lrs, alphas, maps=None, K=128)
            srs = srs.squeeze(0).squeeze(0)
            img = srs.cpu().detach().numpy()*65535.0
            img = (img - np.min(img)).astype(np.uint16)
            # print(folder, np.min(img), np.max(img))
            if config["testing"]["truncate values"]:
                img = np.where(img > 16383 , 16383, img)

            # normalize and safe resulting image in temporary folder (complains on low contrast if not suppressed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(out + '/' + folder + '.png', img)
            print('*', end='', flush='True')

    print('\narchiving: ')

    zf = ZipFile(sub_archive, mode='w')
    try:
        for img in os.listdir(out):
            # ignore the .zip-file itself
            if not img.startswith('imgset'):
                continue
            zf.write(out + '/' + img, arcname=img)
            print('*', end='', flush='True')
    finally:
        zf.close()

    print('\ndone. The submission-file is found at {}. Bye!'.format(sub_archive))


if __name__ == '__main__':

    with open('./config/config.json', "r") as read_file:
        config = json.load(read_file)

    out = config["testing"]["submission_path"]
    path = config["testing"]["test_data_path"]

    # sanity check
    if 'test' not in os.listdir(path):
        raise ValueError('ABORT: your path {} does not contain a folder "test".'.format(path))

    # creating folder for convenience
    if os.path.exists(out):
        os.removedirs(out)

    if out not in os.listdir('.'):
        os.mkdir(out)

    main(path, out, config)

    if config["testing"]["generate_visual_imgs"]:
        print("waiting for generating visual images")
        srpath = config["testing"]["submission_path"]
        outputpath = config["testing"]["visual_imgs_path"]
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)

        for i in os.listdir(srpath):
            if i[-4:] == '.png':
                sr = np.array([io.imread(os.path.join(srpath, i))], dtype=np.uint16)
                plt.figure()
                plt.imshow(sr[0])
                plt.savefig(os.path.join(outputpath, i))
                plt.close()
    
    print('done!')
