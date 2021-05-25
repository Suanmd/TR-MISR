import matplotlib.pyplot as plt
from os.path import join, basename
import torch
import skimage
import json
from glob import glob

import os
import numpy as np
import itertools
from tqdm import tqdm
from Evaluator import cPSNR, patch_iterator, shift_cPSNR
from DataLoader import get_patch

from skimage import io
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
import heapq

from DeepNetworks.TRNet import TRNet
from utils import readBaselineCPSNR
import warnings
warnings.filterwarnings("ignore")

def Draw_result(lr_images, lr_maps, band, lr_index):

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
        mean_value = np.sum(lr_images, axis=(1, 2)) / (np.sum(lr_maps, axis=(1, 2)) + 0.0001)
        lr_images = np.where(lr_images == 0., np.expand_dims(mean_value, (1, 2)), lr_images)
        std_value = np.std(lr_images, axis=(1, 2)) * 64 / (np.sqrt(np.sum(lr_maps, axis=(1, 2))) + 0.0001)
        lr_images = np.where(lr_images != 0., ((lr_images - np.expand_dims(mean_value, (1, 2))) / (np.expand_dims(std_value, (1, 2)) + 0.0001)), 0.)

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

    fusion_model.eval()

    if band == 'RED':
        pth_path = config["val"]["model_path_band_RED"]
        if config["val"]["pth_epoch_num_RED"]:
            print('LOAD RED MODEL: TRNet%s.pth', (str(config["val"]["pth_epoch_num_RED"])))
            fusion_model.load_state_dict(torch.load(os.path.join(pth_path,
                                                                 'TRNet' + str(
                                                                     config["val"]["pth_epoch_num_RED"]) + '.pth')))
        else:
            fusion_model.load_state_dict(torch.load(os.path.join(pth_path, 'TRNet.pth')))
    elif band == 'NIR':
        pth_path = config["val"]["model_path_band_NIR"]
        if config["val"]["pth_epoch_num_NIR"]:
            print('LOAD NIR MODEL: TRNet%s.pth', (str(config["val"]["pth_epoch_num_NIR"])))
            fusion_model.load_state_dict(torch.load(os.path.join(pth_path,
                                                                 'TRNet' + str(
                                                                     config["val"]["pth_epoch_num_NIR"]) + '.pth')))
        else:
            fusion_model.load_state_dict(torch.load(os.path.join(pth_path, 'TRNet.pth')))

    fusion_model.to(device)

    srs = fusion_model(lrs, alphas, maps=None, K=128)

    srs = srs.squeeze(0).squeeze(0)
    img = srs.cpu().detach().numpy() * 65535.0
    img = (img - np.min(img)).astype(np.uint16)
    if config["val"]["truncate values"]:
        img = np.where(img > 16383 , 16383, img)

    return img

# The first three pictures represent SR, HR, SR_detail, HR_detail, and the latter represents lrs

def test_img(imgset_path):

    lr_index = imgset_path.split('/')[-1]
    # print("lr_index: ", lr_index)
    lrpath = imgset_path[:-len(lr_index)]
    band = lrpath.split('/')[-2]
    # print('band:', band)
    save_address = './draw_demo2/' + band + '/' + lr_index
    os.mkdir(save_address)

    temp = glob(lrpath + lr_index + '/QM*.png')
    temp = np.sort(temp)
    idx_names = np.array([t[-7:-4] for t in temp])
    lrc = np.zeros([len(idx_names), 128, 128])
    for i, lrc_fn in zip(range(len(temp)), temp):
        lrc[i] = io.imread(lrc_fn)

    top_k = min(config['training']['min_L'], len(temp))
    clearance = np.sum(lrc, axis=(1,2)) # MAX: 4177920
    i_samples = heapq.nlargest(top_k, range(len(clearance)), clearance.take)
    idx_names = idx_names[i_samples]

    # idx_names = idx_names[0:16]
    lr_images = np.array([io.imread(join(lrpath + lr_index, f'LR{i}.png')) for i in idx_names], dtype=np.uint16)
    lr_maps = np.array([io.imread(join(lrpath + lr_index, f'QM{i}.png')) for i in idx_names], dtype=np.bool)
    HR = np.array(io.imread(join(lrpath + lr_index, 'HR.png')), dtype=np.uint16)
    hr_map = np.array(io.imread(join(lrpath + lr_index, 'SM.png')), dtype=np.bool)
    SR = Draw_result(lr_images, lr_maps, band=band, lr_index=lr_index)
    max_cpnsr = shift_cPSNR(SR, HR, hr_map, border_w=3)
    # print("max_cpnsr: ", max_cpnsr)

    if os.path.exists(os.path.join(config["val"]["val_data_path"], "norm.csv")):
        baseline_cpsnrs = readBaselineCPSNR(os.path.join(config["val"]["val_data_path"], "norm.csv"))
        # print("base_cpnsr: ", baseline_cpsnrs[lr_index])

        print("%s - %s - %.4f - %.4f - %.4f" % (lr_index, band, max_cpnsr, baseline_cpsnrs[lr_index], max_cpnsr/baseline_cpsnrs[lr_index]))
        lrs = skimage.img_as_float(lr_images)
        lrs = lrs / 16383 * 65535 * 255
        lrs = np.where(lrs > 255, 255, lrs)
        lrs = lrs.astype(np.uint8)

        hr = skimage.img_as_float(HR)
        hr = hr / 16383 * 65535 * 255
        hr = np.where(hr > 255, 255, hr)
        hr = hr.astype(np.uint8)

        sr = skimage.img_as_float(SR)
        sr = sr / 16383 * 65535 * 255
        sr = np.where(sr > 255, 255, sr)
        sr = sr.astype(np.uint8)

        for k in range(lrs.shape[0]):
            io.imsave(save_address + '/LRx'+str(k)+'.png', lrs[int(k)])
            io.imsave(save_address + '/QMx'+str(k)+'.png', lr_maps[int(k)])

        io.imsave(save_address + '/HR.png', hr)
        io.imsave(save_address + '/SM.png', hr_map)
        io.imsave(save_address + '/SR.png', sr)

if __name__ =='__main__':

    with open('./config/config.json', "r") as read_file:
        config = json.load(read_file)

    if config["val"]["use_gpu"]:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config["val"]["gpu_num"])
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    os.mkdir('./draw_demo2')
    os.mkdir('./draw_demo2/NIR')
    os.mkdir('./draw_demo2/RED')

    PATH_nir = '/share/home/antai/Project/RAMS_data/val/NIR/'
    PATH_red = '/share/home/antai/Project/RAMS_data/val/RED/'
    # for ii in os.listdir(PATH_nir):
    #     test_img(PATH_nir+ii)

    # for ii in os.listdir(PATH_red):
    #     test_img(PATH_red+ii)
    imgset_path = '/share/home/antai/Project/data/train/NIR/imgset0604'
    test_img(imgset_path)
    imgset_path = '/share/home/antai/Project/data/train/NIR/imgset0651'
    test_img(imgset_path)
    imgset_path = '/share/home/antai/Project/data/train/NIR/imgset0764'
    test_img(imgset_path)
    imgset_path = '/share/home/antai/Project/data/train/NIR/imgset0910'
    test_img(imgset_path)
    imgset_path = '/share/home/antai/Project/data/train/NIR/imgset1091'
    test_img(imgset_path)
    imgset_path = '/share/home/antai/Project/data/train/NIR/imgset1111'
    test_img(imgset_path)
    imgset_path = '/share/home/antai/Project/data/train/RED/imgset0130'
    test_img(imgset_path)
    imgset_path = '/share/home/antai/Project/data/train/RED/imgset0281'
    test_img(imgset_path)
    imgset_path = '/share/home/antai/Project/data/train/RED/imgset0387'
    test_img(imgset_path)
    imgset_path = '/share/home/antai/Project/data/train/RED/imgset0503'
    test_img(imgset_path)