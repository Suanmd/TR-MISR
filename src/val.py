import torch
import warnings
import sys
from glob import glob

import skimage
import matplotlib.pyplot as plt

import heapq
import json
import os
import numpy as np

from skimage import io
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

from DeepNetworks.TRNet import TRNet
import pdb

from utils import readBaselineCPSNR
from DataLoader import get_patch
import itertools
# from pytorch_msssim import ssim
from skimage.metrics import structural_similarity as ssim

def patch_iterator(img, positions, size):
    """Iterator across square patches of `img` located in `positions`."""
    for x, y in positions:
        yield get_patch(img=img, x=x, y=y, size=size)

def cPSNRval(img, hr, hr_map):
    n_clear = np.sum(hr_map)  # number of clear pixels in the high-res patch
    diff = hr - img
    bias = np.sum(diff * hr_map) / n_clear  # brightness bias
    cMSE = np.sum(np.square((diff - bias) * hr_map)) / n_clear
    cPSNR = -10 * np.log10(cMSE)

    return cPSNR

def cSSIMval(img, hr, hr_map):
    n_clear = np.sum(hr_map)
    diff = hr - img
    bias = np.sum(diff * hr_map) / n_clear
    # cSSIM = ssim(torch.from_numpy(((img+bias)*hr_map)).unsqueeze(0).unsqueeze(0),
    #              torch.from_numpy((hr*hr_map)).unsqueeze(0).unsqueeze(0))
    # cSSIM = ssim((img+bias)*hr_map, hr*hr_map, data_range=0.25)
    cSSIM = ssim((img+bias)*hr_map, hr*hr_map)
    
    return cSSIM


def plot_scatter(x, y, label, ):
    plt.scatter(x, y, s=20, color='black', marker='x')
    # lim_min = min(math.floor(np.min(x)), math.floor(np.min(y))) - 1
    # lim_max = max(math.floor(np.max(x)), math.floor(np.max(y))) + 1
    lim_min = 30
    lim_max = 60
    _x_ = np.linspace(0, 100, 100)
    # set background
    if label == 'RED':
        plt.fill_between(_x_,0,_x_, color='red', alpha=.30)
        plt.fill_between(_x_,_x_,100, color='red', alpha=.15)
    elif label == 'NIR':
        plt.fill_between(_x_,0,_x_, color='blue', alpha=.30)
        plt.fill_between(_x_,_x_,100, color='blue', alpha=.15)
    else:
        plt.fill_between(_x_,0,_x_, color='green', alpha=.30)
        plt.fill_between(_x_,_x_,100, color='green', alpha=.15)
    # set title
    # plt.title(title)
    # set grid
    plt.grid(linestyle=":", color='gray')
    # plt.grid(linestyle="--")
    # set label
    plt.xlabel(label + ' cPSNR Bicubic (dB)', size=16)
    plt.ylabel(label + ' cPSNR TR-MISR (dB)', size=16)
    my_x_ticks = np.arange(lim_min, lim_max+0.01, 10)
    my_y_ticks = np.arange(lim_min, lim_max+0.01, 10)
    plt.xticks(my_x_ticks, size=16)
    plt.yticks(my_y_ticks, size=16)
    # set range
    plt.xlim((lim_min-5, lim_max+5))
    plt.ylim((lim_min-5, lim_max+5))

def main(path, config):
    # name of submission archive
    if config["val"]["use_gpu"]:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config["val"]["gpu_num"])
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    psnr = []
    psnr_NIR = []
    psnr_RED = []
    ssim = []
    ssim_NIR = []
    ssim_RED = []
    psnr_baseline = []
    psnr_baseline_NIR = []
    psnr_baseline_RED = []

    for subpath in [path + '/val/RED', path + '/val/NIR']:
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
            hr = np.array(io.imread(subpath + '/' + folder + '/HR.png'), dtype=np.uint16)
            hr = skimage.img_as_float(hr).astype(np.float32)
            hr_map = np.array(io.imread(subpath + '/' + folder + '/SM.png'), dtype=np.bool)

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
                std_value = np.std(lr_images, axis=(1, 2)) * patch_size / (np.sqrt(np.sum(lr_maps, axis=(1, 2))) + 0.0001)
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

            if subpath.split('/')[-1] == 'RED':
                pth_path = config["val"]["model_path_band_RED"]
                if config["val"]["pth_epoch_num_RED"]:
                    fusion_model.load_state_dict(torch.load(os.path.join(pth_path,
                                                           'TRNet'+str(config["val"]["pth_epoch_num_RED"])+'.pth')))
                else:
                    fusion_model.load_state_dict(torch.load(os.path.join(pth_path,'TRNet.pth')))
            elif subpath.split('/')[-1] == 'NIR':
                pth_path = config["val"]["model_path_band_NIR"]
                if config["val"]["pth_epoch_num_NIR"]:
                    fusion_model.load_state_dict(torch.load(os.path.join(pth_path,
                                                           'TRNet'+str(config["val"]["pth_epoch_num_NIR"])+'.pth')))
                else:
                    fusion_model.load_state_dict(torch.load(os.path.join(pth_path,'TRNet.pth')))                
            fusion_model.to(device)

            # Eval
            fusion_model.eval()
            srs = fusion_model(lrs, alphas, maps=None, K=128)
            srs = srs.squeeze(0).squeeze(0)
            img = srs.cpu().detach().numpy()
            img = img - np.min(img)
            if config["val"]["truncate values"]:
                img = np.clip(img, 0, 16383/65535)
     
            border_w = 3
            size = img.shape[0] - (2 * border_w)  # patch size
            img = get_patch(img=img, x=border_w, y=border_w, size=size)

            pos = list(itertools.product(range(2 * border_w + 1), range(2 * border_w + 1)))
            iter_hr = patch_iterator(img=hr, positions=pos, size=size)
            iter_hr_map = patch_iterator(img=hr_map, positions=pos, size=size)
            site_cPSNR = np.array([cPSNRval(img=img, hr=hr, hr_map=hr_map) for hr, hr_map in zip(iter_hr, iter_hr_map)])
            max_cPSNR = np.max(site_cPSNR, axis=0)

            pos2 = list(itertools.product(range(2 * border_w + 1), range(2 * border_w + 1)))
            iter_hr2 = patch_iterator(img=hr, positions=pos2, size=size)
            iter_hr_map2 = patch_iterator(img=hr_map, positions=pos2, size=size)
            site_cSSIM = np.array([cSSIMval(img=img, hr=hr, hr_map=hr_map) for hr, hr_map in zip(iter_hr2, iter_hr_map2)])
            max_cSSIM = np.max(site_cSSIM, axis=0)
            
            if os.path.exists(os.path.join(config["val"]["val_data_path"], "norm.csv")):
                baseline_cpsnrs = readBaselineCPSNR(os.path.join(config["val"]["val_data_path"], "norm.csv"))

            print('norm: %.4f, cPSNR: %.4f, cSSIM: %.4f' % (baseline_cpsnrs[folder], max_cPSNR, max_cSSIM))

            psnr.append(max_cPSNR)
            ssim.append(max_cSSIM)
            psnr_baseline.append(baseline_cpsnrs[folder])
            if subpath.split('/')[-1] == 'NIR':
                psnr_NIR.append(max_cPSNR)
                ssim_NIR.append(max_cSSIM)
                psnr_baseline_NIR.append(baseline_cpsnrs[folder])
            elif subpath.split('/')[-1] == 'RED':
                psnr_RED.append(max_cPSNR)
                ssim_RED.append(max_cSSIM)
                psnr_baseline_RED.append(baseline_cpsnrs[folder])

            del img
            del lrs
            del srs
            del alphas
            


    print('average cPSNR: %.4f' % np.mean(psnr))
    print('average cPSNR_RED: %.4f' % np.mean(psnr_RED))
    print('average cPSNR_NIR: %.4f' % np.mean(psnr_NIR))
    print('average cSSIM: %.4f' % np.mean(ssim))
    print('average cSSIM_RED: %.4f' % np.mean(ssim_RED))
    print('average cSSIM_NIR: %.4f' % np.mean(ssim_NIR))    

    plt.figure(figsize=(20,6)) # dpi=X00
    ax1 = plt.subplot2grid((1,3),(0,0), colspan=1, rowspan=1)
    plot_scatter(psnr_baseline_RED, psnr_RED, label='RED')
    ax2 = plt.subplot2grid((1,3),(0,1), colspan=1, rowspan=1)
    plot_scatter(psnr_baseline_NIR, psnr_NIR, label='NIR')
    ax3 = plt.subplot2grid((1,3),(0,2), colspan=1, rowspan=1)
    plot_scatter(psnr_baseline, psnr, label='ALL')
    plt.savefig(config["val"]["save_fig_name"])

if __name__ == '__main__':

    with open('./config/config.json', "r") as read_file:
        config = json.load(read_file)

    path = config["val"]["val_data_path"]

    main(path, config)
    print('done!')
