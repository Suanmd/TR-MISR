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
import heapq

from skimage import io
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

from DeepNetworks.TRNet import TRNet
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

    lrs = lrs.squeeze(0).cpu().detach().numpy()

    return img, lrs

# The first three pictures represent SR, HR, SR_detail, HR_detail, and the latter represents lrs
def plot_data_with_GT(img_set, lr_index):
    # assert
    assert len(img_set) == 5
    for i in range(2):
        assert img_set[i].shape == (384, 384)
    assert img_set[4].shape == (24, 128, 128)

    plt.figure(figsize=(24, 10))
    ax1 = plt.subplot2grid((5, 12), (0, 0), colspan=3, rowspan=3)
    ax1.imshow(img_set[0], cmap='gray'), ax1.set_title('HR'), ax1.axis('off')
    ax2 = plt.subplot2grid((5, 12), (0, 3), colspan=3, rowspan=3)
    ax2.imshow(img_set[1], cmap='gray'), ax2.set_title('SR'), ax2.axis('off')
    ax3 = plt.subplot2grid((5, 12), (0, 6), colspan=3, rowspan=3)
    ax3.imshow(img_set[2], cmap='gray'), ax3.set_title('Details of HR'), ax3.axis('off')
    ax4 = plt.subplot2grid((5, 12), (0, 9), colspan=3, rowspan=3)
    ax4.imshow(img_set[3], cmap='gray'), ax4.set_title('Details of SR'), ax4.axis('off')
    ax5 = plt.subplot2grid((5, 12), (3, 0), colspan=1, rowspan=1)
    ax5.imshow(img_set[4][0], cmap='gray'), ax5.set_title('LR 0'), ax5.axis('off')
    ax6 = plt.subplot2grid((5, 12), (3, 1), colspan=1, rowspan=1)
    ax6.imshow(img_set[4][1], cmap='gray'), ax6.set_title('LR 1'), ax6.axis('off')
    ax7 = plt.subplot2grid((5, 12), (3, 2), colspan=1, rowspan=1)
    ax7.imshow(img_set[4][2], cmap='gray'), ax7.set_title('LR 2'), ax7.axis('off')
    ax8 = plt.subplot2grid((5, 12), (3, 3), colspan=1, rowspan=1)
    ax8.imshow(img_set[4][3], cmap='gray'), ax8.set_title('LR 3'), ax8.axis('off')
    ax9 = plt.subplot2grid((5, 12), (3, 4), colspan=1, rowspan=1)
    ax9.imshow(img_set[4][4], cmap='gray'), ax9.set_title('LR 4'), ax9.axis('off')
    ax10 = plt.subplot2grid((5, 12), (3, 5), colspan=1, rowspan=1)
    ax10.imshow(img_set[4][5], cmap='gray'), ax10.set_title('LR 5'), ax10.axis('off')
    ax11 = plt.subplot2grid((5, 12), (3, 6), colspan=1, rowspan=1)
    ax11.imshow(img_set[4][6], cmap='gray'), ax11.set_title('LR 6'), ax11.axis('off')
    ax12 = plt.subplot2grid((5, 12), (3, 7), colspan=1, rowspan=1)
    ax12.imshow(img_set[4][7], cmap='gray'), ax12.set_title('LR 7'), ax12.axis('off')
    ax13 = plt.subplot2grid((5, 12), (3, 8), colspan=1, rowspan=1)
    ax13.imshow(img_set[4][8], cmap='gray'), ax13.set_title('LR 8'), ax13.axis('off')
    ax14 = plt.subplot2grid((5, 12), (3, 9), colspan=1, rowspan=1)
    ax14.imshow(img_set[4][9], cmap='gray'), ax14.set_title('LR 9'), ax14.axis('off')
    ax15 = plt.subplot2grid((5, 12), (3, 10), colspan=1, rowspan=1)
    ax15.imshow(img_set[4][10], cmap='gray'), ax15.set_title('LR 10'), ax15.axis('off')
    ax16 = plt.subplot2grid((5, 12), (3, 11), colspan=1, rowspan=1)
    ax16.imshow(img_set[4][11], cmap='gray'), ax16.set_title('LR 11'), ax16.axis('off')
    ax17 = plt.subplot2grid((5, 12), (4, 0), colspan=1, rowspan=1)
    ax17.imshow(img_set[4][12], cmap='gray'), ax17.set_title('LR 12'), ax17.axis('off')
    ax18 = plt.subplot2grid((5, 12), (4, 1), colspan=1, rowspan=1)
    ax18.imshow(img_set[4][13], cmap='gray'), ax18.set_title('LR 13'), ax18.axis('off')
    ax19 = plt.subplot2grid((5, 12), (4, 2), colspan=1, rowspan=1)
    ax19.imshow(img_set[4][14], cmap='gray'), ax19.set_title('LR 14'), ax19.axis('off')
    ax20 = plt.subplot2grid((5, 12), (4, 3), colspan=1, rowspan=1)
    ax20.imshow(img_set[4][15], cmap='gray'), ax20.set_title('LR 15'), ax20.axis('off')
    ax21 = plt.subplot2grid((5, 12), (4, 4), colspan=1, rowspan=1)
    ax21.imshow(img_set[4][16], cmap='gray'), ax21.set_title('LR 16'), ax21.axis('off')
    ax22 = plt.subplot2grid((5, 12), (4, 5), colspan=1, rowspan=1)
    ax22.imshow(img_set[4][17], cmap='gray'), ax22.set_title('LR 17'), ax22.axis('off')
    ax23 = plt.subplot2grid((5, 12), (4, 6), colspan=1, rowspan=1)
    ax23.imshow(img_set[4][18], cmap='gray'), ax23.set_title('LR 18'), ax23.axis('off')
    ax24 = plt.subplot2grid((5, 12), (4, 7), colspan=1, rowspan=1)
    ax24.imshow(img_set[4][19], cmap='gray'), ax24.set_title('LR 19'), ax24.axis('off')
    ax25 = plt.subplot2grid((5, 12), (4, 8), colspan=1, rowspan=1)
    ax25.imshow(img_set[4][20], cmap='gray'), ax25.set_title('LR 20'), ax25.axis('off')
    ax26 = plt.subplot2grid((5, 12), (4, 9), colspan=1, rowspan=1)
    ax26.imshow(img_set[4][21], cmap='gray'), ax26.set_title('LR 21'), ax26.axis('off')
    ax27 = plt.subplot2grid((5, 12), (4, 10), colspan=1, rowspan=1)
    ax27.imshow(img_set[4][22], cmap='gray'), ax27.set_title('LR 22'), ax27.axis('off')
    ax28 = plt.subplot2grid((5, 12), (4, 11), colspan=1, rowspan=1)
    ax28.imshow(img_set[4][23], cmap='gray'), ax28.set_title('LR 23'), ax28.axis('off')

    plt.savefig('./draw_demo/' + lr_index + '.png')
    # plt.show()


def test_img(imgset_path, details_interval_x1_y1_x2_y2):
    lr_index = imgset_path.split('/')[-1]
    print("lr_index: ", lr_index)
    lrpath = imgset_path[:-len(lr_index)]
    band = lrpath.split('/')[-2]
    print('band:', band)

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

    lr_images = np.array([io.imread(join(lrpath + lr_index, f'LR{i}.png')) for i in idx_names], dtype=np.uint16)
    lr_maps = np.array([io.imread(join(lrpath + lr_index, f'QM{i}.png')) for i in idx_names], dtype=np.bool)
    HR = np.array(io.imread(join(lrpath + lr_index, 'HR.png')), dtype=np.uint16)
    hr_map = np.array(io.imread(join(lrpath + lr_index, 'SM.png')), dtype=np.bool)
    SR, lrs = Draw_result(lr_images, lr_maps, band=band, lr_index=lr_index)

    sr = skimage.img_as_float(SR)
    sr = sr / 16383 * 65535 * 255
    sr = np.where(sr > 255, 255, sr)
    sr = sr.astype(np.uint8)

    hr = skimage.img_as_float(HR)
    hr = hr / 16383 * 65535 * 255
    hr = np.where(hr > 255, 255, hr)
    hr = hr.astype(np.uint8)

    lrs = lrs / 16383 * 65535 * 255
    lrs = np.where(lrs > 255, 255, lrs)
    lrs = lrs.astype(np.uint8)

    max_cpnsr = shift_cPSNR(SR, HR, hr_map, border_w=3)
    print("max_cpnsr: ", max_cpnsr)
    img_set = []
    img_set.append(hr)
    img_set.append(sr)
    img_set.append(hr[details_interval_x1_y1_x2_y2[0]:details_interval_x1_y1_x2_y2[1],
                      details_interval_x1_y1_x2_y2[2]:details_interval_x1_y1_x2_y2[3]])
    img_set.append(sr[details_interval_x1_y1_x2_y2[0]:details_interval_x1_y1_x2_y2[1],
                      details_interval_x1_y1_x2_y2[2]:details_interval_x1_y1_x2_y2[3]])
    img_set.append(lrs)

    plot_data_with_GT(img_set, lr_index)

if __name__ =='__main__':

    with open('./config/config.json', "r") as read_file:
        config = json.load(read_file)

    if config["val"]["use_gpu"]:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config["val"]["gpu_num"])
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if not os.path.exists('./draw_demo'):
    	os.mkdir('./draw_demo')

    # Test the first picture
    imgset_path = '/share/home/antai/Project/data/train/RED/imgset0302'
    test_img(imgset_path, details_interval_x1_y1_x2_y2=[62, 215, 102, 255])

    # Test the second picture
    imgset_path = '/share/home/antai/Project/data/train/NIR/imgset0596'
    test_img(imgset_path, details_interval_x1_y1_x2_y2=[203, 356, 0, 153])

    # Test the last picture
    imgset_path = '/share/home/antai/Project/data/train/NIR/imgset0604'
    test_img(imgset_path, details_interval_x1_y1_x2_y2=[13, 13+150, 196, 196+150])
    imgset_path = '/share/home/antai/Project/data/train/NIR/imgset0651'
    test_img(imgset_path, details_interval_x1_y1_x2_y2=[127, 127+150, 143, 143+150])
    imgset_path = '/share/home/antai/Project/data/train/NIR/imgset0764'
    test_img(imgset_path, details_interval_x1_y1_x2_y2=[142, 142+150, 88, 88+150])
    imgset_path = '/share/home/antai/Project/data/train/NIR/imgset0910'
    test_img(imgset_path, details_interval_x1_y1_x2_y2=[163, 163+150, 64, 64+150])
    imgset_path = '/share/home/antai/Project/data/train/NIR/imgset1091'
    test_img(imgset_path, details_interval_x1_y1_x2_y2=[195, 195+150, 230, 230+150])
    imgset_path = '/share/home/antai/Project/data/train/NIR/imgset1111'
    test_img(imgset_path, details_interval_x1_y1_x2_y2=[231, 231_150, 84, 84+150])
    imgset_path = '/share/home/antai/Project/data/train/RED/imgset0130'
    test_img(imgset_path, details_interval_x1_y1_x2_y2=[4, 4+150, 161, 161+150])
    imgset_path = '/share/home/antai/Project/data/train/RED/imgset0281'
    test_img(imgset_path, details_interval_x1_y1_x2_y2=[175, 175+150, 109, 109+150])
    imgset_path = '/share/home/antai/Project/data/train/RED/imgset0387'
    test_img(imgset_path, details_interval_x1_y1_x2_y2=[31, 31+150, 7, 7+150])
    imgset_path = '/share/home/antai/Project/data/train/RED/imgset0503'
    test_img(imgset_path, details_interval_x1_y1_x2_y2=[107, 107+150, 16, 16+150])