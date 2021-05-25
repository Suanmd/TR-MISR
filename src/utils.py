""" Python utilities """
from torch import nn
import torch
from pytorch_msssim import ssim

import csv
import numpy as np
import os
import time

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from skimage import exposure

def readBaselineCPSNR(path):
    """
    Reads the baseline cPSNR scores from `path`.
    Args:
        filePath: str, path/filename of the baseline cPSNR scores
    Returns:
        scores: dict, of {'imagexxx' (str): score (float)}
    """
    scores = dict()
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            scores[row[0].strip()] = float(row[1].strip())
    return scores


def getImageSetDirectories(config, data_dir):
    """
    Returns a list of paths to directories, one for every imageset in `data_dir`.
    Args:
        data_dir: str, path/dir of the dataset
    Returns:
        imageset_dirs: list of str, imageset directories
    """
    imageset_dirs = []
    if config["paths"]["use_all_bands"]:
        for channel_dir in ['RED', 'NIR']:
            path = os.path.join(data_dir, channel_dir)
            for imageset_name in os.listdir(path):
                imageset_dirs.append(os.path.join(path, imageset_name))
    else:
        path = os.path.join(data_dir, config["paths"]["use_band"])
        for imageset_name in os.listdir(path):
            imageset_dirs.append(os.path.join(path, imageset_name))
    return imageset_dirs


class collateFunction():
    """ Util class to create padded batches of data. """

    def __init__(self, config, min_L=32):
        """
        Args:
            min_L: int, pad length
        """
        self.config = config
        self.min_L = min_L

    def __call__(self, batch):
        return self.collateFunction(batch)

    def collateFunction(self, batch):
        """
        Custom collate function to adjust a variable number of low-res images.
        Args:
            batch: list of imageset
        Returns:
            padded_lr_batch: tensor (B, min_L, W, H), low resolution images
            alpha_batch: tensor (B, min_L), low resolution indicator (0 if padded view, 1 otherwise)
            hr_batch: tensor (B, W, H), high resolution images
            hm_batch: tensor (B, W, H), high resolution status maps
            isn_batch: list of imageset names
        """

        lr_batch = []  # batch of low-resolution views
        lm_batch = []
        alpha_batch = []  # batch of indicators (0 if padded view, 1 if genuine view)
        hr_batch = []  # batch of high-resolution views
        hm_batch = []  # batch of high-resolution status maps
        isn_batch = []  # batch of site names

        train_batch = True

        for imageset in batch:

            lrs = imageset['lr']
            lr_maps = imageset['lr_maps']
            L, H, W = lrs.shape

            if L >= self.min_L:  # pad input to top_k
                lr_batch.append(lrs[:self.min_L])
                lm_batch.append(lr_maps[:self.min_L])
                alpha_batch.append(torch.ones(self.min_L))
            else:
                pad = torch.zeros(self.min_L - L, H, W)
                lr_batch.append(torch.cat([lrs, pad], dim=0))
                lm_batch.append(torch.cat([lr_maps, pad], dim=0))
                alpha_batch.append(torch.cat([torch.ones(L), torch.zeros(self.min_L - L)], dim=0))

            hr = imageset['hr']
            if train_batch and hr is not None:
                hr_batch.append(hr)
            else:
                train_batch = False

            hm_batch.append(imageset['hr_map'])
            isn_batch.append(imageset['name'])

        padded_lr_batch = torch.stack(lr_batch, dim=0)
        padded_lm_batch = torch.stack(lm_batch, dim=0)
        alpha_batch = torch.stack(alpha_batch, dim=0)

        if train_batch:
            hr_batch = torch.stack(hr_batch, dim=0)
            hm_batch = torch.stack(hm_batch, dim=0)

        ########## need to fix, and do not use it ##########
        # data_arguments, we need to process padded_lr_batch, padded_lm_batch, hr_batch, hm_batch
        if self.config["training"]["data_arguments"]:
            # print(padded_lr_batch.shape) # [12,16,64,64]
            # print(padded_lm_batch.shape) # [12,16,64,64]
            # print(hr_batch.shape)        # [12,192,192]
            # print(hm_batch.shape)        # [12,192,192]
            np.random.seed(int(1000 * time.time()) % 2**32)
            if np.random.random() <= self.config["training"]["probability of flipping horizontally"]:
                padded_lr_batch = torch.flip(padded_lr_batch, [3]) # Horizontal flip of lr images
                padded_lm_batch = torch.flip(padded_lm_batch, [3]) # Horizontal flip of lm images
                hr_batch = torch.flip(hr_batch, [2]) # Horizontal flip of hr images
                hm_batch = torch.flip(hm_batch, [2]) # Horizontal flip of hm images
            np.random.seed(int(1000 * time.time()) % 2**32)
            if np.random.random() <= self.config["training"]["probability of flipping vertically"]:
                padded_lr_batch = torch.flip(padded_lr_batch, [2]) # Vertical flip of lr images
                padded_lm_batch = torch.flip(padded_lm_batch, [2]) # Vertical flip of lm images
                hr_batch = torch.flip(hr_batch, [1]) # Horizontal flip of hr images
                hm_batch = torch.flip(hm_batch, [1]) # Horizontal flip of hm images
            np.random.seed(int(1000 * time.time()) % 2**32)
            k_num = np.random.choice(a=self.config["training"]["corresponding angles(x90)"],
                                     replace=True,
                                     p=self.config["training"]["probability of rotation"])
            padded_lr_batch = torch.rot90(padded_lr_batch, k=k_num, dims=[2,3]) # Rotate k times ninety degrees counterclockwise of lr images
            padded_lm_batch = torch.rot90(padded_lm_batch, k=k_num, dims=[2,3]) # Rotate k times ninety degrees counterclockwise of lm images
            hr_batch = torch.rot90(hr_batch, k=k_num, dims=[1,2]) # Rotate k times ninety degrees counterclockwise of hr images
            hm_batch = torch.rot90(hm_batch, k=k_num, dims=[1,2]) # Rotate k times ninety degrees counterclockwise of hm images
            np.random.seed(int(1000 * time.time()) % 2**32)

        return padded_lr_batch, padded_lm_batch, alpha_batch, hr_batch, hm_batch, isn_batch


def get_loss(srs, hrs, hr_maps, metric='cMSE'):
    """
    Computes ESA loss for each instance in a batch.
    Args:
        srs: tensor (B, W, H), super resolved images
        hrs: tensor (B, W, H), high-res images
        hr_maps: tensor (B, W, H), high-res status maps
    Returns:
        loss: tensor (B), metric for each super resolved image.
    """
    # ESA Loss: https://kelvins.esa.int/proba-v-super-resolution/scoring/

    if metric == 'L1':
        border = 3
        max_pixels_shifts = 2 * border
        size_image = hrs.shape[1]
        srs = srs.squeeze(1)
        cropped_predictions = srs[:, border:size_image - border, border:size_image - border]

        X = []
        for i in range(max_pixels_shifts + 1):  # range(7)
            for j in range(max_pixels_shifts + 1):  # range(7)
                cropped_labels = hrs[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]
                cropped_y_mask = hr_maps[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]
                cropped_labels_masked = cropped_labels * cropped_y_mask
                cropped_predictions_masked = cropped_predictions * cropped_y_mask
                total_pixels_masked = torch.sum(cropped_y_mask, dim=[1,2])

                # bias brightness
                b = (1.0 / total_pixels_masked) * torch.sum(cropped_labels_masked-cropped_predictions_masked, dim=[1,2])
                b = b.unsqueeze(-1).unsqueeze(-1)

                corrected_cropped_predictions = cropped_predictions_masked + b
                corrected_cropped_predictions = corrected_cropped_predictions * cropped_y_mask

                l1_loss = (1.0 / total_pixels_masked) * torch.sum(torch.abs(cropped_labels_masked-corrected_cropped_predictions) , dim=[1,2])
                X.append(l1_loss)
        X = torch.stack(X)
        min_l1 = torch.min(X, 0)[0]
        loss = -10 * torch.log10(min_l1)

        return loss


    if metric == 'L2':
        border = 3
        max_pixels_shifts = 2 * border
        size_image = hrs.shape[1]
        srs = srs.squeeze(1)
        cropped_predictions = srs[:, border:size_image - border, border:size_image - border]

        X = []
        for i in range(max_pixels_shifts + 1):  # range(7)
            for j in range(max_pixels_shifts + 1):  # range(7)
                cropped_labels = hrs[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]
                cropped_y_mask = hr_maps[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]
                cropped_labels_masked = cropped_labels * cropped_y_mask
                cropped_predictions_masked = cropped_predictions * cropped_y_mask
                total_pixels_masked = torch.sum(cropped_y_mask, dim=[1,2])

                # bias brightness
                b = (1.0 / total_pixels_masked) * torch.sum(cropped_labels_masked-cropped_predictions_masked, dim=[1,2])
                b = b.unsqueeze(-1).unsqueeze(-1)

                corrected_cropped_predictions = cropped_predictions_masked + b
                corrected_cropped_predictions = corrected_cropped_predictions * cropped_y_mask

                corrected_mse = (1.0 / total_pixels_masked) * torch.sum((cropped_labels_masked-corrected_cropped_predictions)**2, dim=[1, 2])
                cPSNR = 10.0 * torch.log10((1.0**2)/corrected_mse)
                X.append(cPSNR)
        X = torch.stack(X)
        max_cPSNR = torch.max(X, 0)[0]

        return max_cPSNR


    if metric == 'SSIM':
        border = 3
        max_pixels_shifts = 2 * border
        size_image = hrs.shape[1]
        srs = srs.squeeze(1)
        cropped_predictions = srs[:, border:size_image - border, border:size_image - border]

        X = []
        for i in range(max_pixels_shifts + 1):  # range(7)
            for j in range(max_pixels_shifts + 1):  # range(7)
                cropped_labels = hrs[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]
                cropped_y_mask = hr_maps[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]
                cropped_labels_masked = cropped_labels * cropped_y_mask
                cropped_predictions_masked = cropped_predictions * cropped_y_mask

                total_pixels_masked = torch.sum(cropped_y_mask, dim=[1,2])

                # bias brightness
                b = (1.0 / total_pixels_masked) * torch.sum(cropped_labels_masked-cropped_predictions_masked, dim=[1,2])
                b = b.unsqueeze(-1).unsqueeze(-1)
                corrected_cropped_predictions = cropped_predictions_masked + b
                corrected_cropped_predictions = corrected_cropped_predictions * cropped_y_mask

                Y = []
                for k in range(corrected_cropped_predictions.shape[0]):
                    cSSIM = ssim(corrected_cropped_predictions[k].unsqueeze(0).unsqueeze(0),
                                 cropped_labels_masked[k].unsqueeze(0).unsqueeze(0),
                                 data_range=1.0,
                                 size_average=False)
                    Y.append(cSSIM)
                Y = torch.stack(Y).squeeze(-1)
                X.append(Y)
        X = torch.stack(X)
        max_cSSIM = torch.max(X, 0)[0]

        return max_cSSIM
    return -1

def get_crop_mask(patch_size, crop_size):
    """
    Computes a mask to crop borders.
    Args:
        patch_size: int, size of patches
        crop_size: int, size to crop (border)
    Returns:
        torch_mask: tensor (1, 1, 3*patch_size, 3*patch_size), mask
    """
    mask = np.ones((1, 1, 3 * patch_size, 3 * patch_size))  # crop_mask for loss (B, C, W, H)
    mask[0, 0, :crop_size, :] = 0
    mask[0, 0, -crop_size:, :] = 0
    mask[0, 0, :, :crop_size] = 0
    mask[0, 0, :, -crop_size:] = 0
    torch_mask = torch.from_numpy(mask).type(torch.FloatTensor)
    return torch_mask