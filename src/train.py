""" Python script to train TR-MSFR for multi frame super resolution (MFSR) """
import torch
import torch.optim as optim

import json
import os
import datetime
import numpy as np
from tqdm import tqdm

import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from DeepNetworks.TRNet import TRNet
from DataLoader import ImagesetDataset
from Evaluator import shift_cPSNR
from utils import getImageSetDirectories, readBaselineCPSNR, collateFunction, get_loss, get_crop_mask
from tensorboardX import SummaryWriter
import pdb

def trainAndGetBestModel(fusion_model, optimizer, dataloaders, baseline_cpsnrs, config):

    np.random.seed(config["training"]["seed"])  # seed all RNGs for reproducibility
    torch.manual_seed(config["training"]["seed"])

    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]

    subfolder_pattern = 'batch_{}_time_{}'.format(batch_size, f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S-%f}")

    checkpoint_dir_run = os.path.join(config["paths"]["checkpoint_dir"], subfolder_pattern)
    os.makedirs(checkpoint_dir_run, exist_ok=True)

    tb_logging_dir = config['paths']['tb_log_file_dir']
    logging_dir = os.path.join(tb_logging_dir, subfolder_pattern)
    os.makedirs(logging_dir, exist_ok=True)

    writer = SummaryWriter(logging_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch_mask = get_crop_mask(patch_size=config["training"]["patch_size"],
                               crop_size=config["training"]["crop"])
    torch_mask = torch_mask.to(device)  # crop borders

    fusion_model.to(device)

    # load model
    if config["training"]["load_model"]:
        if config["training"]["pth_epoch_num"]:
            fusion_model.load_state_dict(torch.load(os.path.join(config["training"]["model_path"], 
                                                   'TRNet'+str(config["training"]["pth_epoch_num"])+'.pth')))
        else:
            fusion_model.load_state_dict(torch.load(os.path.join(config["training"]["model_path"], 'TRNet.pth')))    

    if config["training"]["strategy"] == 0:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['training']['lr_decay'],
                                                   verbose=True, patience=config['training']['lr_step'])
    elif config["training"]["strategy"] == 1:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["num_epochs"], 
                                                   eta_min=optimizer.param_groups[1]['lr']/100.)

    best_score = 100
    min_loss = 100
    patience_flag = 0

    for epoch in tqdm(range(1, num_epochs + 1)):

        # Train
        fusion_model.train()
        train_loss = 0.0  # monitor train loss

        # Iterate over data.
        for lrs, lr_maps, alphas, hrs, hr_maps, names in tqdm(dataloaders['train']):
            optimizer.zero_grad()  # zero the parameter gradients
            lrs = lrs.float().to(device)
            # lr_maps = lr_maps.float().to(device)
            alphas = alphas.float().to(device)
            hr_maps = hr_maps.float().to(device)
            hrs = hrs.float().to(device)
            srs = fusion_model(lrs, alphas, maps=None, K=64)  # fuse multi frames (B, 1, 3*W, 3*H)
            # Training loss
            cropped_mask = torch_mask[0] * hr_maps  # Compute current mask (Batch size, W, H)

            if config['training']['use_all_losses']:
                loss1 = -get_loss(srs, hrs, cropped_mask, metric='L1')
                loss2 = -get_loss(srs, hrs, cropped_mask, metric='L2')
                loss3 = -get_loss(srs, hrs, cropped_mask, metric='SSIM')
                if not config["training"]["use_all_data_to_fight_leaderboard"]:
                    print('loss1: {:.4f}'.format((config['training']['alpha1'] * torch.mean(loss1)).data), end = ' ')
                    print('loss2: {:.4f}'.format((config['training']['alpha2'] * torch.mean(loss2)).data), end = ' ')
                    print('loss3: {:.4f}'.format((config['training']['alpha3'] * torch.mean(loss3)).data))
                loss = config['training']['alpha1'] * loss1 + config['training']['alpha2'] * loss2 + config['training']['alpha3'] * loss3
            else:
                loss = -get_loss(srs, hrs, cropped_mask, metric=config['training']['loss_depend'])
            loss = torch.mean(loss)

            # The full-data training does not need to print so many losses, 
            # Usually we need to print when training the model, and use read_log.py to review the results
            if not config["training"]["use_all_data_to_fight_leaderboard"]:
                tqdm.write('loss:  {:.4f}'.format(loss.data))
            # Backprop
            loss.backward()
            optimizer.step()

            epoch_loss = loss.detach().cpu().numpy() * len(hrs) / len(dataloaders['train'].dataset)
            train_loss += epoch_loss

        # Learning rate decay rule
        if config["training"]["use_all_data_to_fight_leaderboard"]:
            if train_loss <= min_loss:
                min_loss = train_loss
                patience_flag = 0
                # save model
                torch.save(fusion_model.state_dict(), os.path.join(checkpoint_dir_run, 'TRNet.pth'))
            else:
                patience_flag += 1
            
            print('train_loss: ', train_loss)
            print('min_loss: ', min_loss)

            if epoch % 20 == 0:
                torch.save(fusion_model.state_dict(), os.path.join(checkpoint_dir_run, 'TRNet%s.pth' % epoch))

            if config["training"]["strategy"] == 0:
                scheduler.step(train_loss)
            elif config["training"]["strategy"] == 1:
                scheduler.step()
            elif config["training"]["strategy"] == 2:
                # Manually adjust the learning rate
                if patience_flag >= config["training"]["lr_step"]:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * config["training"]["lr_decay"]
                    optimizer.param_groups[1]['lr'] = optimizer.param_groups[1]['lr'] * config["training"]["lr_decay"]
                    patience_flag = 1

            # print the learning rate intuitively
            print('patience_flag: ', patience_flag,
                ' lr_coder: ', optimizer.state_dict()['param_groups'][0]['lr'],
                ' lr_transformer: ', optimizer.state_dict()['param_groups'][1]['lr'])


        if not config["training"]["use_all_data_to_fight_leaderboard"]:
            # Eval
            fusion_model.eval()
            val_score = 0.0  # monitor val score

            for lrs, lr_maps, alphas, hrs, hr_maps, names in dataloaders['val']:
                lrs = lrs.float().to(device)
                # lr_maps = lr_maps.float().to(device)
                alphas = alphas.float().to(device)
                hrs = hrs.numpy()
                hr_maps = hr_maps.numpy()

                srs = fusion_model(lrs, alphas, maps=None, K=128)

                # compute ESA score
                srs = srs[0].detach().cpu().numpy()
                for i in range(srs.shape[0]):
                    if baseline_cpsnrs is None:
                        if config["training"]["truncate values"]:
                            val_score -= shift_cPSNR(np.clip((srs[i] - np.min(srs[i])), 0, 16383/65535), hrs[i], hr_maps[i])
                        else:
                            val_score -= shift_cPSNR(srs[i], hrs[i], hr_maps[i])
                    else:
                        ESA = baseline_cpsnrs[names[i]]
                        # val_score += ESA / shift_cPSNR(srs[i], hrs[i], hr_maps[i])
                        if config["training"]["truncate values"]:
                            val_score -= shift_cPSNR(np.clip((srs[i] - np.min(srs[i])), 0, 16383/65535), hrs[i], hr_maps[i])
                        else:
                            val_score -= shift_cPSNR(srs[i], hrs[i], hr_maps[i])

            val_score /= len(dataloaders['val'].dataset)

            if best_score > val_score:
                torch.save(fusion_model.state_dict(), os.path.join(checkpoint_dir_run, 'TRNet.pth'))
                best_score = val_score
                patience_flag = 0
            else:
                patience_flag += 1

            if epoch % 20 == 0:
                torch.save(fusion_model.state_dict(), os.path.join(checkpoint_dir_run, 'TRNet%s.pth' % epoch))

            print('best_score: ', best_score)
            print('val_score:  ', val_score)
            print('one epoch done!')

            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/val_loss", val_score, epoch)

            if config["training"]["strategy"] == 0:
                scheduler.step(val_score)
            elif config["training"]["strategy"] == 1:
                scheduler.step()
            elif config["training"]["strategy"] == 2:
                # Manually adjust the learning rate
                if patience_flag >= config["training"]["lr_step"]:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * config["training"]["lr_decay"]
                    optimizer.param_groups[1]['lr'] = optimizer.param_groups[1]['lr'] * config["training"]["lr_decay"]
                    patience_flag = 1


    writer.close()


def main(config):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config["training"]["gpu_num"])

    # Reproducibility options
    np.random.seed(config["training"]["seed2"])  # RNG seeds
    torch.manual_seed(config["training"]["seed2"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize the network based on the network configuration
    fusion_model = TRNet(config["network"])

    transformer_params = list(map(id, fusion_model.superres.parameters()))
    coder_params = filter(lambda p: id(p) not in transformer_params, fusion_model.parameters())

    params = [
        {"params": coder_params, "lr": config["training"]["lr_coder"]},
        {"params": fusion_model.superres.parameters(), "lr": config["training"]["lr_transformer"]},
    ]

    if config["training"]["optim"] == 'Adam':
        optimizer = optim.Adam(params)
    elif config["training"]["optim"] == 'SGD':
        optimizer = optim.SGD(params, momentum=0.9, weight_decay=1e-5)

    # ESA dataset
    data_directory = config["paths"]["prefix"]

    baseline_cpsnrs = None
    if os.path.exists(os.path.join(data_directory, "norm.csv")):
        baseline_cpsnrs = readBaselineCPSNR(os.path.join(data_directory, "norm.csv"))

    if config["training"]["use_all_data_to_fight_leaderboard"]:
        train_list = getImageSetDirectories(config, os.path.join(data_directory, "train"))
    else:
        # train_set_directories = getImageSetDirectories(config, os.path.join(data_directory, "train"))
        # train_list, val_list = train_test_split(train_set_directories,
        #                                     test_size=config['training']['val_proportion'],
        #                                     random_state=1, shuffle=True)
        train_list = getImageSetDirectories(config, os.path.join(data_directory, "train"))
        val_list = getImageSetDirectories(config, os.path.join(data_directory, "val"))


    # Dataloaders
    batch_size = config["training"]["batch_size"]
    n_workers = config["training"]["n_workers"]
    n_views = config["training"]["n_views"]
    min_L = config["training"]["min_L"]

    assert config["training"]["create_patches"] == False

    train_dataset = ImagesetDataset(imset_dir=train_list,
                                    config=config["training"],
                                    top_k=n_views,
                                    map_depend=config["training"]["map_depend"],
                                    std_depend=config["training"]["std_depend"])
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers,
                                  collate_fn=collateFunction(config=config, min_L=min_L),
                                  pin_memory=True)

    if config["training"]["use_all_data_to_fight_leaderboard"]:
        dataloaders = {'train': train_dataloader}
    else:
        val_dataset = ImagesetDataset(imset_dir=val_list,
                                      config=config["training"],
                                      top_k=n_views,
                                      map_depend=config["training"]["map_depend"],
                                      std_depend=config["training"]["std_depend"])
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=n_workers,
                                    collate_fn=collateFunction(config=config, min_L=min_L),
                                    pin_memory=True)
        dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    # Train model
    torch.cuda.empty_cache()
    trainAndGetBestModel(fusion_model, optimizer, dataloaders, baseline_cpsnrs, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path of the config file", default='config/config.json')

    args = parser.parse_args()
    assert os.path.isfile(args.config)

    with open(args.config, "r") as read_file:
        config = json.load(read_file)

    main(config)
