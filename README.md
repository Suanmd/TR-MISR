In remote sensing, multi-image super-resolution (MISR) is a challenging problem. The release of the Proba-V Kelvin data set has aroused our great interest.

We believe that multiple images contain more information than a single image, so it is necessary to improve image utilization significantly. Besides, the timespan of multiple images taken by Proba-V is long, and the ordering of the images is unknown, so the interference of image position needs to be eliminated.

In this repository, we demonstrate a novel Transformer-based MISR framework named TR-MISR, which gets the state-of-the-art in the Proba-V super-resolution challenge. TR-MISR does not pursue the complexity of the encoder and decoder but the feature fusion capability of the fusion module. Specifically, we rearrange the feature maps encoded by low-resolution images to a set of feature vectors. By adding an embedding vector, these images can be fused through multi-layers of Transformer with self-attention. Then, we decode the output of the embedding vector to get the high-resolution image.
 
TR-MISR supports predefined model size and number of input images, and no pre-training is required. Overall, TR-MISR is an attempt to use a Transformer for a specific low-level vision task.

Recommended GPU platform: Tesla V100. If the GPU memory is insufficient, please reduce the batch size or choose smaller model hyperparameters as appropriate.
![Fig1. Overview of TR-MISR.](https://github.com/Suanmd/TR-MISR/blob/master/imgs/TR-MISR.png)

#### 0. Setup the environment
-   Setup a python environment and install dependencies, our python version == 3.6.12
```
pip install -r requirements.txt
```
#### 1. Prepare the data set
-   Download the validation/training set assigned by RAMS, [RAMS/probav_data at master · EscVM/RAMS (github.com)](https://github.com/EscVM/RAMS/tree/master/probav_data)
-   Run the _split_data_fit_ script to crop images in each scene for the training set.
```
python ./split_data_fit.py
```
-   Run the _save_clearance_ script to precompute clearance scores for low-resolution images.
```
python ./save_clearance.py
```
-   You can easily get the complete pre-processed dataset on [Google Drive](https://drive.google.com/file/d/1_ZYJqHaXmAZqVlLVxLf118_R5wp7Rt7L/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1vlaisAQS1BAhDhsnZW73pA) (code:gflb).

#### 2. Complete the config file
In the config file, the main settings are shown in the following table.
|item|description| 
|-|--|
|_prefix_|directory for the data set.|
|_use_all_bands_|whether to use all bands, if _False_, then set _use_band_.|
|_use_all_data_to_fight_leaderboard_|if _True_, then use all training set and skip validation.|
|_strategy_|learning rate decay strategy, set _Manual_ by default.|
|_pth_epoch_num_|load the model with the corresponding epoch number.|
|_truncate values_|whether to truncate values that exceed the valid value.|
|_data_arguments_|please set _False_.|
|_all_loss_depend_|if _True_, then set the ratio of the three losses.|
|_model_path_band_|indicate the path of the model.|

#### 3. Train the model
If the above processes are prepared, then training can be carried out.
```
python ./src/train.py
```
If you need to record the training log, run
```
python ./src/train.py 2>&1 | tee re.log
```
The re.log file can be used to print out the details in each epoch.
```
python ./train_all_read_log.py   # for training all data
python ./train_read_log.py       # for training and evaluation
```
You can also view training logs with _tensorboardX_.
```
tensorboard --logdir='tb_logs/'
```
#### 3. Validate the model

-  Fix the model paths trained in the NIR band and RED band, respectively.
-  The _val_ script outputs a val_plot.png to visualize the results of each scene constructed by TR-MISR compared to the baseline.
```
python /src/val.py
```
#### 4. Test the model
The _test_ script is mainly used to submit the results to the leaderboard since the ground truth is not involved in the testing set. The _test_ script will output a submission zip with 16-bit images (located in _'./submission/'_) and a visualization with 8-bit images(located in _'./submission_vis/'_).
```
python /src/test.py
```
The leaderboard is shown as follows:
![Fig2. The leaderboard.](https://github.com/Suanmd/TR-MISR/blob/master/imgs/The_leader_board.png)

#### 5. Plot the results


