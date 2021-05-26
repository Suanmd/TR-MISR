初版，待进一步完善。

在遥感领域，MISR是一个有挑战性的问题。PROBA-V Kelvin Dataset的发布引起了我们研究的极大兴趣。

我们认为，多张图像包含更多的信息，所以对于遥感图像的MISR问题，首先需要解决的就是图像利用率的问题。其次，PROBA-V所拍摄的多图像存在较大的时间跨度，并且图像序列未知，如何尽可能减小图像位置信息的干扰也是一个挑战。

在这个repository中，我们展示了一种灵活的，基于Transformer的MISR解决方案，名字叫TR-MISR。该方案在PROBA-V Super Resolution Challenge上获得了最好的成绩。TR-MISR不追求编码和解码的复杂性，追求的是融合模块的特征融合能力。简单来说，在融合模块中，每个基于感受野编码得到的特征向量经过Transformer进行自注意力计算，并通过一个embedding vector将这些向量融合，在其后的解码模块只需要对embedding vector进行简单的sub-pixel convolution即可输出高分辨率图像的对应区域。TR-MISR支持预定义模型大小和输入图像数量，且无需预训练。总体来说，TR-MISR是一次Transformer用于特定低级视觉任务的尝试。

推荐GPU平台：Tesla V100，如果GPU显存不足，请酌情降低batch size或者选择更小型的模型进行训练测试。

![Fig.1 The overview of TR-MISR. ](README_md_files%5CTR-MISR%20%282%29.png?v=1&type=image)

#### 0. Setup the environment
-   Setup a python environment and install dependencies, our python version == 3.6.12
```
pip install -r requirements.txt
```
#### 1. Prepare the data set
-   下载RAMS所分配的验证/训练集，[RAMS/probav_data at master · EscVM/RAMS (github.com)](https://github.com/EscVM/RAMS/tree/master/probav_data)
-   Run the split_data_fit script to crop images in each scene，设置好路径(对训练集裁剪即可)。
```
python ./split_data_fit.py
```
-   Run the save_clearance script to precompute clearance scores for low-res views
```
python ./save_clearance.py
```
-   You can get the 完整的预处理后的数据集，在XX和XX中。

#### 2. Complete the config
在config文件中，主要的设置包括
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

#### 2. Train the model
以上流程若准备完毕，可以进行训练。
```
python ./src/train.py
```
如果需要保留训练日志re.log，请运行
```
python ./src/train.py 2>&1 | tee re.log
```
保留下来的日志re.log可供train_all_read_log或train_read_log读取并打印出详细信息，若是全数据训练则使用前者。
```
python ./train_all_read_log.py
python ./train_read_log.py
```
#### 3. Validate the model
验证集在给定NIR波段和RED波段分别训练的模型路径后，会输出一个val_plot.png用于可视化每个场景与baseline的比较结果。
```
python /src/val.py
```
#### 4. Test the model
此函数主要是为了提交结果，由于Proba-V数据集中的测试集没有Ground Truth，用此函数是为了提交结果到排行榜上，见[Kelvins - PROBA-V Super Resolution post mortem Leaderboard (esa.int)](https://kelvins.esa.int/proba-v-super-resolution-post-mortem/leaderboard/)。此脚本会输出一个16位的提交压缩包（位于submission）用于提交和一个8位的可视化结果（位于submission_vis）方便查看效果。
```
python /src/test.py
```

![Fig 2. The leader board.](README_md_files%5CThe_leader_board.png?v=1&type=image)

#### 5. Plot the results


