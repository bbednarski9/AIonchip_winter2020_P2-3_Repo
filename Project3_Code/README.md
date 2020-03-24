## Project 3 - Yolo-V3 Network Pruning and Analysis


Follow instructions to download and configure the Pascal VOC object detection test dataset and Yolo-V3 trained weights for pruning and analysis

**Course Project 3 code from - https://github.com/krosac/Pruning_PJ3/tree/master/pascal_voc**


Note 1: This system is configured to run via CUDA GPU compute. The '.ipynb' scripts provided in Google Colab are designed as such.

Follow steps under the section header "PASCAL VOC" only to download test and evaluation dataset and weights

### Code Base Configuration
```
cmd> cd ~/ROOT_PATH
cmd> git clone https://github.com/krosac/Pruning_PJ3
```

Change to pascal_voc directory
```
cd pascal_voc
```

Create weights directory and download pretrained backbone weights for darknet53 to "weights folder". Then download pretrained 'model_27.pth' weights from google drive and move to the same directory
**Darknet53 backbone weights from - https://drive.google.com/file/d/1zoGUFk9Tfoll0vCteSJRGa9I1yEAfU-a/view**
** Pretrained Yolo-V3 weights from - https://drive.google.com/file/d/1PnhVkGkjiBalNK_gBNS0bw9SN39eLcXu/view?usp=sharing**
```
mkdir weights
mv ~/Downloads/darknet53_weights_pytorch.pth /weights
mv ~/Downloads/model_27.pth /weights
```

Next, download this repository and replace mode.py, prune_utils.py and the '.ipynb' notebook into the /pascal_voc directory. Save the original files.
```
cd ~/ROOT_PATH/pascal_voc/
mv mode.py mode_orig.py
mv prune_utils.py prune_utils_orig.py
mv ~/ROOT_PATH/Project3_Code/mode.py .
mv ~/ROOT_PATH/Project3_Code/prune_utils.py .
mv ~/ROOT_PATH/Project3_Code/Pruning_Notebook.ipynb .
```

Following the Project 3 repo, change to "data" direcotry and download pascal voc dataset. Uncompress the tar file and you should find "VOCdevkit" under "data" directory. Meanwhile, check image path names in xx_val.txt and xx_train.txt to make sure training scripts can find them.
```
cd data
sh get_voc_dataset.sh
```

Next, edit line 17 of config.py, changing from 'freeze_backbone=True' to 'freeze_backbone=False'
```
vim config.py
```

Next, upload the entire /pascal_voc directory to google drive (this will take a while... sorry)

The next steps will be run in the Google Colab notebook.
 - Switch to GPU runtime via: Edit > Notebook Settings
 - If not running with Google Colab pro, run the first cell to overflow RAM and get upgraded to 25GB
 - Run the next 5 cells to make sure you install dependencies, import APIs and verify CUDA version and GPU enabled, and mount the drive
 - The next two cells will change directory and run 'voc_label.py' to re-evaluate path names and annotations for all images in test, train and validaiton datasets (this will take a while.)
 - Run the next two cells to change back to the pascal_voc directory.

 The following code blocks can be inserted into subsequent cells of the Google Colab notebook, to train and evaluate models.
```
python3 main.py train --load PRETRAINED_PTH --name=voc --gpu True
```
```
python3 main.py eval --load PRETRAINED_PTH --name=voc --gpu True
```

In order to run threshold-based pruning and evaluation, verify the following changes in /pascal_voc files:
1. mode.py; uncomment lines 38, 87, 178 ; comment lines 36, 37, 85, 86, 179
2. prune_utils.py; determine weight threshold to prune by scaling line 57 value

In order to run percentage-based pruning and evaluation, verify the following changes in /pascal_voc files:
1. mode.py; uncomment lines 37, 86, 179 ; comment lines 36, 38, 85, 87, 178
2. prune_utils.py; determine percentage of weights pruned by scaling line 27 value

Note 2: In downloaded folder, the directory /test_data contains models trained under a variety of different thresholds and percentages. To evaluate them, move the '.pth' file into /pascal_voc/weights and run the above eval function
