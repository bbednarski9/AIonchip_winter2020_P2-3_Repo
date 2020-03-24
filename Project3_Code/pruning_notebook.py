# -*- coding: utf-8 -*-
"""Pruning_Notebook.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yGwwmbIEjkV5zHWrdUzzn80iBpMXxdF5

ECE209 AS AI/ML on Chip Winter 2020
Bryan Bednarski and Matt Nicholas

See direction for using this notebook at the following repo:
https://github.com/bbednarski9/AIonchip_winter2020_P2-3_Repo
"""

# Run this code block to increase ram to 25GB, takes ~30sec to crash
a = []
while(1):
    a.append(1)

# install Tensorflow, will need to run this each time the runtime gets reset
!pip install tensorboardX

# import dependenceies
import os
import random
import numpy as np
import tqdm
random.seed(42)

# Commented out IPython magic to ensure Python compatibility.
# # check python version, bash commands
# %%bash
# python --version

# check CUDA version, make sure runtime is conencted to GPU before this
# Edit > Notebook Settings > GPU
!nvcc --version

# Checks GPU version and status of compute device
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime → "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
  print(gpu_info)

# Mount your google drive
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

# Commented out IPython magic to ensure Python compatibility.
# print current directory, and change to pascal_voc/data top level
!pwd
# this is an example below, feel free to change
# %cd '/content/gdrive/My Drive/209AS_AIML/pascal_voc/data'

# Run voc_label only once to generate image paths and correct annotations for the network
!python voc_label.py

# Commented out IPython magic to ensure Python compatibility.
!pwd
# %cd '/content/gdrive/My Drive/209AS_AIML/pascal_voc'

# Verify files are up to date. Each time we change 'mode.py' or 'prune_utils.py'
# this can be run again to make sure file load goes through
!ls -l

"""# Training Yolo-V3
With whatever pruning funciton is uncommented. See repository for details on changing between threshold and percentage-based methods.
"""

!python3 main.py train --load weights/model_27.pth --name=voc --gpu True

"""# Evaluating Yolo-V3 Inference Accuracy
Can download pretrained weights from repo and insert to evaluate without training
"""

!python3 main.py eval --load weights/model_28.pth --name=voc --gpu True