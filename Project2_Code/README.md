## Project 2 - Yolo-V3 Network Quantization and Evaluation

### Code Base Configuration

Follow instructions to download and configure the tvm compiler and Project 2 code base:

**Course assignment code from https://github.com/krosac/Quantization_PJ2**

**TVM compiler code from https://github.com/apache/incubator-tvm**

**Yolo-V3 Pytorch implementation from https://github.com/eriklindernoren/PyTorch-YOLOv3**


```
cmd> cd ~/ROOT_PATH
cmd> git clone https://github.com/krosac/Quantization_PJ2
```

Download this Project 2 repository for files to insert into working code base and open terminal

Change directories into the repository
```
cmd> cd /Project2_Code
```

Move 'quantization_utils.py' to the project 2 code base and tvm compiler
```
cmd> mv quantization_utils.py ~/ROOT_PATH/Quantization_PJ2/utils
cmd> mv quantization_utils.py ~/ROOT_PATH/tvm/python/tvm/relay/utils
```

move '/image_processing' folder to the following path: ROOT_PATH/Quantization_PJ2
```
cmd> mv -r /image_processing ROOT_PATH/Quantization_PJ2
```

Setup virtual environment according to the dependencies listed in requirements.txt
Source that vitual environment
```
cmd> source VENV_PATH/bin/activate
```

Change to image preprocessing repository and run script:
```
cmd(venv)> cd ~/ROOT_PATH/Quantization_PJ2/image_processing
cmd(venv)> python preprocess.py
```

Change directories back to main project 2 source
```
cmd(venv)> cd ..
```

Run main.py with pre-processed image to get full precision .pb file
```
cmd(venv)> python3 main.py --input shufflenet_v1.onnx --input_shape 1 224 224 3 --output_dir tmp --gen_pb
```

Verify in both copies of quantization_utils that the 'tf_symbolic_convert' function for the custom
quantization function is uncommented

Run main.py with quantization to get quantized .pb file -> label tmp directory accordingly
```
cmd(venv)> python3 main.py  --input yolov3.onnx  --input_shape  1 416 416 3 --output_dir tmp --gen_pb --gen_fx_pb --reference_input /home/bryanbed/Documents/tvm/Quantization_PJ2/preprocessing_trial/preprocessed.npy --output_tensor_names output_fx BiasAdd_58 BiasAdd_66 --preprocess custom  --dump 
```

Edit both copies of quantization_utils so that that the 'tf_symbolic_convert' function for the custom
quantization function is COMMENTED and the version for the built-in tensorflow API quantization is
UNCOMMENTED

Run main.py with quantization to get quantized .pb file -> label tmp directory accordingly
```
cmd(venv)> python3 main.py  --input yolov3.onnx  --input_shape  1 416 416 3 --output_dir tmp --gen_pb --gen_fx_pb --reference_input /home/bryanbed/Documents/tvm/Quantization_PJ2/preprocessing_trial/preprocessed.npy --output_tensor_names output_fx BiasAdd_58 BiasAdd_66 --preprocess custom  --dump
```

Download Pytorch-YoloV3 repository from github source code and add dependencies to virual environment with pip install
```
cmd(venv)> cd ~/ROOT_PATH
cmd(venv)> git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
```

Follow directions in github repo to download dataset, and generate intermediate test results

Once results have been verified, return to downloaded code for this project and move test.py file into Pytorch repo. Verify that results match those in '/Pytorch-YOLOV3_example_output'
```
cmd(venv)> mv test.py test_original.py
cmd(venv)> cd ~/ROOT_PATH/Project2_Code
cmd(venv)> mv test.py ~/ROOT_PATH/Pytorch-YOLOv3
cmd(venv)> cd test.py ~/ROOT_PATH/Pytorch-YOLOv3
```

EVALUATION COMPARISONS x 3

edit line 28 of test.py to the representative path of your non-quantized, built-in quantization or custom quantization test.pb files.

Run test.py to evaluate quantization accuracy
```
cmd(venv)> python test.py
```

Note 1: For custom quantization function, change test.py line 127 to:

'batch_size = 1'

Note 2: If evaluation fails on a specific batch, uncomment lines 50, 52, 53, 54 and change '10' to the back that evaluationw as failing on -> for partial results only.
