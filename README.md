# Handsign-digits-classification-on-KV260

## Get the dataset
https://github.com/ardamavi/Sign-Language-Digits-Dataset

## Source files
My source files for this project are kept here at Github,

https://github.com/lobster1989/Handsign-digits-classification-on-KV260

There are 4 sub-repos in the linked github repo,

code: all source codes are included here, including scripts for training/quantizing/compiling tasks, as well as a host application to be running on ARM core.
output: generated model files.
Sign-Language-Digits-Dataset: dataset should be putted here.
target_zcu102_zcu104_kv260: files prepared to be copied to target board.

## Steps on host machine
### Setup Docker environment
Follow Vitis-AI page to set up Docker. Quantize & Compile should be done inside Vitis-AI docker and conda environment.

https://github.com/Xilinx/Vitis-AI

### Train/quantize/compile model
Below steps are done on host machine within Vitis-AI docker and vitis-ai-tensorflow2 conda environment.

#### Train the model
Run "python3 train.py". 

A custom CNN model is created and trained by this python script. 

For this simple CNN, a accuracy of 0.7682 is abtained on validating dataset at 10th epoch.

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/chao-proj/handsigndigits_end2end/code > python3 train.py

2021-11-15 07:55:10.103323: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1

Found 1653 images belonging to 10 classes.

Found 409 images belonging to 10 classes.

2021-11-15 07:55:11.990663: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1

2021-11-15 07:55:12.032860: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties:

pciBusID: 0000:17:00.0 name: GeForce GTX 1050 Ti computeCapability: 6.1

coreClock: 1.468GHz coreCount: 6 deviceMemorySize: 3.94GiB deviceMemoryBandwidth: 104.43GiB/s

2021-11-15 07:55:12.032914: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1

2021-11-15 07:55:12.035884: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10

2021-11-15 07:55:12.038081: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10

2021-11-15 07:55:12.038523: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10

2021-11-15 07:55:12.040759: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10

2021-11-15 07:55:12.042001: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10

2021-11-15 07:55:12.046426: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7

2021-11-15 07:55:12.047435: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0

2021-11-15 07:55:12.047830: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA

To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

2021-11-15 07:55:12.057500: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 3699850000 Hz

2021-11-15 07:55:12.058631: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556696945db0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:

2021-11-15 07:55:12.058667: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

2021-11-15 07:55:12.159767: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556696864380 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:

2021-11-15 07:55:12.159810: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce GTX 1050 Ti, Compute Capability 6.1

2021-11-15 07:55:12.168971: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties:

pciBusID: 0000:17:00.0 name: GeForce GTX 1050 Ti computeCapability: 6.1

coreClock: 1.468GHz coreCount: 6 deviceMemorySize: 3.94GiB deviceMemoryBandwidth: 104.43GiB/s

2021-11-15 07:55:12.169028: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1

2021-11-15 07:55:12.169075: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10

2021-11-15 07:55:12.169098: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10

2021-11-15 07:55:12.169121: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10

2021-11-15 07:55:12.169143: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10

2021-11-15 07:55:12.169165: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10

2021-11-15 07:55:12.169188: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7

2021-11-15 07:55:12.170444: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0

2021-11-15 07:55:12.170493: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1

2021-11-15 07:55:12.550866: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:

2021-11-15 07:55:12.550889: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      0

2021-11-15 07:55:12.550893: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1276] 0:   N

2021-11-15 07:55:12.551517: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1402] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3244 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:17:00.0, compute capability: 6.1)

Model: "customcnn_model"

_________________________________________________________________

Layer (type)                 Output Shape              Param #

=================================================================

input_1 (InputLayer)         [(None, 100, 100, 1)]     0

_________________________________________________________________

conv2d (Conv2D)              (None, 98, 98, 32)        320

_________________________________________________________________

max_pooling2d (MaxPooling2D) (None, 49, 49, 32)        0

_________________________________________________________________

conv2d_1 (Conv2D)            (None, 47, 47, 64)        18496

_________________________________________________________________

max_pooling2d_1 (MaxPooling2 (None, 23, 23, 64)        0

_________________________________________________________________

conv2d_2 (Conv2D)            (None, 21, 21, 128)       73856

_________________________________________________________________

max_pooling2d_2 (MaxPooling2 (None, 10, 10, 128)       0

_________________________________________________________________

conv2d_3 (Conv2D)            (None, 8, 8, 128)         147584

_________________________________________________________________

max_pooling2d_3 (MaxPooling2 (None, 4, 4, 128)         0

_________________________________________________________________

flatten (Flatten)            (None, 2048)              0

_________________________________________________________________

dropout (Dropout)            (None, 2048)              0

_________________________________________________________________

dense (Dense)                (None, 512)               1049088

_________________________________________________________________

dense_1 (Dense)              (None, 10)                5130

=================================================================

Total params: 1,294,474

Trainable params: 1,294,474

Non-trainable params: 0

_________________________________________________________________

Epoch 1/10

2021-11-15 07:55:13.500503: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10

2021-11-15 07:55:13.649000: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7

2021-11-15 07:55:14.115335: W tensorflow/stream_executor/gpu/asm_compiler.cc:81] Running ptxas --version returned 256

2021-11-15 07:55:14.143671: W tensorflow/stream_executor/gpu/redzone_allocator.cc:314] Internal: ptxas exited with non-zero error code 256, output:

Relying on driver to perform ptx compilation.

Modify $PATH to customize ptxas location.

This message will be only logged once.

48/48 [==============================] - 11s 226ms/step - loss: 2.2957 - acc: 0.1128 - val_loss: 2.1856 - val_acc: 0.2292

Epoch 2/10

48/48 [==============================] - 4s 84ms/step - loss: 1.7664 - acc: 0.3633 - val_loss: 1.8149 - val_acc: 0.3568

Epoch 3/10

48/48 [==============================] - 4s 83ms/step - loss: 1.1517 - acc: 0.5921 - val_loss: 1.2129 - val_acc: 0.5833

Epoch 4/10

48/48 [==============================] - 4s 85ms/step - loss: 0.8129 - acc: 0.6979 - val_loss: 1.1744 - val_acc: 0.5729

Epoch 5/10

48/48 [==============================] - 4s 84ms/step - loss: 0.6780 - acc: 0.7639 - val_loss: 1.1073 - val_acc: 0.6589

Epoch 6/10

48/48 [==============================] - 4s 80ms/step - loss: 0.5517 - acc: 0.8131 - val_loss: 1.1636 - val_acc: 0.6562

Epoch 7/10

48/48 [==============================] - 4s 84ms/step - loss: 0.4344 - acc: 0.8472 - val_loss: 0.7979 - val_acc: 0.7344

Epoch 8/10

48/48 [==============================] - 4s 84ms/step - loss: 0.3883 - acc: 0.8675 - val_loss: 0.9528 - val_acc: 0.7005

Epoch 9/10

48/48 [==============================] - 4s 82ms/step - loss: 0.3665 - acc: 0.8839 - val_loss: 0.7645 - val_acc: 0.7370

Epoch 10/10

48/48 [==============================] - 4s 84ms/step - loss: 0.3273 - acc: 0.8872 - val_loss: 0.7131 - val_acc: 0.7682
```

Learning curve as below,

#### Quantize the model
Run "python3 quantize.py".

''quantized_model.h5'' will be generated in this step.

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/chao-proj/handsigndigits_end2end/code > python3 quantize.py

Load float model..

Model: "customcnn_model"

_________________________________________________________________

Layer (type)                 Output Shape              Param #

=================================================================

input_1 (InputLayer)         [(None, 100, 100, 1)]     0

_________________________________________________________________

conv2d (Conv2D)              (None, 98, 98, 32)        320

_________________________________________________________________

max_pooling2d (MaxPooling2D) (None, 49, 49, 32)        0

_________________________________________________________________

conv2d_1 (Conv2D)            (None, 47, 47, 64)        18496

_________________________________________________________________

max_pooling2d_1 (MaxPooling2 (None, 23, 23, 64)        0

_________________________________________________________________

conv2d_2 (Conv2D)            (None, 21, 21, 128)       73856

_________________________________________________________________

max_pooling2d_2 (MaxPooling2 (None, 10, 10, 128)       0

_________________________________________________________________

conv2d_3 (Conv2D)            (None, 8, 8, 128)         147584

_________________________________________________________________

max_pooling2d_3 (MaxPooling2 (None, 4, 4, 128)         0

_________________________________________________________________

flatten (Flatten)            (None, 2048)              0

_________________________________________________________________

dropout (Dropout)            (None, 2048)              0

_________________________________________________________________

dense (Dense)                (None, 512)               1049088

_________________________________________________________________

dense_1 (Dense)              (None, 10)                5130

=================================================================

Total params: 1,294,474

Trainable params: 1,294,474

Non-trainable params: 0

_________________________________________________________________



model input size: 100 100



Load validation dataset for quantization..

Found 2062 images belonging to 10 classes.



Run quantization..

[VAI INFO] Start CrossLayerEqualization...

10/10 [==============================] - 1s 64ms/step

[VAI INFO] CrossLayerEqualization Done.

[VAI INFO] Start Quantize Calibration...

65/65 [==============================] - 11s 169ms/step

[VAI INFO] Quantize Calibration Done.

[VAI INFO] Start Post-Quantize Adjustment...

[VAI INFO] Post-Quantize Adjustment Done.

[VAI INFO] Quantization Finished.



Saved quantized model as ../output/quantized_model.h5
```

#### Evaluate the quantized model
Evaluating of quantized model can be done within python scripts.

Run "python3 eval_quantize.py".

You might ask why a higher accurary(0.797) is obtained after quantizing? Well that's because I don't have the dataset augmented this time for evaluation, per my guess the validation dataset could be a little "easier".

In the training stage I have the dataset augmented since we have a relatively small dataset so it's likely to get overfit after training.

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/chao-proj/handsigndigits_end2end/code > python3 eval_quantize.py



Load quantized model..

WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.



Compile model..

Found 409 images belonging to 10 classes.



Evaluate model on test Dataset

13/13 [==============================] - 3s 255ms/step - loss: 0.6034 - accuracy: 0.7971

loss: 0.603

acc: 0.797

(vitis-ai-tensorflow2) Vitis-AI /workspace/chao-proj/handsigndigits_end2end/code > python3 eval_quantize.py



Load quantized model..

WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.



Compile model..

Found 409 images belonging to 10 classes.



Evaluate model on test Dataset

13/13 [==============================] - 4s 270ms/step - loss: 0.6033 - accuracy: 0.7971

loss: 0.603

acc: 0.797
```

#### Comile the model
Run "bash -x compile.sh" to compile the quantized model.

A xmodel file will be generated. 

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/chao-proj/handsigndigits_end2end/code > bash -x compile.sh

+ ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json

+ OUTDIR=../output

+ NET_NAME=customcnn

+ MODEL=../output/quantized_model.h5

+ echo -----------------------------------------

-----------------------------------------

+ echo 'COMPILING MODEL FOR ZCU102..'

COMPILING MODEL FOR ZCU102..

+ echo -----------------------------------------

-----------------------------------------

+ compile

+ tee compile.log

+ vai_c_tensorflow2 --model ../output/quantized_model.h5 --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json --output_dir ../output --net_name customcnn

[INFO] Namespace(batchsize=1, inputs_shape=None, layout='NHWC', model_files=['../output/quantized_model.h5'], model_type='tensorflow2', named_inputs_shape=None, out_filename='/tmp/customcnn_org.xmodel', proto=None)

[INFO] tensorflow2 model: /workspace/chao-proj/handsigndigits_end2end/output/quantized_model.h5

[INFO] keras version: 2.4.0

[INFO] Tensorflow Keras model type: functional

[INFO] parse raw model     :100%|██████████| 19/19 [00:00<00:00, 8027.78it/s]

[INFO] infer shape (NHWC)  :100%|██████████| 33/33 [00:00<00:00, 731.94it/s]

[INFO] perform level-0 opt :100%|██████████| 2/2 [00:00<00:00, 212.70it/s]

[INFO] perform level-1 opt :100%|██████████| 2/2 [00:00<00:00, 962.22it/s]

[INFO] generate xmodel     :100%|██████████| 33/33 [00:00<00:00, 450.63it/s]

[INFO] dump xmodel: /tmp/customcnn_org.xmodel

[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA0_B4096_MAX_BG2

[UNILOG][INFO] Compile mode: dpu

[UNILOG][INFO] Debug mode: function

[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA0_B4096_MAX_BG2

[UNILOG][INFO] Graph name: customcnn_model, with op num: 53

[UNILOG][INFO] Begin to compile...

[UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1

[UNILOG][INFO] Compile done.

[UNILOG][INFO] The meta json is saved to "/workspace/chao-proj/handsigndigits_end2end/code/../output/meta.json"

[UNILOG][INFO] The compiled xmodel is saved to "/workspace/chao-proj/handsigndigits_end2end/code/../output/customcnn.xmodel"

[UNILOG][INFO] The compiled xmodel's md5sum is 93022a6b0243ac1251f7acc60c145a3d, and has been saved to "/workspace/chao-proj/handsigndigits_end2end/code/../output/md5sum.txt"

**************************************************

* VITIS_AI Compilation - Xilinx Inc.

**************************************************

+ echo -----------------------------------------

-----------------------------------------

+ echo 'MODEL COMPILED'

MODEL COMPILED

+ echo -----------------------------------------

-----------------------------------------

```

After all steps on host machine finished, copy below files to target repository.

If you are using Xilinx zcu102/zcu104/KV260 official vitis-ai image, the DPU configurations are the same on all 3 boards.  There files can be deployed on each of the 3 boards.


### Steps on target board

Copy target repository to target board.

Run app_mt.py with below command,

```
root@xilinx-zcu102-2021_1:/home/petalinux/Target_zcu102_HandSignDigit# python3 app_mt.py -d Examples/ -m customcnn.xmodel

Command line options:

 --image_dir :  Examples/

 --threads   :  1

 --model     :  customcnn.xmodel

Pre-processing 10 images...

Starting 1 threads...

Throughput=1111.96 fps, total frames = 10, time=0.0090 seconds

Correct:10, Wrong:0, Accuracy:1.0000

```

Luckily, we got all 10 images correctly.



