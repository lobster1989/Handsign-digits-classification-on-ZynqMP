'''
 Quantize the float point model
 Author: chao.zhang
'''

import os
 
# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as models
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_DIR = '../output'
FLOAT_MODEL = 'float_model.h5'
QAUNT_MODEL = 'quantized_model.h5'

image_size = (100,100)
image_dir = '../Sign-Language-Digits-Dataset/Dataset'

# Load the floating point trained model
print('Load float model..')
path = os.path.join(MODEL_DIR, FLOAT_MODEL)
try:
    float_model = models.load_model(path)
    float_model.summary()
except:
    print('\nError:load float model failed!')

# get input dimensions of the floating-point model
height = float_model.input_shape[1]
width = float_model.input_shape[2]
print("\nmodel input size:", height, width)

# get the validation dataset for quantization
print("\nLoad validation dataset for quantization..")
datagen = ImageDataGenerator(
        rescale = 1./255
        )
        
train_generator = datagen.flow_from_directory(
        image_dir,
        target_size = image_size,
        color_mode = "grayscale",
        batch_size = 32,
        class_mode = 'categorical'
        )

# Run quantization
print('\nRun quantization..')
quantizer = vitis_quantize.VitisQuantizer(float_model)
quantized_model = quantizer.quantize_model(
        calib_dataset=train_generator
        )

# Save quantized model
path = os.path.join(MODEL_DIR, QAUNT_MODEL)
quantized_model.save(path)
print('\nSaved quantized model as',path)




