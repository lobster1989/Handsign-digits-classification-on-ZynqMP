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
from tensorflow.keras.preprocessing import image_dataset_from_directory 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_DIR = '../output'
FLOAT_MODEL = 'float_model.h5'
QAUNT_MODEL = 'quantized_model.h5'

image_size = (64,64)
#  image_size = (28,28)
image_dir = '../Sign-Language-Digits-Dataset/Dataset'

# set learning phase for no training: This line must be executed before loading Keras model
#  from tensorflow.keras import backend as K
#  K.set_learning_phase(0)

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
# create image data generator

datagen = ImageDataGenerator(
        validation_split = 0.2, 
        rescale = 1./255
        #  rotation_range=20,
        #  width_shift_range=0.2,
        #  height_shift_range=0.2,
        #  horizontal_flip=True
        )
        

# generate images using flow from directory method
train_generator = datagen.flow_from_directory(
        image_dir,
        subset = 'training',
        target_size = image_size,
        batch_size = 32,
        class_mode = 'categorical'
        )


#  calib_dataset = image_dataset_from_directory(
        #  image_dir, 
        #  label_mode = 'categorical',
        #  color_mode = 'rgb',
        #  #  image_size = image_size
        #  batch_size = 32,
        #  image_size = (height, width)
        #  )

# show some images to see what we have 
#  for images,labels in calib_dataset.take(3):
    
    #  plt.figure(figsize = (10,10))
    #  for i in range(9):
        #  ax = plt.subplot(3,3,i+1)
        #  plt.imshow(images[i].numpy().astype("uint8"))
        #  #  plt.title(int(labels[i]))
        #  plt.axis('off')
    #  plt.show() 

# Run quantization
print('\nRun quantization..')
quantizer = vitis_quantize.VitisQuantizer(float_model)
quantized_model = quantizer.quantize_model(
        calib_dataset=train_generator
        #  calib_steps=1000,
        #  calib_batch_size=32
        )

# Save quantized model
path = os.path.join(MODEL_DIR, QAUNT_MODEL)
quantized_model.save(path)
print('\nSaved quantized model as',path)




