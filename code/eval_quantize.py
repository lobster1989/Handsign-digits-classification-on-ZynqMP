'''
 Evaluate the quantized model
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
QAUNT_MODEL = 'quantized_model.h5'
image_dir = '../Sign-Language-Digits-Dataset/Dataset'
val_image_save_dir = '../output/val_image_save'
image_size = (100,100)
image_shape = (100,100,3)

# Load the quantized model
print('\nLoad quantized model..')
path = os.path.join(MODEL_DIR, QAUNT_MODEL)
with vitis_quantize.quantize_scope():
    model = models.load_model(path)

# Compile the model
print('\nCompile model..')
model.compile(optimizer="rmsprop", 
        loss="categorical_crossentropy",
        metrics=['accuracy']
        )

# get dataset
datagen = ImageDataGenerator(
        validation_split = 0.2, 
        rescale = 1./255
        )
        
# generate images using flow from directory method
val_generator = datagen.flow_from_directory(
        image_dir,
        #  save_to_dir = val_image_save_dir,
        #  save_prefix = 'val',
        subset = 'validation',
        color_mode = "grayscale",
        target_size = image_size,
        batch_size = 32,
        class_mode = 'categorical'
        )

# Evaluate model with test data
print("\nEvaluate model on test Dataset")
loss, acc = model.evaluate(val_generator)  # returns loss and metrics
print("loss: %.3f" % loss)
print("acc: %.3f" % acc)
