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
image_size = (64,64)
image_shape = (64,64,3)

# Load the quantized model
print('\nLoad quantized model..')
path = os.path.join(MODEL_DIR, QAUNT_MODEL)
with vitis_quantize.quantize_scope():
    model = models.load_model(path)

# get dataset
datagen = ImageDataGenerator(
        validation_split = 0.2, 
        rescale = 1./255
        #  rotation_range=20,
        #  width_shift_range=0.2,
        #  height_shift_range=0.2,
        #  horizontal_flip=True
        )
        

# generate images using flow from directory method
val_generator = datagen.flow_from_directory(
        image_dir,
        subset = 'validation',
        target_size = image_size,
        batch_size = 32,
        class_mode = 'categorical'
        )

# Compile the model
print('\nCompile model..')
model.compile(optimizer="rmsprop", 
        loss="categorical_crossentropy",
        metrics=['accuracy']
        )

# Evaluate model with test data
print("\nEvaluate model on test Dataset")
loss, acc = model.evaluate(val_generator)  # returns loss and metrics
print("loss: %.3f" % loss)
print("acc: %.3f" % acc)
