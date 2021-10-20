import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory 

# Settings:
image_size = (64,64)
image_shape = (64,64,3)
#  image_size = (28,28)
#  image_shape = (28,28,3)
image_dir = '../Sign-Language-Digits-Dataset/Dataset'
model_dir = '../output/float_model.h5'

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

val_generator = datagen.flow_from_directory(
        image_dir,
        subset = 'validation',
        target_size = image_size,
        batch_size = 32,
        class_mode = 'categorical'
        )

# list some images to see what we have
#  import matplotlib.pyplot as plt
#  for items in range(10):
    #  (x,y) = next(train_generator)
    #  plt.figure(figsize = (5,5))
    #  plt.imshow(x[0])
    #  plt.show()

    

#  train_dataset =  image_dataset_from_directory(
        #  image_dir,
        #  label_mode = 'categorical',
        #  color_mode = 'rgb',
        #  batch_size = 20,
        #  image_size = image_size,
        #  validation_split = 0.2,
        #  subset = 'training',
        #  seed = 1337
        #  )

#  val_dataset =  image_dataset_from_directory(
        #  image_dir,
        #  label_mode = 'categorical',
        #  color_mode = 'rgb',
        #  batch_size = 20,
        #  image_size = image_size,
        #  validation_split = 0.2,
        #  subset = 'validation',
        #  seed = 1337
        #  )


# build a convolutional model

#  def customcnn():
    #  # create a cnn model
    #  inputs = keras.Input(shape=(28,28,3))
    #  x = layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,3))(inputs)
    #  x = layers.MaxPooling2D((2,2))(x)
    #  x = layers.Conv2D(64, (3,3), activation='relu')(x)
    #  x = layers.MaxPooling2D((2,2))(x)
    #  x = layers.Conv2D(64, (3,3), activation='relu')(x)
    #  x = layers.Flatten()(x)
    #  x = layers.Dense(64, activation='relu')(x)
    #  outputs = layers.Dense(10, activation='softmax')(x)

    #  model = keras.Model(inputs=inputs, outputs=outputs, name='customcnn_model')
    #  model.summary()

    #  # Compile the model
    #  model.compile(optimizer="rmsprop",
            #  loss="categorical_crossentropy",
            #  metrics=['accuracy']
            #  )

    #  return model
#  model = customcnn()
inputs = keras.Input(shape=(64,64,3))
x = layers.Conv2D(32, (3,3), activation='relu', padding='same',input_shape=image_shape)(inputs)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(32, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs, name='customcnn_model')
model.summary()

#  Compile the model
model.compile(optimizer="rmsprop", 
        loss="categorical_crossentropy",
        metrics=['acc']
        )

# train model on training dataset
history = model.fit(
        train_generator,
        #  train_dataset,
        steps_per_epoch=50,
        epochs=10,
        validation_data=val_generator,
        #  validation_data = val_dataset,
        validation_steps=10
        )

model.save(model_dir)

# plot learning curve
#  history_dict = history.history
#  loss_values = history_dict['loss']
#  val_loss_values = history_dict['val_loss']
#  acc = history_dict['acc']
#  val_acc = history_dict['val_acc']
#  epochs = range(1,len(loss_values) + 1)

#  plt.figure(figsize = (6,6))
#  plt.subplot(2,1,1)
#  plt.plot(epochs, loss_values, 'bo', label='Training loss')
#  plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
#  plt.title("Training and validation loss")
#  plt.ylabel('Loss')
#  plt.legend()
#  plt.subplot(2,1,2)
#  plt.plot(epochs, acc, 'ro', label='Training accuracy')
#  plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
#  plt.title("Training and validation accuracy")
#  plt.xlabel('Epochs')
#  plt.ylabel('Accuracy')
#  plt.legend()
#  plt.show()
