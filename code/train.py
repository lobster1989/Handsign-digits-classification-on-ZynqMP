import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# To display images, set this as True
dispaly_images = False

# To draw learning curve, set this as True
display_learning_curve = False

# Settings:
image_size = (100,100)
image_shape = (100,100,1)

image_dir = '../Sign-Language-Digits-Dataset/Dataset'
model_dir = '../output/float_model.h5'

# create image data generator

datagen = ImageDataGenerator(
        validation_split = 0.2, 
        rescale = 1./255,
        rotation_range = 40,
        zoom_range=0.2
        )
        

# generate images using flow from directory method
train_generator = datagen.flow_from_directory(
        image_dir,
        subset = 'training',
        color_mode = "grayscale",
        target_size = image_size,
        batch_size = 32,
        class_mode = 'categorical'
        )

val_generator = datagen.flow_from_directory(
        image_dir,
        subset = 'validation',
        color_mode = "grayscale",
        target_size = image_size,
        batch_size = 32,
        class_mode = 'categorical'
        )

if dispaly_images == True:
    # print some pictures and and check the data
    print("An element from datagen: ")
    (x,y) = next(train_generator)
    print((x,y))

    # Visualize some pictures
    print("Some pictures from dataset: ")
    plt.figure(figsize = (6,6))
    plt.gray()
    for i in range(9):
        plt.subplot(3,3,i+1)
        (x,y) = next(train_generator)
        plt.imshow(x[0])
        #  plt.title('label:' + str(y[0]))
        plt.title('label:' + str(np.argmax(y[0])))
        plt.xticks([])
        plt.yticks([])
    plt.show()

   
# Function: create a custom CNN model
def customcnn():
    inputs = keras.Input(shape=image_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', input_shape=image_shape)(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='customcnn_model')
    model.summary()

    #  Compile the model
    model.compile(optimizer="rmsprop", 
            loss="categorical_crossentropy",
            metrics=['acc']
            )

    return model

# get model
model = customcnn()

# train model on training dataset
history = model.fit(
        train_generator,
        steps_per_epoch=48,
        epochs=10,
        validation_data=val_generator,
        validation_steps=12
        )

model.save(model_dir)

if display_learning_curve == True:
    # plot learning curve
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    epochs = range(1,len(loss_values) + 1)

    plt.figure(figsize = (6,6))
    plt.subplot(2,1,1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title("Training and validation loss")
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(epochs, acc, 'ro', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title("Training and validation accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()



