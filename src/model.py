import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import configparser
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.layers.experimental.preprocessing as exper_preprocess
import numpy as np

from dataset import prepareDatasets

path_to_model = 'models/main_model'


def createModel():
    config = configparser.ConfigParser()
    config.read('src/config.ini')

    img_height = int(config['ML']['img_height'])
    img_width = int(config['ML']['img_width'])
    num_classes = int(config['ML']['num_classes'])
    num_epochs = int(config['ML']['num_init_epochs'])

    train_ds, val_ds, test_ds = prepareDatasets()
    
    # Layers
    data_augmentation_layer = tf.keras.Sequential([
            exper_preprocess.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
            exper_preprocess.RandomRotation(0.1),
            exper_preprocess.RandomZoom(0.1),
    ])
    normalization_layer = exper_preprocess.Rescaling(1./255, input_shape=(img_height, img_width, 3))

    model = tf.keras.models.Sequential([
        data_augmentation_layer,
        normalization_layer,
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    model.fit(train_ds, validation_data=val_ds, epochs=num_epochs)

    model.save(path_to_model)
    pass


def trainModel():
    config = configparser.ConfigParser()
    config.read('src/config.ini')

    epochs = int(config['ML']['num_train_epochs'])

    train_ds, val_ds, test_ds = prepareDatasets()

    model = tf.keras.models.load_model(path_to_model)

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    model.save(path_to_model)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    pass


def predictModel():
    train_ds, val_ds, test_ds = prepareDatasets()
    model = tf.keras.models.load_model(path_to_model)
    image_batch, label_batch = test_ds.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()

    print(label_batch[0])

    plt.imshow(image_batch[0].astype("uint8"))
    plt.title(predictions[0])
    plt.show()

    pass


if __name__ == '__main__':
    # createModel()
    # trainModel()
    predictModel()
    pass
