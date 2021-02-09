import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import configparser
import pathlib
import tensorflow as tf
import wget
import zipfile
import matplotlib.pyplot as plt
import time


def configure_for_performance(dataset):
    # batch_size = 32
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1000)
    # dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def downloadDataset():
    directory = 'datasets/'
    url = 'https://onedrive.live.com/download?cid=DE07050F0B3164E1&resid=DE07050F0B3164E1%2162938&authkey=AIT6kBW-GJwYm3Y'
    # url = 'https://onedrive.live.com/download?cid=DE07050F0B3164E1&resid=DE07050F0B3164E1%2159261&authkey=AMswoY3myXkPX2w'

    if not os.path.exists(directory):
        print(f"Directory {directory} was created, because it didn't exist")
        os.mkdir(directory)
    
    wget.download(url, out=directory)

    # TODO Make 'flower_photos.zip' as a Variable
    with zipfile.ZipFile(directory + 'flower_photos.zip', 'r') as zip:
        zip.extractall(directory)
        pass


def prepareDatasets():
    data_dir = pathlib.Path('datasets/flower_photos/')
    config = configparser.ConfigParser()
    config.read('src/config.ini')
    seed = (int)(time.mktime(time.localtime()) * 963) % 256
    print(f"Seed: {seed}")

    batch_size = int(config['ML']['batch_size'])
    img_height = int(config['ML']['img_height'])
    img_width = int(config['ML']['img_width'])

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(f"Anzahl Bilder: {image_count}")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        shuffle=True,
        validation_split=0.3,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        shuffle=True,
        validation_split=0.3,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    num_batches = tf.data.experimental.cardinality(train_ds)
    test_ds = train_ds.take(num_batches // 10)
    train_ds = train_ds.skip(num_batches // 10)

    # Performancetuning
    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)

    print('Number of train batches: %d' % tf.data.experimental.cardinality(train_ds))
    print('Number of validation batches: %d' % tf.data.experimental.cardinality(val_ds))
    print('Number of test batches: %d' % tf.data.experimental.cardinality(test_ds))

    # for image, _ in test_ds.take(1):
    #     first_image = image[0].numpy().astype("uint8")
    #     plt.imshow(first_image)
    #     plt.show()

    return train_ds, val_ds, test_ds


if __name__ == '__main__':
    downloadDataset()
    # prepareDatasets()
    pass
