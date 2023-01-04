import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations

# setting random seed
tf.random.set_seed(123)

# create dir to save training history and plots
if not os.path.exists('history_plot'):
    os.mkdir('history_plot')

# create dir to save trained models
if not os.path.exists('model'):
    os.mkdir ('model')

# laod training and validation set from tfds
(train_ds, val_ds), dataset_info = tfds.load(
    'fashion_mnist',
    split=['train[:70%]', 'train[70%:]'],
    with_info=True,
    as_supervised=True,
)

# constants
num_classes = dataset_info.features['label'].num_classes
IMG_SIZE = 28
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE
epochs=30
no_augmentation = 0
augs = ['flip', 'rotate', 'bright', 'contrast', 'translate', 'zoom']

# resize layer to makes sure all training/validation/test data are of the same size
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1./255)
])

# define a prepare function
@tf.autograph.experimental.do_not_convert
def prepare(ds, shuffle=False, augment=False):
    # Resize and rescale all datasets.
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y),
                num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    # Batch all datasets.
    ds = ds.batch(batch_size)
    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)

val_ds = prepare(val_ds)

for no_augmentation in range(len(augs)+1):
    # train a model with no data augmentation
    if no_augmentation == 0:
        current_train_ds = prepare(train_ds, shuffle=True)
        # define and train model
        model = tf.keras.Sequential([
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes)
        ])
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        history = model.fit(
            current_train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                patience=5,
                min_delta=0.01,
                mode='min',
                monitor='val_loss',
                restore_best_weights=True,
                verbose=1)
            ]
        )
        # plot training process history
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        e = range(len(acc))
        plt.figure()
        plt.plot(e, acc, label='Training accuracy')
        plt.plot(e, val_acc, label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend(loc=0)
        plt.savefig("history_plot/no_aug.png")
        plt.clf()
        pd.DataFrame({'acc': acc, 'val_acc': val_acc}).to_csv("history_plot/no_aug.csv")
        # save the model
        model.save("model/no_aug")
    else:
        # loop through all possible combination of data augmentation
        for aug in list(combinations(augs, no_augmentation)):
            data_augmentation = tf.keras.Sequential(name='augmentation')
            if 'flip' in aug:
                data_augmentation.add(layers.RandomFlip("horizontal_and_vertical"))
            if 'rotate' in aug:
                data_augmentation.add(layers.RandomRotation(0.4))
            if 'bright' in aug:
                data_augmentation.add(layers.RandomBrightness(0.2))
            if 'contrast' in aug:
                data_augmentation.add(layers.RandomContrast(0.2))
            if 'translate' in aug:
                data_augmentation.add(layers.RandomTranslation(0.2, 0.2))
            if 'zoom' in aug:
                data_augmentation.add(layers.RandomZoom(0.2))

            current_train_ds = prepare(train_ds, shuffle=True, augment=True)
            # define and train model
            model = tf.keras.Sequential([
                layers.Conv2D(16, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(32, activation='relu'),
                layers.Dense(num_classes)
            ])
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            history = model.fit(
                current_train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    patience=5,
                    min_delta=0.01,
                    mode='min',
                    monitor='val_loss',
                    restore_best_weights=True,
                    verbose=1)
                ]
            )
            # plot training process history
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            e = range(len(acc))
            plt.figure()
            plt.plot(e, acc, label='Training accuracy')
            plt.plot(e, val_acc, label='Validation accuracy')
            plt.title('Training and validation accuracy')
            plt.legend(loc=0)
            plt.savefig(f"history_plot/{'_'.join(aug)}.png")
            plt.clf()
            pd.DataFrame({'acc': acc, 'val_acc': val_acc}).to_csv(f"history_plot/{'_'.join(aug)}.csv")
            # save model
            model.save(f"model/{'_'.join(aug)}")
