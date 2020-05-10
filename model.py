import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.regularizers import l2
import math
import argparse

INPUT_WIDTH = INPUT_HEIGHT = 224

class DatasetGenerator:
    def __init__(self):
        self.test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, samplewise_center=True)
        self.train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, validation_split=0.0, samplewise_center=True)

    def get_test_set(self, base_path, batch_size):
        classes = os.listdir(f'{base_path}/test')
        return self.test_generator.flow_from_directory(directory=f'{base_path}/test',
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            target_size=(224, 224),
                                                            classes=classes)
    
    def get_train_set(self, base_path, batch_size):
        classes = os.listdir(f'{base_path}/train')
        return self.train_generator.flow_from_directory(directory=f'{base_path}/train',
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        target_size=(224, 224),
                                                        classes=classes)
    
    def get_validation_set(self, base_path, batch_size):
        classes = os.listdir(f'{base_path}/train')
        return self.train_generator.flow_from_directory(directory=f'{base_path}/train',
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        target_size=(224, 224),
                                                        classes=classes,
                                                        subset="validation")

def mobilenet():
    premade = tf.keras.applications.MobileNet(
            input_shape=(224, 224, 3), include_top=False, pooling='max')
    model = tf.keras.Sequential([
        premade,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=500, activation='relu'),
        tf.keras.layers.Dense(units=30, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def mobilenetv2():
    premade = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, pooling='max')
    model = tf.keras.Sequential([
        premade,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=500, activation='relu'),
        tf.keras.layers.Dense(units=30, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model():
    m = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3)),
        tf.keras.layers.Conv2D(64, (5, 5), activation='relu', kernel_regularizer=l2()),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(64, (5, 5), activation='relu', kernel_regularizer=l2()),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2()),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2()),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(500, activation='relu', kernel_regularizer=l2()),
        tf.keras.layers.Dense(30, activation='softmax', kernel_regularizer=l2())
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return m

def get_num_samples(data_dir, classes):
    count = 0
    for c in classes:
        count += len(os.listdir(f'{data_dir}/{c}'))

    return count

def main(epochs, batch_size, data_dir):
    generator = DatasetGenerator()

    callback = tf.keras.callbacks.ModelCheckpoint(
            'mobilenetv2.hdf5',
            monitor='val_accuracy',
            save_best_only=True)

    m = mobilenetv2()
    m.fit(
        generator.get_train_set(data_dir, batch_size),
        epochs=epochs,
        validation_data=generator.get_test_set(data_dir, batch_size),
        callbacks=[callback])
    
    m.save('mobilenetv2_end.hdf5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_dir', type=str)

    args, _ = parser.parse_known_args()

    main(args.epochs, args.batch_size, args.data_dir)
