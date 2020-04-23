import tensorflow as tf
import matplotlib.pyplot as plt
import os
import math

INPUT_WIDTH = INPUT_HEIGHT = 225

def get_input_flow(data_dir, batch_size, classes):
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
    return image_generator.flow_from_directory(directory=data_dir,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        target_size=(225, 225),
                                                        classes=classes)

def model():
    m = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3)),
        tf.keras.layers.Conv2D(16, (5, 5), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(4, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return m

def get_num_samples(data_dir, classes):
    count = 0
    for c in classes:
        count += len(os.listdir(f'{data_dir}/{c}'))

    return count

def main():
    epochs = 10
    classes = ['echeveria']
    samples = get_num_samples('data/', classes)
    batch_size = 32

    flow = get_input_flow('data', batch_size, classes)
    m = model()
    
    m.fit(flow, steps_per_epoch=math.ceil(samples / batch_size), epochs=epochs)

if __name__ == '__main__':
    main()