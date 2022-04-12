#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt

from tensorflow.keras.layers import LSTM, Dense, Conv1D, Lambda

time_step = []
sunspots = []

with open('./Sunspots.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        sunspots.append(float(row[2]))
        time_step.append(int(row[0]))

series = np.array(sunspots)
time = np.array(time_step)

split_time = 3000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
y_valid = series[split_time:]

window_size = 60
batch_size = 256
shuffle_buffer_size = 3000

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
    Conv1D(filters=60, kernel_size=5, strides=1, padding='causal', activation=tf.nn.relu, input_shape=[None, 1]),
    LSTM(60, return_sequences=True),
    LSTM(60, return_sequences=True),
    Dense(30, activation=tf.nn.relu),
    Dense(10, activation=tf.nn.relu),
    Dense(1),
    Lambda(lambda x: x*400),
])

model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9),  metrics=['mae', 'accuracy'])
history = model.fit(dataset, epochs=100, verbose=1)

plt.plot(history.history['mae'])
plt.show()