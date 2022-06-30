#!/usr/bin/env python3

import plotly.express as px
import tensorflow as tf
import numpy as np
import csv

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

split_time = int(len(time)*0.90)
time_train = time[:split_time]
x_train = series[:split_time]
time_test = time[split_time:]
y_test = series[split_time:]

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

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=60, kernel_size=5, strides=1, padding='causal', activation=tf.nn.relu, input_shape=[None, 1]),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.Dense(30, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x*400),
])

model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9),  metrics=['mae'])
history = model.fit(dataset, epochs=100, verbose=1)

performance_plot = px.line(history.history['mae'], title='Mean Absolute Error during training')
performance_plot.show()

forecast = model_forecast(model, series[..., np.newaxis], window_size)
forecast = forecast[split_time - window_size:-1, -1, 0]

fig = px.line(x=time_test, y=[forecast, y_test], title='Forecast for Sunspots')
fig.update_layout({'title':'Forecast for Sunspots', 'yaxis':{'title':'Sunspots'}, 'xaxis':{'title':'months'},'showlegend':False})
newnames = {'wide_variable_0':'Forecast', 'wide_variable_1': 'Real'}
fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                            )
                   )
fig.show()