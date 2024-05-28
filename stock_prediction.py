"""
Author : Priyansh Saxena
Dated : 21 December 2023

Visit the Notebook for better explanation of the code
"""

import pandas as pd
import numpy as np
from datetime import datetime
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import TimeseriesGenerator
from plotly import graph_objs as go
from sklearn.metrics import r2_score
import yfinance as yf

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

def download_and_process_data(stock_name):
    df = yf.download(stock_name, period='max')
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    close_data = df['Close'].values
    close_data = close_data.reshape((-1, 1))
    info = yf.Ticker(stock_name)
    return df, close_data, info

def split_data(close_data, df):
    split_percent = 80 / 100
    split = int(split_percent * len(close_data))
    close_train = close_data[:split]
    close_test = close_data[split:]
    date_train = df['Date'][:split]
    date_test = df['Date'][split:]
    return close_train, close_test, date_train, date_test

def sequence_to_supervised(look_back, close_train, close_test):
    train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)
    test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)
    return train_generator, test_generator

def train_model(look_back, train_generator, epochs):
    lstm_model = Sequential()
    lstm_model.add(LSTM(10, activation='relu', input_shape=(look_back, 1)))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit_generator(train_generator, epochs=epochs)
    return lstm_model

def plot_train_test_graph(stock, model, test_generator, close_train, close_test, date_train, date_test):
    prediction = model.predict_generator(test_generator)
    close_train = close_train.reshape((-1))
    close_test = close_test.reshape((-1))
    prediction = prediction.reshape((-1))
    
    trace1 = go.Scatter(
        x=date_train,
        y=close_train,
        mode='lines',
        name='Data'
    )
    trace2 = go.Scatter(
        x=date_test,
        y=prediction,
        mode='lines',
        name='Prediction',
        line=dict(color='red')
    )
    trace3 = go.Scatter(
        x=date_test,
        y=close_test,
        mode='lines',
        name='Ground Truth'
    )
    
    layout = go.Layout(
        title=stock,
        xaxis={'title': "Date"},
        yaxis={'title': "Close"}
    )
    
    figure = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    figure.update_layout(
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors["background"],
        font_color=colors['text']
    )
    
    return figure, r2_score(close_test[:-15], prediction)

def predict(num_prediction, model, close_data, look_back):
    prediction_list = close_data[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    
    prediction_list = prediction_list[look_back-1:]
    return prediction_list

def predict_dates(num_prediction, df):
    last_date = df['Date'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

def predicting(close_data, model, look_back, df):
    close_data = close_data.reshape((-1))
    num_prediction = 30
    forecast = predict(num_prediction, model, close_data, look_back)
    forecast_dates = predict_dates(num_prediction, df)
    return close_data, forecast, forecast_dates

def plot_future_prediction(model, test_generator, close_train, close_test, df, forecast_dates, forecast):
    prediction = model.predict_generator(test_generator)
    close_train = close_train.reshape((-1))
    close_test = close_test.reshape((-1))
    prediction = prediction.reshape((-1))
    
    trace1 = go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Data'
    )
    trace2 = go.Scatter(
        x=forecast_dates,
        y=forecast,
        mode='lines',
        name='Prediction'
    )

    layout = go.Layout(
        title="FUTURE PREDICTION",
        xaxis={'title': "Date"},
        yaxis={'title': "Close"}
    )
    
    figure = go.Figure(data=[trace1, trace2], layout=layout)
    figure.update_layout(
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors["background"],
        font_color=colors['text']
    )
    
    return figure

