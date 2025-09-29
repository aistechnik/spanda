import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This relates to plotting datetime values with matplotlib:
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import MinMaxScaler
register_matplotlib_converters()

import altair as alt

# load a sample dataset as a pandas DataFrame
#from vega_datasets import data
#cars = data.cars()

# make the chart
#bars = alt.Chart(cars).mark_point().encode(
#    x='Horsepower',
#    y='Miles_per_Gallon',
#    color='Origin',
#).interactive()

#st.altair_chart(bars, theme='streamlit', use_container_width=True)

# Define section
data_sec = st.container()

with data_sec:
    ## Загрузка данных
    df = pd.read_csv('stat2_data.csv',index_col=0)

    # 
    numb_lins = len(df)
    print(numb_lins)

    # EDA:
    df.dropna(inplace=True)
    #len(df)

    # Извлечение даннах из входново файла .csv file
    y = df['value'].values.astype(float)

    # Определение размера тестового набора данных
    test_size = 12

    # Создание наборов данных train и test
    train_set = y[:-test_size]
    test_set = y[-test_size:]

    #print(f"shape of train_set : {train_set.shape}")
    #print(f"shape of test_set : {test_set.shape}")

    # Создание экземпляр масштабатора с диапазоном от -1 до 1
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Normalize the training set
    train_norm = scaler.fit_transform(train_set.reshape(-1, 1))

    #train_norm[:10]

    # Convert train_norm from an array to a tensor
    train_norm = torch.FloatTensor(train_norm).view(-1)

    # Define a window size
    window_size = 12

    # Define function to create seq/label tuples
    def input_data(seq, ws):  # ws is the window size
        out = []
        L = len(seq)
        for i in range(L-ws):
            window = seq[i:i+ws]
            label = seq[i+ws:i+ws+1]
            out.append((window,label))
        return out

    # Apply the input_data function to train_norm
    train_data = input_data(train_norm,window_size)
    len(train_data)  # this should equal 325-12-12

    #train_data[:2]

    # We will be using an LSTM layer of size (1, 1000).


    class KTGPredictor(nn.Module):

        def __init__(self, input_size=1, hidden_size=100, output_size=1):
            super().__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size)
            self.linear = nn.Linear(hidden_size, output_size)
            self.hidden = (torch.zeros(1, 1, self.hidden_size),
                       torch.zeros(1, 1, self.hidden_size))

        def forward(self, seq):
            out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
            pred = self.linear(out.view(len(seq), -1))
            return pred[-1]


        # ## Instantiate the model and define the loss and optimizer

    predictorModel = KTGPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(predictorModel.parameters(), lr=0.001)

    print(predictorModel)


    # ## Обучение модели

    epochs = 50

    import time
    start_time = time.time()

    for epoch in range(epochs):

        # extract the sequence & label from the training data
        for seq, y_train in train_data:

            # reset the parameters and hidden states
            optimizer.zero_grad()
            predictorModel.hidden = (torch.zeros(1,1,predictorModel.hidden_size),
                        torch.zeros(1,1,predictorModel.hidden_size))

            y_pred = predictorModel(seq)

            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

        # print training result
        print(f'Epoch: {epoch+1:2} Loss: {loss.item():10.8f}')

    print(f'\nDuration: {time.time() - start_time:.0f} seconds')


    # ## Model's inference

    future = 12

    # Add the last window of training values to the list of predictions
    preds = train_norm[-window_size:].tolist()

    # Set the model to evaluation mode
    predictorModel.eval()

    for i in range(future):
        seq = torch.FloatTensor(preds[-window_size:])
        with torch.no_grad():
            predictorModel.hidden = (torch.zeros(1,1,predictorModel.hidden_size),
                        torch.zeros(1,1,predictorModel.hidden_size))
            preds.append(predictorModel(seq).item())

    #preds[window_size:]  


    true_prediction = scaler.inverse_transform(np.array(preds[window_size:]).reshape(1, -1))
    true_prediction = true_prediction.squeeze()

    #df['value'][-12:]
    #x = np.arange(100)
    #numb_12 = numb_lins-test_size
    x = np.arange(numb_lins)
    #print(x)

    source = pd.DataFrame({
    'x': x,
    'f(x)': df['value']
    })

    tser = alt.Chart(source).mark_line().encode(
        x='x',
        y='f(x)'
    )
    
    x = np.arange(12)
    source = pd.DataFrame({
    'x': x,
    'f(x)': true_prediction
    })

    pred = alt.Chart(source).mark_line().encode(
        x='x',
        y='f(x)'
    )
    
    st.altair_chart(tser)
    st.altair_chart(pred)
