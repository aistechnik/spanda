
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly

##from statist.stat1 import get_stat1
from statist.stat2 import get_stat2

# Define functions
def load_data(path):
    dataset = pd.read_csv(path)
    return dataset


##with st.sidebar:
# Load data
uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])
if uploaded_file is not None:
    data_df = load_data(uploaded_file)
    print(uploaded_file.name)
    ##df = pd.read_csv(uploaded_file, index_col=False)


# Define section
data_sec = st.container()

# Set up the data section that users will interact with
with data_sec:
    #data.title("On this page, you can preview the dataset and view daily sales across stores")
    #data_path = st.text_input("Укажите путь к файлу:", "")

    # Load the dataset
    #data_path = "data1/bpm/20250901-01000001_1.csv"
    #data_path="data1/uterus/20250901-01000001_2.csv"
    if uploaded_file is not None:
        data_df.to_csv('input_data.csv', index=False)
        #data_df = load_data(data_path)
        get_stat2(uploaded_file.name)
        ##get_stat1(data_path)
        

        ##st.write("View the Dataset below")

        if st.button("Просмотр данных"):
            data_sec.write(data_df)
        
        ##st.write("Graph showing daily sales can be viewed below")
        if st.button("Просмотр графика"):

            # Set the "date" column as the index
            data_df = data_df.set_index("time_sec")

            # Display the line chart with dates on the x-axis
            #st.subheader("A Chart of the Daily Sales Across Favorita Stores")
            #st.line_chart(load_df["value"])
            fig = px.line(x=data_df.index, y=data_df.value, title="График") 
            st.plotly_chart(fig)  

##Adding visua#lof model prediction        
##st.write("View the model's prediction")
#if st.button("Model's graph"):

    # we predict with the model
    #result = model.predict(test_df)

    # Create a Plotly line chart for the model's predictions
    #fig = px.line(x=test_df.index, y=result, title="Model's Forecast")

    # Display the chart using st.plotly_chart()
    #st.subheader("A plot of model's forecast")
    #st.plotly_chart(fig)

    







