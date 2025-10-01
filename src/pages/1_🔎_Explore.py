import altair as alt
import streamlit as st
import numpy as np
import pandas as pd

from statist.stat2 import get_stat2

# Define functions
def load_data(path):
    dataset = pd.read_csv(path)
    return dataset


# Load data
uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])
if uploaded_file is not None:
    data_df = load_data(uploaded_file)
    print(uploaded_file.name)

# Define section
data_sec = st.container()

with data_sec:

    if uploaded_file is not None:
        data_df.to_csv('input_data.csv', index=False)
        get_stat2(uploaded_file.name)

        numb_lins = len(data_df)
        print(numb_lins)

        if st.button("Просмотр данных"):
            data_sec.write(data_df)
        
        if st.button("Просмотр графика"):
            finp = open('type_data.txt', 'r')
            typed = finp.readline().strip()
            print(typed)

            df = data_df.set_index("time_sec")
            source = pd.DataFrame({
                'time_sec': df.index,
                typed: data_df['value']
            })

            tser = alt.Chart(source).mark_line().encode(
                x='time_sec',
                y=typed
            ).properties(
                title='График исходных данных '+ typed
            )

            st.altair_chart(tser)
            finp.close()
