import pandas as pd
import streamlit as st
df1 = pd.read_csv("https://raw.githubusercontent.com/vrunm/nlp-datasets/main/earnings_calls_data_sentences.csv")
st.dataframe(data=df1, width=None, height=None, *, use_container_width=False)
