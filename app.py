import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing

st.title('🏠House Price prediction using ML')
st.image('https://i.pinimg.com/originals/4b/7e/69/4b7e69a0eb1cf87c5487634c35c4c552.gif')

df=pd.read_csv('house_data.csv')
x = df.iloc[:,:-3]
y = df.iloc[:,-1]

st.sidebar.title('🏠 select house feature')
st.sidebar.image('https://i.pinimg.com/originals/4b/7e/69/4b7e69a0eb1cf87c5487634c35c4c552.gif')
all_value = []
for i in x:
  ans = st.sidebar.slider(f'select {i} value',) 
  all_value.append(ans)

st.write(all_value)




