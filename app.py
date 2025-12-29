import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

st.title('🏠House Price prediction using ML')
st.sidebar.image('https://i.pinimg.com/originals/4b/7e/69/4b7e69a0eb1cf87c5487634c35c4c552.gif')

df=pd.read_csv('house_data.csv')
X = df.iloc[:,:-3]
y = df.iloc[:,-1]

st.sidebar.title('🏠 select house feature')
st.image('https://i.pinimg.com/originals/4b/7e/69/4b7e69a0eb1cf87c5487634c35c4c552.gif')
all_value = []
for i in X:
  min_value = int(X[i].min())
  max_value = int(X[i].max())
  ans = st.sidebar.slider(f'select {i} value',min_value,max_value) 
  all_value.append(ans)

# st.write(all_value)
scaler = StandardScaler()

scaled_X = scaler.fit_transform(X)
final_value = scaler.transform([all_value])
@st.cache_data
def model_run(X,y):
 model = RandomForestRegressor()
 model.fit(X,y)
 return model
model = model_run(X,y)  
house_price = model.predict(final_value)[0]
with st.spinner('predicting House price'):
  time.sleep(3)
  msg = f'''house price is: ${round(house_price*100000,2)}'''
  st.success(msg)
  st.markdown('''**Design and devlopment by; Arun shukla**''')
  


















