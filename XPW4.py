# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.title('a title')
st.markdown('Xin Peng')

f = st.file_uploader('upload a csv file',type:=['csv'])
def cbn(c):
    try:
        pd.to_numeric(df[c])
        return True
    except:
        return False

if f is not None:
    df = pd.read_csv(f)
    df = df.applymap(lambda x: np.nan if x == " " else x)
    st.write(df)
    
    col = [x for x in df.columns if cbn(x)]
    st.write(col)
    df[col] = df[col].apply(pd.to_numeric,axis=0)
    
    sx = st.selectbox('select numeric column for the x-axis:',col)
    st.write(f'x = {sx}')
    sy = st.selectbox('select numeric column for the y-axis:',col)
    st.write(f'y = {sy}')
    sl = st.slider('choose how many rows:',0,df.shape[0])
    st.write(f'row = {sl}')
    
    m = alt.Chart(df).mark_circle().encode(
        x=sx, y=sy, size=sy, color=sx, tooltip=[sx, sy])
    st.altair_chart(m ,use_container_width = sl)
    
    
    



