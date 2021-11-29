#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:43:42 2021

@author: pengxin1
"""

import streamlit as st
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import altair as alt
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

rng = np.random.default_rng()

# function to highlight
def colo(val):
    if val < -5:
        c = 'red'
    elif val > 5:
        c = 'blue'
    else:
        c = 'black'
    return 'color: %s' %c


st.title('Final Project')
st.markdown('**Xin Peng**')

st.write('The app is used to analyze a dataset that contains a number of measurements taken on a random sample of 252 American adult men.')


source = pd.read_csv('datatable.csv') # clean dataframe
source = source[source.notna().all(axis=1)]

num_cols = [c for c in source.columns if is_numeric_dtype(source[c])]
df = source[num_cols]

fat = (df['percent fat1'] + df['percent fat2'])/2 # take the mean of fat1 and fat2
dff = df.loc[:,'age':'wrist circum (cm)']
dff['percent fat'] = fat


st.markdown('The dataset that removed NaN value and only contain numerical colunms would be like:')
st.write(dff) # show the dataset
st.markdown('Now, we are going to find the linear relationship between different colunms.')


nc = [c for c in dff.columns]

aran =[ [20 for x in dff['age'] if 20<=x<30], [30 for x in dff['age'] if 30<=x<40],  [40 for x in dff['age'] if 40<=x<50], 
      [50 for x in dff['age'] if 50<=x<60], [60 for x in dff['age'] if 60<=x<70], [70 for x in dff['age'] if 70<=x]]
ageran = [x for y in aran for x in y]
dff['age range'] = ageran # assign the data by age range, prepare for plot


st.markdown('**Firstly, plot every picture with the chosen x column:**')
xvar = st.selectbox('Select the column you want for the x-axis:',nc)
ncc = nc.copy()
ncc.remove(xvar)


c1 = alt.Chart(dff).mark_circle().encode(
        alt.X(xvar, type='quantitative'),
        alt.Y(alt.repeat("row"), type='quantitative'),
        color='age range:N'
    ).properties(
        width=100,
        height=100
    ).repeat(
        row=ncc[0:3]
    )

c2 = alt.Chart(dff).mark_circle().encode(
        alt.X(xvar, type='quantitative'),
        alt.Y(alt.repeat("row"), type='quantitative'),
        color='age range:N'
    ).properties(
        width=100,
        height=100
    ).repeat(
        row=ncc[3:6]
    )

        
c3 = alt.Chart(dff).mark_circle().encode(
        alt.X(xvar, type='quantitative'),
        alt.Y(alt.repeat("row"), type='quantitative'),
        color='age range:N'
    ).properties(
        width=100,
        height=100
    ).repeat(
        row=ncc[6:9]
    )
        
c4 = alt.Chart(dff).mark_circle().encode(
        alt.X(xvar, type='quantitative'),
        alt.Y(alt.repeat("row"), type='quantitative'),
        color='age range:N'
    ).properties(
        width=100,
        height=100
    ).repeat(
        row=ncc[9:]
    )

m = alt.hconcat(c1,c2,c3,c4)


st.write('These are the charts of your choice:')
st.altair_chart(m)



if xvar is not None:
    st.markdown('**Now, move to one specific picture.**')
    yvar = st.selectbox('Select the column you want for the y-axis:',ncc)
    st.write(f'Your x is {xvar}, your y is {yvar}, and your chart is:')
    m2 = alt.Chart(dff).mark_circle().encode(
        x=xvar, y=yvar,
        color='age range:N')
    st.altair_chart(m2)
    st.markdown('Now let us find the linear relationship of the two columns.')
    reg = LinearRegression()
    X = dff[ xvar ].values[:,np.newaxis]
    y = dff[ yvar ].values
    reg.fit(X, y)
    co = reg.coef_
    inte = reg.intercept_
    st.markdown('**The linear function would be roughly like:**')
    st.write(f'{float(co)} x + {inte} = y')
    
    # plot the line and the original data
    st.write('And the plot would be:')
    fig,ax = plt.subplots()
    ax.scatter(X, y,color='b')
    ax.plot(X, reg.predict(X),color='r')
    st.pyplot(fig)
    
    # calculate error
    st.markdown('We can find the error between the function and the true value.')
    n = st.slider('Number of sample you want to calculate:',10,40,12,2)
    st.write(f'We will randomly choose {n} samples.')
    l = len(dff[xvar])
    ran = rng.choice(l, size=n, replace=False)
    sam = dff.iloc[ran]
    sap = sam[[xvar,yvar]]
    st.markdown('**The x and true y data of your random sample are:**')
    st.write(sam[[xvar,yvar]]) # show the dataframe
    pred = [float(co*y+inte) for y in sam[xvar]]
    sap['Predict value'] = pred
    diff = pred - sam[yvar]
    sap['Difference'] = diff
    # color the dataframe
    sapcol = sap.style.applymap(colo,subset=['Difference'])
    st.markdown('**The predict value and the difference are here:**')
    st.markdown('(The difference value bigger than 5 are highlighten)')
    st.write(sapcol)
    
    # predict value
    st.markdown('**You can also try your own predict by the model.**')
    inpt = int(st.number_input(
        label="Enter the number as x value:", value=10))
    ans = reg.predict([[inpt]])
    st.write(f'The predict value of {inpt} would be {float(ans)}.')
    
    










