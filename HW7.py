import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

st.title('KMeans')
st.markdown('Xin Peng')

sli = st.slider('choose the number of iterations',1,10)
st.write(f'iterations = {sli}.')

X, _ = make_blobs(n_samples=300, centers=5, n_features=2, random_state = 1)
df = pd.DataFrame(X, columns = list("ab"))
starting_points = np.array([[0,0],[-2,0],[-4,0],[0,2],[0,4]])

for i in range(sli):
    kmeans = KMeans(n_clusters = 5 , max_iter= i+1, init=starting_points, n_init = 1)
    kmeans.fit(X);
    df["c"] = kmeans.predict(X)

    starting_points = pd.DataFrame(kmeans.cluster_centers_, columns = list("ab"))

    
    
chart1 = alt.Chart(df).mark_circle().encode(
    x = "a",
    y = "b",
    color = "c:N"
    )

chart_centers = alt.Chart(starting_points).mark_point().encode(
    x = "a",
    y = "b",
    color = alt.value("black"),
    shape = alt.value("diamond"),
    )


st.altair_chart(chart1 + chart_centers)


