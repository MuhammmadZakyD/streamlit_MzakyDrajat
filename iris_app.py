import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

st.title("Iris Flower Classification")

# Load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Input user
sepal_length = st.slider('Sepal length', 4.0, 8.0, 5.0)
sepal_width = st.slider('Sepal width', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal length', 1.0, 7.0, 4.0)
petal_width = st.slider('Petal width', 0.1, 2.5, 1.0)

# Prediksi
clf = RandomForestClassifier()
clf.fit(iris.data, iris.target)
prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])

st.write("Prediction:", iris.target_names[prediction][0])
