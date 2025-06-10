import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Load dan Latih Kedua Model
# ---------------------------
@st.cache_resource
def load_models():
    df = pd.read_csv("data_balita.csv")

    # Encode Jenis Kelamin
    le_gender = LabelEncoder()
    df['Jenis Kelamin'] = le_gender.fit_transform(df['Jenis Kelamin'])

    # Encode Status Gizi
    le_status = LabelEncoder()
    df['Status Gizi'] = le_status.fit_transform(df['Status Gizi'])

    # Fitur dan Label
    X = df[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']]
    y = df['Status Gizi']

    # Decision Tree
    model_dt = DecisionTreeClassifier()
    model_dt.fit(X, y)

    # KNN
    model_knn = KNeighborsClassifier(n_neighbors=3)
    model_knn.fit(X, y)

    return model_dt, model_knn, le_status, le_gender

model_dt, model_knn, le_status, le_gender = load_models()

# ---------------------------
# UI Streamlit
# ---------------------------
st.title("🔍 Perbandingan Prediksi: Decision Tree vs KNN")

umur = st.number_input("Umur Balita (bulan)", min_value=0, max_value=60, value=24)
jenis_kelamin = st.selectbox("Jenis Kelamin", ["laki-laki", "perempuan"])
tinggi = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=120.0, value=80.0)

if st.button("Bandingkan Prediksi"):
    jk_encoded = le_gender.transform([jenis_kelamin])[0]
    input_data = [[umur, jk_encoded, tinggi]]

    pred_dt = model_dt.predict(input_data)[0]
    pred_knn = model_knn.predict(input_data)[0]

    hasil_dt = le_status.inverse_transform([pred_dt])[0]
    hasil_knn = le_status.inverse_transform([pred_knn])[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌳 Decision Tree")
        st.success(f"Status Gizi: **{hasil_dt.upper()}**")
    with col2:
        st.subheader("📊 KNN")
        st.info(f"Status Gizi: **{hasil_knn.upper()}**")

    if hasil_dt == hasil_knn:
        st.warning("🔁 Kedua model memberikan hasil yang sama.")
    else:
        st.success("✅ Model memberikan hasil yang berbeda. Analisis bisa dilakukan.")