import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def load_model(filename):
  model = joblib.load(filename)
  return model

def show_raw_data(df):
    """Raw Dataset."""
    st.subheader("Raw Dataset")
    
    num_rows = st.slider("Jumlah baris yang ditampilkan: ", 5, len(df), 5)
    st.write(df.head(num_rows))
    
    # Checkbox untuk menampilkan seluruh dataset
    if st.checkbox("Tampilkan semua data"):
        st.write(df)

def show_data_visualization(df):
    """Data Visualization."""
    st.subheader("📈 Visualisasi Data")

    chart_type = st.selectbox("Pilih jenis visualisasi:", ["Histogram", "Boxplot", "Scatter Plot"])

    num_columns = df.select_dtypes(include=['number']).columns
    col1 = st.selectbox("Pilih Kolom 1:", num_columns)

    if chart_type == "Histogram":
        fig, ax = plt.subplots()
        sns.histplot(df[col1], kde=True, ax=ax)
        st.pyplot(fig)

    elif chart_type == "Boxplot":
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col1], ax=ax)
        st.pyplot(fig)

    elif chart_type == "Scatter Plot":
        col2 = st.selectbox("Pilih Kolom 2:", num_columns)
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[col1], y=df[col2], data=df, ax=ax)
        st.pyplot(fig)


def show_user_input():
    """User Input"""
    st.subheader("Masukkan Data untuk Prediksi")

    age = st.slider("Usia", 5, 100, 25)
    height = st.slider("Tinggi (m)", 1.2, 2.2, 1.7)
    weight = st.slider("Berat (kg)", 30, 200, 70)
    FCVC = st.slider("Frekuensi Konsumsi Sayur (0-3)", 0.0, 3.0, 2.0)
    NCP = st.slider("Jumlah Makan dalam Sehari", 1.0, 5.0, 3.0)
    CH2O = st.slider("Asupan Air (Liter/Hari)", 0.5, 5.0, 2.0)
    FAF = st.slider("Frekuensi Aktivitas Fisik (0-3)", 0.0, 3.0, 1.0)
    TUE = st.slider("Waktu di Depan Layar (0-2)", 0.0, 2.0, 1.0)

    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    family_history = st.selectbox("Riwayat Obesitas dalam Keluarga", ["yes", "no"])
    FAVC = st.selectbox("Sering Konsumsi Makanan Cepat Saji?", ["yes", "no"])
    CAEC = st.selectbox("Frekuensi Konsumsi Makanan di Antara Makan Besar", ["no", "Sometimes", "Frequently", "Always"])
    SMOKE = st.selectbox("Apakah Anda Merokok?", ["yes", "no"])
    SCC = st.selectbox("Mengontrol Kalori yang Dikonsumsi?", ["yes", "no"])
    CALC = st.selectbox("Frekuensi Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
    MTRANS = st.selectbox("Moda Transportasi", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

    binary_map = {"Male": 1, "Female": 0, "yes": 1, "no": 0}
    label_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    mtrans_map = {"Public_Transportation": 0, "Automobile": 1, "Walking": 2, "Motorbike": 3, "Bike": 4}

    user_data = pd.DataFrame({
        "Gender": [binary_map[gender]],
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "family_history_with_overweight": [binary_map[family_history]],
        "FAVC": [binary_map[FAVC]],
        "FCVC": [FCVC],
        "NCP": [NCP],
        "CAEC": [label_map[CAEC]],
        "SMOKE": [binary_map[SMOKE]],
        "SCC": [binary_map[SCC]],
        "FAF": [FAF],
        "TUE": [TUE],
        "CH2O": [CH2O],
        "CALC": [label_map[CALC]],
        "MTRANS": [mtrans_map[MTRANS]],
    })

    st.subheader("Data yang Anda Masukkan")
    st.write(user_data)

    return user_data

def load_model(filename):
    model = pickle.load(open(filename, "rb"))
    return model

def main():
    st.title('Obesity Prediction')

    st.write('2702264576 - Stephanie Nadya Assignment Model Deployment')

    file_path = "ObesityDataSet_raw_and_data_sinthetic.csv"
    df = pd.read_csv(file_path)

    st.sidebar.header("Navigasi")
    menu = st.sidebar.radio("Pilih Menu:", ["Lihat Data", "Visualisasi Data", "Prediksi Obesitas"])

    # Panggil function jika user memilih menu "Lihat Data"
    if menu == "Lihat Data":
        show_raw_data(df)
    if menu == "Visualisasi Data":
        show_data_visualization(df)
    
    
    model = load_model("trained_model.pkl")
    user_input = show_user_input()

    if st.button("Prediksi"):
        prediction_proba = model.predict_proba(user_input)

        class_labels = model.classes_ 
        prob_df = pd.DataFrame(prediction_proba, columns=class_labels)
        prob_df = prob_df.transpose().reset_index()
        prob_df.columns = ['Class', 'Probability']

        st.subheader("Probabilitas Klasifikasi")
        st.write(prob_df)
        predicted_class = class_labels[prediction[0]]
        st.subheader("📈 Prediksi Akhir")
        st.success(f"The predicted output is: **{predicted_class}**")

if __name__ == "__main__":
    main()
  
