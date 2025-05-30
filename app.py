from flask import Flask, render_template, request
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore

app = Flask(__name__)

# Load model dan scaler
model = load_model("diabetes_model_new.keras")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        jenis_kelamin = 1 if request.form['jenis_kelamin'] == 'Laki-laki' else 0
        usia = float(request.form['usia'])
        hipertensi = int(request.form['hipertensi'])
        penyakit_jantung = int(request.form['penyakit_jantung'])
        riwayat_merokok = request.form['riwayat_merokok']
        if riwayat_merokok == 'Tidak Pernah':
            riwayat = 0
        elif riwayat_merokok == 'Masih Merokok':
            riwayat = 1
        else:
            riwayat = 2
        bmi = float(request.form['bmi'])
        hba1c = float(request.form['hba1c'])
        glukosa = float(request.form['glukosa'])

        data = np.array([[jenis_kelamin, usia, hipertensi, penyakit_jantung, riwayat, bmi, hba1c, glukosa]])
        data_scaled = scaler.transform(data)
        pred = model.predict(data_scaled)[0][0]
        prediction = "Positif Diabetes" if pred >= 0.5 else "Negatif Diabetes"

    return render_template("index.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)