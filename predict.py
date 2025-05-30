import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. Load data uji
data_baru = pd.read_csv('kumpulan_data_prediksi_diabetes.csv')

# 2. Pemetaan label kategori
gender_map = {'Laki-laki': 1, 'Perempuan': 0}
smoke_map = {
    'Tidak Pernah': 0,
    'Masih Merokok': 1,
    'Pernah Merokok': 2,
    'Pernah': 3,
    'Tidak Diketahui': 4,
    'Tidak Saat Ini': 4
}

# 3. Encoding kolom kategori
data_baru['Jenis Kelamin'] = data_baru['Jenis Kelamin'].map(gender_map)
data_baru['Riwayat Merokok'] = data_baru['Riwayat Merokok'].map(lambda x: smoke_map.get(x, 4))

# 4. Simpan kolom target jika ada
if 'Diabetes' in data_baru.columns:
    y_true = data_baru['Diabetes'].map({'Tidak': 0, 'Ya': 1})
    data_baru = data_baru.drop(columns=['Diabetes'])
else:
    y_true = None

# 5. Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 6. Normalisasi
X_scaled = scaler.transform(data_baru)

# 7. Load model
model = load_model('diabetes_model_new.keras')

# 8. Prediksi
predictions = model.predict(X_scaled)
predicted_labels = ['Ya' if p >= 0.5 else 'Tidak' for p in predictions]

# 9. Tambahkan hasil ke DataFrame
data_baru['Prediksi Diabetes'] = predicted_labels

# 10. Evaluasi jika tersedia label
if y_true is not None:
    y_pred = [1 if p >= 0.5 else 0 for p in predictions]
    acc = accuracy_score(y_true, y_pred)
    print(f"Akurasi pada data uji: {acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    labels = ['Negatif', 'Positif']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.title('Confusion Matrix Heatmap')
    plt.show()

    data_baru['Label Asli'] = y_true.map({0: 'Tidak', 1: 'Ya'})

# 11. Simpan hasil
data_baru.to_csv('hasil_prediksi.csv', index=False)

# 12. Tampilkan hasil
print(data_baru[['Usia', 'BMI', 'Kadar Glukosa Darah', 'Prediksi Diabetes']].head(10))
