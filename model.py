import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.callbacks import Callback  # type: ignore
import matplotlib.pyplot as plt
import pickle

# Callback untuk stop training di epoch tertentu
class StopAtEpoch(Callback):
    def __init__(self, stop_epoch):
        super().__init__()
        self.stop_epoch = stop_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 == self.stop_epoch:
            self.model.stop_training = True

# 1. Load Data
data = pd.read_csv('kumpulan_data_prediksi_diabetes.csv')

# 2. Encoding Fitur Kategori
label_encoder = LabelEncoder()
data['Jenis Kelamin'] = label_encoder.fit_transform(data['Jenis Kelamin'])
data['Riwayat Merokok'] = label_encoder.fit_transform(data['Riwayat Merokok'])
data['Diabetes'] = label_encoder.fit_transform(data['Diabetes'])  # 'Ya' = 1, 'Tidak' = 0

# 3. Pisahkan Fitur dan Target
X = data.drop(columns=['Diabetes'])
y = data['Diabetes']

# 4. Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Membangun Model Backpropagation
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 7. Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 8. Inisiasi Callback stop di epoch 46
stop_at_46 = StopAtEpoch(stop_epoch=46)

# 9. Training Model
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=5,
    validation_data=(X_test, y_test),
    callbacks=[stop_at_46]
)

# 10. Evaluasi Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss pada data uji: {loss:.4f}")
print(f"Akurasi pada data uji: {accuracy:.4f}")

# 11. Simpan model ke format `.keras` (direkomendasikan)
model.save('diabetes_model_new.keras')  # Simpan sebagai file, bukan folder

# 12. Simpan scaler agar bisa digunakan saat prediksi nanti
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 13. Visualisasi Loss dan Akurasi per Epoch
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.show()