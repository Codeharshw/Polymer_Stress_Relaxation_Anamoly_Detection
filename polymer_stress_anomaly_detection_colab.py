"""
polymer_stress_anomaly_detection_colab.py

Dual-method anomaly detection for polymer stress data, optimized for GPU execution (e.g., Google Colab T4).
1. IQR-based outlier detection (statistical baseline)
2. LSTM Autoencoder (learns temporal patterns, detects damage/recovery signatures)

Usage in Google Colab:
  1. Set Runtime -> Change runtime type -> GPU (T4 preferred)
  2. Run the upload cell to select your CSV file.
  3. Execute all cells below to perform the full analysis.

Author: [Codeharshw]
License: MIT
"""

# -----------------------
# 0) IMPORTS & FILE UPLOAD
# -----------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from google.colab import files  # For file upload in Colab
import io

print("="*60)
print("FILE UPLOAD & GPU SETUP")
print("="*60)

# Upload CSV file
uploaded = files.upload()
if not uploaded:
    raise ValueError("⚠ Please upload a CSV file to continue.")

# Get the uploaded file name dynamically
uploaded_filename = list(uploaded.keys())[0]
print(f"\n✓ File uploaded successfully: {uploaded_filename}")

# Read the uploaded CSV
df_full = pd.read_csv(io.BytesIO(uploaded[uploaded_filename]))
print(f"\nLoaded data with columns: {list(df_full.columns)}")
print(f"Total rows: {len(df_full)}")

# -----------------------
# GPU SETUP
# -----------------------
print(f"\nTensorFlow Version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"✓ {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs found and configured.")
        print(f"  Details: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("\n⚠ No GPU found. Go to 'Runtime' -> 'Change runtime type' -> 'GPU'.")


# -----------------------
# USER PARAMETERS
# -----------------------
TIME_COL = "Time (s)"                 # column name for time
STRESS_COL = "Normal Stress (Pa)"     # column name for stress

seq_len = 200
stride = 20
batch_size = 64
epochs = 100
threshold_pct = 98.5
iqr_k = 1.5
use_full_sequence_labeling = True


# -----------------------
# STEP 1: LOAD & PREVIEW DATA
# -----------------------
print("\n" + "="*60)
print("STEP 1: DATA PREVIEW")
print("="*60)

df = df_full[[TIME_COL, STRESS_COL]].copy()
df.columns = ['time', 'stress']
print(f"\nData range:")
print(f"  Time: {df['time'].min():.2f} to {df['time'].max():.2f} s")
print(f"  Stress: {df['stress'].min():.2e} to {df['stress'].max():.2e} Pa")

# -----------------------
# STEP 2: IQR OUTLIER DETECTION
# -----------------------
print("\n" + "="*60)
print("STEP 2: IQR-BASED OUTLIER DETECTION")
print("="*60)

def detect_outliers_iqr(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (series < lower) | (series > upper)

df['is_iqr_outlier'] = detect_outliers_iqr(df['stress'], k=iqr_k)
print(f"\nIQR outliers detected: {df['is_iqr_outlier'].sum()} / {len(df)}")

# Plot IQR outliers
plt.figure(figsize=(14,4))
plt.plot(df['time'], df['stress'], lw=0.8, color='steelblue', alpha=0.7)
plt.scatter(df.loc[df['is_iqr_outlier'],'time'], df.loc[df['is_iqr_outlier'],'stress'],
            color='orange', s=20, label='IQR Outliers', alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Normal Stress (Pa)')
plt.title('IQR-Based Outlier Detection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# -----------------------
# STEP 3: PREPARE CLEAN DATA
# -----------------------
df_clean = df.loc[~df['is_iqr_outlier']].copy()
clean_idx = df_clean.index.values
stress_vals = df_clean['stress'].values.reshape(-1,1).astype(np.float32)

seq_starts = list(range(0, len(stress_vals)-seq_len+1, stride))
if not seq_starts:
    raise ValueError("Not enough data for given seq_len. Try reducing seq_len.")

sequences = []
center_global_indices = []
for s in seq_starts:
    seq = stress_vals[s:s+seq_len]
    sequences.append(seq)
    center_global_indices.append(int(clean_idx[s + seq_len//2]))

sequences = np.stack(sequences)
print(f"Sequences built: {len(sequences)}")


# -----------------------
# STEP 4: SCALE & SPLIT
# -----------------------
scaler = StandardScaler()
scaler.fit(stress_vals)
sequences_scaled = scaler.transform(sequences.reshape(-1,1)).reshape(sequences.shape)

split = int(0.8 * len(sequences_scaled))
X_train, X_val = sequences_scaled[:split], sequences_scaled[split:]


# -----------------------
# STEP 5: LSTM AUTOENCODER
# -----------------------
timesteps, n_features = seq_len, 1

inp = keras.Input(shape=(timesteps, n_features))
x = layers.LSTM(128, activation='tanh', return_sequences=True, dropout=0.2)(inp)
x = layers.LSTM(64, activation='tanh', return_sequences=True, dropout=0.2)(x)
x = layers.LSTM(32, activation='tanh', return_sequences=False)(x)
latent = layers.RepeatVector(timesteps)(x)
x = layers.LSTM(32, activation='tanh', return_sequences=True)(latent)
x = layers.LSTM(64, activation='tanh', return_sequences=True, dropout=0.2)(x)
x = layers.LSTM(128, activation='tanh', return_sequences=True, dropout=0.2)(x)
out = layers.TimeDistributed(layers.Dense(n_features))(x)

model = keras.Model(inp, out)
model.compile(optimizer='adam', loss='mse')
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
history = model.fit(
    X_train, X_train,
    epochs=epochs, batch_size=batch_size,
    validation_data=(X_val, X_val),
    callbacks=[early_stop],
    verbose=1
)

plt.figure(figsize=(10,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Training History")
plt.show()


# -----------------------
# STEP 6: ANOMALY DETECTION
# -----------------------
X_all = sequences_scaled
X_rec = model.predict(X_all, batch_size=batch_size, verbose=0)
seq_mse = np.mean(np.square(X_all - X_rec), axis=(1,2))

threshold = np.percentile(seq_mse, threshold_pct)
is_anom_seq = seq_mse > threshold

df['is_anomaly_lstm'] = False
if use_full_sequence_labeling:
    for i, s in enumerate(seq_starts):
        if is_anom_seq[i]:
            start_global = clean_idx[s]
            end_global = clean_idx[min(s+seq_len-1, len(clean_idx)-1)]
            df.loc[start_global:end_global, 'is_anomaly_lstm'] = True
else:
    centers = [center_global_indices[i] for i in range(len(is_anom_seq)) if is_anom_seq[i]]
    df.loc[centers, 'is_anomaly_lstm'] = True

print(f"LSTM anomalies detected: {df['is_anomaly_lstm'].sum()}")


# -----------------------
# STEP 7: VISUALIZATION
# -----------------------
plt.figure(figsize=(14,4))
plt.plot(df['time'], df['stress'], color='steelblue', lw=0.8, alpha=0.7)
plt.scatter(df.loc[df['is_anomaly_lstm'],'time'], df.loc[df['is_anomaly_lstm'],'stress'],
            color='red', s=25, alpha=0.7, label='LSTM Anomalies')
plt.xlabel('Time (s)')
plt.ylabel('Normal Stress (Pa)')
plt.title('LSTM Anomaly Detection Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# -----------------------
# STEP 8: SAVE RESULTS
# -----------------------
output_file = "polymer_stress_anomaly_results.csv"
df.to_csv(output_file, index=False)
print(f"\n✓ Results saved to '{output_file}' — you can now download it below:")

from google.colab import files
files.download(output_file)
