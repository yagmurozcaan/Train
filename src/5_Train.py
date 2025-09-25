import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalAveragePooling1D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import regularizers

# --- Dosya yolları ---
FEATURE_DIR = r"C:\Users\Yagmur\Desktop\newobs\features"
X_train_path = os.path.join(FEATURE_DIR, "X_train.npy")
y_train_path = os.path.join(FEATURE_DIR, "y_train.npy")
X_val_path = os.path.join(FEATURE_DIR, "X_val.npy")
y_val_path = os.path.join(FEATURE_DIR, "y_val.npy")

# --- Verileri yükle ---
X_train = np.load(X_train_path)
y_train = np.load(y_train_path)
X_val = np.load(X_val_path)
y_val = np.load(y_val_path)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")

# --- CNN modeli (regularization + dropout ile) ---
def create_light_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, kernel_size=1, activation='relu',
               kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.3),
        Conv1D(16, kernel_size=1, activation='relu',
               kernel_regularizer=regularizers.l2(0.001)),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.4),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model

input_shape = (X_train.shape[1], X_train.shape[2])
model = create_light_cnn(input_shape)
model.summary()

# --- Callback'ler ---
checkpoint_path = "light_cnn_best_model.keras"
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
]

# --- Modeli eğit ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,   # küçük dataset → batch küçült
    callbacks=callbacks,
    verbose=1,
    shuffle=True
)

# --- Eğitim geçmişini görselleştir ---
def plot_history(history):
    metrics = ['accuracy', 'loss', 'precision', 'recall']
    plt.figure(figsize=(16, 8))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        plt.title(metric)
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_history(history)

# --- Keras modelini .h5 formatına kaydet ---
h5_path = "light_cnn_model.h5"
loaded_model = load_model(checkpoint_path)
loaded_model.save(h5_path)
print(f"Keras modeli .h5 formatında kaydedildi: {h5_path}")
