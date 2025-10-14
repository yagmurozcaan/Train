import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib
from datetime import datetime

class AutismDetectionTrainer:
    def __init__(self, features_dir="data/features", 
                 csv_path=r"data\final_balanced_clean_dataset.csv"):
        self.features_dir = features_dir
        self.csv_path = csv_path
        self.model = None
        self.history = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.X_val, self.y_val = None, None

    def load_data(self):
        print("Önceden bölünmüş feature dosyaları yükleniyor...")
        try:
            self.X_train = np.load(os.path.join(self.features_dir, "X_train.npy"))
            self.y_train = np.load(os.path.join(self.features_dir, "y_train_binary.npy"))
            self.X_val = np.load(os.path.join(self.features_dir, "X_val.npy"))
            self.y_val = np.load(os.path.join(self.features_dir, "y_val_binary.npy"))
            self.X_test = np.load(os.path.join(self.features_dir, "X_test.npy"))
            self.y_test = np.load(os.path.join(self.features_dir, "y_test_binary.npy"))
            
            # Video splitting bilgileri
            self.train_videos = np.load(os.path.join(self.features_dir, "train_videos.npy"), allow_pickle=True)
            self.val_videos = np.load(os.path.join(self.features_dir, "val_videos.npy"), allow_pickle=True)
            self.test_videos = np.load(os.path.join(self.features_dir, "test_videos.npy"), allow_pickle=True)
            
            print(f"Train: {self.X_train.shape}, {self.y_train.shape}")
            print(f"Validation: {self.X_val.shape}, {self.y_val.shape}")
            print(f"Test: {self.X_test.shape}, {self.y_test.shape}")
            
            # NaN kontrolü
            for dataset_name, dataset in [("X_train", self.X_train), ("X_val", self.X_val), ("X_test", self.X_test)]:
                if np.isnan(dataset).any():
                    print(f"NaN tespit edildi ({dataset_name}), temizleniyor...")
                    if dataset_name == "X_train":
                        self.X_train = np.nan_to_num(self.X_train)
                    elif dataset_name == "X_val":
                        self.X_val = np.nan_to_num(self.X_val)
                    elif dataset_name == "X_test":
                        self.X_test = np.nan_to_num(self.X_test)
            
            return True
        except FileNotFoundError as e:
            print(f"Hata: Feature dosyaları bulunamadı! {e}")
            return False

    def create_model(self, input_shape):
        """
        T zaman serisi için Bidirectional LSTM modeli
        """
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(Masking(mask_value=0.))
        
        # LSTM katmanları
        model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)))
        model.add(Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.2)))
        
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification

        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'AUC']
        )
        return model

    def get_callbacks(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return [
            EarlyStopping(
                monitor='val_loss',  # Binary loss ana hedef
                mode='min',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                mode='min',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'best_model_{timestamp}.keras',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ]

    def train_model(self):
        if not self.load_data():
            return False

        input_shape = (self.X_train.shape[1], self.X_train.shape[2])  # T, d+n_categories
        self.model = self.create_model(input_shape)
        self.model.summary()

        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=75,
            batch_size=32,
            callbacks=self.get_callbacks(),
            shuffle=True,
            verbose=1
        )
        return True

    def evaluate_model(self):
        if self.model is None:
            print("Önce modeli eğitin!")
            return
        from sklearn.metrics import precision_recall_curve

        y_pred_proba = self.model.predict(self.X_test)
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = f1_scores.argmax()
        best_threshold = thresholds[best_idx]
        print(f"En iyi threshold (F1): {best_threshold:.2f}")

        y_pred = (y_pred_proba > best_threshold).astype("int32")
        print("\n--- Classification Report ---")
        print(classification_report(self.y_test, y_pred, target_names=['Sağlıklı', 'Otizm']))

        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Sağlıklı', 'Otizm'])
        disp.plot(cmap='Blues', values_format='d')
        plt.show()

        # Model sonuçlarını yazdır
        print("\n--- Son Epoch Sonuçları ---")
        print(f"Train Loss: {self.history.history['loss'][-1]:.4f}")
        print(f"Train Accuracy: {self.history.history['accuracy'][-1]:.4f}")
        print(f"Validation Loss: {self.history.history['val_loss'][-1]:.4f}")
        print(f"Validation Accuracy: {self.history.history['val_accuracy'][-1]:.4f}")

    def save_results(self):
        if self.model is None:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model.save(f'autism_detection_model_{timestamp}.keras')
        np.save(f'training_history_{timestamp}.npy', self.history.history)
        model_info = {
            'input_shape': (self.X_train.shape[1], self.X_train.shape[2]),
            'classes': ['Sağlıklı', 'Otizm'],
            'test_accuracy': self.history.history['val_accuracy'][-1],
            'test_loss': self.history.history['val_loss'][-1],
            'training_date': timestamp
        }
        joblib.dump(model_info, f'model_info_{timestamp}.pkl')
        print(f"Model ve eğitim sonuçları kaydedildi.")

def main():
    trainer = AutismDetectionTrainer()
    success = trainer.train_model()
    if success:
        trainer.evaluate_model()
        trainer.save_results()

if __name__ == "__main__":
    main()
