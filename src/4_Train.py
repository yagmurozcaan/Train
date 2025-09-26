import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib
import pandas as pd
from datetime import datetime
import csv

class AutismDetectionTrainer:
    def __init__(self, features_dir="data/features", 
                 csv_path=r"C:\Users\Casper\Desktop\newobs\data\final_balanced_clean_dataset.csv"):
        """
        Otizm tespiti modeli eğitici sınıfı
        """
        self.features_dir = features_dir
        self.csv_path = csv_path
        self.model = None
        self.history = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.X_val, self.y_val = None, None
        
    def load_data(self):
        """
        Önceden bölünmüş feature dosyalarını yükler
        """
        print("Önceden bölünmüş feature dosyaları yükleniyor...")
        try:
            # Train/Val/Test dosyalarını yükle
            self.X_train = np.load(os.path.join(self.features_dir, "X_train.npy"))
            self.y_train = np.load(os.path.join(self.features_dir, "y_train.npy"))
            self.X_val = np.load(os.path.join(self.features_dir, "X_val.npy"))
            self.y_val = np.load(os.path.join(self.features_dir, "y_val.npy"))
            self.X_test = np.load(os.path.join(self.features_dir, "X_test.npy"))
            self.y_test = np.load(os.path.join(self.features_dir, "y_test.npy"))
            
            # Video splitting bilgilerini yükle
            self.train_videos = np.load(os.path.join(self.features_dir, "train_videos.npy"), allow_pickle=True)
            self.val_videos = np.load(os.path.join(self.features_dir, "val_videos.npy"), allow_pickle=True)
            self.test_videos = np.load(os.path.join(self.features_dir, "test_videos.npy"), allow_pickle=True)
            
            print(f"Train: {self.X_train.shape}, {self.y_train.shape}")
            print(f"Validation: {self.X_val.shape}, {self.y_val.shape}")
            print(f"Test: {self.X_test.shape}, {self.y_test.shape}")
            print(f"Train videoları: {len(self.train_videos)}")
            print(f"Validation videoları: {len(self.val_videos)}")
            print(f"Test videoları: {len(self.test_videos)}")
            
            # NaN değerleri kontrol et ve temizle
            for dataset_name, dataset in [("X_train", self.X_train), ("X_val", self.X_val), ("X_test", self.X_test)]:
                if np.isnan(dataset).any():
                    print(f"NaN değerler tespit edildi ({dataset_name}), temizleniyor...")
                    if dataset_name == "X_train":
                        self.X_train = np.nan_to_num(self.X_train)
                    elif dataset_name == "X_val":
                        self.X_val = np.nan_to_num(self.X_val)
                    elif dataset_name == "X_test":
                        self.X_test = np.nan_to_num(self.X_test)
            
            # Veri sızıntısı kontrolü
            self.check_data_leakage()
            
            return True
            
        except FileNotFoundError as e:
            print(f"Hata: Feature dosyaları bulunamadı! {e}")
            print("Önce 3_Feature_Extraction.py dosyasını çalıştırmalısınız.")
            return False
    
    
    def check_data_leakage(self):
        """
        Veri sızıntısı kontrolü yapar
        """
        print("\n" + "="*50)
        print("VERI SIZINTISI KONTROLÜ")
        print("="*50)
        
        # Video setlerinin kesişimini kontrol et
        train_set = set(self.train_videos)
        val_set = set(self.val_videos)
        test_set = set(self.test_videos)
        
        # Kesişimleri kontrol et
        train_val_intersection = train_set.intersection(val_set)
        train_test_intersection = train_set.intersection(test_set)
        val_test_intersection = val_set.intersection(test_set)
        
        print(f"Train videoları: {len(train_set)}")
        print(f"Validation videoları: {len(val_set)}")
        print(f"Test videoları: {len(test_set)}")
        
        if train_val_intersection:
            print(f"⚠️  UYARI: Train-Val kesişimi: {len(train_val_intersection)} video")
            print(f"Kesişen videolar: {list(train_val_intersection)[:5]}")
        else:
            print("✓ Train-Val kesişimi yok")
            
        if train_test_intersection:
            print(f"⚠️  UYARI: Train-Test kesişimi: {len(train_test_intersection)} video")
            print(f"Kesişen videolar: {list(train_test_intersection)[:5]}")
        else:
            print("✓ Train-Test kesişimi yok")
            
        if val_test_intersection:
            print(f"⚠️  UYARI: Val-Test kesişimi: {len(val_test_intersection)} video")
            print(f"Kesişen videolar: {list(val_test_intersection)[:5]}")
        else:
            print("✓ Val-Test kesişimi yok")
        
        # Toplam benzersiz video sayısı
        all_videos = train_set.union(val_set).union(test_set)
        print(f"\nToplam benzersiz video: {len(all_videos)}")
        
        if not (train_val_intersection or train_test_intersection or val_test_intersection):
            print("🎉 VERİ SIZINTISI YOK! Video seviyesinde tamamen bağımsız bölme yapıldı.")
        else:
            print("❌ VERİ SIZINTISI VAR! Aynı videolar farklı setlerde bulunuyor.")
    
    
    def create_model(self, input_shape, model_type='cnn'):
        """
        Farklı model mimarileri oluşturur
        """
        # CNN için input shape'i düzelt (zaman serisi çok kısa olduğu için)
        if model_type == 'cnn':
            # Zaman serisi boyutu 3 olduğu için daha basit bir CNN
            model = Sequential()
            model.add(Input(shape=input_shape))
            model.add(Conv1D(64, kernel_size=1, activation='relu'))  # kernel_size=1 kullanıyoruz
            model.add(GlobalAveragePooling1D())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(1, activation='sigmoid'))
            
        elif model_type == 'lstm':
            model = Sequential()
            model.add(Input(shape=input_shape))
            model.add(Masking(mask_value=0.))
            model.add(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
            model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
            
        elif model_type == 'dense':
            # Basit Dense model - zaman serisi boyutu küçük olduğu için
            model = Sequential()
            model.add(Input(shape=input_shape))
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            
        elif model_type == 'simple':
            # Çok basit model
            model = Sequential()
            model.add(Input(shape=input_shape))
            model.add(GlobalAveragePooling1D())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        return model
    
    def get_callbacks(self):
        """
        Eğitim callback'lerini döndürür
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'best_model_{timestamp}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
    
    def train_model(self, model_type='dense'):
        """
        Modeli eğitir - önceden bölünmüş verilerle
        """
        # Verileri yükle
        if not self.load_data():
            return False
        
        print(f"Sınıf dağılımı - Eğitim: {np.bincount(self.y_train.astype(int))}")
        print(f"Sınıf dağılımı - Validation: {np.bincount(self.y_val.astype(int))}")
        print(f"Sınıf dağılımı - Test: {np.bincount(self.y_test.astype(int))}")
        
        # Modeli oluştur
        print(f"{model_type.upper()} modeli oluşturuluyor...")
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        self.model = self.create_model(input_shape, model_type)
        self.model.summary()
        
        # Modeli eğit
        print("Model eğitiliyor...")
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=100,
            batch_size=32,
            callbacks=self.get_callbacks(),
            verbose=1,
            shuffle=True
        )
        
        return True

    def cross_validate(self, model_type='dense', n_splits=5):
        """
        Çapraz doğrulama yapar - video seviyesinde
        """
        if not self.load_data():
            return
        
        print("⚠️  Video seviyesinde çapraz doğrulama yapılamaz çünkü veriler zaten video seviyesinde bölünmüş.")
        print("Bunun yerine normal eğitim yapılacak.")
        return None

    def evaluate_model(self):
        """
        Modeli değerlendirir
        """
        if self.model is None:
            print("Önce modeli eğitin!")
            return
        
        print("Model değerlendiriliyor...")
        y_pred_proba = self.model.predict(self.X_test)
        y_pred = (y_pred_proba > 0.5).astype("int32")
        
        # Metrikler
        print("=" * 60)
        print("MODEL DEĞERLENDİRME SONUÇLARI")
        print("=" * 60)
        print(classification_report(self.y_test, y_pred, target_names=['Sağlıklı', 'Otizm']))
        
        # Karışıklık matrisi
        cm = confusion_matrix(self.y_test, y_pred)
        print("Karışıklık Matrisi:")
        print(cm)
        
        # Görselleştirme
        self.plot_confusion_matrix(cm)
        self.plot_training_history()
        
        return y_pred, y_pred_proba

    def plot_training_history(self):
        """
        Eğitim geçmişini görselleştirir
        """
        if self.history is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = [
            ('accuracy', 'Doğruluk'),
            ('loss', 'Kayıp'),
            ('precision', 'Precision'),
            ('recall', 'Recall')
        ]
        
        for i, (metric, title) in enumerate(metrics):
            row, col = i // 2, i % 2
            if metric in self.history.history:
                axes[row, col].plot(self.history.history[metric], label=f'Eğitim {title}')
                axes[row, col].plot(self.history.history[f'val_{metric}'], label=f'Doğrulama {title}')
                axes[row, col].set_title(f'Model {title}')
                axes[row, col].set_ylabel(title)
                axes[row, col].set_xlabel('Epok')
                axes[row, col].legend()
                axes[row, col].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix(self, cm):
        """
        Karışıklık matrisini görselleştirir
        """
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Sağlıklı', 'Otizm'])
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Karışıklık Matrisi')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_results(self):
        """
        Sonuçları kaydeder
        """
        if self.model is None:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Modeli kaydet
        model_path = f'autism_detection_model_{timestamp}.keras'
        self.model.save(model_path)
        
        # Eğitim geçmişi
        np.save(f'training_history_{timestamp}.npy', self.history.history)
        
        # Model bilgileri
        model_info = {
            'input_shape': (self.X_train.shape[1], self.X_train.shape[2]),
            'classes': ['Sağlıklı', 'Otizm'],
            'test_accuracy': self.history.history['val_accuracy'][-1],
            'test_loss': self.history.history['val_loss'][-1],
            'training_date': timestamp,
            'data_shape': {
                'X_train': self.X_train.shape,
                'X_test': self.X_test.shape,
                'y_train': self.y_train.shape,
                'y_test': self.y_test.shape
            }
        }
        
        joblib.dump(model_info, f'model_info_{timestamp}.pkl')
        
        # Sonuç raporu
        final_accuracy = self.history.history['val_accuracy'][-1]
        print("=" * 60)
        print("EĞİTİM TAMAMLANDI!")
        print("=" * 60)
        print(f"Son test doğruluğu: {final_accuracy:.4f}")
        print(f"Model kaydedildi: {model_path}")
        print(f"Eğitim geçmişi: training_history_{timestamp}.npy")
        print(f"Model bilgileri: model_info_{timestamp}.pkl")

def main():
    """
    Ana eğitim fonksiyonu
    """
    print("Otizm Tespiti Model Eğitimine Başlanıyor...")
    print("=" * 50)
    
    # Eğitici oluştur
    trainer = AutismDetectionTrainer()
    
    # Otomatik olarak Dense model ile eğitim yap
    print("Dense model ile eğitime başlanıyor...")
    success = trainer.train_model(model_type='dense')
    
    if success:
        # Değerlendirme
        trainer.evaluate_model()
        
        # Kaydetme
        trainer.save_results()

if __name__ == "__main__":
    main()