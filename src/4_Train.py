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
    def __init__(self, X_path="X_features.npy", y_path="y.npy", 
                 csv_path=r"C:\Users\Casper\Desktop\newobs\data\final_balanced_clean_dataset.csv"):
        """
        Otizm tespiti modeli eğitici sınıfı
        """
        self.X_path = X_path
        self.y_path = y_path
        self.csv_path = csv_path
        self.model = None
        self.history = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.video_ids = None
        self.video_to_segment_map = None
        
    def load_data(self):
        """
        Verileri yükler ve hazırlar
        """
        print("Veriler yükleniyor...")
        try:
            X = np.load(self.X_path)
            y = np.load(self.y_path)
            print(f"Veri şekli: X={X.shape}, y={y.shape}")
            
            # NaN değerleri kontrol et ve temizle
            if np.isnan(X).any():
                print("NaN değerler tespit edildi, temizleniyor...")
                X = np.nan_to_num(X)
            
            # Video ID'leri yükle ve veri sızıntısı kontrolü yap
            self.load_video_info()
            self.check_data_leakage()
            
            return X, y
            
        except FileNotFoundError:
            print("Hata: X_features.npy veya y.npy dosyaları bulunamadı!")
            print("Önce feature_extraction.py dosyasını çalıştırmalısınız.")
            return None, None
    
    def load_video_info(self):
        """
        CSV'den video ID'leri yükler ve segmentlerle eşleştirir
        """
        print("Video bilgileri yükleniyor...")
        self.video_ids = []
        self.video_to_segment_map = {}
        
        try:
            # Önce segment dosya isimlerini al
            segment_files = []
            autism_dir = r"C:\Users\Casper\Desktop\newobs\data\download_videos\autism"
            healthy_dir = r"C:\Users\Casper\Desktop\newobs\data\download_videos\healthy"
            
            for label_dir in [autism_dir, healthy_dir]:
                if os.path.exists(label_dir):
                    for file in os.listdir(label_dir):
                        if file.endswith('.mp4'):
                            video_id = file.split('.')[0]
                            segment_files.append(video_id)
            
            # CSV'den video bilgilerini oku
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    video_id = row['video_id'].strip()
                    self.video_ids.append(video_id)
                    
                    # Eğer bu video_id segmentlerde varsa, mapping yap
                    if video_id in segment_files:
                        self.video_to_segment_map[i] = video_id
            
            print(f"Toplam {len(self.video_ids)} video bulundu")
            print(f"Toplam {len(self.video_to_segment_map)} segment-video eşleşmesi")
            
        except Exception as e:
            print(f"Video bilgileri yüklenirken hata: {e}")
    
    def check_data_leakage(self):
        """
        Veri sızıntısı kontrolü yapar
        """
        print("\n" + "="*50)
        print("VERI SIZINTISI KONTROLÜ")
        print("="*50)
        
        if not self.video_to_segment_map:
            print("Video-segment eşleşmesi bulunamadı")
            return
        
        # Benzersiz video sayısı
        unique_videos = len(set(self.video_to_segment_map.values()))
        total_segments = len(self.video_to_segment_map)
        
        print(f"Benzersiz videolar: {unique_videos}")
        print(f"Toplam segmentler: {total_segments}")
        print(f"Ortalama segment/video: {total_segments/unique_videos:.2f}")
        
        # Video başına segment dağılımı
        video_segment_count = {}
        for video_id in self.video_to_segment_map.values():
            video_segment_count[video_id] = video_segment_count.get(video_id, 0) + 1
        
        print(f"\nSegment dağılımı:")
        print(f"Max segment/video: {max(video_segment_count.values())}")
        print(f"Min segment/video: {min(video_segment_count.values())}")
        print(f"Ortalama segment/video: {np.mean(list(video_segment_count.values())):.2f}")
        
        # Çok segmentli videoları kontrol et
        multi_segment_videos = {k: v for k, v in video_segment_count.items() if v > 1}
        print(f"Çok segmentli videolar: {len(multi_segment_videos)}")
        
        if multi_segment_videos:
            print("İlk 5 çok segmentli video:")
            for i, (video_id, count) in enumerate(list(multi_segment_videos.items())[:5]):
                print(f"  {video_id}: {count} segment")
    
    def create_independent_test_split(self, test_size=0.2):
        print("\nVideo seviyesinde bağımsız test seti oluşturuluyor...")
        
        # Benzersiz videoları al
        unique_videos = list(set(self.video_to_segment_map.values()))
        
        # Videoları rastgele ama stratify şekilde böl
        from sklearn.model_selection import train_test_split
        
        # Her video için etiket bul
        video_labels = {}
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row['video_id'].strip()
                label = 1 if row['label'].strip().lower() == 'autism' else 0
                video_labels[video_id] = label
        
        # Sadece segmenti olan videoları filtrele
        available_videos = [v for v in unique_videos if v in video_labels]
        available_labels = [video_labels[v] for v in available_videos]
        
        # Videoları böl
        train_videos, test_videos = train_test_split(
            available_videos, 
            test_size=test_size, 
            random_state=42, 
            stratify=available_labels
        )
        
        # Segment indekslerini ayır
        train_indices = []
        test_indices = []
        
        for idx, video_id in self.video_to_segment_map.items():
            if video_id in train_videos:
                train_indices.append(idx)
            elif video_id in test_videos:
                test_indices.append(idx)
        
        return train_indices, test_indices
    
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
    
    def train_model(self, model_type='dense', test_size=0.2, independent_split=True):
        """
        Modeli eğitir
        """
        # Verileri yükle
        X, y = self.load_data()
        if X is None:
            return False
        
        # Veriyi böl
        print("Veri train-test olarak ayrılıyor...")
        
        if independent_split:
            # Video seviyesinde bağımsız bölme
            train_indices, test_indices = self.create_independent_test_split(test_size)
            
            if train_indices and test_indices:
                self.X_train, self.X_test = X[train_indices], X[test_indices]
                self.y_train, self.y_test = y[train_indices], y[test_indices]
                print("✓ Video seviyesinde bağımsız bölme yapıldı")
            else:
                print("✗ Bağımsız bölme yapılamadı, normal bölme kullanılıyor")
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y, shuffle=True
                )
        else:
            # Normal bölme
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y, shuffle=True
            )
        
        print(f"Eğitim verisi: {self.X_train.shape}")
        print(f"Test verisi: {self.X_test.shape}")
        print(f"Sınıf dağılımı - Eğitim: {np.bincount(self.y_train.astype(int))}")
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
            validation_data=(self.X_test, self.y_test),
            epochs=100,
            batch_size=32,
            callbacks=self.get_callbacks(),
            verbose=1,
            shuffle=True
        )
        
        return True

    def cross_validate(self, model_type='dense', n_splits=5):
        """
        Çapraz doğrulama yapar
        """
        X, y = self.load_data()
        if X is None:
            return
        
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        fold = 1
        
        for train_idx, val_idx in kfold.split(X, y):
            print(f"\nFold {fold}/{n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = self.create_model((X.shape[1], X.shape[2]), model_type)
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                verbose=0,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
            )
            
            score = model.evaluate(X_val, y_val, verbose=0)
            cv_scores.append(score[1])  # accuracy
            
            print(f"Fold {fold} Doğruluk: {score[1]:.4f}")
            fold += 1
        
        print(f"\nÇapraz Doğrulama Sonuçları: {cv_scores}")
        print(f"Ortalama Doğruluk: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
        
        return cv_scores

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
    
    # Otomatik olarak 5-fold çapraz doğrulama yap (Dense model ile)
    print("5-fold çapraz doğrulama yapılıyor (Dense model)...")
    cv_scores = trainer.cross_validate(model_type='dense', n_splits=5)
    
    # Otomatik olarak Dense model ile eğitim yap
    print("\nDense model ile eğitime başlanıyor...")
    success = trainer.train_model(model_type='dense', independent_split=True)
    
    if success:
        # Değerlendirme
        trainer.evaluate_model()
        
        # Kaydetme
        trainer.save_results()

if __name__ == "__main__":
    main()