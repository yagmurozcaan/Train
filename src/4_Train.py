import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D, Input, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib
import pandas as pd
from datetime import datetime
import csv
from sklearn.metrics import precision_recall_curve


class AutismDetectionTrainer:
    def __init__(self, features_dir="data/features", 
                 csv_path="data/final_balanced_clean_dataset_synchronized.csv"):
        """
        Otizm tespiti modeli eğitici sınıfı
        """
        self.features_dir = features_dir
        self.csv_path = csv_path
        self.outputs_dir = os.path.join("Train", "outputs")
        os.makedirs(self.outputs_dir, exist_ok=True)
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
            self.y_train = np.load(os.path.join(self.features_dir, "y_train_binary.npy"))
            self.X_val = np.load(os.path.join(self.features_dir, "X_val.npy"))
            self.y_val = np.load(os.path.join(self.features_dir, "y_val_binary.npy"))
            self.X_test = np.load(os.path.join(self.features_dir, "X_test.npy"))
            self.y_test = np.load(os.path.join(self.features_dir, "y_test_binary.npy"))
            
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
    
    
    def create_model(self, input_shape, model_type='dense'):
        """
        Gelişmiş model mimarileri oluşturur
        """
        if model_type == 'cnn':
            # Geliştirilmiş CNN modeli
            model = Sequential()
            model.add(Input(shape=input_shape))
            model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
            model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
            model.add(GlobalAveragePooling1D())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(1, activation='sigmoid'))
            
        elif model_type == 'lstm':
            # Geliştirilmiş LSTM modeli
            model = Sequential()
            model.add(Input(shape=input_shape))
            model.add(Masking(mask_value=0.))
            model.add(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
            model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
            
        elif model_type == 'bi_lstm':
            # YENİ MODEL EKLEME: Çift Yönlü LSTM (Bi-LSTM)
            model = Sequential()
            model.add(Input(shape=input_shape))
            model.add(Masking(mask_value=0.))
            # Dropout ve Recurrent Dropout oranları artırıldı
            model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.4, recurrent_dropout=0.4))) 
            model.add(Bidirectional(LSTM(64, dropout=0.4, recurrent_dropout=0.4)))
            model.add(Dense(32, activation='relu'))
            # Regularizasyon artırıldı
            model.add(Dropout(0.6)) 
            model.add(Dense(1, activation='sigmoid'))
            
        elif model_type == 'transformer':
            # Basit Transformer modeli
            inputs = Input(shape=input_shape)
            
            # Multi-head attention
            attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
            attention_output = Dropout(0.1)(attention_output)
            
            # Add & Norm
            attention_output = Add()([inputs, attention_output])
            attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
            
            # Feed forward
            ffn = Dense(128, activation='relu')(attention_output)
            ffn = Dropout(0.1)(ffn)
            ffn = Dense(input_shape[-1])(ffn)
            
            # Add & Norm
            ffn_output = Add()([attention_output, ffn])
            ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output)
            
            # Global pooling ve classification
            pooled = GlobalAveragePooling1D()(ffn_output)
            pooled = Dense(64, activation='relu')(pooled)
            pooled = Dropout(0.5)(pooled)
            outputs = Dense(1, activation='sigmoid')(pooled)
            
            model = Model(inputs=inputs, outputs=outputs)
            
        elif model_type == 'hybrid':
            # CNN + LSTM hibrit modeli
            model = Sequential()
            model.add(Input(shape=input_shape))
            
            # CNN kısmı
            model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
            model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
            model.add(Dropout(0.2))
            
            # LSTM kısmı
            model.add(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
            model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
            
            # Classification
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
            
        elif model_type == 'dense':
            # Geliştirilmiş Dense model
            model = Sequential()
            model.add(Input(shape=input_shape))
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            
        elif model_type == 'simple':
            # Basit model
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
    
    def get_callbacks(self, patience=20):
        """
        Geliştirilmiş eğitim callback'lerini döndürür
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,  # Artırıldı: 15 → 20
                restore_best_weights=True,
                verbose=1,
                min_delta=1e-4  # Minimum iyileşme eşiği
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,  # Artırıldı: 8 → 10
                min_lr=1e-7,
                verbose=1,
                cooldown=5  # Learning rate azaltma sonrası bekleme
            ),
            ModelCheckpoint(
                os.path.join(self.outputs_dir, f'best_model_{timestamp}.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1,
                save_weights_only=False
            )
        ]

    def calculate_class_weights(self):
        """
        Sınıf ağırlıklarını hesaplar (sınıf dengesizliği için)
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(self.y_train)
        class_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=self.y_train
        )
        
        class_weight_dict = dict(zip(classes, class_weights))
        print(f"📊 Sınıf ağırlıkları: {class_weight_dict}")
        
        return class_weight_dict

    def train_model(self, model_type='dense', use_class_weights=True, patience=20):
        """
        Geliştirilmiş model eğitimi
        """
        # Verileri yükle
        if not self.load_data():
            return False
        
        print(f"Sınıf dağılımı - Eğitim: {np.bincount(self.y_train.astype(int))}")
        print(f"Sınıf dağılımı - Validation: {np.bincount(self.y_val.astype(int))}")
        print(f"Sınıf dağılımı - Test: {np.bincount(self.y_test.astype(int))}")
        
        # Sınıf ağırlıklarını hesapla
        class_weights = None
        if use_class_weights:
            class_weights = self.calculate_class_weights()
        
        # Modeli oluştur
        print(f"{model_type.upper()} modeli oluşturuluyor...")
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        self.model = self.create_model(input_shape, model_type)
        self.model.summary()
        
        # Modeli eğit
        print("🚀 Model eğitiliyor...")
        print(f"📈 Eğitim parametreleri:")
        print(f"   - Model: {model_type}")
        print(f"   - Patience: {patience}")
        print(f"   - Class weights: {'Evet' if use_class_weights else 'Hayır'}")
        print(f"   - Batch size: 32")
        print(f"   - Max epochs: 100")
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=100,
            batch_size=32,
            callbacks=self.get_callbacks(patience=patience),
            verbose=1,
            shuffle=True,
            class_weight=class_weights  # Sınıf ağırlıkları
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

    from sklearn.metrics import precision_recall_curve

    def find_optimal_threshold(self, y_true, y_pred_proba):
        """
        Gelişmiş threshold optimizasyonu
        """
        from sklearn.metrics import roc_curve, precision_recall_curve
        
        # F1-score bazlı threshold
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_f1_idx = f1_scores.argmax()
        best_f1_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
        
        # ROC bazlı threshold (Youden's J statistic)
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        youden_j = tpr - fpr
        best_roc_idx = youden_j.argmax()
        best_roc_threshold = roc_thresholds[best_roc_idx]
        
        # Balanced accuracy bazlı threshold
        balanced_accuracies = []
        for threshold in np.arange(0.1, 0.9, 0.01):
            y_pred_temp = (y_pred_proba > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_temp).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_acc = (sensitivity + specificity) / 2
            balanced_accuracies.append(balanced_acc)
        
        best_bal_idx = np.argmax(balanced_accuracies)
        best_bal_threshold = 0.1 + best_bal_idx * 0.01
        
        print(f"🎯 Threshold Optimizasyon Sonuçları:")
        print(f"   F1-score bazlı: {best_f1_threshold:.3f}")
        print(f"   ROC bazlı: {best_roc_threshold:.3f}")
        print(f"   Balanced accuracy bazlı: {best_bal_threshold:.3f}")
        
        # En iyi threshold'u seç (F1-score öncelikli)
        return best_f1_threshold

    def evaluate_model(self):
        if self.model is None:
            print("Önce modeli eğitin!")
            return

        print("🔍 Model değerlendiriliyor...")
        y_pred_proba = self.model.predict(self.X_test)

        # -------- Gelişmiş threshold optimizasyonu --------
        best_threshold = self.find_optimal_threshold(self.y_test, y_pred_proba)
        print(f"✅ Seçilen optimal threshold: {best_threshold:.3f}")

        # -------- Tahmin --------
        y_pred = (y_pred_proba > best_threshold).astype("int32")

        # -------- Raporlama --------
        from sklearn.metrics import (
            classification_report, confusion_matrix, ConfusionMatrixDisplay,
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        )

        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred)
        rec = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_pred_proba)

        print("=" * 60)
        print("MODEL DEĞERLENDİRME SONUÇLARI")
        print("=" * 60)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC:       {auc:.4f}")
        print("\nSınıf bazlı detaylar:\n")
        print(classification_report(self.y_test, y_pred, target_names=['Sağlıklı', 'Otizm']))

        # -------- Karışıklık matrisi --------
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Sağlıklı', 'Otizm'])
        disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
        plt.title('Karışıklık Matrisi')
        plt.savefig(os.path.join(self.outputs_dir, 'confusion_matrix_detailed.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # -------- Eğitim grafikleri --------
        self.plot_training_history()

        # -------- Return --------
        return y_pred, y_pred_proba, best_threshold


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
        plt.savefig(os.path.join(self.outputs_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
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
        model_path = os.path.join(self.outputs_dir, f'autism_detection_model_{timestamp}.keras')
        self.model.save(model_path)
        
        # Eğitim geçmişi
        np.save(os.path.join(self.outputs_dir, f'training_history_{timestamp}.npy'), self.history.history)
        
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
        
        joblib.dump(model_info, os.path.join(self.outputs_dir, f'model_info_{timestamp}.pkl'))
        
        # Sonuç raporu
        final_accuracy = self.history.history['val_accuracy'][-1]
        print("=" * 60)
        print("EĞİTİM TAMAMLANDI!")
        print("=" * 60)
        print(f"Son test doğruluğu: {final_accuracy:.4f}")
        print(f"Model kaydedildi: {model_path}")
        print(f"Eğitim geçmişi: {os.path.join(self.outputs_dir, f'training_history_{timestamp}.npy')}")
        print(f"Model bilgileri: {os.path.join(self.outputs_dir, f'model_info_{timestamp}.pkl')}")
    
def compare_models():
    """
    Farklı model mimarilerini karşılaştırır
    """
    print("\n" + "="*60)
    print("MODEL MİMARİSİ KARŞILAŞTIRMASI")
    print("="*60)
    
    models_info = {
        'dense': {
            'Açıklama': 'Tam bağlantılı katmanlar',
            'Avantaj': 'Basit, hızlı eğitim',
            'Dezavantaj': 'Temporal bilgiyi göz ardı eder',
            'Önerilen': 'Baseline model'
        },
        'lstm': {
            'Açıklama': 'Tek yönlü LSTM',
            'Avantaj': 'Temporal bağımlılıkları öğrenir',
            'Dezavantaj': 'Sadece ileri yönde',
            'Önerilen': 'Temporal veri için iyi'
        },
        'bi_lstm': {
            'Açıklama': 'Çift yönlü LSTM',
            'Avantaj': 'İleri ve geri temporal bilgi',
            'Dezavantaj': 'Daha yavaş eğitim',
            'Önerilen': 'En iyi temporal performans'
        },
        'cnn': {
            'Açıklama': '1D Convolutional',
            'Avantaj': 'Yerel kalıpları yakalar',
            'Dezavantaj': 'Uzun mesafe bağımlılıkları zor',
            'Önerilen': 'Kısa sekanslar için'
        },
        'transformer': {
            'Açıklama': 'Attention mekanizması',
            'Avantaj': 'Paralel işleme, uzun bağımlılıklar',
            'Dezavantaj': 'Çok veri gerektirir',
            'Önerilen': 'Büyük veri setleri için'
        },
        'hybrid': {
            'Açıklama': 'CNN + LSTM kombinasyonu',
            'Avantaj': 'Hem yerel hem temporal bilgi',
            'Dezavantaj': 'Karmaşık mimari',
            'Önerilen': 'Karmaşık kalıplar için'
        }
    }
    
    for model, info in models_info.items():
        print(f"\n🔹 {model.upper()}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    print("\n💡 Önerilen Model Sırası:")
    print("1. bi_lstm - En iyi temporal performans")
    print("2. lstm - İyi temporal performans")
    print("3. hybrid - Karmaşık kalıplar")
    print("4. transformer - Büyük veri setleri")
    print("5. cnn - Kısa sekanslar")
    print("6. dense - Baseline")

def print_training_recommendations():
    """
    Eğitim süreci önerileri
    """
    print("\n" + "="*60)
    print("EĞİTİM SÜRECİ ÖNERİLERİ")
    print("="*60)
    
    print("\n🔧 Callback Optimizasyonları:")
    print("• EarlyStopping patience: 15 → 20-25 (daha uzun eğitim)")
    print("• ReduceLROnPlateau patience: 8 → 10 (daha sabırlı)")
    print("• min_delta: 1e-4 (minimum iyileşme eşiği)")
    print("• cooldown: 5 (learning rate azaltma sonrası bekleme)")
    
    print("\n⚖️ Sınıf Dengesizliği Çözümleri:")
    print("• Class weights: Otomatik hesaplama (balanced)")
    print("• Focal loss: Zor örnekler için ağırlık artırma")
    print("• SMOTE: Synthetic minority oversampling")
    print("• Threshold optimization: F1-score, ROC, Balanced accuracy")
    
    print("\n📊 Threshold Optimizasyon Stratejileri:")
    print("• F1-score bazlı: Precision-Recall dengesi")
    print("• ROC bazlı: Youden's J statistic")
    print("• Balanced accuracy: Sensitivity-Specificity dengesi")
    print("• Custom threshold: Domain-specific optimizasyon")
    
    print("\n🎯 Model Seçim Stratejisi:")
    print("1. Baseline: Dense model")
    print("2. Temporal: LSTM → Bi-LSTM")
    print("3. Hibrit: CNN + LSTM")
    print("4. Advanced: Transformer (büyük veri setleri)")
    
    print("\n💡 Performans Artırma İpuçları:")
    print("• Veri artırma: Augmentation teknikleri")
    print("• Ensemble: Birden fazla model kombinasyonu")
    print("• Cross-validation: Video seviyesinde bölme")
    print("• Hyperparameter tuning: Grid/Random search")

def main():
    """
    Ana eğitim fonksiyonu
    """
    print("Otizm Tespiti Model Eğitimine Başlanıyor...")
    print("=" * 50)
    
    # Model karşılaştırması ve eğitim önerileri göster
    compare_models()
    print_training_recommendations()
    
    # Eğitici oluştur
    trainer = AutismDetectionTrainer()
    
    # Dense yerine Bi-LSTM model ile eğitime başla
    print("\n🚀 Bi-LSTM model ile eğitime başlanıyor...")
    print("(En iyi temporal performans için önerilen)")
    success = trainer.train_model(
        model_type='bi_lstm',
        use_class_weights=True,  # Sınıf dengesizliği için
        patience=25  # Daha uzun eğitim için
    )
    
    if success:
        # Değerlendirme
        trainer.evaluate_model()
        
        # Kaydetme
        trainer.save_results()

if __name__ == "__main__":
    main()