import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split

# --- EfficientNet Model Seçimi ve Fine-tuning Stratejisi ---
def create_feature_extractor(model_name='EfficientNetB0', fine_tune_layers=20):
    """
    Farklı EfficientNet modelleri ve fine-tuning stratejileri
    
    Args:
        model_name: 'EfficientNetB0', 'EfficientNetB3', 'EfficientNetB5', 'ResNet50', 'InceptionV3'
        fine_tune_layers: Fine-tune edilecek katman sayısı
    """
    if model_name == 'EfficientNetB0':
        from tensorflow.keras.applications import EfficientNetB0
        base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    elif model_name == 'EfficientNetB3':
        from tensorflow.keras.applications import EfficientNetB3
        base_model = EfficientNetB3(weights='imagenet', include_top=False, pooling='avg')
    elif model_name == 'EfficientNetB5':
        from tensorflow.keras.applications import EfficientNetB5
        base_model = EfficientNetB5(weights='imagenet', include_top=False, pooling='avg')
    elif model_name == 'ResNet50':
        from tensorflow.keras.applications import ResNet50
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    elif model_name == 'InceptionV3':
        from tensorflow.keras.applications import InceptionV3
        base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    else:
        raise ValueError(f"Desteklenmeyen model: {model_name}")
    
    # Tüm katmanları dondur
    for layer in base_model.layers:
        layer.trainable = False
    
    # Belirtilen sayıda katmanı fine-tune için aç
    if fine_tune_layers > 0:
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True
    
    print(f"✓ {model_name} katmanları fine-tune için ayarlandı (son {fine_tune_layers} katman)")
    return base_model

# Model seçimi - deneysel olarak değiştirilebilir
SELECTED_MODEL = 'EfficientNetB0'  # EfficientNetB3, EfficientNetB5, ResNet50, InceptionV3
FINE_TUNE_LAYERS = 20  # 10, 20, 30 veya 0 (tüm katmanları dondur)

base_model = create_feature_extractor(SELECTED_MODEL, FINE_TUNE_LAYERS)

# --- Feature extraction ---
def extract_features(frames, model=None):
    """
    Geliştirilmiş feature extraction
    """
    if model is None:
        model = base_model
    
    features = []
    for frame in frames:
        img = image.img_to_array(frame)
        img = np.expand_dims(img, axis=0)
        
        # Model-specific preprocessing
        if 'EfficientNet' in SELECTED_MODEL:
            img = preprocess_input(img)
        elif SELECTED_MODEL == 'ResNet50':
            from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
            img = resnet_preprocess(img)
        elif SELECTED_MODEL == 'InceptionV3':
            from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
            img = inception_preprocess(img)
        
        feat = model.predict(img, verbose=0)
        features.append(feat.flatten())
    return np.array(features)  # (T, d)

# --- CSV'deki category feature'ını ekleme ---
def add_category_features(X_features, segment_video_map, csv_path):
    df = pd.read_csv(csv_path)
    categories = sorted(df['category'].unique())
    cat_to_index = {cat: i for i, cat in enumerate(categories)}
    
    extra_features_list = []
    for seg in segment_video_map:
        idx = seg['csv_index']
        cat = df.loc[idx, 'category']
        cat_feat = np.zeros(len(categories))
        cat_feat[cat_to_index[cat]] = 1
        extra_features_list.append(cat_feat)
    
    extra_features_array = np.array(extra_features_list)  # (N, n_categories)
    N, T, d = X_features.shape
    extra_features_time = np.repeat(extra_features_array[:, np.newaxis, :], T, axis=1)  # (N, T, n_categories)
    
    X_augmented = np.concatenate([X_features, extra_features_time], axis=2)
    return X_augmented

# --- Feature dataset oluşturma ---
def build_feature_dataset(
    X_path="data/segments/X.npy",
    y_binary_path="data/segments/y_binary.npy",
    y_category_path="data/segments/y_category.npy",
    map_path="data/segments/segment_video_map.npy",
    csv_path="data/final_balanced_clean_dataset_synchronized.csv",
    out_dir="data/features"
):
    X = np.load(X_path)
    y_binary = np.load(y_binary_path)
    y_category = np.load(y_category_path)
    segment_video_map = np.load(map_path, allow_pickle=True)

    N, T, H, W, C = X.shape
    feature_list = []

    # Feature çıkar
    for i in range(N):
        feats = extract_features(X[i])
        feature_list.append(feats)
        if i % 100 == 0:
            print(f"{i}/{N} segment işlendi...")

    X_features = np.array(feature_list)  # (N, T, d)

    # CSV category feature ekle
    X_features = add_category_features(X_features, segment_video_map, csv_path)
    print("Feature dataset şekli (EffNet + category feature):", X_features.shape)

    # Video bazlı split
    video_ids = [seg['video_id'] for seg in segment_video_map]
    unique_videos = list(set(video_ids))
    video_labels = {vid: y_binary[i] for i, vid in enumerate(video_ids)}

    trainval_videos, test_videos = train_test_split(
        unique_videos, test_size=0.15, random_state=42,
        stratify=[video_labels[v] for v in unique_videos]
    )

    train_videos, val_videos = train_test_split(
        trainval_videos, test_size=0.1765, random_state=42,
        stratify=[video_labels[v] for v in trainval_videos]
    )

    train_idx = [i for i, vid in enumerate(video_ids) if vid in train_videos]
    val_idx   = [i for i, vid in enumerate(video_ids) if vid in val_videos]
    test_idx  = [i for i, vid in enumerate(video_ids) if vid in test_videos]

    X_train, y_train_bin, y_train_cat = X_features[train_idx], y_binary[train_idx], y_category[train_idx]
    X_val, y_val_bin, y_val_cat       = X_features[val_idx], y_binary[val_idx], y_category[val_idx]
    X_test, y_test_bin, y_test_cat    = X_features[test_idx], y_binary[test_idx], y_category[test_idx]

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train_binary.npy"), y_train_bin)
    np.save(os.path.join(out_dir, "y_train_category.npy"), y_train_cat)
    np.save(os.path.join(out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(out_dir, "y_val_binary.npy"), y_val_bin)
    np.save(os.path.join(out_dir, "y_val_category.npy"), y_val_cat)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_test_binary.npy"), y_test_bin)
    np.save(os.path.join(out_dir, "y_test_category.npy"), y_test_cat)
    np.save(os.path.join(out_dir, "segment_video_map.npy"), segment_video_map)
    
    # EKLENMESİ GEREKEN SATIRLAR:
    np.save(os.path.join(out_dir, "train_videos.npy"), train_videos)
    np.save(os.path.join(out_dir, "val_videos.npy"), val_videos)
    np.save(os.path.join(out_dir, "test_videos.npy"), test_videos)

    print("✓ Feature dataset kaydedildi.")
    print("Train:", X_train.shape, y_train_bin.shape, y_train_cat.shape)
    print("Validation:", X_val.shape, y_val_bin.shape, y_val_cat.shape)
    print("Test:", X_test.shape, y_test_bin.shape, y_test_cat.shape)

# --- Model Karşılaştırma ve Optimizasyon Önerileri ---
def print_model_recommendations():
    """
    Model seçimi için öneriler
    """
    print("\n" + "="*60)
    print("FEATURE EXTRACTION MODEL ÖNERİLERİ")
    print("="*60)
    print("Mevcut seçim:", SELECTED_MODEL)
    print("Fine-tune katmanları:", FINE_TUNE_LAYERS)
    
    print("\n📊 Model Karşılaştırması:")
    models_info = {
        'EfficientNetB0': {'Boyut': 'Küçük', 'Hız': 'Hızlı', 'Doğruluk': 'İyi', 'Bellek': 'Düşük'},
        'EfficientNetB3': {'Boyut': 'Orta', 'Hız': 'Orta', 'Doğruluk': 'Çok İyi', 'Bellek': 'Orta'},
        'EfficientNetB5': {'Boyut': 'Büyük', 'Hız': 'Yavaş', 'Doğruluk': 'Mükemmel', 'Bellek': 'Yüksek'},
        'ResNet50': {'Boyut': 'Orta', 'Hız': 'Orta', 'Doğruluk': 'İyi', 'Bellek': 'Orta'},
        'InceptionV3': {'Boyut': 'Orta', 'Hız': 'Orta', 'Doğruluk': 'İyi', 'Bellek': 'Orta'}
    }
    
    for model, info in models_info.items():
        print(f"\n{model}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    print("\n🔧 Fine-tuning Stratejileri:")
    print("• 0 katman: Sadece feature extraction (hızlı, düşük performans)")
    print("• 10 katman: Hafif fine-tuning (dengeli)")
    print("• 20 katman: Orta fine-tuning (mevcut seçim)")
    print("• 30+ katman: Ağır fine-tuning (yavaş, yüksek performans)")
    
    print("\n💡 Öneriler:")
    print("1. Küçük veri seti → EfficientNetB0 + 10-20 katman")
    print("2. Orta veri seti → EfficientNetB3 + 20-30 katman")
    print("3. Büyük veri seti → EfficientNetB5 + 30+ katman")
    print("4. Hız öncelikli → EfficientNetB0 + 0-10 katman")
    print("5. Doğruluk öncelikli → EfficientNetB5 + 30+ katman")

if __name__ == "__main__":
    print_model_recommendations()
    build_feature_dataset()
