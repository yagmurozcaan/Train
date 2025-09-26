import os
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split

# EfficientNetB0 modelini yükle (ImageNet pretrained)
base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

# EfficientNet katmanlarını dondur (freeze)
for layer in base_model.layers:
    layer.trainable = False

print("✓ EfficientNetB0 katmanları donduruldu (frozen)")
print(f"Toplam katman sayısı: {len(base_model.layers)}")
print(f"Trainable parametre sayısı: {sum([layer.count_params() for layer in base_model.layers if layer.trainable])}")

def extract_features(frames):
    """
    frames: (T, H, W, 3) numpy array
    return: (T, d) feature array
    """
    features = []
    for frame in frames:
        img = image.img_to_array(frame)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = base_model.predict(img, verbose=0)
        features.append(feat.flatten())
    return np.array(features)  # shape (T, d)

def build_feature_dataset_tracked(X_path="data/segments/X.npy", y_path="data/segments/y.npy", map_path="data/segments/segment_video_map.npy", out_dir="data/features"):
    # Ham segmentleri ve eşleşmeleri yükle
    X = np.load(X_path)
    y = np.load(y_path)
    segment_video_map = np.load(map_path, allow_pickle=True)

    N, T, H, W, C = X.shape
    feature_list = []

    # Özellik çıkarma
    for i in range(N):
        feats = extract_features(X[i])
        feature_list.append(feats)
        if i % 100 == 0:
            print(f"{i}/{N} segment işlendi...")

    X_features = np.array(feature_list)  # (N, T, d)
    print("Feature dataset şekli:", X_features.shape, y.shape)

    # Video bazlı split yapmak için önce hangi segment hangi videoya ait
    video_ids = [m['video_id'] for m in segment_video_map]
    unique_videos = list(set(video_ids))
    
    # Her video için etiket bilgisini al
    video_labels = {}
    for i, video_id in enumerate(video_ids):
        video_labels[video_id] = y[i]
    
    # Stratified video splitting için
    from sklearn.model_selection import train_test_split
    
    # Önce test set (%20) video bazlı ve stratified
    test_videos, trainval_videos = train_test_split(
        unique_videos, 
        test_size=0.8, 
        random_state=42,
        stratify=[video_labels[v] for v in unique_videos]
    )

    trainval_indices = [i for i, vid in enumerate(video_ids) if vid in trainval_videos]
    test_indices = [i for i, vid in enumerate(video_ids) if vid in test_videos]

    X_trainval, y_trainval = X_features[trainval_indices], y[trainval_indices]
    X_test, y_test = X_features[test_indices], y[test_indices]

    # Validation set (%20 toplam, video bazlı ve stratified)
    train_videos, val_videos = train_test_split(
        trainval_videos,
        test_size=0.25,  # %20 of total
        random_state=42,
        stratify=[video_labels[v] for v in trainval_videos]
    )

    train_indices = [i for i, vid in enumerate(video_ids) if vid in train_videos]
    val_indices = [i for i, vid in enumerate(video_ids) if vid in val_videos]

    X_train, y_train = X_features[train_indices], y[train_indices]
    X_val, y_val = X_features[val_indices], y[val_indices]

    # Çıkış klasörü
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Dosyaları kaydet
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(out_dir, "y_val.npy"), y_val)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)
    np.save(os.path.join(out_dir, "segment_video_map.npy"), segment_video_map)
    
    # Video splitting bilgilerini de kaydet
    np.save(os.path.join(out_dir, "train_videos.npy"), train_videos)
    np.save(os.path.join(out_dir, "val_videos.npy"), val_videos)
    np.save(os.path.join(out_dir, "test_videos.npy"), test_videos)

    print("Train/Validation/Test dosyaları kaydedildi:")
    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)
    print(f"Train videoları: {len(train_videos)}")
    print(f"Validation videoları: {len(val_videos)}")
    print(f"Test videoları: {len(test_videos)}")
    print("✓ Video seviyesinde bağımsız bölme yapıldı - veri sızıntısı yok!")
    print("✓ EfficientNetB0 (FROZEN) ile özellik çıkarımı tamamlandı!")
    print("✓ Pretrained ağırlıklar korundu - transfer learning aktif!")

if __name__ == "__main__":
    build_feature_dataset_tracked()
