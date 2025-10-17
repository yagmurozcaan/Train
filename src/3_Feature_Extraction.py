"""
Feature Extraction Module for NEUROLOOK Project
Extracts visual features using EfficientNetB0 and combines with category features.
Creates train/validation/test splits for LSTM model training with video-based splitting.
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split

base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[-10:]:
    layer.trainable = True

print("✓ EfficientNetB0 katmanları fine-tune için ayarlandı")

def extract_features(frames):
    features = []
    for frame in frames:
        img = image.img_to_array(frame)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = base_model.predict(img, verbose=0)
        features.append(feat.flatten())
    return np.array(features)

def add_category_features(X_features, segment_video_map, csv_path):
    df = pd.read_csv(csv_path)

    categories = sorted([c for c in df['category'].unique() if c.lower() != 'normal_behavior'])
    cat_to_index = {cat: i for i, cat in enumerate(categories)}
    
    extra_features_list = []
    for seg in segment_video_map:
        idx = seg['csv_index']
        cat = df.loc[idx, 'category']
        label = df.loc[idx, 'label']

        cat_feat = np.zeros(len(categories))
        if label.lower() == 'autism' and cat.lower() != 'normal_behavior':
            if cat in cat_to_index:
                cat_feat[cat_to_index[cat]] = 1

        extra_features_list.append(cat_feat)
    
    extra_features_array = np.array(extra_features_list)
    N, T, d = X_features.shape
    extra_features_time = np.repeat(extra_features_array[:, np.newaxis, :], T, axis=1)

    X_augmented = np.concatenate([X_features, extra_features_time], axis=2)
    print("Feature dataset kategorileri (normal_behavior hariç):", categories)
    return X_augmented

def build_feature_dataset(
    X_path="data/segments/X.npy",
    y_binary_path="data/segments/y_binary.npy",
    y_category_path="data/segments/y_category.npy",
    map_path="data/segments/segment_video_map.npy",
    csv_path="data/final_balanced_clean_dataset.csv",
    out_dir="data/features"
):
    X = np.load(X_path)
    y_binary = np.load(y_binary_path)
    y_category = np.load(y_category_path)
    segment_video_map = np.load(map_path, allow_pickle=True)

    N, T, H, W, C = X.shape
    feature_list = []

    for i in range(N):
        feats = extract_features(X[i])
        feature_list.append(feats)
        if i % 100 == 0:
            print(f"{i}/{N} segment işlendi...")

    X_features = np.array(feature_list)

    X_features = add_category_features(X_features, segment_video_map, csv_path)
    print("Feature dataset şekli (EffNet + category feature):", X_features.shape)

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

    print("✓ Feature dataset kaydedildi.")
    print("Train:", X_train.shape, y_train_bin.shape, y_train_cat.shape)
    print("Validation:", X_val.shape, y_val_bin.shape, y_val_cat.shape)
    print("Test:", X_test.shape, y_test_bin.shape, y_test_cat.shape)

if __name__ == "__main__":
    build_feature_dataset()
