import os
import cv2
import pandas as pd
import numpy as np

# --- Parametreler ---
CSV_FILE = r"data\final_balanced_clean_dataset_synchronized.csv"
VIDEO_DIR = r"data\download_videos"
OUTPUT_DIR = r"data/segments"

# FPS ve T değerleri - deneysel olarak optimize edilebilir
DEFAULT_FPS = 30
T_OPTIONS = [16, 32, 64]  # Farklı T değerleri deneyebilirsiniz
SELECTED_T = 32  # Şu anki seçim

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Veri Yükle ---
df = pd.read_csv(CSV_FILE)

# Category'yi one-hot için hazırlayalım
categories = sorted(df["category"].unique())
cat_to_index = {cat: i for i, cat in enumerate(categories)}
print("Kategori listesi:", categories)

X = []                # Segment kareleri
y_binary = []         # Autism vs Healthy
y_category = []       # One-hot davranış kategorisi
segment_video_map = []  # Segment-video eşleşmesi

# --- Segment Extraction ---
for index, row in df.iterrows():
    video_name = row['video_id']
    label_text = row['label']       # autism / healthy
    category_text = row['category'] # davranış türü

    # Binary label
    label_bin = 1 if label_text.lower() == "autism" else 0

    # Category label (one-hot)
    label_cat = np.zeros(len(categories), dtype=np.int32)
    label_cat[cat_to_index[category_text]] = 1

    start_time = float(row['start_time'])
    end_time = float(row['end_time'])

    video_path = os.path.join(VIDEO_DIR, label_text, f"{video_name}.mp4")
    if not os.path.exists(video_path):
        print(f"{video_path} bulunamadı, geçiliyor.")
        continue

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Gerçek FPS'yi al (öneri: FPS kontrolü)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0:
        actual_fps = DEFAULT_FPS
        print(f"⚠️ Video FPS alınamadı, varsayılan {DEFAULT_FPS} kullanılıyor: {video_name}")
    else:
        print(f"📹 Video FPS: {actual_fps:.2f} - {video_name}")

    start_frame = int(start_time * actual_fps)
    end_frame = min(int(end_time * actual_fps), total_frames - 1)

    # T kareyi eşit aralıklarla seç (öneri: farklı stratejiler deneyebilirsiniz)
    indices = np.linspace(start_frame, end_frame, num=SELECTED_T, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (224, 224))  # Feature extraction uyumlu boyut
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == SELECTED_T:
        frames_array = np.array(frames)
        X.append(frames_array)
        y_binary.append(label_bin)
        y_category.append(label_cat)
        segment_video_map.append({
            "video_id": video_name,
            "csv_index": index,
            "label": label_text,
            "category": category_text
        })

        # Opsiyonel: segmentleri dosya olarak kaydet
        category_dir = os.path.join(OUTPUT_DIR, label_text)
        os.makedirs(category_dir, exist_ok=True)
        out_path = os.path.join(category_dir, f"{video_name}_{start_time}_{end_time}.npy")
        np.save(out_path, frames_array)
        print(f"{video_name} segment kaydedildi: {start_time}-{end_time}s, shape: {frames_array.shape}")

# --- Numpy array kaydet ---
X = np.array(X)
y_binary = np.array(y_binary)
y_category = np.array(y_category)

np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_binary.npy"), y_binary)
np.save(os.path.join(OUTPUT_DIR, "y_category.npy"), y_category)
np.save(os.path.join(OUTPUT_DIR, "segment_video_map.npy"), segment_video_map)

print("✓ X, y_binary, y_category ve segment_video_map kaydedildi.")
print("Dataset şekli:", X.shape, y_binary.shape, y_category.shape)

# --- Deneysel T değeri test fonksiyonu ---
def test_different_T_values():
    """
    Farklı T değerlerini test etmek için kullanılabilir
    """
    print("\n" + "="*50)
    print("T DEĞERİ OPTİMİZASYON ÖNERİSİ")
    print("="*50)
    print("Mevcut T değeri:", SELECTED_T)
    print("Test edilebilecek T değerleri:", T_OPTIONS)
    print("\nÖneriler:")
    print("1. T=16: Daha hızlı eğitim, daha az bellek kullanımı")
    print("2. T=32: Mevcut seçim (dengeli)")
    print("3. T=64: Daha detaylı temporal bilgi, daha yavaş eğitim")
    print("\nFarklı T değerlerini test etmek için:")
    print("- SELECTED_T değerini değiştirin")
    print("- Bu scripti yeniden çalıştırın")
    print("- Model performansını karşılaştırın")

if __name__ == "__main__":
    test_different_T_values()
