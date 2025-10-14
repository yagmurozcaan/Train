import os
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# --- Parametreler ---
CSV_FILE = r"data/final_balanced_clean_dataset.csv"
VIDEO_DIR = r"data/download_videos"
OUTPUT_DIR = r"data/segments"

FPS = 30
T = 32  # Her segmentten alınacak kare sayısı

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- MediaPipe Holistic ---
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(static_image_mode=False,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

# --- Veri Yükle ---
df = pd.read_csv(CSV_FILE)

# 🔹 normal_behavior hariç kategoriler
categories = sorted([c for c in df['category'].unique() if c.lower() != 'normal_behavior'])
cat_to_index = {cat: i for i, cat in enumerate(categories)}
print("Kategori listesi (normal_behavior hariç):", categories)

X = []                
y_binary = []
y_category = []
segment_video_map = []
landmark_dir = os.path.join(OUTPUT_DIR, "landmarks")
os.makedirs(landmark_dir, exist_ok=True)

# --- Segment Extraction + Landmark normalize ---
for index, row in tqdm(df.iterrows(), total=len(df), desc="Segmentler işleniyor"):
    video_name = row['video_id']
    label_text = row['label']
    category_text = row['category']

    # 1 = autism, 0 = healthy
    label_bin = 1 if label_text.lower() == "autism" else 0

    # 🔹 sadece autism ve normal_behavior olmayanlar için one-hot
    label_cat = np.zeros(len(categories), dtype=np.int32)
    if label_bin == 1 and category_text.lower() != "normal_behavior":
        if category_text in cat_to_index:
            label_cat[cat_to_index[category_text]] = 1

    start_time = float(row['start_time'])
    end_time = float(row['end_time'])

    video_path = os.path.join(VIDEO_DIR, label_text, f"{video_name}.mp4")
    if not os.path.exists(video_path):
        print(f"{video_path} bulunamadı, geçiliyor.")
        continue

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_time * FPS)
    end_frame = min(int(end_time * FPS), total_frames - 1)
    indices = np.linspace(start_frame, end_frame, num=T, dtype=int)

    frames = []
    landmarks_segment = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        frames.append(frame_resized)

        # MediaPipe Holistic
        results = holistic_model.process(frame_rgb)
        landmarks_frame = {}

        # Face
        if results.face_landmarks:
            face = np.array([[lm.x, lm.y] for lm in results.face_landmarks.landmark])
            landmarks_frame["face"] = face

        # Hands
        hands = []
        if results.left_hand_landmarks:
            hands.append(np.array([[lm.x, lm.y] for lm in results.left_hand_landmarks.landmark]))
        if results.right_hand_landmarks:
            hands.append(np.array([[lm.x, lm.y] for lm in results.right_hand_landmarks.landmark]))
        landmarks_frame["hands"] = hands

        # Pose
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark])
            # Normalize: center ve scale
            center = pose[0]  # pelvis
            pose_centered = pose - center
            scale = np.linalg.norm(pose[11] - pose[12])  # omuz mesafesi
            pose_normalized = pose_centered / (scale + 1e-6)
            landmarks_frame["pose"] = pose_normalized

        landmarks_segment.append(landmarks_frame)

    cap.release()

    if len(frames) == T:
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

        # Landmark kaydet
        landmark_path = os.path.join(landmark_dir, f"{video_name}_{start_time}_{end_time}_landmarks.npy")
        np.save(landmark_path, landmarks_segment)

        # Segment RGB kaydet
        category_dir = os.path.join(OUTPUT_DIR, label_text)
        os.makedirs(category_dir, exist_ok=True)
        segment_path = os.path.join(category_dir, f"{video_name}_{start_time}_{end_time}.npy")
        np.save(segment_path, frames_array)

        print(f"{video_name} segment kaydedildi: {start_time}-{end_time}s, shape: {frames_array.shape}")

# --- Numpy array kaydet ---
X = np.array(X)
y_binary = np.array(y_binary)
y_category = np.array(y_category)
np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_binary.npy"), y_binary)
np.save(os.path.join(OUTPUT_DIR, "y_category.npy"), y_category)
np.save(os.path.join(OUTPUT_DIR, "segment_video_map.npy"), segment_video_map)

print("✓ Segment extraction ve normalize landmark kaydı tamamlandı.")
