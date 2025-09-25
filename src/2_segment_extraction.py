import os
import cv2
import csv
import numpy as np

CSV_FILE = r"data/final_balanced_clean_dataset.csv"
VIDEO_BASE_DIR = r"data/download_videos"
FPS = 30
T = 3  # Her segmentten alınacak kare sayısı

def extract_segment_from_range(video_path, start_sec, end_sec, T=T):
    cap = cv2.VideoCapture(video_path)
    segments = []

    start_frame = int(start_sec * FPS)
    end_frame = int(end_sec * FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame >= total_frames:
        cap.release()
        return segments

    end_frame = min(end_frame, total_frames - 1)
    indices = np.linspace(start_frame, end_frame, num=T, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    if len(frames) == T:
        segments.append(np.array(frames))

    cap.release()
    return segments

def build_dataset_from_csv_tracked():
    X, y = [], []
    segment_video_map = []  # Her segmentin hangi videoya ait olduğunu saklamak için

    with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            video_id = row["video_id"]
            start_time = float(row["start_time"])
            end_time = float(row["end_time"])
            label_text = row["label"]
            label = 1 if label_text.lower() == "autism" else 0

            if label_text.lower() == "autism":
                video_path = os.path.join(VIDEO_BASE_DIR, "autism", f"{video_id}.mp4")
            else:
                video_path = os.path.join(VIDEO_BASE_DIR, "healthy", f"{video_id}.mp4")

            if not os.path.exists(video_path):
                print(f"Video bulunamadı: {video_path}")
                continue

            segments = extract_segment_from_range(video_path, start_time, end_time, T=T)
            for seg in segments:
                X.append(seg)
                y.append(label)
                segment_video_map.append({
                    "video_id": video_id,
                    "csv_index": row_idx
                })

            print(f"İşlendi: {video_id}, segment sayısı: {len(segments)}")

    X = np.array(X)
    y = np.array(y)
    print("Dataset şekli:", X.shape, y.shape)

    # Segment-video eşleşmesini kaydet
    np.save("X.npy", X)
    np.save("y.npy", y)
    np.save("segment_video_map.npy", segment_video_map)
    print("X, y ve segment_video_map kaydedildi.")

if __name__ == "__main__":
    build_dataset_from_csv_tracked()
