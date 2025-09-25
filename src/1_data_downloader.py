import os
import csv
from yt_dlp import YoutubeDL

def download_videos(csv_file="data/final_balanced_clean_dataset.csv", 
                    output_dir="data/download_videos"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    videos = {}
    
    # CSV'den link, video_id ve label okuma
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_url = row["video_url"].strip()
            video_id = row["video_id"].strip()
            label = row["label"].strip() if "label" in row else "unknown"
            
            if video_url not in videos:
                videos[video_url] = {"id": video_id, "label": label}
    
    for video_url, info in videos.items():
        video_id = info["id"]
        label = info["label"]
        
        # Label'e göre klasör oluşturma
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        output_file = os.path.join(label_dir, f"{video_id}.mp4")
        
        # Dosya varsa indirme
        if os.path.exists(output_file):
            print(f"Zaten var, atlanıyor: {video_id} ({label})")
            continue

        try:
            print(f"İndiriliyor: {video_id} -> {video_url} | Kategori: {label}")
            
            ydl_opts = {
                'outtmpl': output_file,
                'format': 'mp4'
            }
            
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            print(f"Tamamlandı: {video_id} ({label})")
        except Exception as e:
            print(f"Hata oluştu ({video_id} - {label}): {e}")

if __name__ == "__main__":
    download_videos()
