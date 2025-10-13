import os
import csv
import sys
import subprocess
from yt_dlp import YoutubeDL

def install_yt_dlp():
    """yt-dlp kontrolü"""
    try:
        import yt_dlp
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
            return True
        except Exception as e:
            print(f"❌ yt-dlp kurulamadı: {e}")
            return False

def download_videos(csv_file="data/final_balanced_clean_dataset.csv",
                    output_dir="data/download_videos"):
    """Hem Instagram hem YouTube videolarını indirir"""
    if not install_yt_dlp():
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    failed_videos = []

    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row.get("video_id", "").strip()
            video_url = row.get("video_url", "").strip()
            label = row.get("label", "unknown").strip()

            if not video_id or not video_url:
                print(f"⚠️ Eksik veri atlandı: {row}")
                continue

            # --- Instagram kontrolü ---
            if "instagram.com" in video_url.lower():
                platform = "instagram"
            # --- YouTube kontrolü ---
            elif "youtube.com/shorts/" in video_url:
                vid = video_url.split("shorts/")[-1].split("?")[0]
                video_url = f"https://www.youtube.com/watch?v={vid}"
                platform = "youtube"
            elif "youtube.com/watch" in video_url or "youtu.be/" in video_url:
                platform = "youtube"
            else:
                print(f"⚠️ Desteklenmeyen link atlandı: {video_url}")
                continue

            # Label klasörü
            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            output_file = os.path.join(label_dir, f"{video_id}.%(ext)s")

            # Daha önce indirilmiş mi?
            if any(os.path.exists(os.path.join(label_dir, f"{video_id}.{ext}")) for ext in ["mp4", "webm", "mkv"]):
                print(f"✅ Zaten var, atlanıyor: {video_id}")
                continue

            try:
                print(f"⬇️ [{platform.upper()}] İndiriliyor: {video_id} -> {video_url}")

                ydl_opts = {
                    'outtmpl': output_file,
                    'format': 'best[ext=mp4]/best',
                    'quiet': False,
                    'noprogress': False,
                    'ignoreerrors': True,
                    'no_warnings': False,
                }

                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])

                print(f"🎉 Tamamlandı: {video_id}")

            except Exception as e:
                print(f"❌ Hata ({video_id}): {e}")
                failed_videos.append({
                    "video_id": video_id,
                    "video_url": video_url,
                    "label": label,
                    "error": str(e)
                })

    # Başarısız videoları kaydet
    if failed_videos:
        fail_file = os.path.join(output_dir, "failed_videos.csv")
        with open(fail_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["video_id", "video_url", "label", "error"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(failed_videos)

        print(f"⚠️ Başarısız olanlar kaydedildi: {fail_file}")


if __name__ == "__main__":
    download_videos()
