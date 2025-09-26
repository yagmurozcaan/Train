import os
import csv
import subprocess
import sys
from pathlib import Path

def install_yt_dlp():
    """yt-dlp'yi yükler"""
    try:
        import yt_dlp
        print("✓ yt-dlp zaten yüklü")
        return True
    except ImportError:
        print("yt-dlp yükleniyor...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
            print("✓ yt-dlp başarıyla yüklendi")
            return True
        except Exception as e:
            print(f"❌ yt-dlp yüklenirken hata: {e}")
            return False

def download_instagram_videos(csv_file=r"newobs\data\final_balanced_clean_dataset.csv", 
                           output_dir=r"newobs\data\download_videos"):
    """
    Instagram videolarını indirir
    """
    # yt-dlp'yi kontrol et ve yükle
    if not install_yt_dlp():
        return False
    
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        print("❌ yt-dlp import edilemedi")
        return False
    
    # Çıkış klasörünü oluştur
    autism_dir = os.path.join(output_dir, "autism")
    os.makedirs(autism_dir, exist_ok=True)
    
    print(f"📁 İndirme klasörü: {autism_dir}")
    
    # Instagram videolarını bul
    instagram_videos = []
    
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video_id"].strip()
            video_url = row["video_url"].strip()
            label = row["label"].strip()
            
            # Instagram videolarını filtrele
            if video_id.startswith("instagram_autism_") and label == "autism":
                instagram_videos.append({
                    "id": video_id,
                    "url": video_url,
                    "label": label
                })
    
    print(f"📱 {len(instagram_videos)} Instagram videosu bulundu")
    
    if not instagram_videos:
        print("❌ Instagram videosu bulunamadı")
        return False
    
    # Her Instagram videosunu indir
    success_count = 0
    failed_count = 0
    
    for video in instagram_videos:
        video_id = video["id"]
        video_url = video["url"]
        label = video["label"]
        
        output_file = os.path.join(autism_dir, f"{video_id}.mp4")
        
        # Dosya varsa atla
        if os.path.exists(output_file):
            print(f"⏭️  Zaten var: {video_id}")
            success_count += 1
            continue
        
        try:
            print(f"📥 İndiriliyor: {video_id}")
            print(f"🔗 URL: {video_url}")
            
            # yt-dlp konfigürasyonu
            ydl_opts = {
                'outtmpl': output_file,
                'format': 'best[ext=mp4]/best',
                'writesubtitles': False,
                'writeautomaticsub': False,
                'ignoreerrors': True,
                'no_warnings': False,
                'extract_flat': False,
                'noplaylist': True,
                'quiet': False,
                'verbose': True
            }
            
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            # Dosyanın indirilip indirilmediğini kontrol et
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print(f"✅ Başarılı: {video_id}")
                success_count += 1
            else:
                print(f"❌ Dosya oluşturulamadı: {video_id}")
                failed_count += 1
                
        except Exception as e:
            print(f"❌ Hata ({video_id}): {e}")
            failed_count += 1
    
    # Sonuçları yazdır
    print("\n" + "="*50)
    print("📊 İNDİRME SONUÇLARI")
    print("="*50)
    print(f"✅ Başarılı: {success_count}")
    print(f"❌ Başarısız: {failed_count}")
    print(f"📱 Toplam: {len(instagram_videos)}")
    
    if success_count > 0:
        print(f"\n📁 İndirilen dosyalar: {autism_dir}")
        print("🎯 Instagram videoları başarıyla indirildi!")
        print(f"📂 Tam yol: {os.path.abspath(autism_dir)}")
    
    return success_count > 0

def check_instagram_videos(csv_file=r"newobs\data\final_balanced_clean_dataset.csv"):
    """
    Instagram videolarını listeler
    """
    print("📱 Instagram Videosu Listesi:")
    print("-" * 50)
    
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            video_id = row["video_id"].strip()
            video_url = row["video_url"].strip()
            label = row["label"].strip()
            
            if video_id.startswith("instagram_autism_") and label == "autism":
                count += 1
                print(f"{count:2d}. {video_id}")
                print(f"    🔗 {video_url}")
                print()
    
    print(f"📊 Toplam {count} Instagram videosu bulundu")

def main():
    """
    Ana fonksiyon
    """
    print("📱 Instagram Video İndirici")
    print("=" * 50)
    
    # CSV dosyasını kontrol et
    csv_file = r"newobs\data\final_balanced_clean_dataset.csv"
    if not os.path.exists(csv_file):
        print(f"❌ CSV dosyası bulunamadı: {csv_file}")
        return
    
    # Instagram videolarını listele
    check_instagram_videos(csv_file)
    
    # İndirme işlemini başlat
    print("🚀 İndirme işlemi başlatılıyor...")
    success = download_instagram_videos(csv_file)
    
    if success:
        print("\n🎉 Instagram videoları başarıyla indirildi!")
        print("📁 Sonraki adım: Segment extraction çalıştırın")
    else:
        print("\n❌ İndirme işlemi başarısız!")

if __name__ == "__main__":
    main()
