import os
import shutil
from bing_image_downloader import downloader

def get_real_dataset():
    classes = {
        'clean': 'clean pristine river water landscape',
        'plastic': 'plastic bottles garbage floating in river water',
        'algae': 'severe thick green algae bloom on water'
    }
    
    base_dir = '../dataset'
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    
    for cls, query in classes.items():
        print(f"\\nDownloading 35 images for {cls}...")
        try:
            downloader.download(query, limit=35, output_dir=base_dir, adult_filter_off=False, force_replace=False, timeout=5, verbose=False)
            
            # The folder downloaded is named exactly after the query
            downloaded_dir = os.path.join(base_dir, query)
            target_dir = os.path.join(base_dir, cls)
            
            if os.path.exists(downloaded_dir):
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                os.rename(downloaded_dir, target_dir)
                print(f"Successfully processed {cls}")
        except Exception as e:
            print(f"Error downloading {cls}: {e}")

if __name__ == '__main__':
    get_real_dataset()
