import os
import random
from PIL import Image, ImageDraw

def create_synthetic_dataset(base_dir='../dataset', samples_per_class=20):
    classes = ['clean', 'oil', 'plastic', 'algae']
    
    os.makedirs(base_dir, exist_ok=True)
    
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        for i in range(samples_per_class):
            # Base water color
            img = Image.new('RGB', (224, 224), color=(0, 105, 148))
            draw = ImageDraw.Draw(img)
            
            if cls == 'clean':
                # Add light blue waves
                for _ in range(5):
                    x1, y1 = random.randint(0, 224), random.randint(0, 224)
                    draw.line([(x1, y1), (x1+40, y1)], fill=(20, 150, 200), width=3)
            elif cls == 'oil':
                # Add black/brown oil slicks
                for _ in range(3):
                    x, y = random.randint(0, 180), random.randint(0, 180)
                    draw.ellipse([x, y, x+60, y+40], fill=(30, 30, 30))
            elif cls == 'plastic':
                # Add bright colored debris
                for _ in range(8):
                    x, y = random.randint(0, 200), random.randint(0, 200)
                    r = random.randint(150, 255)
                    g = random.randint(0, 255)
                    b = random.randint(0, 255)
                    draw.rectangle([x, y, x+15, y+15], fill=(r, g, b))
            elif cls == 'algae':
                # Add green patches
                for _ in range(15):
                    x, y = random.randint(0, 200), random.randint(0, 200)
                    draw.ellipse([x, y, x+30, y+30], fill=(34, 139, 34))
                    
            img.save(os.path.join(cls_dir, f"{cls}_{i}.jpg"))
            
if __name__ == '__main__':
    create_synthetic_dataset()
    print("Synthetic dataset generated successfully!")
