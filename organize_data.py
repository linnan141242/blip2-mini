import os
import json
import shutil
from collections import defaultdict

captions_file = "captions.txt"
images_dir = "Images"
output_dir = "my_mini_data"
output_images = os.path.join(output_dir, "images")
output_captions = os.path.join(output_dir, "captions.json")

os.makedirs(output_images, exist_ok=True)

data = defaultdict(list)
with open(captions_file, "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            img_name = parts[0]
            caption = ",".join(parts[1:]).strip()
            data[img_name].append(caption)

count = 0
final_data = {}
for img_name, captions in data.items():
    if count >= 200:
        break
    src = os.path.join(images_dir, img_name)
    dst = os.path.join(output_images, img_name)
    if os.path.exists(src):
        shutil.copy(src, dst)
        final_data[img_name] = captions
        count += 1

with open(output_captions, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2)

print(f"整理完成！共 {count} 张图片，保存在 {output_dir} 文件夹")