import json
import os
from collections import Counter

# Шляхи до файлів
COCO_FILE = "/home/azureuser/label-studio/coco-dataset/_annotations.coco.json"
OUTPUT_FILE = "/home/azureuser/label-studio/mydata/label_studio_tasks.json"

# Префікс для файлів у Local Storage Label Studio
# (відповідає /label-studio/data/images/fpv_3_5 в контейнері, тобто ./mydata/images/fpv_3_5 на хості)
IMAGE_PREFIX = "/data/local-files/?d=images/fpv_3_5/"

print("Зчитування COCO JSON...")
with open(COCO_FILE, "r", encoding="utf-8") as f:
    coco = json.load(f)

# Категорії (класи), пропускаємо кореневу supercategory-заглушку без анотацій
categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}

# Інформація про зображення
images = {img["id"]: img for img in coco.get("images", [])}

# Групування анотацій за image_id
annotations_by_img = {}
for ann in coco.get("annotations", []):
    annotations_by_img.setdefault(ann["image_id"], []).append(ann)

tasks = []
all_tags = Counter()
all_labels = Counter()

for img_id, img_info in images.items():
    img_filename = os.path.basename(img_info["file_name"])
    img_w = img_info.get("width")
    img_h = img_info.get("height")
    image_url = IMAGE_PREFIX + img_filename

    results = []

    # 1. Теги зображення (Roboflow зберігає їх у extra.user_tags)
    img_tags = img_info.get("extra", {}).get("user_tags") or img_info.get("tags") or []
    if isinstance(img_tags, str):
        img_tags = [img_tags]

    if img_tags:
        all_tags.update(img_tags)
        results.append({
            "from_name": "tags",
            "to_name": "image",
            "type": "choices",
            "value": {"choices": img_tags},
        })

    # 2. Bounding boxes
    for ann in annotations_by_img.get(img_id, []):
        category_name = categories.get(ann["category_id"], "object")
        bbox = ann.get("bbox")

        if not bbox or len(bbox) != 4 or not img_w or not img_h:
            continue

        x_px, y_px, w_px, h_px = bbox
        all_labels[category_name] += 1

        results.append({
            "id": f"ann_{ann['id']}",
            "type": "rectanglelabels",
            "from_name": "label",
            "to_name": "image",
            "original_width": img_w,
            "original_height": img_h,
            "value": {
                "x": (x_px / img_w) * 100,
                "y": (y_px / img_h) * 100,
                "width": (w_px / img_w) * 100,
                "height": (h_px / img_h) * 100,
                "rotation": 0,
                "rectanglelabels": [category_name],
            },
        })

    tasks.append({
        "data": {"image": image_url},
        "annotations": [{"result": results}] if results else [],
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(tasks, f, ensure_ascii=False, indent=2)

print(f"Успішно оброблено {len(tasks)} завдань у {OUTPUT_FILE}")
print(f"Класи об'єктів: {dict(all_labels)}")
print(f"Теги зображень: {dict(all_tags)}")
