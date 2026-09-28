import json
import os
from collections import Counter
from dotenv import load_dotenv

from label_studio_sdk import LabelStudio


# ============================================================
# CONFIG
# ============================================================
load_dotenv() 

LABEL_STUDIO_URL = "http://localhost:8080"
LABEL_STUDIO_API_KEY = os.environ.get("LABEL_STUDIO_API_TOKEN", "")

if not LABEL_STUDIO_API_KEY:
    raise RuntimeError("Set LABEL_STUDIO_API_TOKEN before running this script.")

PROJECT_ID = 1

COCO_FILE = (
    "/home/azureuser/label-studio/"
    "coco-dataset/_annotations.coco.json"
)


# ============================================================
# CLIENT
# ============================================================

client = LabelStudio(
    base_url=LABEL_STUDIO_URL,
    api_key=LABEL_STUDIO_API_KEY,
)


# ============================================================
# LOAD COCO
# ============================================================

print("Loading COCO...")

with open(COCO_FILE, "r", encoding="utf-8") as f:
    coco = json.load(f)


categories = {
    category["id"]: category["name"]
    for category in coco.get("categories", [])
}

images = {
    image["id"]: image
    for image in coco.get("images", [])
}


annotations_by_image = {}

for annotation in coco.get("annotations", []):
    image_id = annotation["image_id"]

    annotations_by_image.setdefault(
        image_id,
        []
    ).append(annotation)


print(f"COCO images: {len(images)}")
print(f"COCO annotations: {len(coco.get('annotations', []))}")
print(f"COCO categories: {len(categories)}")


# ============================================================
# HELPERS
# ============================================================

def normalize_filename(filename):
    if not filename:
        return None

    return os.path.basename(filename).strip().lower()


def get_task_filename(task):
    """
    Extract filename from Label Studio task.

    Example:

    /data/local-files/?d=images/fpv_3_5/image001.jpg

    -> image001.jpg
    """

    image = task.data.get("image")

    if not image:
        return None

    if "?d=" in image:
        image = image.split("?d=", 1)[1]

    return os.path.basename(image)


# ============================================================
# GET TASKS
# ============================================================

print()
print("Loading Label Studio tasks...")

tasks = list(
    client.tasks.list(
        project=PROJECT_ID,
        page_size=1000,
    )
)

print(f"Label Studio tasks: {len(tasks)}")


# ============================================================
# INDEX TASKS BY FILENAME
# ============================================================

tasks_by_filename = {}

for task in tasks:

    filename = get_task_filename(task)

    if not filename:
        continue

    filename = normalize_filename(filename)

    if filename in tasks_by_filename:
        print(
            f"WARNING: duplicate Label Studio task "
            f"for {filename}"
        )

    tasks_by_filename[filename] = task


# ============================================================
# IMPORT ANNOTATIONS
# ============================================================

stats = Counter()


for image_id, image_info in images.items():

    filename = normalize_filename(
        image_info.get("file_name")
    )

    if not filename:

        stats["invalid_coco_images"] += 1
        continue


    # --------------------------------------------------------
    # Find existing Label Studio task
    # --------------------------------------------------------

    task = tasks_by_filename.get(filename)

    if not task:

        print(
            f"[NOT FOUND] {filename}"
        )

        stats["tasks_not_found"] += 1
        continue


    task_id = task.id


    # --------------------------------------------------------
    # Check existing annotations
    # --------------------------------------------------------

    existing_annotations = list(
        client.annotations.list(
            id=task_id
        )
    )

    if existing_annotations:

        print(
            f"[SKIP] {filename} "
            f"(task={task_id}, already annotated)"
        )

        stats["already_annotated"] += 1
        continue


    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    width = image_info.get("width")
    height = image_info.get("height")

    if not width or not height:

        print(
            f"[SKIP] {filename} "
            f"(missing dimensions)"
        )

        stats["missing_dimensions"] += 1
        continue


    results = []


    # ========================================================
    # ROBoflow IMAGE TAGS
    # ========================================================

    tags = []
    # tags = (
    #     image_info.get("extra", {}).get("user_tags")
    #     or image_info.get("tags")
    #     or []
    # )

    # if isinstance(tags, str):
    #     tags = [tags]


    # if tags:

    #     results.append({
    #         "id": f"tags_{image_id}",
    #         "from_name": "tags",
    #         "to_name": "image",
    #         "type": "choices",
    #         "value": {
    #             "choices": tags
    #         },
    #     })


    # ========================================================
    # COCO BOUNDING BOXES
    # ========================================================

    coco_annotations = annotations_by_image.get(
        image_id,
        []
    )


    for annotation in coco_annotations:

        category_name = categories.get(
            annotation["category_id"],
            "object"
        )

        bbox = annotation.get("bbox")

        if not bbox or len(bbox) != 4:
            continue


        x, y, box_width, box_height = bbox


        results.append({
            "id": f"ann_{annotation['id']}",

            "type": "rectanglelabels",

            "from_name": "label",
            "to_name": "image",

            "original_width": width,
            "original_height": height,

            "value": {
                "x": x / width * 100,
                "y": y / height * 100,
                "width": box_width / width * 100,
                "height": box_height / height * 100,
                "rotation": 0,

                "rectanglelabels": [
                    category_name
                ],
            },
        })


    # ========================================================
    # NOTHING TO IMPORT
    # ========================================================

    if not results:

        print(
            f"[EMPTY] {filename}"
        )

        stats["empty"] += 1
        continue


    # ========================================================
    # CREATE ANNOTATION
    # ========================================================

    client.annotations.create(
        id=task_id,
        result=results,
    )

    print(
        f"[OK] {filename} "
        f"(task={task_id}, "
        f"boxes={len(coco_annotations)}, "
        f"tags={len(tags)})"
    )

    stats["imported"] += 1


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("DONE")
print("=" * 60)

for key, value in stats.items():
    print(f"{key:25} {value}")