import os
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

from flask import Flask, request, jsonify
from ultralytics import YOLO
import json

app = Flask(__name__)

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
MODEL_NAME = os.environ.get("YOLO_MODEL", "best.pt")
MODEL_PATH = MODEL_DIR / MODEL_NAME

# Shared Label Studio data
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/app/data")).resolve()

print(f"Loading YOLO model: {MODEL_PATH}", flush=True)
print(f"Data root: {DATA_ROOT}", flush=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully", flush=True)
print(f"Classes: {model.names}", flush=True)


def resolve_image_path(image_url: str) -> Path:
    """
    Convert Label Studio local-files URL:

        /data/local-files/?d=images/foo/bar.jpg

    into actual filesystem path:

        /app/data/images/foo/bar.jpg
    """

    print(f"Resolving image URL: {image_url}", flush=True)

    parsed = urlparse(image_url)

    # Label Studio local files endpoint
    if parsed.path == "/data/local-files/":
        query = parse_qs(parsed.query)

        if "d" not in query or not query["d"]:
            raise FileNotFoundError(
                f"Missing 'd' parameter in Label Studio URL: {image_url}"
            )

        relative_path = unquote(query["d"][0]).lstrip("/")

        file_path = (DATA_ROOT / relative_path).resolve()

        # Security: prevent path traversal outside DATA_ROOT
        try:
            file_path.relative_to(DATA_ROOT)
        except ValueError:
            raise FileNotFoundError(
                f"Invalid path outside DATA_ROOT: {relative_path}"
            )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Image does not exist: {file_path}"
            )

        return file_path

    # In case Label Studio sends a direct filesystem path
    path = Path(image_url)

    if path.exists():
        return path

    raise FileNotFoundError(
        f"Unsupported or missing image path: {image_url}"
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "UP",
        "model": str(MODEL_PATH),
        "classes": model.names,
        "data_root": str(DATA_ROOT)
    })


@app.route("/setup", methods=["POST"])
def setup():
    return jsonify({
        "model_version": MODEL_NAME
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    tasks = data.get("tasks", [])

    predictions = []

    for task in tasks:

        image_url = task.get("data", {}).get("image")

        if not image_url:
            predictions.append({
                "result": [],
                "score": 0
            })
            continue

        try:
            image_path = resolve_image_path(image_url)

            print(
                f"Predicting image: {image_path}",
                flush=True
            )

            results = model.predict(
                source=str(image_path),
                device="cpu",
                verbose=False
            )

            result = results[0]
            print(
                f"YOLO detected {len(result.boxes) if result.boxes is not None else 0} objects",
                flush=True
            )
            height, width = result.orig_shape

            task_results = []

            if result.boxes is not None:
                for box in result.boxes:

                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    task_results.append({
                        "id": f"yolo_{cls_id}_{len(task_results)}",
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",

                        "original_width": width,
                        "original_height": height,
                        "image_rotation": 0,

                        "value": {
                            "x": x1 / width * 100,
                            "y": y1 / height * 100,
                            "width": (x2 - x1) / width * 100,
                            "height": (y2 - y1) / height * 100,
                            "rotation": 0,
                            "rectanglelabels": [
                                model.names[cls_id]
                            ]
                        },

                        "score": confidence
                    })

            predictions.append({
                "model_version": MODEL_NAME,
                "result": task_results,
                "score": max(
                    [x["score"] for x in task_results],
                    default=0
                )
            })

        except Exception as e:

            print(
                f"Prediction failed: {e}",
                flush=True
            )

            predictions.append({
                "result": [],
                "score": 0
            })


    print(
        "PREDICTION RESPONSE:",
        json.dumps(predictions, indent=2),
        flush=True
    )
    return jsonify({
        "results": predictions
    })

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=9090,
        threaded=True
    )