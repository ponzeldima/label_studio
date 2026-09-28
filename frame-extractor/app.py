"""Dataset Hub: frame extraction and Label Studio dataset export."""

from __future__ import annotations

import re
import shutil
import uuid
import zipfile
import os
from collections import defaultdict
from io import BytesIO
from math import ceil
from datetime import datetime
from pathlib import Path
from random import Random
from urllib.parse import unquote, urlparse

import cv2
import requests
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

DATA_DIR = Path("/data")
IMAGES_DIR = DATA_DIR / "images"
STAGING_ROOT = Path("/tmp/frame-staging")
UPLOAD_ROOT = Path("/tmp/frame-uploads")
FRAMES_PER_PAGE = 24
MAX_SUBFOLDER_LENGTH = 80
PREVIEW_WIDTH = 360
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://app:8080")
LABEL_STUDIO_API_TOKEN = os.getenv("LABEL_STUDIO_API_TOKEN", "")
LOCAL_STORAGE_ENDPOINT = "/api/storages/localfiles/"
CLASS_IDS = {"fpv": 0, "bird": 1}


def configured_label_studio() -> tuple[str, str]:
    return LABEL_STUDIO_URL.rstrip("/"), LABEL_STUDIO_API_TOKEN


def label_studio_get(path: str, params: dict[str, object] | None = None) -> object:
    base_url, token = configured_label_studio()
    if not token:
        raise RuntimeError("Set LABEL_STUDIO_API_TOKEN in .env and restart dataset-hub.")
    response = requests.get(
        f"{base_url}{path}", params=params, headers={"Authorization": f"Token {token}"}, timeout=60
    )
    response.raise_for_status()
    return response.json()


def response_items(response: object) -> list[dict[str, object]]:
    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]
    if isinstance(response, dict):
        return [item for item in response.get("results", []) if isinstance(item, dict)]
    return []


def load_storages() -> list[dict[str, object]]:
    storages = response_items(label_studio_get(LOCAL_STORAGE_ENDPOINT))
    for storage in storages:
        storage["storage_type"] = "localfiles"
    return storages


def load_project_tasks(project_id: int) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    page = 1
    while True:
        response = label_studio_get("/api/tasks", {"project": project_id, "page": page, "page_size": 1000})
        tasks.extend(response_items(response))
        if not isinstance(response, dict) or not response.get("next"):
            return tasks
        page += 1


def storage_id_for_task(task: dict[str, object]) -> int | None:
    meta = task.get("meta", {})
    if not isinstance(meta, dict):
        return None
    for key in ("source_storage_id", "storage_id", "import_storage_id"):
        value = meta.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def task_belongs_to_storage(task: dict[str, object], storage: dict[str, object]) -> bool:
    storage_id = storage_id_for_task(task)
    if storage_id is not None:
        return storage_id == int(storage["id"])
    image = task_image_reference(task)
    if not image:
        return False
    image_path = unquote(image.split("?d=", 1)[1]) if "?d=" in image else image
    for key in ("prefix", "path", "bucket", "container"):
        value = storage.get(key)
        if not isinstance(value, str) or not value:
            continue
        normalized = value.rstrip("/")
        if normalized in image_path or image_path.startswith(Path(normalized).name + "/"):
            return True
    title = storage.get("title")
    return isinstance(title, str) and title in image_path


def completed_annotation(task: dict[str, object]) -> dict[str, object] | None:
    annotations = task.get("annotations", [])
    if not isinstance(annotations, list):
        return None
    for annotation in reversed(annotations):
        if isinstance(annotation, dict) and not annotation.get("was_cancelled", False):
            return annotation
    return None


def yolo_labels(annotation: dict[str, object]) -> str:
    lines: list[str] = []
    for result in annotation.get("result", []):
        if not isinstance(result, dict) or result.get("type") != "rectanglelabels":
            continue
        value = result.get("value", {})
        if not isinstance(value, dict):
            continue
        labels = value.get("rectanglelabels", [])
        if not isinstance(labels, list) or not labels or labels[0] not in CLASS_IDS:
            continue
        try:
            x, y = float(value["x"]) / 100, float(value["y"]) / 100
            width, height = float(value["width"]) / 100, float(value["height"]) / 100
        except (KeyError, TypeError, ValueError):
            continue
        lines.append(f"{CLASS_IDS[labels[0]]} {x + width / 2:.6f} {y + height / 2:.6f} {width:.6f} {height:.6f}")
    return "\n".join(lines)


def task_image_reference(task: dict[str, object]) -> str | None:
    data = task.get("data", {})
    image = data.get("image") if isinstance(data, dict) else None
    return image if isinstance(image, str) else None


def image_bytes(task: dict[str, object]) -> tuple[bytes, str]:
    image = task_image_reference(task)
    if not image:
        raise RuntimeError("Task does not contain an image field.")
    image_path = unquote(image.split("?d=", 1)[1]) if "?d=" in image else ""
    local_path = DATA_DIR / image_path
    if image_path and local_path.is_file():
        return local_path.read_bytes(), local_path.suffix or ".jpg"
    base_url, token = configured_label_studio()
    url = image if image.startswith(("http://", "https://")) else f"{base_url}{image}"
    response = requests.get(url, headers={"Authorization": f"Token {token}"}, timeout=120)
    response.raise_for_status()
    return response.content, Path(urlparse(url).path).suffix or ".jpg"


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "storage"


def build_yolo_archive(selected_storages: list[dict[str, object]], tasks_by_storage: dict[int, list[dict[str, object]]], splits: tuple[int, int, int]) -> tuple[bytes, dict[str, int]]:
    export_tasks: list[tuple[str, dict[str, object], dict[str, object]]] = []
    for storage in selected_storages:
        storage_id = int(storage["id"])
        prefix = safe_filename(str(storage.get("title") or f"storage_{storage_id}"))
        for task in tasks_by_storage.get(storage_id, []):
            if annotation := completed_annotation(task):
                export_tasks.append((prefix, task, annotation))
    Random(42).shuffle(export_tasks)
    train_end = round(len(export_tasks) * splits[0] / 100)
    valid_end = train_end + round(len(export_tasks) * splits[1] / 100)
    split_names = ["train"] * train_end + ["valid"] * (valid_end - train_end) + ["test"] * (len(export_tasks) - valid_end)
    counts: dict[str, int] = defaultdict(int)
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("data.yaml", "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\nnames:\n  0: 'fpv'\n  1: 'bird'\n")
        for index, ((prefix, task, annotation), split) in enumerate(zip(export_tasks, split_names)):
            data, suffix = image_bytes(task)
            stem = Path(unquote(task_image_reference(task) or "image.jpg").split("?", 1)[0]).stem
            output_name = f"{prefix}_{safe_filename(stem)}_{task.get('id', index)}{suffix.lower()}"
            zip_file.writestr(f"{split}/images/{output_name}", data)
            zip_file.writestr(f"{split}/labels/{Path(output_name).stem}.txt", yolo_labels(annotation))
            counts[split] += 1
    return archive.getvalue(), counts


def session_staging_dir() -> Path:
    session_id = st.session_state.setdefault("session_id", uuid.uuid4().hex)
    return STAGING_ROOT / session_id


def session_upload_dir() -> Path:
    session_id = st.session_state.setdefault("session_id", uuid.uuid4().hex)
    return UPLOAD_ROOT / session_id


def clear_frame_staging() -> None:
    shutil.rmtree(session_staging_dir(), ignore_errors=True)


def clear_staging() -> None:
    clear_frame_staging()
    shutil.rmtree(session_upload_dir(), ignore_errors=True)


def safe_subfolder_name(value: str) -> str | None:
    name = value.strip()
    if not name or len(name) > MAX_SUBFOLDER_LENGTH:
        return None
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        return None
    return name


def get_video_info(video_path: Path) -> tuple[float, float, int, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open the uploaded video.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return frame_count / fps if frame_count else 0, fps, frame_count, width, height


def get_preview_frame(video_path: Path, timestamp: float) -> Image.Image:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Could not read a preview frame.")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes):02d}:{remaining_seconds:04.1f}"


def default_folder_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def crop_bounds(crop_box: dict[str, float] | None, preview_size: tuple[int, int], source_size: tuple[int, int]) -> tuple[int, int, int] | None:
    if not crop_box:
        return None
    preview_width, preview_height = preview_size
    source_width, source_height = source_size
    scale = source_width / preview_width
    left = max(0, int(crop_box["left"] * scale))
    top = max(0, int(crop_box["top"] * scale))
    displayed_width = crop_box["width"] * scale
    displayed_height = crop_box["height"] * scale
    side = max(1, int(min(displayed_width, displayed_height)))
    side = min(side, source_width, source_height)
    left = min(left, source_width - side)
    top = min(top, source_height - side)
    return left, top, side


def largest_square_coords(image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return left, left + side, top, top + side


def extract_frames(
    video_path: Path,
    mode: str,
    value: int,
    start_time: float,
    end_time: float,
    crop: tuple[int, int, int] | None,
) -> list[Path]:
    staging_dir = session_staging_dir()
    clear_frame_staging()
    staging_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(source_fps / value)), 1) if mode == "fps" else value
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_paths: list[Path] = []
    progress = st.progress(0.0, text="Extracting frames...")

    start_frame = max(0, int(start_time * source_fps))
    end_frame = min(total_frames, int(end_time * source_fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    while True:
        ok, frame = capture.read()
        if not ok or frame_index >= end_frame:
            break
        if (frame_index - start_frame) % step == 0:
            if crop:
                left, top, side = crop
                frame = frame[top : top + side, left : left + side]
            frame_path = staging_dir / f"frame_{len(frame_paths):05d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError("Could not write an extracted frame.")
            frame_paths.append(frame_path)
        frame_index += 1
        if end_frame > start_frame:
            progress.progress(
                min((frame_index - start_frame) / (end_frame - start_frame), 1.0),
                text="Extracting frames...",
            )

    capture.release()
    progress.empty()
    return frame_paths


def reset_frames(clear_upload: bool = False) -> None:
    clear_frame_staging()
    st.session_state.frame_paths = []
    st.session_state.removed = set()
    st.session_state.page = 0
    st.session_state.folder_name_reset_pending = True
    if clear_upload:
        shutil.rmtree(session_upload_dir(), ignore_errors=True)
        st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1


def reset_frames_and_folder_name(clear_upload: bool = False) -> None:
    reset_frames(clear_upload=clear_upload)
    st.session_state.folder_name = default_folder_name()


def init_state() -> None:
    st.session_state.setdefault("frame_paths", [])
    st.session_state.setdefault("removed", set())
    st.session_state.setdefault("video_stem", "video")
    st.session_state.setdefault("page", 0)
    st.session_state.setdefault("video_signature", None)
    st.session_state.setdefault("default_folder_name", default_folder_name())
    st.session_state.setdefault("folder_name_reset_pending", False)
    st.session_state.setdefault("success_message", None)
    st.session_state.setdefault("uploader_key", 0)
    if st.session_state.folder_name_reset_pending:
        st.session_state.default_folder_name = default_folder_name()
        st.session_state.folder_name_reset_pending = False


def render_frame_extractor() -> None:
    st.title("Video Frame Extractor")
    init_state()
    if success_message := st.session_state.success_message:
        st.success(success_message)
        st.session_state.success_message = None

    uploaded_video = st.file_uploader(
        "Video",
        type=["mp4", "mov", "avi", "mkv", "m4v", "webm"],
        key=f"uploaded_video_{st.session_state.uploader_key}",
    )
    if uploaded_video is None:
        return

    upload_dir = session_upload_dir()
    upload_path = upload_dir / Path(uploaded_video.name).name
    video_signature = (uploaded_video.name, uploaded_video.size)
    if st.session_state.video_signature != video_signature:
        reset_frames_and_folder_name(clear_upload=False)
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path.write_bytes(uploaded_video.getbuffer())
        st.session_state.video_signature = video_signature

    try:
        duration, source_fps, source_frames, source_width, source_height = get_video_info(upload_path)
    except RuntimeError as error:
        st.error(str(error))
        return

    st.video(uploaded_video)
    st.caption(
        f"Source: {format_duration(duration)} | {source_width} x {source_height} px | "
        f"{source_fps:.2f} FPS | {source_frames} frames"
    )
    start_time, end_time = st.slider(
        "Video segment to extract",
        min_value=0.0,
        max_value=max(duration, 0.1),
        value=(0.0, max(duration, 0.1)),
        step=0.1,
    )
    st.caption(
        f"Start: {format_duration(start_time)} | End: {format_duration(end_time)} | "
        f"Duration: {format_duration(end_time - start_time)}"
    )

    crop_enabled = st.checkbox("Crop to a square", value=False)
    crop: tuple[int, int, int] | None = None
    if crop_enabled:
        preview = get_preview_frame(upload_path, start_time)
        preview.thumbnail((PREVIEW_WIDTH, PREVIEW_WIDTH))
        preview_width, preview_height = preview.size
        st.caption(f"Preview at {format_duration(start_time)}. Move or resize the square selection.")
        crop_box = st_cropper(
            preview,
            aspect_ratio=(1, 1),
            default_coords=largest_square_coords(preview.size),
            return_type="box",
            box_color="#00b4ff",
            should_resize_image=False,
            key=f"cropper_{video_signature}_{start_time}",
        )
        crop = crop_bounds(crop_box, preview.size, (source_width, source_height))
        if crop:
            st.caption(f"Output size: {crop[2]} x {crop[2]} px")

    mode_label = st.radio(
        "Sampling method", ["Frames per second", "Every Nth frame"], horizontal=True
    )
    mode = "fps" if mode_label == "Frames per second" else "every_nth"
    value = st.number_input(
        "Frames per second" if mode == "fps" else "Every Nth frame",
        min_value=1,
        max_value=60 if mode == "fps" else 1000,
        value=2 if mode == "fps" else 15,
    )
    segment_duration = end_time - start_time
    sampling_step = max(int(round(source_fps / int(value))), 1) if mode == "fps" else int(value)
    selected_source_frames = max(0, int(end_time * source_fps) - int(start_time * source_fps))
    expected_count = ceil(selected_source_frames / sampling_step)
    output_size = f"{crop[2]} x {crop[2]} px" if crop else f"{source_width} x {source_height} px"
    st.info(
        f"Selected duration: {format_duration(segment_duration)} | Output: {expected_count} frames | "
        f"Frame size: {output_size}"
    )

    if st.button("Extract frames", type="primary"):
        try:
            st.session_state.frame_paths = extract_frames(
                upload_path, mode, int(value), start_time, end_time, crop
            )
            st.session_state.removed = set()
            st.session_state.video_stem = Path(uploaded_video.name).stem
            st.session_state.page = 0
        except RuntimeError as error:
            reset_frames()
            st.error(f"Extraction failed: {error}")

    frame_paths: list[Path] = st.session_state.frame_paths
    if not frame_paths:
        return

    st.subheader(f"Review frames ({len(frame_paths)})")
    total_pages = max((len(frame_paths) - 1) // FRAMES_PER_PAGE + 1, 1)
    st.session_state.page = min(st.session_state.page, total_pages - 1)
    previous_column, page_column, next_column = st.columns([1, 2, 1])
    with previous_column:
        if st.button("Previous", disabled=st.session_state.page == 0):
            st.session_state.page -= 1
    with page_column:
        st.write(f"Page {st.session_state.page + 1} of {total_pages}")
    with next_column:
        if st.button("Next", disabled=st.session_state.page == total_pages - 1):
            st.session_state.page += 1

    start = st.session_state.page * FRAMES_PER_PAGE
    page_paths = frame_paths[start : start + FRAMES_PER_PAGE]
    for row_start in range(0, len(page_paths), 6):
        for column, frame_path in zip(st.columns(6), page_paths[row_start : row_start + 6]):
            with column:
                st.image(str(frame_path), use_container_width=True)
                is_removed = st.checkbox(
                    "Remove", key=f"remove_{frame_path.name}", value=frame_path in st.session_state.removed
                )
                if is_removed:
                    st.session_state.removed.add(frame_path)
                else:
                    st.session_state.removed.discard(frame_path)

    kept_paths = [path for path in frame_paths if path not in st.session_state.removed]
    st.subheader("Save selected frames")
    subfolder_input = st.text_input(
        "Folder name", value=st.session_state.default_folder_name, key="folder_name"
    )
    target_name = safe_subfolder_name(subfolder_input)
    if subfolder_input and target_name is None:
        st.error("Use 1-80 letters, numbers, hyphens, or underscores; begin with a letter or number.")

    save_column, cancel_column = st.columns(2)
    with save_column:
        if st.button("Save frames", type="primary", disabled=not kept_paths):
            target_dir = IMAGES_DIR / target_name if target_name else None
            if target_dir is None:
                st.error("Enter a valid folder name.")
            elif target_dir.exists():
                st.error("This folder already exists. Choose another name.")
            else:
                target_dir.mkdir(parents=True)
                for index, frame_path in enumerate(kept_paths):
                    output_name = f"{st.session_state.video_stem}_frame_{index:05d}.jpg"
                    shutil.copy2(frame_path, target_dir / output_name)
                saved_count = len(kept_paths)
                reset_frames(clear_upload=True)
                st.session_state.success_message = f"Saved {saved_count} frames to images/{target_name}."
                st.rerun()
    with cancel_column:
        if st.button(
            "Discard frames",
            on_click=lambda: reset_frames_and_folder_name(clear_upload=True),
        ):
            st.rerun()


def render_dataset_export() -> None:
    st.title("Dataset export")
    st.caption("Download annotated images from Label Studio Local Files storages as a YOLO dataset.")
    try:
        storages = load_storages()
    except (RuntimeError, requests.RequestException) as error:
        st.info(str(error))
        return
    if not storages:
        st.warning("No Local Files storages were found.")
        return
    tasks_by_project: dict[int, list[dict[str, object]]] = {}
    for storage in storages:
        project = storage.get("project")
        if isinstance(project, int) and project not in tasks_by_project:
            tasks_by_project[project] = load_project_tasks(project)
    tasks_by_storage: dict[int, list[dict[str, object]]] = defaultdict(list)
    storages_by_project: dict[int, list[dict[str, object]]] = defaultdict(list)
    for storage in storages:
        if isinstance(storage.get("project"), int):
            storages_by_project[storage["project"]].append(storage)
    for project, tasks in tasks_by_project.items():
        project_storages = storages_by_project[project]
        for task in tasks:
            matches = [storage for storage in project_storages if task_belongs_to_storage(task, storage)]
            if not matches and len(project_storages) == 1:
                matches = project_storages
            for storage in matches:
                tasks_by_storage[int(storage["id"])].append(task)
    rows = []
    for storage in storages:
        storage_id = int(storage["id"])
        storage_tasks = tasks_by_storage.get(storage_id, [])
        rows.append({"Storage": str(storage.get("title") or f"Storage {storage_id}"), "Type": str(storage.get("storage_type", "unknown")).upper(), "Images": len(storage_tasks), "Annotated": sum(completed_annotation(task) is not None for task in storage_tasks)})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    labels = {int(storage["id"]): str(storage.get("title") or f"Storage {storage['id']}") for storage in storages}
    selected_ids = st.multiselect("Local Files storages", list(labels), default=list(labels), format_func=labels.get)
    train, valid, test = st.columns(3)
    with train:
        train_percent = st.number_input("Train (%)", 0, 100, 70)
    with valid:
        valid_percent = st.number_input("Validation (%)", 0, 100, 20)
    with test:
        test_percent = st.number_input("Test (%)", 0, 100, 10)
    if train_percent + valid_percent + test_percent != 100:
        st.error("Train, validation, and test percentages must total 100.")
        return
    selected_storages = [storage for storage in storages if int(storage["id"]) in selected_ids]
    annotated_count = sum(completed_annotation(task) is not None for storage in selected_storages for task in tasks_by_storage.get(int(storage["id"]), []))
    st.caption(f"{annotated_count} annotated images will be exported. Unannotated tasks are skipped.")
    if st.button("Build YOLO dataset", type="primary", disabled=not selected_storages or not annotated_count):
        try:
            archive, counts = build_yolo_archive(selected_storages, tasks_by_storage, (train_percent, valid_percent, test_percent))
            st.session_state.yolo_archive = archive
            st.session_state.yolo_counts = dict(counts)
        except (RuntimeError, requests.RequestException) as error:
            st.error(f"Dataset export failed: {error}")
    if archive := st.session_state.get("yolo_archive"):
        counts = st.session_state.get("yolo_counts", {})
        st.download_button("Download YOLO dataset (.zip)", archive, "yolo-dataset.zip", "application/zip", type="primary")
        st.caption(f"Ready: {counts.get('train', 0)} train, {counts.get('valid', 0)} validation, {counts.get('test', 0)} test images.")


def main() -> None:
    st.set_page_config(page_title="Dataset Hub", layout="wide")
    st.sidebar.header("Dataset Hub")
    frame_tab, export_tab = st.tabs(["Frame extraction", "Dataset export"])
    with frame_tab:
        render_frame_extractor()
    with export_tab:
        render_dataset_export()


if __name__ == "__main__":
    main()