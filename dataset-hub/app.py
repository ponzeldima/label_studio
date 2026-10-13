"""Dataset Hub: frame extraction and Label Studio dataset export."""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import tempfile
import uuid
import zipfile
import os
from collections import defaultdict
from dataclasses import dataclass
from math import ceil, floor
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Callable
from urllib.parse import unquote, urlparse

import cv2
import httpx
import psycopg
import requests
import streamlit as st
from label_studio_sdk import LabelStudio
from label_studio_sdk.core.api_error import ApiError
from PIL import Image
from streamlit_cropper import st_cropper

STAGING_ROOT = Path("/tmp/frame-staging")
UPLOAD_ROOT = Path("/tmp/frame-uploads")
EXPORT_ROOT = Path("/tmp/dataset-export")
ARCHIVE_SCHEMA_VERSION = 1
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 10 * 1024 * 1024 * 1024
FRAMES_PER_PAGE = 24
MAX_SUBFOLDER_LENGTH = 80
PREVIEW_WIDTH = 360
DATA_SOURCE = os.getenv("DATA_SOURCE", "api").strip().lower()
if DATA_SOURCE not in {"api", "postgres"}:
    DATA_SOURCE = "api"
CLASS_IDS = {"fpv": 0, "bird": 1}
# Errors that can surface from any Label Studio API call (SDK-raised or raw httpx/requests).
LS_API_ERRORS = (RuntimeError, requests.RequestException, httpx.HTTPError, ApiError)


@dataclass(frozen=True)
class LabelStudioInstance:
    key: str
    label: str
    data_dir: Path
    url: str
    api_token: str
    project_id: int
    postgres_host: str
    postgres_db: str
    postgres_user: str
    postgres_password: str
    postgres_port: int


LABEL_STUDIO_INSTANCES = {
    "main": LabelStudioInstance(
        key="main",
        label="Main Label Studio",
        data_dir=Path(os.getenv("MAIN_DATA_DIR", "/data/main")),
        url=os.getenv("LABEL_STUDIO_URL", "http://app:8080"),
        api_token=os.getenv("LABEL_STUDIO_API_TOKEN", ""),
        project_id=int(os.getenv("LABEL_STUDIO_PROJECT_ID", "1")),
        postgres_host=os.getenv("POSTGRES_HOST", "db"),
        postgres_db=os.getenv("POSTGRES_DB", "postgres"),
        postgres_user=os.getenv("POSTGRES_USER", "postgres"),
        postgres_password=os.getenv("POSTGRES_PASSWORD", ""),
        postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
    ),
    "freelance": LabelStudioInstance(
        key="freelance",
        label="Freelancer Label Studio",
        data_dir=Path(os.getenv("FREELANCE_DATA_DIR", "/data/freelance")),
        url=os.getenv("FREELANCE_LABEL_STUDIO_URL", "http://app-freelance:8080"),
        api_token=os.getenv("FREELANCE_LABEL_STUDIO_API_TOKEN", ""),
        project_id=int(os.getenv("FREELANCE_LABEL_STUDIO_PROJECT_ID", "1")),
        postgres_host=os.getenv("FREELANCE_POSTGRES_HOST", "db-freelance"),
        postgres_db=os.getenv("FREELANCE_POSTGRES_DB", "postgres"),
        postgres_user=os.getenv("FREELANCE_POSTGRES_USER", "postgres"),
        postgres_password=os.getenv("FREELANCE_POSTGRES_PASSWORD", ""),
        postgres_port=int(os.getenv("FREELANCE_POSTGRES_PORT", "5432")),
    ),
}


def selected_instance_key() -> str:
    return st.session_state.get("label_studio_instance", "main")


def selected_instance() -> LabelStudioInstance:
    return LABEL_STUDIO_INSTANCES[selected_instance_key()]


def configured_label_studio(instance: LabelStudioInstance | None = None) -> tuple[str, str]:
    target = instance or selected_instance()
    return target.url.rstrip("/"), target.api_token


def configured_postgres(instance: LabelStudioInstance | None = None) -> tuple[str, str, str, str, int]:
    target = instance or selected_instance()
    return (
        target.postgres_host,
        target.postgres_db,
        target.postgres_user,
        target.postgres_password,
        target.postgres_port,
    )


def postgres_connection(instance: LabelStudioInstance | None = None):
    host, dbname, user, password, port = configured_postgres(instance)
    return psycopg.connect(
        host=host,
        dbname=dbname,
        user=user,
        password=password,
        port=port,
        connect_timeout=5,
    )


def label_studio_client(instance: LabelStudioInstance | None = None) -> LabelStudio:
    target = instance or selected_instance()
    base_url, token = configured_label_studio(target)
    if not token:
        raise RuntimeError(f"Set the API token for {target.label} in .env and restart dataset-hub.")
    return LabelStudio(base_url=base_url, api_key=token)


def sdk_dict(item: object) -> dict[str, object]:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "dict"):
        return item.dict()
    return {}


# ============================================================
# TAGS: apply/remove image-level tags without affecting bbox annotation state
# ============================================================
TAGS_FROM_NAME = "tags"
TAGS_TO_NAME = "image"
# Non-greedy match keeps this scoped to the single <Choices name="tags"> block.
TAGS_CHOICES_RE = re.compile(r'(<Choices\b[^>]*\bname="tags"[^>]*>)(.*?)(</Choices>)', re.DOTALL)
CHOICE_VALUE_RE = re.compile(r'<Choice\b[^>]*\bvalue="([^"]*)"')


def load_label_config() -> str:
    instance = selected_instance()
    if DATA_SOURCE == "postgres":
        with postgres_connection(instance) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT label_config FROM project WHERE id = %s", (instance.project_id,))
                row = cur.fetchone()
        if not row:
            raise RuntimeError(f"Project {instance.project_id} not found in {instance.label}.")
        return row[0] or ""
    project = sdk_dict(label_studio_client(instance).projects.get(instance.project_id))
    return str(project.get("label_config") or "")


def available_tags(label_config: str) -> list[str]:
    match = TAGS_CHOICES_RE.search(label_config)
    if not match:
        return []
    return CHOICE_VALUE_RE.findall(match.group(2))


def update_label_config_tags(label_config: str, tags: list[str]) -> str:
    match = TAGS_CHOICES_RE.search(label_config)
    if not match:
        raise RuntimeError('No <Choices name="tags"> block found in the project label config.')
    body = "\n" + "".join(f'        <Choice value="{tag}"/>\n' for tag in tags) + "    "
    return label_config[: match.start()] + match.group(1) + body + match.group(3) + label_config[match.end():]


def save_label_config(new_config: str) -> None:
    instance = selected_instance()
    label_studio_client(instance).projects.update(instance.project_id, label_config=new_config)


def find_tags_result(result: list[dict[str, object]]) -> dict[str, object] | None:
    for item in result:
        if isinstance(item, dict) and item.get("from_name") == TAGS_FROM_NAME and item.get("type") == "choices":
            return item
    return None


def result_tags(result: list[dict[str, object]]) -> list[str]:
    item = find_tags_result(result)
    if not item:
        return []
    value = item.get("value", {})
    choices = value.get("choices", []) if isinstance(value, dict) else []
    return sorted(choices) if isinstance(choices, list) else []


def apply_tag_diff(
    result: list[dict[str, object]], add: set[str], remove: set[str]
) -> tuple[list[dict[str, object]], bool]:
    """Return (new_result, changed); new_result is a fresh list, safe to send back to Label Studio."""
    result = [dict(item) for item in result]
    tags_item = find_tags_result(result)
    current = set(result_tags(result))
    updated = (current | add) - remove
    if updated == current:
        return result, False
    if not updated:
        return [item for item in result if item is not tags_item], True
    if tags_item is None:
        result.append({"from_name": TAGS_FROM_NAME, "to_name": TAGS_TO_NAME, "type": "choices", "value": {"choices": sorted(updated)}})
    else:
        tags_item["value"] = {"choices": sorted(updated)}
    return result, True


def task_current_tags(task: dict[str, object]) -> list[str]:
    annotation = completed_annotation(task)
    return result_tags(annotation.get("result", [])) if annotation else []


def tasks_grouped_by_storage(
    storages: list[dict[str, object]], tasks: list[dict[str, object]]
) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for task in tasks:
        matches = [storage for storage in storages if task_belongs_to_storage(task, storage)]
        if not matches and len(storages) == 1:
            matches = storages
        for storage in matches:
            grouped[int(storage["id"])].append(task)
    return grouped


def tag_counts(tasks: list[dict[str, object]], tags: list[str]) -> dict[str, int]:
    counts = {tag: 0 for tag in tags}
    for task in tasks:
        for tag in task_current_tags(task):
            if tag in counts:
                counts[tag] += 1
    return counts


def apply_tags_to_tasks(
    tasks: list[dict[str, object]],
    add: set[str],
    remove: set[str],
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[dict[str, int], list[str]]:
    # Only annotated tasks are touched; tasks without an annotation yet are skipped
    # (no drafts) so tagging never gets in the way of the normal prediction-to-annotation flow.
    counts: dict[str, int] = defaultdict(int)
    errors: list[str] = []
    client = label_studio_client()
    total = len(tasks)
    for index, task in enumerate(tasks):
        task_id = int(task["id"])
        try:
            annotation = completed_annotation(task)
            if not annotation:
                counts["skipped"] += 1
            else:
                new_result, changed = apply_tag_diff(annotation.get("result", []), add, remove)
                if changed:
                    client.annotations.update(int(annotation["id"]), result=new_result)
                counts["updated" if changed else "unchanged"] += 1
        except LS_API_ERRORS as error:
            # Keep processing the rest of the storage; one flaky task shouldn't abort the whole batch.
            counts["failed"] += 1
            if len(errors) < 5:
                errors.append(f"Task {task_id}: {error}")
        if progress_callback:
            progress_callback(index + 1, total, str(task.get("id")))
    return dict(counts), errors


@st.cache_data(ttl=300, show_spinner=False)
def load_storages(instance_key: str) -> list[dict[str, object]]:
    instance = LABEL_STUDIO_INSTANCES[instance_key]
    if DATA_SOURCE == "postgres":
        return load_storages_postgres(instance)
    client = label_studio_client(instance)
    storages: list[dict[str, object]] = []
    for storage in client.import_storage.local.list(project=instance.project_id):
        storage_data = sdk_dict(storage)
        storage_data["storage_type"] = "localfiles"
        storages.append(storage_data)
    return storages


@st.cache_data(ttl=300, show_spinner=False)
def load_project_tasks(instance_key: str) -> list[dict[str, object]]:
    instance = LABEL_STUDIO_INSTANCES[instance_key]
    if DATA_SOURCE == "postgres":
        return load_project_tasks_postgres(instance, instance.project_id)
    return [
        sdk_dict(task)
        for task in label_studio_client(instance).tasks.list(project=instance.project_id, page_size=1000)
    ]


def load_storages_postgres(instance: LabelStudioInstance) -> list[dict[str, object]]:
    query = """
        SELECT
            localfilesmixin_ptr_id AS id,
            project_id,
            title,
            status,
            meta,
            created_at,
            last_sync_count
        FROM io_storages_localfilesimportstorage
        WHERE project_id = %s
        ORDER BY id
    """
    with postgres_connection(instance) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (instance.project_id,))
            rows = cur.fetchall()
    storages: list[dict[str, object]] = []
    for row in rows:
        storage = {
            "id": int(row[0]),
            "project_id": row[1],
            "title": row[2],
            "status": row[3],
            "meta": row[4] or {},
            "created_at": row[5],
            "last_sync_count": row[6],
            "storage_type": "localfiles",
        }
        storages.append(storage)
    return storages


def load_project_tasks_postgres(
    instance: LabelStudioInstance, project_id: int
) -> list[dict[str, object]]:
    query = """
        SELECT
            t.id,
            t.project_id,
            t.data,
            t.meta,
            l.storage_id,
            tc.id AS completion_id,
            tc.result,
            tc.was_cancelled,
            tc.updated_at
        FROM task t
        LEFT JOIN io_storages_localfilesimportstoragelink l
            ON l.task_id = t.id
        LEFT JOIN LATERAL (
            SELECT tc.*
            FROM task_completion tc
            WHERE tc.task_id = t.id
              AND tc.was_cancelled = false
            ORDER BY tc.updated_at DESC, tc.id DESC
            LIMIT 1
        ) tc ON true
        WHERE t.project_id = %s
        ORDER BY t.id
    """
    with postgres_connection(instance) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (project_id,))
            rows = cur.fetchall()

    tasks_by_id: dict[int, dict[str, object]] = {}
    for row in rows:
        task_id, project_id_value, data, meta, storage_id, completion_id, result, was_cancelled, updated_at = row
        task = tasks_by_id.setdefault(
            int(task_id),
            {
                "id": int(task_id),
                "project_id": int(project_id_value),
                "data": data or {},
                "meta": meta or {},
                "annotations": [],
            },
        )
        if storage_id is not None:
            task["meta"] = dict(task["meta"])
            task["meta"]["source_storage_id"] = int(storage_id)
        if result is not None and not was_cancelled:
            task["annotations"] = [{
                "id": completion_id,
                "result": result,
                "was_cancelled": bool(was_cancelled),
                "updated_at": updated_at,
            }]
    return list(tasks_by_id.values())


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
        relative_path = normalized.split("/data/", 1)[-1].lstrip("/")
        if image_path.startswith(relative_path + "/") or image_path == relative_path:
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
    if not isinstance(data, dict):
        return None

    for key in ("image", "$undefined$"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value

    if len(data) == 1:
        value = next(iter(data.values()))
        if isinstance(value, str) and value:
            return value
    return None


def image_bytes(task: dict[str, object]) -> tuple[bytes, str]:
    image = task_image_reference(task)
    if not image:
        raise RuntimeError("Task does not contain an image field.")
    image_path = unquote(image.split("?d=", 1)[1]) if "?d=" in image else ""
    local_path = selected_instance().data_dir / image_path
    if image_path and local_path.is_file():
        return local_path.read_bytes(), local_path.suffix or ".jpg"
    base_url, token = configured_label_studio()
    url = image if image.startswith(("http://", "https://")) else f"{base_url}{image}"
    response = requests.get(url, headers={"Authorization": f"Token {token}"}, timeout=120)
    response.raise_for_status()
    return response.content, Path(urlparse(url).path).suffix or ".jpg"


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "storage"


def task_source_filename(task: dict[str, object]) -> str:
    reference = task_image_reference(task) or ""
    image_path = reference.split("?d=", 1)[1] if "?d=" in reference else urlparse(reference).path
    filename = Path(unquote(image_path.split("?", 1)[0])).name
    return filename or f"task_{task.get('id', 'unknown')}"


def storage_archive_name(storage: dict[str, object]) -> str:
    return safe_filename(str(storage.get("title") or f"storage-{storage['id']}"))


def task_storage_relative_path(task: dict[str, object], storage: dict[str, object]) -> Path:
    reference = task_image_reference(task)
    if not reference:
        raise RuntimeError(f"Task {task.get('id')} does not contain an image field.")
    image_path = unquote(reference.split("?d=", 1)[1]) if "?d=" in reference else urlparse(reference).path
    image_path = image_path.split("?", 1)[0].lstrip("/")
    if not image_path:
        raise RuntimeError(f"Task {task.get('id')} has an empty image path.")
    storage_path = str(storage.get("path") or storage.get("prefix") or "").rstrip("/")
    marker = "/data/"
    storage_relative = storage_path.split(marker, 1)[1].strip("/") if marker in storage_path else ""
    if storage_relative and image_path.startswith(storage_relative + "/"):
        image_path = image_path[len(storage_relative) + 1:]
    path = Path(image_path)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"Task {task.get('id')} has an unsafe image path.")
    storage_name = storage_archive_name(storage)
    for prefix in (("images", storage_name), (storage_name,)):
        if path.parts[:len(prefix)] == prefix:
            path = Path(*path.parts[len(prefix):])
            break
    if path == Path("."):
        raise RuntimeError(f"Task {task.get('id')} has an empty storage-relative image path.")
    return path


def archive_member_path(storage_name: str, relative_path: Path) -> str:
    return f"{storage_name}/images/{relative_path.as_posix()}"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = floor(position)
    upper = ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "median": percentile(values, 0.5),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
    }


def rectangle_objects(annotation: dict[str, object]) -> list[tuple[str, float]]:
    objects: list[tuple[str, float]] = []
    results = annotation.get("result", [])
    if not isinstance(results, list):
        return objects
    for result in results:
        if not isinstance(result, dict) or result.get("type") != "rectanglelabels":
            continue
        value = result.get("value", {})
        if not isinstance(value, dict):
            continue
        labels = value.get("rectanglelabels", [])
        if not isinstance(labels, list):
            continue
        try:
            area_fraction = (float(value["width"]) * float(value["height"])) / 10_000
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 < area_fraction <= 1:
            continue
        for label in labels:
            if isinstance(label, str):
                objects.append((label, area_fraction))
    return objects


def storage_archive_summary(
    storage_name: str,
    entries: list[tuple[Path, dict[str, object], dict[str, object] | None, bytes]],
) -> tuple[dict[str, object], str, str, str]:
    tag_image_counts: dict[str, int] = defaultdict(int)
    object_counts: dict[str, int] = defaultdict(int)
    object_areas: list[float] = []
    image_widths: list[float] = []
    image_heights: list[float] = []
    image_bytes: list[float] = []
    aspect_ratios: list[float] = []
    image_rows: list[dict[str, object]] = []
    for relative_path, task, annotation, data in entries:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
        image_widths.append(width)
        image_heights.append(height)
        image_bytes.append(len(data))
        if height:
            aspect_ratios.append(width / height)
        tags = task_current_tags(task) if annotation else []
        for tag in tags:
            tag_image_counts[tag] += 1
        objects = rectangle_objects(annotation) if annotation else []
        for label, area_fraction in objects:
            object_counts[label] += 1
            object_areas.append(area_fraction)
        image_rows.append(
            {
                "image_path": relative_path.as_posix(),
                "width": width,
                "height": height,
                "bytes": len(data),
                "aspect_ratio": width / height if height else None,
                "tag_count": len(tags),
                "object_count": len(objects),
            }
        )
    summary: dict[str, object] = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "storage_name": storage_name,
        "images": {
            "count": len(entries),
            "width_pixels": distribution(image_widths),
            "height_pixels": distribution(image_heights),
            "bytes": distribution(image_bytes),
            "aspect_ratio": distribution(aspect_ratios),
        },
        "tags": {"image_count_by_tag": dict(sorted(tag_image_counts.items()))},
        "objects": {
            "count": sum(object_counts.values()),
            "count_by_label": dict(sorted(object_counts.items())),
            "bounding_box_area_fraction_of_image": distribution(object_areas),
        },
    }
    return (
        summary,
        analytics_csv([{"tag": tag, "image_count": count} for tag, count in sorted(tag_image_counts.items())]),
        analytics_csv([{"label": label, "object_count": count} for label, count in sorted(object_counts.items())]),
        analytics_csv(image_rows),
    )


def build_storage_archive(
    selected_storages: list[dict[str, object]],
    tasks_by_storage: dict[int, list[dict[str, object]]],
    include_unannotated: bool,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[Path, int]:
    export_items: list[tuple[dict[str, object], Path, dict[str, object], dict[str, object] | None]] = []
    for storage in selected_storages:
        for task in tasks_by_storage.get(int(storage["id"]), []):
            annotation = completed_annotation(task)
            if annotation is None and not include_unannotated:
                continue
            export_items.append((storage, task_storage_relative_path(task, storage), task, annotation))
    if not export_items:
        raise ValueError("No images match the export selection.")
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    archive_fd, archive_path_str = tempfile.mkstemp(suffix=".zip", dir=EXPORT_ROOT)
    os.close(archive_fd)
    archive_path = Path(archive_path_str)
    archive_path.chmod(0o644)
    entries_by_storage: dict[int, list[tuple[Path, dict[str, object], dict[str, object] | None, bytes]]] = defaultdict(list)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for index, (storage, relative_path, task, annotation) in enumerate(export_items, start=1):
            data, _ = image_bytes(task)
            zip_file.writestr(archive_member_path(storage_archive_name(storage), relative_path), data)
            entries_by_storage[int(storage["id"])].append((relative_path, task, annotation, data))
            if progress_callback:
                progress_callback(index, len(export_items), relative_path.name)
        archive_manifest: dict[str, object] = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "storage_count": len(selected_storages),
            "storages": [],
        }
        for storage in selected_storages:
            storage_id = int(storage["id"])
            storage_name = storage_archive_name(storage)
            entries = entries_by_storage[storage_id]
            native_annotations = []
            image_tags = {}
            coco_images = []
            coco_annotations = []
            category_names: set[str] = set()
            annotation_id = 1
            for image_id, (relative_path, task, annotation, _) in enumerate(entries, start=1):
                image_tags[relative_path.as_posix()] = task_current_tags(task) if annotation else []
                native_annotations.append(
                    {
                        "image_path": relative_path.as_posix(),
                        "result": annotation.get("result", []) if annotation else [],
                    }
                )
                image_data, _ = image_bytes(task)
                with Image.open(io.BytesIO(image_data)) as image:
                    width, height = image.size
                coco_images.append({"id": image_id, "file_name": relative_path.as_posix(), "width": width, "height": height})
                for label, _ in rectangle_objects(annotation) if annotation else []:
                    category_names.add(label)
                for result in (annotation or {}).get("result", []):
                    if not isinstance(result, dict) or result.get("type") != "rectanglelabels":
                        continue
                    value = result.get("value", {})
                    if not isinstance(value, dict):
                        continue
                    labels = value.get("rectanglelabels", [])
                    if not isinstance(labels, list):
                        continue
                    try:
                        x = float(value["x"]) * width / 100
                        y = float(value["y"]) * height / 100
                        box_width = float(value["width"]) * width / 100
                        box_height = float(value["height"]) * height / 100
                    except (KeyError, TypeError, ValueError):
                        continue
                    for label in labels:
                        if isinstance(label, str):
                            coco_annotations.append(
                                {"id": annotation_id, "image_id": image_id, "category_name": label, "bbox": [x, y, box_width, box_height], "area": box_width * box_height, "iscrowd": 0}
                            )
                            annotation_id += 1
            categories = [{"id": index, "name": name} for index, name in enumerate(sorted(category_names), start=1)]
            category_ids = {category["name"]: category["id"] for category in categories}
            for item in coco_annotations:
                item["category_id"] = category_ids[str(item.pop("category_name"))]
            summary, tags_csv, objects_csv, images_csv = storage_archive_summary(storage_name, entries)
            root = f"{storage_name}/"
            zip_file.writestr(root + "annotations/label-studio.json", json.dumps(native_annotations, indent=2))
            zip_file.writestr(root + "annotations/coco.json", json.dumps({"images": coco_images, "annotations": coco_annotations, "categories": categories}, indent=2))
            zip_file.writestr(root + "tags/image-tags.json", json.dumps(image_tags, indent=2))
            zip_file.writestr(root + "summary/summary.json", json.dumps(summary, indent=2))
            zip_file.writestr(root + "summary/tags.csv", tags_csv)
            zip_file.writestr(root + "summary/objects.csv", objects_csv)
            zip_file.writestr(root + "summary/images.csv", images_csv)
            archive_manifest["storages"].append({"name": storage_name, "image_count": len(entries)})
        zip_file.writestr("manifest.json", json.dumps(archive_manifest, indent=2))
    return archive_path, len(export_items)


def safe_zip_members(zip_file: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = [member for member in zip_file.infolist() if not member.is_dir()]
    total_size = sum(member.file_size for member in members)
    if total_size > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise ValueError("Archive exceeds the 10 GB uncompressed import limit.")
    for member in members:
        path = Path(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Archive contains an unsafe path: {member.filename}")
    return members


def archive_storage_root(archive: zipfile.ZipFile) -> str:
    members = safe_zip_members(archive)
    try:
        manifest = json.loads(archive.read("manifest.json"))
    except KeyError:
        roots = {Path(member.filename).parts[0] for member in members}
        if len(roots) != 1:
            raise ValueError(
                "Archive without manifest.json must contain exactly one top-level storage folder."
            )
        return roots.pop()
    except json.JSONDecodeError as error:
        raise ValueError("Archive has an invalid manifest.json.") from error

    storages = manifest.get("storages", []) if isinstance(manifest, dict) else []
    if not isinstance(storages, list) or len(storages) != 1:
        raise ValueError("Import accepts an archive containing exactly one storage.")
    storage_name = storages[0].get("name") if isinstance(storages[0], dict) else None
    if (
        not isinstance(storage_name, str)
        or not storage_name
        or safe_filename(storage_name) != storage_name
    ):
        raise ValueError("Archive manifest has no valid storage name.")
    return storage_name


def imported_archive_data(uploaded_file: object) -> tuple[str, list[dict[str, object]], dict[str, list[str]], zipfile.ZipFile]:
    if not hasattr(uploaded_file, "read"):
        raise ValueError("Choose a ZIP archive to import.")
    archive = zipfile.ZipFile(uploaded_file)
    try:
        storage_name = archive_storage_root(archive)
        annotations = json.loads(archive.read(f"{storage_name}/annotations/label-studio.json"))
        tags = json.loads(archive.read(f"{storage_name}/tags/image-tags.json"))
    except (KeyError, json.JSONDecodeError) as error:
        archive.close()
        raise ValueError("Archive is missing valid Label Studio annotations or tags.") from error
    except ValueError:
        archive.close()
        raise
    if not isinstance(annotations, list) or not isinstance(tags, dict):
        archive.close()
        raise ValueError("Archive annotations or tags have an invalid format.")
    return storage_name, annotations, {str(key): value for key, value in tags.items() if isinstance(value, list)}, archive


def import_storage_archive(uploaded_file: object, new_storage_name: str) -> dict[str, int]:
    new_storage_name = new_storage_name.strip()
    if safe_filename(new_storage_name) != new_storage_name:
        raise ValueError("Storage name may contain only letters, numbers, underscores, and hyphens.")
    load_storages.clear()
    existing_storages = load_storages(selected_instance_key())
    if any(str(storage.get("title")) == new_storage_name for storage in existing_storages):
        raise ValueError(f'A storage named "{new_storage_name}" already exists.')
    source_storage_name, annotations, tags_by_path, archive = imported_archive_data(uploaded_file)
    try:
        known_tags = set(available_tags(load_label_config()))
        archive_tags = {tag for tags in tags_by_path.values() for tag in tags if isinstance(tag, str)}
        missing_tags = sorted(archive_tags - known_tags)
        if missing_tags:
            raise ValueError("Target project is missing archive tags: " + ", ".join(missing_tags))
        destination = selected_instance().data_dir / "images" / new_storage_name
        if destination.exists():
            raise ValueError(f'Local directory for "{new_storage_name}" already exists.')
        if not any(
            member.filename.startswith(f"{source_storage_name}/images/")
            for member in safe_zip_members(archive)
        ):
            raise ValueError("Archive contains no images.")
        client = label_studio_client()
        destination.mkdir(parents=True)
        storage_id: int | None = None
        try:
            for member in safe_zip_members(archive):
                prefix = f"{source_storage_name}/images/"
                if member.filename.startswith(prefix):
                    relative_path = Path(member.filename[len(prefix):])
                    target = destination / relative_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
            storage = client.import_storage.local.create(
                project=selected_instance().project_id,
                title=new_storage_name,
                path=f"/label-studio/data/images/{new_storage_name}",
                use_blob_urls=True,
            )
            storage_data = sdk_dict(storage)
            storage_id = int(storage_data["id"])
            client.import_storage.local.sync(storage_id)
        except (OSError,) + LS_API_ERRORS:
            if storage_id is not None:
                client.import_storage.local.delete(storage_id)
            shutil.rmtree(destination)
            raise
        load_storages.clear()
        load_project_tasks.clear()
        synced_tasks = load_project_tasks(selected_instance_key())
        imported_tasks = [
            task for task in synced_tasks
            if task_belongs_to_storage(task, {"id": storage_id, "title": new_storage_name, "path": f"/label-studio/data/images/{new_storage_name}"})
        ]
        tasks_by_path = {}
        for task in imported_tasks:
            reference = task_image_reference(task) or ""
            image_path = unquote(reference.split("?d=", 1)[1]) if "?d=" in reference else reference
            marker = f"images/{new_storage_name}/"
            if marker in image_path:
                tasks_by_path[image_path.split(marker, 1)[1]] = task
        unmatched_paths = sorted(set(tags_by_path) - set(tasks_by_path))
        if unmatched_paths:
            client.import_storage.local.delete(storage_id)
            shutil.rmtree(destination)
            load_storages.clear()
            load_project_tasks.clear()
            raise RuntimeError(
                f"Storage sync did not create tasks for {len(unmatched_paths)} archived image(s), "
                f"including {unmatched_paths[0]}."
            )
        annotation_by_path = {
            item.get("image_path"): item.get("result", [])
            for item in annotations
            if isinstance(item, dict) and isinstance(item.get("image_path"), str) and isinstance(item.get("result"), list)
        }
        created = 0
        skipped = 0
        for image_path, task in tasks_by_path.items():
            result = annotation_by_path.get(image_path)
            if result is None:
                skipped += 1
                continue
            client.annotations.create(int(task["id"]), result=result, project=selected_instance().project_id)
            created += 1
        return {"images": len(tasks_by_path), "annotations_created": created, "without_annotations": skipped}
    finally:
        archive.close()


def natural_sort_key(value: str) -> list[tuple[int, int | str]]:
    return [
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", value)
    ]


def annotated_tasks_in_filename_order(
    tasks: list[dict[str, object]],
) -> list[tuple[dict[str, object], dict[str, object]]]:
    annotated_tasks = [
        (task, annotation)
        for task in tasks
        if (annotation := completed_annotation(task)) is not None
    ]
    return sorted(
        annotated_tasks,
        key=lambda item: (natural_sort_key(task_source_filename(item[0])), int(item[0]["id"])),
    )


def allocate_by_percentage(total: int, percentages: tuple[int, ...]) -> tuple[int, ...]:
    if sum(percentages) != 100:
        raise ValueError("Split percentages must total 100.")
    raw_counts = [total * percentage / 100 for percentage in percentages]
    counts = [int(count) for count in raw_counts]
    remaining = total - sum(counts)
    remainders = sorted(
        range(len(percentages)),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in remainders[:remaining]:
        counts[index] += 1
    return tuple(counts)


def sequence_split_tasks(
    tasks: list[dict[str, object]], splits: tuple[int, int, int, int, int]
) -> dict[str, list[tuple[dict[str, object], dict[str, object]]]]:
    train_percent, train_buffer_percent, valid_percent, valid_buffer_percent, test_percent = splits
    ordered_tasks = annotated_tasks_in_filename_order(tasks)
    train_buffer_count = round(len(ordered_tasks) * train_buffer_percent / 100)
    valid_buffer_count = round(len(ordered_tasks) * valid_buffer_percent / 100)
    retained_count = len(ordered_tasks) - train_buffer_count - valid_buffer_count
    if ordered_tasks and retained_count <= 0:
        raise ValueError("The buffer percentages leave no images to export.")
    train_count, valid_count, test_count = allocate_by_percentage(
        retained_count, (train_percent, valid_percent, test_percent)
    )
    train_end = train_count
    train_buffer_end = train_end + train_buffer_count
    valid_end = train_buffer_end + valid_count
    valid_buffer_end = valid_end + valid_buffer_count
    return {
        "train": ordered_tasks[:train_end],
        "skip_train_to_valid": ordered_tasks[train_end:train_buffer_end],
        "valid": ordered_tasks[train_buffer_end:valid_end],
        "skip_valid_to_test": ordered_tasks[valid_end:valid_buffer_end],
        "test": ordered_tasks[valid_buffer_end : valid_buffer_end + test_count],
    }


def random_split_tasks(
    selected_storages: list[dict[str, object]],
    tasks_by_storage: dict[int, list[dict[str, object]]],
    splits: tuple[int, int, int],
) -> dict[int, dict[str, list[tuple[dict[str, object], dict[str, object]]]]]:
    random_tasks: list[tuple[int, dict[str, object], dict[str, object]]] = []
    for storage in selected_storages:
        storage_id = int(storage["id"])
        for task in tasks_by_storage.get(storage_id, []):
            if annotation := completed_annotation(task):
                random_tasks.append((storage_id, task, annotation))
    Random(42).shuffle(random_tasks)
    train_end = round(len(random_tasks) * splits[0] / 100)
    valid_end = train_end + round(len(random_tasks) * splits[1] / 100)
    split_names = (
        ["train"] * train_end
        + ["valid"] * (valid_end - train_end)
        + ["test"] * (len(random_tasks) - valid_end)
    )
    plans: dict[int, dict[str, list[tuple[dict[str, object], dict[str, object]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (storage_id, task, annotation), split in zip(random_tasks, split_names):
        plans[storage_id][split].append((task, annotation))
    return plans


def annotation_has_objects(annotation: dict[str, object]) -> bool:
    results = annotation.get("result", [])
    return isinstance(results, list) and any(
        isinstance(result, dict) and result.get("type") == "rectanglelabels"
        for result in results
    )


def annotation_labels(annotation: dict[str, object]) -> list[str]:
    labels: list[str] = []
    results = annotation.get("result", [])
    if not isinstance(results, list):
        return labels
    for result in results:
        if not isinstance(result, dict) or result.get("type") != "rectanglelabels":
            continue
        value = result.get("value", {})
        rectangle_labels = value.get("rectanglelabels", []) if isinstance(value, dict) else []
        if isinstance(rectangle_labels, list):
            labels.extend(label for label in rectangle_labels if isinstance(label, str))
    return labels


def analytics_row(
    scope: str,
    tasks: list[dict[str, object]],
    tags: list[str],
) -> dict[str, int | str]:
    annotations = [annotation for task in tasks if (annotation := completed_annotation(task))]
    labels = [label for annotation in annotations for label in annotation_labels(annotation)]
    row: dict[str, int | str] = {
        "Scope": scope,
        "Images count": len(annotations),
        "Empty images": sum(not annotation_has_objects(annotation) for annotation in annotations),
        "Labels count": len(labels),
        "FPV labels": labels.count("fpv"),
        "Bird labels": labels.count("bird"),
    }
    for tag in tags:
        row[tag] = sum(tag in result_tags(annotation.get("result", [])) for annotation in annotations)
    return row


def split_analytics_rows(
    plan: dict[str, list[tuple[dict[str, object], dict[str, object]]]],
    tags: list[str],
    include_buffers: bool = True,
) -> list[dict[str, int | str]]:
    split_definitions = [
        ("train", "Train", True),
        ("valid", "Validation", True),
        ("test", "Test", True),
    ]
    if include_buffers:
        split_definitions.insert(1, ("skip_train_to_valid", "Buffer 1", False))
        split_definitions.insert(3, ("skip_valid_to_test", "Buffer 2", False))
    return [
        analytics_row(
            label,
            [task for task, _ in plan.get(split, [])],
            tags,
        )
        for split, label, exported in split_definitions
    ]


def analytics_csv(rows: list[dict[str, int | str]]) -> str:
    fieldnames: list[str] = []
    for row in rows:
        for fieldname in row:
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def export_summary(
    selected_storages: list[dict[str, object]],
    tasks_by_storage: dict[int, list[dict[str, object]]],
    plans: dict[int, dict[str, list[tuple[dict[str, object], dict[str, object]]]]],
    tags: list[str],
    include_buffers: bool,
) -> str:
    split_definitions = [("train", "Train"), ("valid", "Validation"), ("test", "Test")]
    if include_buffers:
        split_definitions.insert(1, ("skip_train_to_valid", "Buffer 1"))
        split_definitions.insert(3, ("skip_valid_to_test", "Buffer 2"))
    exported_tasks = [
        task
        for plan in plans.values()
        for split in ("train", "valid", "test")
        for task, _ in plan.get(split, [])
    ]
    totals = analytics_row("Export", exported_tasks, tags)
    total_images = int(totals["Images count"])
    background_images = int(totals["Empty images"])
    background_percent = background_images / total_images * 100 if total_images else 0
    lines = [
        f"Total images: {total_images} | Total labels: {totals['Labels count']}",
        f"Background images: {background_images} ({background_percent:.1f}%)",
        "",
        "Per-class instances:",
        f"  fpv  {totals['FPV labels']}",
        f"  bird  {totals['Bird labels']}",
    ]
    if tags:
        lines.extend(["", "Image tags:"])
        lines.extend(f"  {tag}  {totals[tag]}" for tag in tags)
    lines.extend(["", "Dataset source splits:"])
    for storage in selected_storages:
        storage_id = int(storage["id"])
        storage_name = str(storage.get("title") or f"Storage {storage_id}")
        annotated_count = len(annotated_tasks_in_filename_order(tasks_by_storage.get(storage_id, [])))
        lines.append(f"  {storage_name} ({annotated_count} annotated images):")
        plan = plans.get(storage_id, {})
        for split, label in split_definitions:
            count = len(plan.get(split, []))
            percent = count / annotated_count * 100 if annotated_count else 0
            lines.append(f"    {label}: {count} ({percent:.1f}%)")
    return "\n".join(lines) + "\n"


def build_yolo_archive(
    selected_storages: list[dict[str, object]],
    tasks_by_storage: dict[int, list[dict[str, object]]],
    split_mode: str,
    tags: list[str],
    random_splits: tuple[int, int, int] | None = None,
    storage_splits: dict[int, tuple[int, int, int, int, int]] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[Path, dict[str, int]]:
    export_tasks: list[tuple[str, str, int, dict[str, object], dict[str, object], str]] = []
    manifest_rows: list[tuple[str, int, str, int, str]] = []
    plans: dict[int, dict[str, list[tuple[dict[str, object], dict[str, object]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    if split_mode == "random":
        if random_splits is None:
            raise ValueError("Random split percentages are required.")
        random_tasks: list[tuple[int, str, str, int, dict[str, object], dict[str, object]]] = []
        for storage in selected_storages:
            storage_id = int(storage["id"])
            storage_name = str(storage.get("title") or f"Storage {storage_id}")
            prefix = safe_filename(storage_name)
            for source_order, task in enumerate(tasks_by_storage.get(storage_id, []), start=1):
                if annotation := completed_annotation(task):
                    random_tasks.append((storage_id, prefix, storage_name, source_order, task, annotation))
        Random(42).shuffle(random_tasks)
        train_end = round(len(random_tasks) * random_splits[0] / 100)
        valid_end = train_end + round(len(random_tasks) * random_splits[1] / 100)
        split_names = (
            ["train"] * train_end
            + ["valid"] * (valid_end - train_end)
            + ["test"] * (len(random_tasks) - valid_end)
        )
        for item, split in zip(random_tasks, split_names):
            storage_id, prefix, storage_name, source_order, task, annotation = item
            export_tasks.append((prefix, storage_name, source_order, task, annotation, split))
            manifest_rows.append((storage_name, int(task["id"]), task_source_filename(task), source_order, split))
            plans[storage_id][split].append((task, annotation))
    elif split_mode == "sequential":
        if storage_splits is None:
            raise ValueError("Per-storage split percentages are required.")
        for storage in selected_storages:
            storage_id = int(storage["id"])
            storage_name = str(storage.get("title") or f"Storage {storage_id}")
            prefix = safe_filename(storage_name)
            ordered_tasks = annotated_tasks_in_filename_order(tasks_by_storage.get(storage_id, []))
            source_orders = {int(task["id"]): index for index, (task, _) in enumerate(ordered_tasks, start=1)}
            split_tasks = sequence_split_tasks(tasks_by_storage.get(storage_id, []), storage_splits[storage_id])
            for split, assigned_tasks in split_tasks.items():
                plans[storage_id][split].extend(assigned_tasks)
                for task, annotation in assigned_tasks:
                    source_order = source_orders[int(task["id"])]
                    manifest_rows.append((storage_name, int(task["id"]), task_source_filename(task), source_order, split))
                    if split in {"train", "valid", "test"}:
                        export_tasks.append((prefix, storage_name, source_order, task, annotation, split))
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")
    counts: dict[str, int] = defaultdict(int)
    total_tasks = len(export_tasks)

    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    # Write straight to disk so the whole dataset never has to fit in RAM at once.
    archive_fd, archive_path_str = tempfile.mkstemp(suffix=".zip", dir=EXPORT_ROOT)
    os.close(archive_fd)
    archive_path = Path(archive_path_str)
    archive_path.chmod(0o644)  # mkstemp creates 0600; nginx (different user) needs read access to serve it
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("data.yaml", "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\nnames:\n  0: 'fpv'\n  1: 'bird'\n")
        manifest = io.StringIO(newline="")
        manifest_writer = csv.writer(manifest, lineterminator="\n")
        manifest_writer.writerow(["storage", "task_id", "source_image", "source_order", "assignment"])
        manifest_writer.writerows(manifest_rows)
        zip_file.writestr("split_manifest.csv", manifest.getvalue())
        storage_rows = []
        overall_tasks: list[dict[str, object]] = []
        overall_plan: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = defaultdict(list)
        storage_split_rows = []
        for storage in selected_storages:
            storage_id = int(storage["id"])
            storage_name = str(storage.get("title") or f"Storage {storage_id}")
            storage_rows.append(analytics_row(storage_name, tasks_by_storage.get(storage_id, []), tags))
            overall_tasks.extend(tasks_by_storage.get(storage_id, []))
            for split, split_tasks in plans[storage_id].items():
                overall_plan[split].extend(split_tasks)
            for row in split_analytics_rows(plans[storage_id], tags, include_buffers=split_mode == "sequential"):
                storage_split_rows.append({"Storage": storage_name, **row})
        storage_rows.append(analytics_row("All selected storages", overall_tasks, tags))
        overall_split_rows = split_analytics_rows(
            overall_plan, tags, include_buffers=split_mode == "sequential"
        )
        zip_file.writestr("dataset_analytics.csv", analytics_csv(storage_rows))
        zip_file.writestr("overall_split_analytics.csv", analytics_csv(overall_split_rows))
        zip_file.writestr("storage_split_analytics.csv", analytics_csv(storage_split_rows))
        zip_file.writestr(
            "dataset_summary.txt",
            export_summary(
                selected_storages,
                tasks_by_storage,
                plans,
                tags,
                include_buffers=split_mode == "sequential",
            ),
        )
        for index, (prefix, _, _, task, annotation, split) in enumerate(export_tasks):
            data, suffix = image_bytes(task)
            stem = Path(unquote(task_image_reference(task) or "image.jpg").split("?", 1)[0]).stem
            output_name = f"{prefix}_{safe_filename(stem)}_{task.get('id', index)}{suffix.lower()}"
            zip_file.writestr(f"{split}/images/{output_name}", data)
            del data
            zip_file.writestr(f"{split}/labels/{Path(output_name).stem}.txt", yolo_labels(annotation))
            counts[split] += 1
            if progress_callback:
                progress_callback(index + 1, total_tasks, output_name)
    return archive_path, counts


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
    st.session_state.setdefault("dataset_export_in_progress", False)
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
            target_dir = selected_instance().data_dir / "images" / target_name if target_name else None
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
    st.caption("Download annotated images from Label Studio Local Files storages as one YOLO dataset.")
    st.caption(f"Data source: {DATA_SOURCE.upper()} (set DATA_SOURCE=api or DATA_SOURCE=postgres in .env to switch manually)")
    try:
        with st.spinner("Loading Local Files storages and annotation statistics..."):
            storages = load_storages(selected_instance_key())
            if not storages:
                st.warning("No Local Files storages were found.")
                return
            tasks = load_project_tasks(selected_instance_key())
            tags = available_tags(load_label_config())
            tasks_by_storage: dict[int, list[dict[str, object]]] = defaultdict(list)
            for task in tasks:
                matches = [storage for storage in storages if task_belongs_to_storage(task, storage)]
                if not matches and len(storages) == 1:
                    matches = storages
                for storage in matches:
                    tasks_by_storage[int(storage["id"])].append(task)
    except LS_API_ERRORS + (psycopg.Error,) as error:
        st.info(
            f"Failed to load data with DATA_SOURCE={DATA_SOURCE}. "
            "Set DATA_SOURCE=api in .env and restart dataset-hub to use the Label Studio API fallback. "
            f"Details: {error}"
        )
        return
    rows = []
    for storage in storages:
        storage_id = int(storage["id"])
        storage_tasks = tasks_by_storage.get(storage_id, [])
        rows.append({"Storage": str(storage.get("title") or f"Storage {storage_id}"), "Type": str(storage.get("storage_type", "unknown")).upper(), "Images": len(storage_tasks), "Annotated": sum(completed_annotation(task) is not None for task in storage_tasks)})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    labels = {int(storage["id"]): str(storage.get("title") or f"Storage {storage['id']}") for storage in storages}
    selected_ids = st.multiselect("Local Files storages", list(labels), default=list(labels), format_func=labels.get)
    selected_storages = [storage for storage in storages if int(storage["id"]) in selected_ids]
    split_mode = "sequential" if st.toggle("Split each storage sequentially", key="dataset_export_sequential") else "random"
    random_splits: tuple[int, int, int] | None = None
    storage_splits: dict[int, tuple[int, int, int, int, int]] = {}
    sequential_plans: dict[int, dict[str, list[tuple[dict[str, object], dict[str, object]]]]] = {}
    random_plans: dict[int, dict[str, list[tuple[dict[str, object], dict[str, object]]]]] = {}
    invalid_split = False

    if split_mode == "random":
        st.caption("Uses the existing deterministic random split across all selected storages.")
        train, valid, test = st.columns(3)
        with train:
            train_percent = st.number_input("Train (%)", 0, 100, 100, key="random_train_percent")
        with valid:
            valid_percent = st.number_input("Validation (%)", 0, 100, 0, key="random_valid_percent")
        with test:
            test_percent = st.number_input("Test (%)", 0, 100, 0, key="random_test_percent")
        if train_percent + valid_percent + test_percent != 100:
            st.error("Train, validation, and test percentages must total 100.")
            invalid_split = True
        random_splits = (int(train_percent), int(valid_percent), int(test_percent))
        if not invalid_split:
            random_plans = random_split_tasks(selected_storages, tasks_by_storage, random_splits)
    else:
        st.caption(
            "Frames are sorted by filename and assigned in order. Buffers are removed before the train, validation, and test split is calculated."
        )
        for storage in selected_storages:
            storage_id = int(storage["id"])
            annotated_tasks = annotated_tasks_in_filename_order(tasks_by_storage.get(storage_id, []))
            with st.container(border=True):
                st.subheader(f"{labels[storage_id]} ({len(annotated_tasks)} annotated images)")
                train_column, train_buffer_column, valid_column, valid_buffer_column, test_column = st.columns(5)
                with train_column:
                    train_percent = st.number_input("Train (%)", 0, 100, 100, key=f"storage_{storage_id}_train")
                with train_buffer_column:
                    train_buffer_percent = st.number_input("Buffer after train (%)", 0, 100, 0, key=f"storage_{storage_id}_train_buffer")
                with valid_column:
                    valid_percent = st.number_input("Validation (%)", 0, 100, 0, key=f"storage_{storage_id}_valid")
                with valid_buffer_column:
                    valid_buffer_percent = st.number_input("Buffer after validation (%)", 0, 100, 0, key=f"storage_{storage_id}_valid_buffer")
                with test_column:
                    test_percent = st.number_input("Test (%)", 0, 100, 0, key=f"storage_{storage_id}_test")
                splits = (
                    int(train_percent),
                    int(train_buffer_percent),
                    int(valid_percent),
                    int(valid_buffer_percent),
                    int(test_percent),
                )
                if train_percent + valid_percent + test_percent != 100:
                    st.error("Train, validation, and test percentages must total 100.")
                    invalid_split = True
                else:
                    try:
                        plan = sequence_split_tasks(tasks_by_storage.get(storage_id, []), splits)
                    except ValueError as error:
                        st.error(str(error))
                        invalid_split = True
                    else:
                        storage_splits[storage_id] = splits
                        sequential_plans[storage_id] = plan
                        st.caption(
                            f"Train: {len(plan['train'])} | Buffer: {len(plan['skip_train_to_valid'])} | "
                            f"Validation: {len(plan['valid'])} | Buffer: {len(plan['skip_valid_to_test'])} | "
                            f"Test: {len(plan['test'])}"
                        )
                        analytics_column, preview_column = st.columns(2)
                        with analytics_column:
                            with st.expander("Split analytics"):
                                st.dataframe(
                                    split_analytics_rows(plan, tags),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                        with preview_column:
                            preview_signature = tuple(
                                (split, int(task["id"]))
                                for split in ("train", "valid", "test")
                                for task, _ in plan[split]
                            )
                            preview_state_key = f"storage_{storage_id}_preview_signature"
                            with st.expander("Split previews"):
                                if st.button("Load / refresh previews", key=f"storage_{storage_id}_load_previews"):
                                    st.session_state[preview_state_key] = preview_signature
                                if st.session_state.get(preview_state_key) != preview_signature:
                                    st.caption("Open this section and load previews after confirming the split percentages.")
                                else:
                                    for split in ("train", "valid", "test"):
                                        split_tasks = plan[split]
                                        st.write(f"{split.title()} ({len(split_tasks)} images)")
                                        if not split_tasks:
                                            st.caption("No images assigned to this split.")
                                            continue
                                        first_column, last_column = st.columns(2)
                                        for column, position, item in (
                                            (first_column, "First", split_tasks[0]),
                                            (last_column, "Last", split_tasks[-1]),
                                        ):
                                            task, _ = item
                                            with column:
                                                try:
                                                    data, _ = image_bytes(task)
                                                    st.image(data, caption=f"{position}: {task_source_filename(task)}", use_container_width=True)
                                                except LS_API_ERRORS as error:
                                                    st.warning(f"Could not load {position.lower()} image: {error}")

    annotated_count = sum(
        completed_annotation(task) is not None
        for storage in selected_storages
        for task in tasks_by_storage.get(int(storage["id"]), [])
    )
    if split_mode == "sequential":
        export_count = sum(
            len(plan[split])
            for plan in sequential_plans.values()
            for split in ("train", "valid", "test")
        )
        skipped_count = sum(
            len(plan[split])
            for plan in sequential_plans.values()
            for split in ("skip_train_to_valid", "skip_valid_to_test")
        )
        st.caption(f"{export_count} annotated images will be exported; {skipped_count} will be skipped as buffers.")
    else:
        export_count = annotated_count
        st.caption(f"{annotated_count} annotated images will be exported. Unannotated tasks are skipped.")
    plans = sequential_plans if split_mode == "sequential" else random_plans
    storage_analytics = []
    overall_tasks: list[dict[str, object]] = []
    overall_plan: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = defaultdict(list)
    for storage in selected_storages:
        storage_id = int(storage["id"])
        storage_tasks = tasks_by_storage.get(storage_id, [])
        plan = plans.get(storage_id, {})
        storage_analytics.append(analytics_row(labels[storage_id], storage_tasks, tags))
        overall_tasks.extend(storage_tasks)
        for split, split_tasks in plan.items():
            overall_plan[split].extend(split_tasks)
    if selected_storages:
        storage_analytics.append(analytics_row("All selected storages", overall_tasks, tags))
        st.subheader("Dataset analytics")
        st.dataframe(storage_analytics, use_container_width=True, hide_index=True)
        st.subheader("Overall split analytics")
        st.dataframe(
            split_analytics_rows(overall_plan, tags, include_buffers=split_mode == "sequential"),
            use_container_width=True,
            hide_index=True,
        )
    build_disabled = (
        not selected_storages
        or not export_count
        or invalid_split
        or st.session_state.dataset_export_in_progress
    )
    if st.button("Build YOLO dataset", type="primary", disabled=build_disabled):
        st.session_state.dataset_export_in_progress = True
        progress_bar = st.progress(0.0, text="Preparing YOLO dataset...")
        status = st.empty()

        def update_export_progress(completed: int, total: int, output_name: str) -> None:
            progress_bar.progress(completed / total, text=f"Exporting image {completed} of {total}...")
            status.caption(f"Current file: {output_name}")

        try:
            archive_path, counts = build_yolo_archive(
                selected_storages,
                tasks_by_storage,
                split_mode,
                tags,
                random_splits=random_splits,
                storage_splits=storage_splits,
                progress_callback=update_export_progress,
            )
            if old_path := st.session_state.get("yolo_archive_path"):
                Path(old_path).unlink(missing_ok=True)
            st.session_state.yolo_archive_path = str(archive_path)
            st.session_state.yolo_counts = dict(counts)
        except LS_API_ERRORS as error:
            st.error(f"Dataset export failed: {error}")
        finally:
            st.session_state.dataset_export_in_progress = False
    if archive_path := st.session_state.get("yolo_archive_path"):
        counts = st.session_state.get("yolo_counts", {})
        # Served as a static file by dataset-hub-proxy so the ZIP never has to be loaded into Streamlit's memory.
        st.link_button("Download YOLO dataset (.zip)", f"/dataset-hub/downloads/{Path(archive_path).name}", type="primary")
        st.caption(f"Ready: {counts.get('train', 0)} train, {counts.get('valid', 0)} validation, {counts.get('test', 0)} test images.")


def render_storage_archive() -> None:
    st.title("Storage archive")
    st.caption(
        "Export Local Files storages with source images, Label Studio annotations, COCO annotations, "
        "image tags, and summary statistics. Import accepts exactly one exported storage per archive."
    )
    try:
        with st.spinner("Loading Local Files storages..."):
            storages = load_storages(selected_instance_key())
            tasks = load_project_tasks(selected_instance_key())
    except LS_API_ERRORS + (psycopg.Error,) as error:
        st.error(f"Failed to load Local Files storages: {error}")
        return
    if not storages:
        st.warning("No Local Files storages were found.")
        return
    tasks_by_storage = tasks_grouped_by_storage(storages, tasks)
    labels = {
        int(storage["id"]): str(storage.get("title") or f"Storage {storage['id']}")
        for storage in storages
    }
    rows = [
        {
            "Storage": labels[int(storage["id"])],
            "Images": len(tasks_by_storage.get(int(storage["id"]), [])),
            "Annotated": sum(
                completed_annotation(task) is not None
                for task in tasks_by_storage.get(int(storage["id"]), [])
            ),
        }
        for storage in storages
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Export storage archive")
    selected_ids = st.multiselect(
        "Storages to export",
        options=list(labels),
        default=list(labels),
        format_func=labels.get,
        key="storage_archive_export_ids",
    )
    selected_storages = [storage for storage in storages if int(storage["id"]) in selected_ids]
    include_unannotated = st.checkbox(
        "Include all images, including images without annotations",
        value=False,
        key="storage_archive_include_unannotated",
        help="When unchecked, images without a completed annotation are excluded from the archive.",
    )
    selected_tasks = [
        task
        for storage in selected_storages
        for task in tasks_by_storage.get(int(storage["id"]), [])
    ]
    selected_count = (
        len(selected_tasks)
        if include_unannotated
        else sum(completed_annotation(task) is not None for task in selected_tasks)
    )
    st.caption(
        f"{selected_count} image(s) will be exported."
        if include_unannotated
        else f"{selected_count} annotated image(s) will be exported; images without annotations will be skipped."
    )
    if st.button(
        "Build storage archive",
        type="primary",
        disabled=not selected_storages or not selected_count or st.session_state.get("storage_archive_exporting", False),
    ):
        st.session_state.storage_archive_exporting = True
        progress = st.progress(0.0, text="Preparing storage archive...")
        status = st.empty()

        def update_progress(completed: int, total: int, name: str) -> None:
            progress.progress(completed / total, text=f"Archiving image {completed} of {total}...")
            status.caption(f"Current file: {name}")

        try:
            archive_path, image_count = build_storage_archive(
                selected_storages,
                tasks_by_storage,
                include_unannotated,
                progress_callback=update_progress,
            )
            if old_path := st.session_state.get("storage_archive_path"):
                Path(old_path).unlink(missing_ok=True)
            st.session_state.storage_archive_path = str(archive_path)
            st.session_state.storage_archive_image_count = image_count
        except LS_API_ERRORS + (OSError, ValueError, zipfile.BadZipFile) as error:
            st.error(f"Storage archive export failed: {error}")
        finally:
            st.session_state.storage_archive_exporting = False
    if archive_path := st.session_state.get("storage_archive_path"):
        st.link_button(
            "Download storage archive (.zip)",
            f"/dataset-hub/downloads/{Path(archive_path).name}",
            type="primary",
        )
        st.caption(f"Ready: {st.session_state.get('storage_archive_image_count', 0)} image(s).")

    st.divider()
    st.subheader("Import one storage archive")
    st.caption(
        "The archive must contain one storage. The requested storage name must be new and all archive tags "
        "must already exist in the active project."
    )
    with st.form("storage_archive_import", clear_on_submit=True):
        uploaded_file = st.file_uploader("Storage archive (.zip)", type=["zip"])
        new_storage_name = st.text_input("New storage name")
        submit_import = st.form_submit_button("Upload, sync, and import annotations", type="primary")
    if submit_import:
        try:
            with st.spinner("Uploading images, syncing Local Files storage, and importing annotations..."):
                result = import_storage_archive(uploaded_file, new_storage_name)
            st.success(
                f'Imported "{new_storage_name}": {result["images"]} images synced, '
                f'{result["annotations_created"]} annotations created, '
                f'{result["without_annotations"]} images without annotations.'
            )
        except LS_API_ERRORS + (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
            st.error(f"Storage archive import failed: {error}")


def render_tag_settings() -> None:
    st.title("Tag settings")
    st.caption('Manage the list of available image tags (the <Choices name="tags"> block in the labeling template).')
    try:
        label_config = load_label_config()
    except LS_API_ERRORS + (psycopg.Error,) as error:
        st.error(f"Failed to load the label config: {error}")
        return
    tags = available_tags(label_config)
    if not tags:
        st.warning('No <Choices name="tags"> block was found in the project label config.')
        return
    st.write(f"Current tags ({len(tags)}):")
    st.dataframe([{"Tag": tag} for tag in tags], use_container_width=True, hide_index=True)

    with st.form("add_tag_form", clear_on_submit=True):
        new_tag = st.text_input("New tag value").strip()
        if st.form_submit_button("Add tag") and new_tag:
            if new_tag in tags:
                st.warning(f'Tag "{new_tag}" already exists.')
            else:
                try:
                    save_label_config(update_label_config_tags(label_config, tags + [new_tag]))
                    st.success(f'Added tag "{new_tag}".')
                    st.rerun()
                except LS_API_ERRORS as error:
                    st.error(f"Failed to update the label config: {error}")

    remove_selection = st.multiselect("Remove tags", tags, key="tag_settings_remove_tags")
    affected_tasks: list[dict[str, object]] = []
    if remove_selection:
        try:
            with st.spinner("Checking tag usage across storages..."):
                storages = load_storages(selected_instance_key())
                tasks = load_project_tasks(selected_instance_key())
            tasks_by_storage = tasks_grouped_by_storage(storages, tasks)
        except LS_API_ERRORS + (psycopg.Error,) as error:
            st.error(f"Failed to check tag usage: {error}")
            return

        selected_tags = set(remove_selection)
        usage_rows = []
        for storage in storages:
            storage_id = int(storage["id"])
            used_tasks = [task for task in tasks_by_storage[storage_id] if selected_tags & set(task_current_tags(task))]
            affected_tasks.extend(used_tasks)
            if used_tasks:
                usage_rows.append({
                    "Storage": str(storage.get("title") or f"Storage {storage_id}"),
                    "Images with selected tags": len(used_tasks),
                })
        affected_tasks = list({int(task["id"]): task for task in affected_tasks}.values())
        st.warning(
            f"Removing these tags will also remove them from {len(affected_tasks)} annotated image(s)."
        )
        if usage_rows:
            st.dataframe(usage_rows, use_container_width=True, hide_index=True)
        else:
            st.caption("The selected tags are not currently used in any annotated image.")

    confirm_removal = st.checkbox(
        "I confirm removing the selected tags from all affected annotations and the project tag list.",
        disabled=not remove_selection,
        key="confirm_tag_settings_removal",
    )
    if st.button("Remove selected tags", disabled=not remove_selection or not confirm_removal):
        progress_bar = st.progress(0.0, text="Removing tags from annotations...")
        status = st.empty()

        def update_progress(done: int, total: int, name: str) -> None:
            progress_bar.progress(done / total, text=f"Processing image {done} of {total}...")
            status.caption(f"Task ID: {name}")

        try:
            counts, errors = apply_tags_to_tasks(affected_tasks, set(), set(remove_selection), update_progress)
            if counts.get("failed"):
                st.error(
                    f"The project tag list was not changed because {counts['failed']} annotation update(s) failed:\n"
                    + "\n".join(errors)
                )
                return
            save_label_config(update_label_config_tags(label_config, [tag for tag in tags if tag not in remove_selection]))
            load_project_tasks.clear()
            load_storages.clear()
            st.success(
                f"Removed {len(remove_selection)} tag(s) from the project and {counts.get('updated', 0)} image(s)."
            )
            st.rerun()
        except LS_API_ERRORS as error:
            st.error(f"Failed to remove tags: {error}")


def render_manage_tags() -> None:
    st.title("Manage tags")
    st.caption(
        "Apply or remove image tags for every annotated task in a Local Files storage. "
        "Tasks that don't have an annotation yet are skipped."
    )
    try:
        tags = available_tags(load_label_config())
        with st.spinner("Loading storages and tasks..."):
            storages = load_storages(selected_instance_key())
            if not storages:
                st.warning("No Local Files storages were found.")
                return
            tasks = load_project_tasks(selected_instance_key())
    except LS_API_ERRORS + (psycopg.Error,) as error:
        st.error(f"Failed to load data: {error}")
        return
    if not tags:
        st.warning('No tags are defined yet. Add some in "Tag settings" first.')
        return

    labels = {int(storage["id"]): str(storage.get("title") or f"Storage {storage['id']}") for storage in storages}
    tasks_by_storage = tasks_grouped_by_storage(storages, tasks)
    tag_summary_rows = []
    for storage in storages:
        storage_id = int(storage["id"])
        storage_counts = tag_counts(tasks_by_storage[storage_id], tags)
        tag_summary_rows.append({"Storage": labels[storage_id], "Images": len(tasks_by_storage[storage_id]), **storage_counts})
    st.dataframe(tag_summary_rows, use_container_width=True, hide_index=True)

    storage_id = st.selectbox("Storage", list(labels), format_func=labels.get)
    storage_tasks = tasks_by_storage[storage_id]
    st.caption(f"{len(storage_tasks)} image(s) in this storage.")

    add_tags = st.multiselect("Add tags", tags, key="manage_tags_add_tags")
    remove_tags = st.multiselect("Remove tags", tags, key="manage_tags_remove_tags")
    overlap = set(add_tags) & set(remove_tags)
    if overlap:
        st.error(f"Tags cannot be added and removed at the same time: {', '.join(sorted(overlap))}")
        return

    disabled = not storage_tasks or (not add_tags and not remove_tags)
    confirm = st.checkbox(
        f'I confirm applying this change to all {len(storage_tasks)} image(s) in "{labels[storage_id]}".',
        disabled=disabled,
    )
    if st.button("Apply", type="primary", disabled=disabled or not confirm):
        progress_bar = st.progress(0.0, text="Applying tags...")
        status = st.empty()

        def update_progress(done: int, total: int, name: str) -> None:
            progress_bar.progress(done / total, text=f"Processing task {done} of {total}...")
            status.caption(f"Task ID: {name}")

        try:
            counts, errors = apply_tags_to_tasks(storage_tasks, set(add_tags), set(remove_tags), update_progress)
            load_project_tasks.clear()
            st.success(
                f"Updated {counts.get('updated', 0)} task(s), "
                f"{counts.get('unchanged', 0)} already matched, "
                f"{counts.get('skipped', 0)} skipped (not annotated yet)."
            )
            if counts.get("failed"):
                st.error(f"{counts['failed']} task(s) failed:\n" + "\n".join(errors))
        except LS_API_ERRORS as error:
            st.error(f"Failed to apply tags: {error}")


def main() -> None:
    st.set_page_config(page_title="Dataset Hub", layout="wide")
    st.sidebar.header("Dataset Hub")
    instance_key = st.sidebar.selectbox(
        "Label Studio instance",
        options=list(LABEL_STUDIO_INSTANCES),
        format_func=lambda key: LABEL_STUDIO_INSTANCES[key].label,
        key="label_studio_instance",
    )
    st.sidebar.caption(f"Project {LABEL_STUDIO_INSTANCES[instance_key].project_id}")
    frame_tab, export_tab, storage_archive_tab, manage_tags_tab, tag_settings_tab = st.tabs(
        ["Frame extraction", "Dataset export", "Storage archive", "Manage tags", "Tag settings"]
    )
    with frame_tab:
        render_frame_extractor()
    with export_tab:
        render_dataset_export()
    with storage_archive_tab:
        render_storage_archive()
    with manage_tags_tab:
        render_manage_tags()
    with tag_settings_tab:
        render_tag_settings()


if __name__ == "__main__":
    main()