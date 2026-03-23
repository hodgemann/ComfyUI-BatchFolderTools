"""
ComfyUI-BatchFolderTools  v0.8.0
Batch folder processing utilities for ComfyUI.

Nodes:
  - Load Image From Folder (Batch)  → Iterates images in a folder by index
  - Load Video From Folder (Batch)  → Iterates videos in a folder, extracts frames
  - Save Text File                  → Writes text to a file matching the source name
  - Queue Next                      → Re-queues the workflow for the next iteration
"""

__version__ = "0.8.0"

import copy
import json
import threading
import urllib.request
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageOps, ImageSequence

# ─── Constants ───────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif",
}

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v",
}

SORT_METHODS = [
    "alphabetical", "alphabetical_reverse",
    "modified_newest", "modified_oldest",
    "created_newest", "created_oldest",
]

VIDEO_FRAME_MODES = [
    "first_frame",
    "last_frame",
    "all_frames",
    "evenly_spaced",
    "frame_range",
]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_image_files(folder_path: str, sort_by: str = "alphabetical") -> list[Path]:
    """Return sorted list of image file Paths in a folder."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"[BatchFolder] Folder not found: {folder_path}")

    files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not files:
        raise FileNotFoundError(
            f"[BatchFolder] No image files found in: {folder_path}\n"
            f"  Supported formats: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )

    if sort_by == "alphabetical":
        files.sort(key=lambda p: p.name.lower())
    elif sort_by == "alphabetical_reverse":
        files.sort(key=lambda p: p.name.lower(), reverse=True)
    elif sort_by == "modified_newest":
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    elif sort_by == "modified_oldest":
        files.sort(key=lambda p: p.stat().st_mtime)
    elif sort_by == "created_newest":
        files.sort(key=lambda p: p.stat().st_ctime, reverse=True)
    elif sort_by == "created_oldest":
        files.sort(key=lambda p: p.stat().st_ctime)
    else:
        files.sort(key=lambda p: p.name.lower())

    return files


def _load_image_as_tensor(image_path: Path) -> torch.Tensor:
    """
    Load an image file directly from disk and return a ComfyUI IMAGE
    tensor [1, H, W, C] float32 in range 0-1.

    Nothing is copied to ComfyUI's input/cache directory.
    """
    img = Image.open(str(image_path))

    for frame in ImageSequence.Iterator(img):
        img = frame
        break

    img = ImageOps.exif_transpose(img)

    if img.mode == "I":
        img = img.point(lambda p: p * (1.0 / 255.0))

    img = img.convert("RGB")

    img_array = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_array).unsqueeze(0)

    return tensor


def _get_video_files(folder_path: str, sort_by: str = "alphabetical") -> list[Path]:
    """Return sorted list of video file Paths in a folder."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"[BatchFolder] Folder not found: {folder_path}")

    files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not files:
        raise FileNotFoundError(
            f"[BatchFolder] No video files found in: {folder_path}\n"
            f"  Supported formats: {', '.join(sorted(VIDEO_EXTENSIONS))}"
        )

    if sort_by == "alphabetical":
        files.sort(key=lambda p: p.name.lower())
    elif sort_by == "alphabetical_reverse":
        files.sort(key=lambda p: p.name.lower(), reverse=True)
    elif sort_by == "modified_newest":
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    elif sort_by == "modified_oldest":
        files.sort(key=lambda p: p.stat().st_mtime)
    elif sort_by == "created_newest":
        files.sort(key=lambda p: p.stat().st_ctime, reverse=True)
    elif sort_by == "created_oldest":
        files.sort(key=lambda p: p.stat().st_ctime)
    else:
        files.sort(key=lambda p: p.name.lower())

    return files


def _extract_video_frames(video_path: Path, frame_mode: str,
                          frame_count: int = 16,
                          start_frame: int = 0,
                          end_frame: int = -1) -> torch.Tensor:
    """
    Extract frames from a video file and return as a ComfyUI IMAGE
    tensor [N, H, W, C] float32 in range 0-1, plus video metadata.

    Returns: (tensor, fps, duration_seconds, total_frame_count)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "[BatchFolder] opencv-python (cv2) is required for video loading.\n"
            "  Install it:  pip install opencv-python"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"[BatchFolder] Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0  # fallback
    duration = total_frames / fps if total_frames > 0 else 0.0

    if total_frames <= 0:
        cap.release()
        raise IOError(f"[BatchFolder] Video has 0 frames: {video_path}")

    # Determine which frame indices to extract
    if frame_mode == "first_frame":
        indices = [0]

    elif frame_mode == "last_frame":
        indices = [total_frames - 1]

    elif frame_mode == "all_frames":
        indices = list(range(total_frames))

    elif frame_mode == "evenly_spaced":
        count = min(frame_count, total_frames)
        if count <= 1:
            indices = [0]
        else:
            indices = [int(round(i * (total_frames - 1) / (count - 1)))
                       for i in range(count)]

    elif frame_mode == "frame_range":
        actual_start = max(0, min(start_frame, total_frames - 1))
        actual_end = total_frames - 1 if end_frame < 0 else min(end_frame, total_frames - 1)
        if actual_end < actual_start:
            actual_end = actual_start
        indices = list(range(actual_start, actual_end + 1))

    else:
        indices = [0]

    # Extract frames
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # cv2 returns BGR, convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_float = frame_rgb.astype(np.float32) / 255.0
            frames.append(frame_float)

    cap.release()

    if not frames:
        raise IOError(f"[BatchFolder] Could not read any frames from: {video_path}")

    # Stack into [N, H, W, C] tensor
    tensor = torch.from_numpy(np.stack(frames, axis=0))

    return tensor, fps, duration, total_frames


def _requeue_prompt(prompt_data: dict):
    """
    Re-submit the current workflow to ComfyUI's prompt queue.
    Runs in a background thread so the current execution finishes first.

    Two cache-busting tweaks (no effect on actual results):
    1. Bump the 'index' widget on the loader node (ignored in sequential
       mode) — prevents ComfyUI prompt-level deduplication.
    2. Bump the 'seed' on any VLM node — prevents VLM-internal prompt
       caching from returning stale results for different inputs.
    """
    LOADER_CLASS_TYPES = {"FolderImageLoader_BFT", "FolderVideoLoader_BFT"}
    VLM_CLASS_TYPES = {"AILab_QwenVL_Advanced", "AILab_QwenVL"}

    def _do_requeue():
        import time
        import random
        time.sleep(0.5)
        try:
            modified = copy.deepcopy(prompt_data)

            for node in modified.values():
                if not isinstance(node, dict):
                    continue
                ct = node.get("class_type", "")

                # Bust ComfyUI prompt cache
                if ct in LOADER_CLASS_TYPES:
                    old_idx = node.get("inputs", {}).get("index", 0)
                    node["inputs"]["index"] = old_idx + 1

                # Bust VLM internal prompt cache
                if ct in VLM_CLASS_TYPES:
                    node["inputs"]["seed"] = random.randint(1, 2**31)

            payload = json.dumps({"prompt": modified}).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:8188/prompt",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"[BatchFolder] Re-queue failed: {e}")

    threading.Thread(target=_do_requeue, daemon=True).start()


# ─── Node: Load Image From Folder (Batch) ───────────────────────────────────

class FolderImageLoader:
    """
    Loads one image at a time from a folder by index.

    In sequential mode the counter auto-advances each execution.
    Wire the Queue Next node at the end of your workflow to
    automatically process every image in the folder.

    Images are loaded directly from disk — nothing is cached in
    ComfyUI's input directory.
    """

    _counters: dict[str, int] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Absolute path to image folder. Supports /mnt/ WSL paths, Linux, macOS, Windows, and network shares.",
                }),
                "mode": (["sequential", "manual"], {
                    "default": "sequential",
                    "tooltip": "sequential: auto-advances each execution. manual: uses the index widget.",
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999999,
                    "step": 1,
                    "tooltip": "Image index (0-based). Only used in manual mode.",
                }),
                "sort_by": (SORT_METHODS, {
                    "default": "alphabetical",
                    "tooltip": "How to sort the files before indexing.",
                }),
                "reset_counter": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reset the sequential counter back to 0.",
                }),
                "skip_captioned": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip images that already have a caption file. Checks for .txt/.caption/.cap next to the image. Use this to resume an interrupted batch.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "folder_path", "filename_stem", "filename_full",
                    "current_index", "total_images")
    FUNCTION = "load_image"
    CATEGORY = "📂 BatchFolder"
    DESCRIPTION = (
        "Loads one image at a time from a folder. In sequential mode "
        "the index auto-advances each execution. Enable skip_captioned "
        "to resume an interrupted batch — already-captioned images are "
        "skipped with zero GPU cost."
    )

    @classmethod
    def IS_CHANGED(cls, folder_path, mode, index, sort_by, reset_counter,
                   skip_captioned):
        if mode == "sequential":
            return float("nan")
        return f"{folder_path}|{index}|{sort_by}|{skip_captioned}"

    def load_image(self, folder_path: str, mode: str, index: int,
                   sort_by: str, reset_counter: bool, skip_captioned: bool):

        folder_path = folder_path.strip()
        if not folder_path:
            raise ValueError(
                "[BatchFolder] folder_path is empty — paste the "
                "absolute path to your image folder."
            )

        images = _get_image_files(folder_path, sort_by)
        total = len(images)

        if mode == "sequential":
            key = folder_path
            if reset_counter or key not in self._counters:
                self._counters[key] = 0

            # Skip past already-captioned images
            if skip_captioned:
                while self._counters[key] < total:
                    img = images[self._counters[key]]
                    has_caption = any(
                        img.with_suffix(ext).exists()
                        for ext in [".txt", ".caption", ".cap"]
                    )
                    if has_caption:
                        print(f"[BatchFolder] SKIP (captioned): {img.name}")
                        self._counters[key] += 1
                    else:
                        break

            idx = self._counters[key]

            if idx >= total:
                self._counters[key] = 0
                print("")
                print("[BatchFolder] ═══════════════════════════════════════")
                print(f"[BatchFolder] ✅  ALL {total} IMAGES PROCESSED — DONE")
                print("[BatchFolder] ═══════════════════════════════════════")
                print("")
                raise InterruptedError(
                    f"All {total} images processed. Done!"
                )

            self._counters[key] = idx + 1
        else:
            idx = index % total if total > 0 else 0

        img_path = images[idx]
        tensor = _load_image_as_tensor(img_path)

        print(f"[BatchFolder] [{idx + 1}/{total}] {img_path.name}")

        return (tensor, folder_path, img_path.stem, img_path.name, idx, total)


# ─── Node: Load Video From Folder (Batch) ───────────────────────────────────

class FolderVideoLoader:
    """
    Loads one video at a time from a folder by index, extracting frames
    as an IMAGE batch tensor.

    Frame extraction modes:
      - first_frame:   just the first frame (for thumbnails, previews)
      - last_frame:    just the last frame
      - all_frames:    every frame in the video
      - evenly_spaced: N frames evenly sampled across the video
      - frame_range:   a contiguous range from start to end frame

    In sequential mode the counter auto-advances each execution.
    Wire Queue Next at the end of your workflow to loop through
    every video in the folder.
    """

    _counters: dict[str, int] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Absolute path to video folder. Supports /mnt/ WSL paths, Linux, macOS, Windows, network shares.",
                }),
                "frame_mode": (VIDEO_FRAME_MODES, {
                    "default": "first_frame",
                    "tooltip": "What to extract: first_frame, last_frame, all_frames, evenly_spaced, or frame_range.",
                }),
                "frame_count": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 9999,
                    "step": 1,
                    "tooltip": "Number of frames to extract. Only used in evenly_spaced mode.",
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999999,
                    "step": 1,
                    "tooltip": "First frame index. Only used in frame_range mode.",
                }),
                "end_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 9999999,
                    "step": 1,
                    "tooltip": "Last frame index (-1 = end of video). Only used in frame_range mode.",
                }),
                "mode": (["sequential", "manual"], {
                    "default": "sequential",
                    "tooltip": "sequential: auto-advances each execution. manual: uses the index widget.",
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999999,
                    "step": 1,
                    "tooltip": "Video index (0-based). Only used in manual mode.",
                }),
                "sort_by": (SORT_METHODS, {
                    "default": "alphabetical",
                    "tooltip": "How to sort the files before indexing.",
                }),
                "reset_counter": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reset the sequential counter back to 0.",
                }),
                "skip_captioned": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip videos that already have a caption file. Checks for .txt/.caption/.cap next to the video. Use this to resume an interrupted batch.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "INT", "INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("FRAMES", "folder_path", "filename_stem", "filename_full",
                    "current_index", "total_videos", "frame_count", "fps", "duration_seconds")
    FUNCTION = "load_video"
    CATEGORY = "📂 BatchFolder"
    DESCRIPTION = (
        "Loads one video at a time from a folder, extracting frames as "
        "an IMAGE batch. Enable skip_captioned to resume an interrupted "
        "batch — already-captioned videos are skipped with zero GPU cost."
    )

    @classmethod
    def IS_CHANGED(cls, folder_path, frame_mode, frame_count, start_frame,
                   end_frame, mode, index, sort_by, reset_counter,
                   skip_captioned):
        if mode == "sequential":
            return float("nan")
        return f"{folder_path}|{index}|{sort_by}|{frame_mode}|{frame_count}|{start_frame}|{end_frame}|{skip_captioned}"

    def load_video(self, folder_path: str, frame_mode: str, frame_count: int,
                   start_frame: int, end_frame: int, mode: str, index: int,
                   sort_by: str, reset_counter: bool, skip_captioned: bool):

        folder_path = folder_path.strip()
        if not folder_path:
            raise ValueError(
                "[BatchFolder] folder_path is empty — paste the "
                "absolute path to your video folder."
            )

        videos = _get_video_files(folder_path, sort_by)
        total = len(videos)

        if mode == "sequential":
            key = f"video:{folder_path}"
            if reset_counter or key not in self._counters:
                self._counters[key] = 0

            # Skip past already-captioned videos
            if skip_captioned:
                while self._counters[key] < total:
                    vid = videos[self._counters[key]]
                    has_caption = any(
                        vid.with_suffix(ext).exists()
                        for ext in [".txt", ".caption", ".cap"]
                    )
                    if has_caption:
                        print(f"[BatchFolder] SKIP (captioned): {vid.name}")
                        self._counters[key] += 1
                    else:
                        break

            idx = self._counters[key]

            if idx >= total:
                self._counters[key] = 0
                print("")
                print("[BatchFolder] ═══════════════════════════════════════")
                print(f"[BatchFolder] ✅  ALL {total} VIDEOS PROCESSED — DONE")
                print("[BatchFolder] ═══════════════════════════════════════")
                print("")
                raise InterruptedError(
                    f"All {total} videos processed. Done!"
                )

            self._counters[key] = idx + 1
        else:
            idx = index % total if total > 0 else 0

        video_path = videos[idx]
        tensor, fps, duration, total_source_frames = _extract_video_frames(
            video_path, frame_mode, frame_count, start_frame, end_frame)

        extracted_frames = tensor.shape[0]
        print(f"[BatchFolder] [{idx + 1}/{total}] {video_path.name} "
              f"({extracted_frames} frame{'s' if extracted_frames != 1 else ''}, "
              f"{fps:.1f}fps, {duration:.1f}s, mode={frame_mode})")

        return (tensor, folder_path, video_path.stem, video_path.name,
                idx, total, extracted_frames, fps, duration)


# ─── Node: Save Text File ───────────────────────────────────────────────────

class SaveTextFile:
    """
    Saves a text string to a file matching the source image name.
    Wire folder_path → output_folder and filename_stem from the
    FolderImageLoader node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Text to save.",
                }),
                "filename_stem": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Base filename without extension. Wire from FolderImageLoader.",
                }),
                "output_folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "forceInput": False,
                    "tooltip": "Where to save files. Wire from FolderImageLoader's folder_path, or type a different path.",
                }),
                "file_extension": ([".txt", ".caption", ".cap"], {
                    "default": ".txt",
                    "tooltip": "File extension.",
                }),
                "overwrite_existing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Overwrite if file already exists.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_text",)
    OUTPUT_NODE = True
    FUNCTION = "save_text"
    CATEGORY = "📂 BatchFolder"
    DESCRIPTION = (
        "Saves text as a file matching the source image name. "
        "Wire filename_stem and output_folder from FolderImageLoader."
    )

    def save_text(self, text: str, filename_stem: str, output_folder: str,
                  file_extension: str, overwrite_existing: bool):

        output_folder = output_folder.strip()
        if not output_folder:
            raise ValueError(
                "[BatchFolder] output_folder is empty — wire it from "
                "FolderImageLoader's folder_path, or type a path."
            )

        out_dir = Path(output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{filename_stem}{file_extension}"
        file_path = out_dir / filename

        if file_path.exists() and not overwrite_existing:
            print(f"[BatchFolder] SKIP (exists): {file_path}")
            return (text,)

        clean_text = text.strip()
        file_path.write_text(clean_text, encoding="utf-8")

        print(f"[BatchFolder] Saved: {file_path}  ({len(clean_text)} chars)")

        return (clean_text,)


# ─── Node: Queue Next ───────────────────────────────────────────────────────

class QueueNext:
    """
    Re-queues the workflow for the next iteration.

    Place this at the end of any workflow that uses FolderImageLoader.
    It accepts any input (IMAGE or STRING) — just wire the final
    output of your chain here.  After execution completes, it
    re-submits the workflow so the next image in the folder is
    processed.

    When FolderImageLoader runs out of images it raises an error
    BEFORE this node runs, so no re-queue happens and the workflow
    stops cleanly.  Click Queue Prompt once and walk away.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Wire an IMAGE output here to trigger after image processing.",
                }),
                "text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Wire a STRING output here to trigger after text processing.",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "queue_next"
    CATEGORY = "📂 BatchFolder"
    DESCRIPTION = (
        "Place at the end of your workflow. After each execution, "
        "automatically queues the next run. When FolderImageLoader "
        "runs out of images, the workflow stops on its own."
    )

    def queue_next(self, image=None, text=None, prompt=None):
        if prompt is not None:
            _requeue_prompt(prompt)
        return {}
