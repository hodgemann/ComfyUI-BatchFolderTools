# ComfyUI-BatchFolderTools

**v0.8.0**

Batch-process a folder of images or videos through any ComfyUI workflow.  Captioning, upscaling, style transfer, video analysis, frame extraction — whatever your workflow does to a single file, these nodes make it do that to an entire folder.  **Click Queue Prompt once and walk away.**  Resume interrupted batches instantly with zero wasted GPU time.

Files load directly from disk.  Nothing is copied into ComfyUI's input or cache directory.

## Nodes

All nodes appear under **📂 BatchFolder** in the node browser.

### Load Image From Folder (Batch)

Loads one image at a time from a folder.  In sequential mode the counter auto-advances each execution.

**Outputs:** `IMAGE`, `folder_path`, `filename_stem`, `filename_full`, `current_index`, `total_images`

**Supported formats:** PNG, JPG, JPEG, WebP, BMP, TIFF, GIF

### Load Video From Folder (Batch)

Loads one video at a time from a folder, extracting frames as an IMAGE batch tensor.

**Frame modes:** `first_frame`, `last_frame`, `all_frames`, `evenly_spaced` (default 16 frames), `frame_range`

**Outputs:** `FRAMES` (IMAGE batch), `folder_path`, `filename_stem`, `filename_full`, `current_index`, `total_videos`, `frame_count`, `fps`, `duration_seconds`

**Supported formats:** MP4, AVI, MOV, MKV, WebM, FLV, WMV, M4V

**Requires:** `opencv-python` (`pip install opencv-python`)

### Save Text File

Saves text to a file matching the source filename.

### Queue Next

The loop engine.  Place at the end of any workflow.  After each execution it re-queues the workflow.  When the loader runs out of files, execution stops cleanly.  Has both `image` and `text` inputs — wire whichever is the last output of your chain.

## Installation

### Via Git Clone

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hodgemann/ComfyUI-BatchFolderTools.git
pip install opencv-python  # if not already installed
```

Restart ComfyUI.

### Via ComfyUI Manager

Search for `Batch Folder Tools` in Install Custom Nodes, install, and restart.

### Via Comfy Registry

```bash
pip install comfy-cli
comfy node install batch-folder-tools
```

## Included Example Workflows

| File | What it does |
|------|-------------|
| `batch_caption_qwenvl_mod.json` | Image captioning — loads images, generates detailed captions via QwenVL-Mod, saves .txt files |
| `batch_video_caption_qwenvl_mod.json` | Video captioning — loads videos, samples 16 evenly-spaced frames, generates captions, saves .txt files |

Both require [ComfyUI-QwenVL-Mod](https://github.com/huchukato/ComfyUI-QwenVL-Mod).  Pre-wired and ready to use — paste your folder path and click Queue Prompt.

## Usage

1. Paste the full path to your folder into `folder_path`.
2. Set mode to `sequential`.
3. Click **Queue Prompt** once.
4. Walk away.  Console shows progress.  When done: `✅ DONE`.
5. To stop mid-run, press Escape or click Cancel.

## Resuming an Interrupted Batch

If the run gets interrupted (crash, power loss, you stopped it manually), set **`skip_captioned`** to `true` on the loader node and click Queue Prompt.  The loader checks for an existing `.txt`, `.caption`, or `.cap` file next to each image/video — any file that already has a caption is skipped entirely with zero GPU cost.  It picks up right where it left off.

## Supported Path Formats

| Platform | Example |
|----------|---------|
| WSL | `/mnt/c/Users/me/images` or `/mnt/d/datasets` |
| Linux | `/home/user/images` or `/data/training_set` |
| macOS | `/Users/me/images` or `/Volumes/External/photos` |
| Windows | `C:\Users\me\images` or `D:\datasets` |
| Network shares | Any mounted path readable by the OS |

## Tips

- `keep_model_loaded: true` on VLM nodes — loads once, stays in VRAM.
- `max_tokens: 2048` for detailed descriptions.
- `temperature: 0.3` for consistent output; `0.6+` for variety.
- `skip_captioned: true` to resume interrupted batches.
- Filenames don't need to be sequential.

## Changelog

### v0.8.0
- Added `skip_captioned` to both loaders — skips files that already have a caption, with zero GPU cost.  Resume interrupted batches instantly.
- Added `fps` and `duration_seconds` outputs to video loader.
- Rewrote captioning prompts to only describe what is present (never mentions absences).
- Removed audio inference from video prompt (QwenVL is vision-only).
- Fixed duplicate captions by randomizing VLM seed on each re-queue.
- Added GitHub Actions workflow for Comfy Registry auto-publishing.
- Fixed pyproject.toml for Comfy Registry spec compliance.

### v0.7.x
- Added video loader with 5 frame modes.  Separated Queue Next into its own node.

### v0.6.0
- Renamed to ComfyUI-BatchFolderTools.

### v0.5.x
- Self-re-queuing via hidden PROMPT input.  Cache-busting.

### v0.1.0–v0.4.x
- Initial development.

## License

[MIT](LICENSE)
