# ComfyUI-BatchFolderTools

**v0.7.2**

Batch-process a folder of images or videos through any ComfyUI workflow.  Captioning, upscaling, style transfer, video analysis, frame extraction — whatever your workflow does to a single file, these nodes make it do that to an entire folder.  **Click Queue Prompt once and walk away.**

Files load directly from disk.  Nothing is copied into ComfyUI's input or cache directory.

## Nodes

All nodes appear under **📂 BatchFolder** in the node browser.

### Load Image From Folder (Batch)

Loads one image at a time from a folder.  In sequential mode the counter auto-advances each execution.

**Outputs:** `IMAGE`, `folder_path`, `filename_stem`, `filename_full`, `current_index`, `total_images`

**Supported formats:** PNG, JPG, JPEG, WebP, BMP, TIFF, GIF

### Load Video From Folder (Batch)

Loads one video at a time from a folder, extracting frames as an IMAGE batch tensor.

**Frame modes:**

| Mode | What it does |
|------|-------------|
| `first_frame` | Single frame from the start of the video |
| `last_frame` | Single frame from the end of the video |
| `all_frames` | Every frame (can be large — watch your VRAM) |
| `evenly_spaced` | N frames sampled evenly across the video (default 16) |
| `frame_range` | Contiguous range from start_frame to end_frame |

**Outputs:** `FRAMES` (IMAGE batch), `folder_path`, `filename_stem`, `filename_full`, `current_index`, `total_videos`, `frame_count`

**Supported formats:** MP4, AVI, MOV, MKV, WebM, FLV, WMV, M4V

**Requires:** `opencv-python` (`pip install opencv-python` — often already installed in ComfyUI environments)

### Save Text File

Saves text to a file matching the source filename.

### Queue Next

The loop engine.  Place at the end of any workflow.  After each execution it re-queues the workflow.  When the loader runs out of files, execution stops cleanly.  Has both `image` and `text` inputs — wire whichever is the last output of your chain.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hodgemann/ComfyUI-BatchFolderTools.git
pip install opencv-python  # if not already installed
```

Restart ComfyUI.

## Included Example Workflows

| File | What it does |
|------|-------------|
| `batch_caption_qwenvl_mod.json` | Image captioning — loads images sequentially, generates detailed captions via QwenVL-Mod, saves .txt files |
| `batch_video_caption_qwenvl_mod.json` | Video captioning — loads videos, samples 16 evenly-spaced frames, generates captions covering temporal flow, camera work, and visual style, saves .txt files |

Both workflows are pre-wired and ready to use.  Just paste your folder path and click Queue Prompt.

## Usage Examples

### Image Captioning
```
FolderImageLoader → QwenVL-Mod → Save Text File → Queue Next
```

### Video Captioning (first frame)
```
FolderVideoLoader (first_frame) → QwenVL-Mod → Save Text File → Queue Next
```

### Video Captioning (multi-frame analysis)
```
FolderVideoLoader (evenly_spaced, 16 frames) → QwenVL-Mod (video input) → Save Text File → Queue Next
```

### Batch Image Upscaling
```
FolderImageLoader → Upscale Model → Save Image → Queue Next
```

### Batch Frame Extraction
```
FolderVideoLoader (all_frames) → Save Image → Queue Next
```

### Steps

1. Paste the full path to your folder into `folder_path`.
2. Set mode to `sequential`.
3. Click **Queue Prompt** once.
4. Walk away.  Console shows progress.  When done: `✅ DONE`.

### Supported Path Formats

| Platform | Example |
|----------|---------|
| WSL | `/mnt/c/Users/me/images` or `/mnt/d/datasets` |
| Linux | `/home/user/images` or `/data/training_set` |
| macOS | `/Users/me/images` or `/Volumes/External/photos` |
| Windows | `C:\Users\me\images` or `D:\datasets` |
| Network shares | Any mounted path readable by the OS |

### Sort Options

alphabetical, alphabetical_reverse, modified_newest, modified_oldest, created_newest, created_oldest

## Changelog

### v0.7.2
- Fixed duplicate captions across different videos — re-queue now also bumps VLM seed to bust QwenVL's internal prompt cache.
- Removed audio inference from video captioning prompt (QwenVL is vision-only).

### v0.7.1
- **Added Load Video From Folder (Batch)** with five frame extraction modes: first_frame, last_frame, all_frames, evenly_spaced, frame_range.
- Added `batch_video_caption_qwenvl_mod.json` example workflow for video captioning.
- Queue Next re-queue now handles both image and video loader nodes.
- Added opencv-python dependency.
- 4 nodes, 2 example workflows.

### v0.6.0
- Renamed to ComfyUI-BatchFolderTools.  Split Queue Next into its own node.

### v0.5.x
- Self-re-queuing.  Cache-busting.

### v0.1.0–v0.4.x
- Initial development.

## License

[MIT](LICENSE)
