# Changelog

All notable changes to **ComfyUI-BatchFolderTools** are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/). The version in `pyproject.toml` and `__version__` in
`nodes.py` are kept in sync.

## [0.8.0]

### Added
- `skip_captioned` on both loaders: skips files that already have a `.txt`, `.caption` or `.cap` file, so interrupted batches resume with no wasted GPU time.
- `fps` and `duration_seconds` outputs on the video loader.
- GitHub Actions workflow for Comfy Registry publishing.

### Changed
- Captioning prompts only describe what is present.
- Removed audio inference from the video prompt (QwenVL is vision-only).
- `pyproject.toml` follows the Comfy Registry spec.

### Fixed
- Duplicate captions: the VLM seed is randomized on each re-queue.

## [0.7.x]
- Video loader with 5 frame modes. Queue Next split into its own node.

## [0.6.0]
- Renamed to ComfyUI-BatchFolderTools.

## [0.5.x]
- Self-re-queuing via hidden PROMPT input. Cache-busting.

## [0.1.0–0.4.x]
- Initial development.

[0.8.0]: https://github.com/hodgemann/ComfyUI-BatchFolderTools/releases/tag/v0.8.0
