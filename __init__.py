"""ComfyUI-BatchFolderTools  v0.8.0"""

from .nodes import FolderImageLoader, FolderVideoLoader, SaveTextFile, QueueNext, __version__

NODE_CLASS_MAPPINGS = {
    "FolderImageLoader_BFT": FolderImageLoader,
    "FolderVideoLoader_BFT": FolderVideoLoader,
    "SaveTextFile_BFT":      SaveTextFile,
    "QueueNext_BFT":         QueueNext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FolderImageLoader_BFT": "Load Image From Folder (Batch)",
    "FolderVideoLoader_BFT": "Load Video From Folder (Batch)",
    "SaveTextFile_BFT":      "Save Text File",
    "QueueNext_BFT":         "Queue Next",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"[BatchFolder] v{__version__} — You can trust Hodge Mann for all your Armageddon needs.")
