# src/video_manager.py
# ═══════════════════════════════════════════════════════════
# VIDEO MANAGER
# Downloads videos from Google Drive on first use.
# Caches them locally so they don't re-download every time.
# ═══════════════════════════════════════════════════════════

import os
import json
import requests
import streamlit as st
from pathlib import Path


# Local cache directory for downloaded videos
CACHE_DIR = Path("data/cache")


def get_video_ids() -> dict:
    """Load video ID mapping from video_ids.json."""
    try:
        with open("video_ids.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def get_gdrive_download_url(file_id: str) -> str:
    """
    Convert Google Drive file ID to direct download URL.
    Uses the export=download format which works for most files.
    """
    return f"https://drive.google.com/uc?id={file_id}&export=download"


def download_video(file_id: str,
                   local_path: Path,
                   filename: str) -> bool:
    """
    Download a video from Google Drive to local cache.

    Handles Google Drive's virus scan warning for large files.
    Shows progress bar in Streamlit.

    Args:
        file_id:    Google Drive file ID
        local_path: Where to save the file
        filename:   Display name for progress bar

    Returns:
        True if successful, False if failed
    """
    url = get_gdrive_download_url(file_id)

    try:
        session  = requests.Session()
        response = session.get(url, stream=True)

        # Handle Google's "large file" confirmation page
        # Google shows a warning for files > 100MB
        for key, value in response.cookies.items():
            if "download_warning" in key:
                # Re-request with confirmation token
                params   = {"confirm": value, "id": file_id}
                url2     = "https://drive.google.com/uc"
                response = session.get(url2, params=params, stream=True)
                break

        # Check we got a valid response
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            # Got HTML instead of video — link is not public
            return False

        # Get file size for progress bar
        total_size = int(response.headers.get("content-length", 0))

        # Create parent directories if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress bar
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunks

        if total_size > 0:
            progress = st.progress(0, text=f"Downloading {filename}...")

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = min(downloaded / total_size, 1.0)
                        mb  = downloaded / (1024*1024)
                        tot = total_size  / (1024*1024)
                        progress.progress(
                            pct,
                            text=f"Downloading {filename}: "
                                 f"{mb:.1f}/{tot:.1f} MB"
                        )

        if total_size > 0:
            progress.progress(1.0, text=f"✅ {filename} ready")

        return True

    except Exception as e:
        st.error(f"Download failed for {filename}: {e}")
        return False


def ensure_video_available(video_path: str) -> str:
    """
    Make sure a video file is available locally.

    If it already exists → return path immediately.
    If not → download from Google Drive → return path.

    This is the MAIN function called by app.py.

    Args:
        video_path: Original path from scenarios.json
                    e.g. "data/chokepoint/videos/P1E_S1_C1.mp4"

    Returns:
        Local path to the video file (might be in cache)
        Returns None if download fails
    """
    # Check if file already exists locally
    if os.path.exists(video_path):
        return video_path

    # Get the filename from the path
    filename = os.path.basename(video_path)

    # Check cache
    cache_path = CACHE_DIR / filename
    if cache_path.exists():
        return str(cache_path)

    # Need to download from Google Drive
    video_ids = get_video_ids()

    if filename not in video_ids:
        st.error(f"No Google Drive ID configured for: {filename}")
        st.info("Add the file ID to video_ids.json")
        return None

    file_id = video_ids[filename]

    st.info(f"📥 Downloading {filename} from Google Drive...")
    success = download_video(file_id, cache_path, filename)

    if success:
        return str(cache_path)
    else:
        return None


def check_all_videos(scenarios: list) -> dict:
    """
    Check which videos are available locally vs need downloading.

    Returns:
        dict {scenario_id: {"available": bool, "cached": bool}}
    """
    video_ids = get_video_ids()
    status    = {}

    for s in scenarios:
        sid       = s["id"]
        vpath     = s["video_path"]
        filename  = os.path.basename(vpath)
        local_ok  = os.path.exists(vpath)
        cached_ok = (CACHE_DIR / filename).exists()
        has_id    = filename in video_ids

        status[sid] = {
            "available": local_ok or cached_ok,
            "local":     local_ok,
            "cached":    cached_ok,
            "has_drive_id": has_id,
            "filename":  filename
        }

    return status