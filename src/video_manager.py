# src/video_manager.py

import os
import json
import requests
import streamlit as st
from pathlib import Path

CACHE_DIR = Path("/tmp/securegate_videos")


def get_video_ids() -> dict:
    """Load video ID mapping."""
    try:
        with open("video_ids.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("video_ids.json not found in repository")
        return {}
    except json.JSONDecodeError as e:
        st.error(f"video_ids.json is invalid JSON: {e}")
        return {}


def download_from_gdrive(file_id: str,
                          dest_path: Path,
                          filename: str) -> bool:
    """
    Download file from Google Drive.
    Handles both small files and large files
    (large files show a virus scan warning page).
    """
    CHUNK_SIZE = 32768

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    try:
        session  = requests.Session()
        url      = "https://drive.google.com/uc?export=download"
        params   = {"id": file_id}
        response = session.get(url, params=params, stream=True,
                               timeout=30)

        token = get_confirm_token(response)
        if token:
            params["confirm"] = token
            response = session.get(url, params=params, stream=True,
                                   timeout=30)

        # Check content type
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type and \
           "video" not in content_type:
            st.error(
                f"Google Drive returned HTML for {filename}. "
                f"Make sure sharing is set to "
                f"'Anyone with the link'."
            )
            return False

        # Get size
        total = int(response.headers.get("Content-Length", 0))

        # Download
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = 0

        prog_bar = st.progress(0.0,
                               text=f"⬇️ Downloading {filename}...")

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(
                    chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = min(downloaded / total, 1.0)
                        mb  = downloaded / 1_048_576
                        tmb = total      / 1_048_576
                        prog_bar.progress(
                            pct,
                            text=(f"⬇️ {filename}: "
                                  f"{mb:.1f} / {tmb:.1f} MB")
                        )

        prog_bar.progress(1.0, text=f"✅ {filename} downloaded")

        # Verify file is not empty
        if dest_path.stat().st_size < 1000:
            dest_path.unlink()
            st.error(
                f"Downloaded file is too small — "
                f"Drive link may be broken for {filename}"
            )
            return False

        return True

    except requests.exceptions.Timeout:
        st.error(f"Timeout downloading {filename}. Try again.")
        return False
    except Exception as e:
        st.error(f"Download error for {filename}: {e}")
        return False


def ensure_video_available(video_path: str) -> str:
    """
    Ensure video file is available locally.

    Priority order:
    1. File exists at original path → use it
    2. File exists in /tmp cache → use it
    3. Download from Google Drive → cache it → use it

    Returns local path string or None if unavailable.
    """
    # 1. Check original path
    if os.path.exists(video_path) and \
       os.path.getsize(video_path) > 1000:
        return video_path

    filename   = os.path.basename(video_path)
    cache_path = CACHE_DIR / filename

    # 2. Check cache
    if cache_path.exists() and cache_path.stat().st_size > 1000:
        return str(cache_path)

    # 3. Download from Google Drive
    video_ids = get_video_ids()

    if not video_ids:
        st.error("No video IDs configured. "
                 "Check video_ids.json in your repo.")
        return None

    if filename not in video_ids:
        st.error(
            f"No Google Drive ID for: **{filename}**\n\n"
            f"Add it to `video_ids.json` in your repository.\n\n"
            f"Current keys in video_ids.json: "
            f"{list(video_ids.keys())}"
        )
        return None

    file_id = video_ids[filename]

    if not file_id or file_id == "YOUR_FILE_ID_HERE":
        st.error(
            f"File ID not set for {filename}. "
            f"Update video_ids.json with the real Google Drive ID."
        )
        return None

    st.info(f"📥 First time loading **{filename}** — "
            f"downloading from Google Drive...")

    success = download_from_gdrive(file_id, cache_path, filename)

    if success:
        st.success(f"✅ {filename} ready")
        return str(cache_path)

    return None