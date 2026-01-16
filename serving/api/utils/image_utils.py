import base64
from config.settings import IMAGE_DIR


def decode_base64_image(base64_string: str) -> bytes:
    """
    Decodes a Base64 string into bytes.
    Removes data URI header (e.g., 'data:image/jpeg;base64,') if present.
    """
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    return base64.b64decode(base64_string)


def save_defect_image(image_data: bytes, image_id: str, timestamp: str) -> str:
    """
    Saves image bytes to disk with a filename based on ID and timestamp.
    Returns the relative path to the saved image.

    Format: images/defects/{timestamp}_{image_id}.jpg
    Note: Timestamp usually contains ':', which is invalid in filenames on Windows
          and problematic on Mac/Linux. We will sanitize it.
    """
    # Sanitize timestamp for filename (replace : with -)
    sanitized_timestamp = timestamp.replace(":", "-")

    filename = f"{sanitized_timestamp}_{image_id}.jpg"

    # Ensure directory exists (redundant if main.py does it, but safer)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    file_path = IMAGE_DIR / filename

    with open(file_path, "wb") as f:
        f.write(image_data)

    # Return relative path for storage in DB
    return str(file_path.relative_to(IMAGE_DIR.parent.parent))
