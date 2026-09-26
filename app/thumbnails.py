from pathlib import Path

from PIL import Image

THUMB_SIZE = 400


class ThumbnailStore:
    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, photo_id: int) -> Path:
        return self.dir / f"{photo_id}.jpg"

    def save(self, photo_id: int, img: Image.Image):
        thumb = img.copy()
        thumb.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        thumb.save(self.path(photo_id), "JPEG", quality=82)

    def delete(self, photo_id: int):
        self.path(photo_id).unlink(missing_ok=True)
