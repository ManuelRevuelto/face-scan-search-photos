from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


@dataclass
class PhotoRef:
    source_id: str      # identificador estable dentro de la fuente (fileId de Drive, ruta relativa...)
    name: str
    checksum: str       # cambia si el fichero cambia -> permite indexado incremental
    modified_at: str
    link: str | None = None  # URL para abrir el original (si la fuente la ofrece)


class PhotoSource(ABC):
    type: str = ""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def list_photos(self) -> Iterator[PhotoRef]:
        ...

    @abstractmethod
    def read_bytes(self, source_id: str) -> bytes:
        ...
