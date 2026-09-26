import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .base import IMAGE_EXTENSIONS, PhotoRef, PhotoSource


class LocalFolderSource(PhotoSource):
    """Carpeta local, unidad de red o NAS (p. ej. D:/Fotos o \\\\NAS\\fotos)."""

    type = "local"

    def __init__(self, name: str, path: str):
        super().__init__(name)
        self.root = Path(path)

    def list_photos(self) -> Iterator[PhotoRef]:
        # Se comprueba aquí y no en __init__ para que un NAS apagado no impida arrancar la app
        if not self.root.is_dir():
            raise FileNotFoundError(f"[{self.name}] La carpeta no existe o no es accesible: {self.root}")
        for dirpath, _, filenames in os.walk(self.root):
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                rel = p.relative_to(self.root).as_posix()
                yield PhotoRef(
                    source_id=rel,
                    name=rel,
                    checksum=f"{st.st_size}-{st.st_mtime_ns}",
                    modified_at=datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                )

    def read_bytes(self, source_id: str) -> bytes:
        path = (self.root / source_id).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("Ruta fuera de la carpeta de la fuente")
        return path.read_bytes()
