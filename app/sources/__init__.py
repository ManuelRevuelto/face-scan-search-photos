from .base import PhotoRef, PhotoSource
from .gdrive import GoogleDriveSource
from .gdrive_public import PublicDriveFolderSource
from .local import LocalFolderSource

SOURCE_TYPES = {
    "local": LocalFolderSource,
    "gdrive": GoogleDriveSource,
    "gdrive_public": PublicDriveFolderSource,
}


def build_sources(cfg: dict) -> dict[str, PhotoSource]:
    sources = {}
    for spec in cfg.get("sources", []):
        spec = dict(spec)
        kind = spec.pop("type")
        name = spec.pop("name")
        if kind not in SOURCE_TYPES:
            raise ValueError(f"Tipo de fuente desconocido: {kind}")
        sources[name] = SOURCE_TYPES[kind](name=name, **spec)
    return sources


__all__ = ["PhotoRef", "PhotoSource", "build_sources", "SOURCE_TYPES"]
