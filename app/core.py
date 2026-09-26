from functools import cached_property

from .config import load_config, resolve
from .db import Database
from .face_engine import FaceEngine
from .indexer import Indexer
from .search import FaceSearcher
from .sources import build_sources
from .thumbnails import ThumbnailStore


class Services:
    """Construye y comparte los componentes de la app (usado por la CLI y por la API)."""

    def __init__(self, config_path: str | None = None):
        self.cfg = load_config(config_path)
        self.db = Database(resolve(self.cfg["database"]))
        self.thumbs = ThumbnailStore(resolve(self.cfg["thumbnails_dir"]))
        self.sources = build_sources(self.cfg)
        self.searcher = FaceSearcher(self.db)

    @cached_property
    def engine(self) -> FaceEngine:
        # Carga perezosa: el modelo tarda unos segundos (y la 1ª vez se descarga, ~280 MB)
        return FaceEngine(
            model_dir=resolve(self.cfg["models_dir"]),
            det_size=self.cfg["det_size"],
            providers=self.cfg["providers"],
            max_side=self.cfg["max_side"],
        )

    @cached_property
    def indexer(self) -> Indexer:
        return Indexer(self.db, self.engine, self.sources, self.thumbs, self.cfg["download_workers"])

    def search(self, reference_images: list[bytes], threshold: float | None = None, limit: int = 500):
        refs = [self.engine.reference_embedding(data) for data in reference_images]
        threshold = self.cfg["threshold"] if threshold is None else threshold
        return self.searcher.search(refs, threshold, limit)
