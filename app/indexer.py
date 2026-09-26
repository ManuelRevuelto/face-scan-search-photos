import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from .db import Database
from .face_engine import FaceEngine, load_image
from .sources import PhotoSource
from .thumbnails import ThumbnailStore

log = logging.getLogger(__name__)


class Indexer:
    """Recorre las fuentes y guarda los embeddings de las caras de cada foto nueva o modificada.

    Cada foto se confirma en la BD al terminarla, así que se puede parar y reanudar en cualquier momento.
    """

    def __init__(self, db: Database, engine: FaceEngine, sources: dict[str, PhotoSource],
                 thumbs: ThumbnailStore, download_workers: int = 6):
        self.db = db
        self.engine = engine
        self.sources = sources
        self.thumbs = thumbs
        self.workers = max(1, download_workers)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.status = self._empty_status()

    @staticmethod
    def _empty_status():
        return {"running": False, "source": None, "phase": "idle", "total": 0, "done": 0,
                "skipped": 0, "faces": 0, "errors": 0, "removed": 0, "message": "",
                "started_at": None, "finished_at": None}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, source_names: list[str] | None = None) -> bool:
        if self.running:
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, args=(source_names,), daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()

    def run(self, source_names: list[str] | None = None, progress=None):
        self.status = {**self._empty_status(), "running": True, "started_at": time.time()}
        failures = []
        try:
            for name in source_names or list(self.sources):
                if self._stop.is_set():
                    break
                try:
                    self._index_source(self.sources[name], progress)
                except Exception as e:  # noqa: BLE001 - una fuente caída no debe impedir indexar las demás
                    log.exception("Error indexando la fuente %s", name)
                    failures.append(f"{name}: {e}")
        finally:
            if self._stop.is_set():
                msg = "Detenido"
            elif failures:
                msg = "Errores: " + " | ".join(failures)
            else:
                msg = "Completado"
            self.status.update(running=False, phase="idle", finished_at=time.time(), message=msg)
        return self.status

    def _index_source(self, source: PhotoSource, progress):
        st = self.status
        st.update(source=source.name, phase="listando", total=0, done=0, skipped=0)
        known = self.db.known_photos(source.name)
        refs = list(source.list_photos())
        listed = {r.source_id for r in refs}

        # Fotos que ya no existen en la fuente
        gone = [sid for sid in known if sid not in listed]
        for pid in self.db.delete_photos(source.name, gone):
            self.thumbs.delete(pid)
        st["removed"] += len(gone)

        todo = [r for r in refs if known.get(r.source_id) != (r.checksum, "ok")]
        st.update(phase="indexando", total=len(todo), skipped=len(refs) - len(todo))
        log.info("[%s] %d fotos, %d por indexar", source.name, len(refs), len(todo))

        # Descargas en paralelo (I/O) e inferencia en este hilo, con una ventana acotada en memoria
        with ThreadPoolExecutor(self.workers) as pool:
            pending = deque()
            it = iter(todo)
            while True:
                while len(pending) < self.workers * 2 and not self._stop.is_set():
                    ref = next(it, None)
                    if ref is None:
                        break
                    pending.append((ref, pool.submit(source.read_bytes, ref.source_id)))
                if not pending:
                    break
                ref, fut = pending.popleft()
                self._process(source, ref, fut)
                st["done"] += 1
                if progress:
                    progress(st)
            for _, fut in pending:
                fut.cancel()

    def _process(self, source, ref, fut):
        try:
            img = load_image(fut.result(), self.engine.max_side)
            faces = self.engine.detect(img)
            pid = self.db.save_photo(source.name, ref, img.width, img.height, faces)
            self.thumbs.save(pid, img)
            self.status["faces"] += len(faces)
        except Exception as e:  # noqa: BLE001 - una foto corrupta no debe parar el indexado
            log.warning("[%s] %s: %s", source.name, ref.name, e)
            self.db.save_photo(source.name, ref, None, None, [], status="error", error=str(e)[:500])
            self.status["errors"] += 1
