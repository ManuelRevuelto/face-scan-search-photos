import threading

import numpy as np

from .db import Database


class FaceSearcher:
    """Mantiene todos los embeddings en memoria y busca por similitud coseno (fuerza bruta)."""

    def __init__(self, db: Database):
        self.db = db
        self._lock = threading.Lock()
        self._version = None
        self._emb = self._photo_ids = self._face_ids = None

    def _ensure_loaded(self):
        version = self.db.faces_version()
        with self._lock:
            if version != self._version:
                self._emb, self._photo_ids, self._face_ids = self.db.load_embeddings()
                self._version = version
            return self._emb, self._photo_ids, self._face_ids

    def search(self, references: list[np.ndarray], threshold: float, limit: int = 500) -> list[dict]:
        """Devuelve una entrada por foto (su cara más parecida), ordenadas por similitud descendente.

        Con varias fotos de referencia se usa la máxima similitud contra cualquiera de ellas.
        """
        emb, photo_ids, face_ids = self._ensure_loaded()
        if len(emb) == 0 or not references:
            return []
        q = np.stack(references).astype(np.float32)          # (k, 512)
        scores = (emb @ q.T).max(axis=1)                      # (N,)
        idx = np.nonzero(scores >= threshold)[0]
        idx = idx[np.argsort(-scores[idx])]
        results, seen = [], set()
        for i in idx:
            pid = int(photo_ids[i])
            if pid in seen:
                continue
            seen.add(pid)
            results.append({"photo_id": pid, "face_id": int(face_ids[i]), "score": float(scores[i])})
            if len(results) >= limit:
                break
        return results
