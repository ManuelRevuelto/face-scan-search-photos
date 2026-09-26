import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id          INTEGER PRIMARY KEY,
    source      TEXT NOT NULL,
    source_id   TEXT NOT NULL,
    name        TEXT,
    checksum    TEXT,
    modified_at TEXT,
    link        TEXT,
    width       INTEGER,
    height      INTEGER,
    status      TEXT,
    error       TEXT,
    indexed_at  TEXT,
    UNIQUE (source, source_id)
);
CREATE TABLE IF NOT EXISTS faces (
    id        INTEGER PRIMARY KEY,
    photo_id  INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,   -- bbox relativa (0-1)
    det_score REAL,
    embedding BLOB NOT NULL               -- float32[512] normalizado
);
CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id);
"""


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def query(self, sql: str, params=()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def known_photos(self, source: str) -> dict[str, tuple[str, str]]:
        """source_id -> (checksum, status)"""
        rows = self.query("SELECT source_id, checksum, status FROM photos WHERE source = ?", (source,))
        return {r["source_id"]: (r["checksum"], r["status"]) for r in rows}

    def save_photo(self, source, ref, width, height, faces, status="ok", error=None) -> int:
        """Inserta o actualiza la foto y reemplaza sus caras. Devuelve photo_id."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO photos (source, source_id, name, checksum, modified_at, link,
                                       width, height, status, error, indexed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (source, source_id) DO UPDATE SET
                       name = excluded.name, checksum = excluded.checksum,
                       modified_at = excluded.modified_at, link = excluded.link,
                       width = excluded.width, height = excluded.height,
                       status = excluded.status, error = excluded.error,
                       indexed_at = excluded.indexed_at""",
                (source, ref.source_id, ref.name, ref.checksum, ref.modified_at, ref.link,
                 width, height, status, error, now),
            )
            photo_id = self._conn.execute(
                "SELECT id FROM photos WHERE source = ? AND source_id = ?", (source, ref.source_id)
            ).fetchone()[0]
            self._conn.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))
            self._conn.executemany(
                "INSERT INTO faces (photo_id, x1, y1, x2, y2, det_score, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(photo_id, *f.bbox, f.det_score, f.embedding.astype(np.float32).tobytes()) for f in faces],
            )
        return photo_id

    def delete_photos(self, source: str, source_ids: list[str]) -> list[int]:
        if not source_ids:
            return []
        ids = []
        with self._lock, self._conn:
            for sid in source_ids:
                row = self._conn.execute(
                    "SELECT id FROM photos WHERE source = ? AND source_id = ?", (source, sid)
                ).fetchone()
                if row:
                    ids.append(row[0])
                    self._conn.execute("DELETE FROM photos WHERE id = ?", (row[0],))
        return ids

    def load_embeddings(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Devuelve (embeddings[N,512], photo_ids[N], face_ids[N])."""
        rows = self.query("SELECT id, photo_id, embedding FROM faces")
        if not rows:
            return np.zeros((0, 512), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64)
        emb = np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        return emb, np.array([r["photo_id"] for r in rows]), np.array([r["id"] for r in rows])

    def faces_version(self) -> tuple[int, int]:
        r = self.query("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM faces")[0]
        return r[0], r[1]

    def get_photo(self, photo_id: int):
        rows = self.query("SELECT * FROM photos WHERE id = ?", (photo_id,))
        return rows[0] if rows else None

    def get_face(self, face_id: int):
        rows = self.query("SELECT x1, y1, x2, y2 FROM faces WHERE id = ?", (face_id,))
        return rows[0] if rows else None

    def stats(self) -> dict:
        r = self.query(
            """SELECT source,
                      COUNT(*) AS photos,
                      SUM(status = 'ok') AS ok,
                      SUM(status = 'error') AS errors,
                      (SELECT COUNT(*) FROM faces f JOIN photos p2 ON p2.id = f.photo_id
                        WHERE p2.source = p.source) AS faces
               FROM photos p GROUP BY source"""
        )
        return {row["source"]: dict(row) for row in r}
