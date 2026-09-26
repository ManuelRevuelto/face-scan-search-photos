"""Carpeta pública de Google Drive ("cualquiera con el enlace"), sin OAuth.

- Con `api_key`: lista los ficheros con la Drive API (fiable, incluye md5).
- Sin `api_key`: lee la vista pública embebida de la carpeta (sin configurar nada en Google Cloud,
  pero depende del HTML de Google y podría romperse si lo cambian).

Las fotos se descargan como versión reducida (`download_size` px de lado mayor) para ahorrar ancho de banda.
"""
import html
import http.client
import logging
import re
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Iterator

from .base import IMAGE_EXTENSIONS, PhotoRef, PhotoSource

log = logging.getLogger(__name__)

EMBED_URL = "https://drive.google.com/embeddedfolderview?id={id}"
THUMB_URL = "https://drive.google.com/thumbnail?id={id}&sz=w{size}"
VIEW_URL = "https://drive.google.com/file/d/{id}/view"
FOLDER_MIME = "application/vnd.google-apps.folder"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"

_ENTRY_RE = re.compile(
    r'<div class="flip-entry" id="entry-(?P<id>[\w-]+)".*?<a href="(?P<href>[^"]+)".*?'
    r'<div class="flip-entry-title">(?P<title>[^<]*)</div>.*?'
    r'<div class="flip-entry-last-modified"><div>(?P<modified>[^<]*)</div>',
    re.S,
)


def _http_get(url: str, retries: int = 5, timeout: int = 60) -> bytes:
    delay = 2.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code not in (403, 429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            log.debug("HTTP %s en %s, reintentando en %.0fs", e.code, url, delay)
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
            # Incluye cortes de conexión de Drive (RemoteDisconnected) en descargas masivas
            if attempt == retries - 1:
                raise
        time.sleep(delay)
        delay *= 2
    raise RuntimeError("inalcanzable")


class PublicDriveFolderSource(PhotoSource):
    type = "gdrive_public"

    def __init__(self, name: str, folder_id: str, api_key: str = "", download_size: int = 2000):
        super().__init__(name)
        self.folder_id = self._parse_id(folder_id)
        self.api_key = api_key or ""
        self.download_size = download_size

    @staticmethod
    def _parse_id(value: str) -> str:
        # Acepta tanto el ID como la URL completa de la carpeta
        m = re.search(r"folders/([\w-]+)", value)
        return m.group(1) if m else value.strip()

    # --- listado ---------------------------------------------------------
    def list_photos(self) -> Iterator[PhotoRef]:
        lister = self._list_api if self.api_key else self._list_embedded
        pending = deque([(self.folder_id, "")])
        seen = set()
        while pending:
            folder, prefix = pending.popleft()
            if folder in seen:
                continue
            seen.add(folder)
            for kind, fid, name, checksum, modified in lister(folder):
                if kind == "folder":
                    pending.append((fid, f"{prefix}{name}/"))
                elif "." not in name or "." + name.rsplit(".", 1)[-1].lower() in IMAGE_EXTENSIONS:
                    yield PhotoRef(fid, f"{prefix}{name}", checksum, modified, VIEW_URL.format(id=fid))

    def _list_embedded(self, folder_id: str):
        page = _http_get(EMBED_URL.format(id=folder_id)).decode("utf-8", "replace")
        if "flip-entry" not in page and "flip-view" not in page:
            raise PermissionError(f"[{self.name}] La carpeta {folder_id} no es pública o no existe")
        for m in _ENTRY_RE.finditer(re.sub(r">\s+<", "><", page)):
            kind = "folder" if "/drive/folders/" in m["href"] else "file"
            modified = html.unescape(m["modified"])
            # Sin API no hay md5: el ID de Drive cambia si se sube un fichero nuevo.
            # La fecha no entra en el checksum porque la vista pública cambia de formato ("12:58 am" -> "Sep 23")
            yield kind, m["id"], html.unescape(m["title"]), m["id"], modified

    def _list_api(self, folder_id: str):
        import json
        import urllib.parse

        token = ""
        while True:
            params = {
                "q": f"'{folder_id}' in parents and trashed = false",
                "fields": "nextPageToken, files(id, name, mimeType, md5Checksum, modifiedTime)",
                "pageSize": 1000, "key": self.api_key,
                "supportsAllDrives": "true", "includeItemsFromAllDrives": "true",
            }
            if token:
                params["pageToken"] = token
            data = json.loads(_http_get("https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(params)))
            for f in data.get("files", []):
                if f["mimeType"] == FOLDER_MIME:
                    yield "folder", f["id"], f["name"], "", ""
                elif f["mimeType"].startswith("image/"):
                    yield "file", f["id"], f["name"], f.get("md5Checksum") or f.get("modifiedTime", ""), f.get("modifiedTime", "")
            token = data.get("nextPageToken")
            if not token:
                return

    # --- descarga --------------------------------------------------------
    def read_bytes(self, source_id: str) -> bytes:
        data = _http_get(THUMB_URL.format(id=source_id, size=self.download_size))
        if data[:15].lstrip().lower().startswith(b"<!doctype") or data[:5].lower() == b"<html":
            raise PermissionError("Drive devolvió una página HTML en lugar de la imagen (¿acceso restringido?)")
        return data
