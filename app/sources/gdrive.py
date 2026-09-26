import io
import threading
from collections import deque
from pathlib import Path
from typing import Iterator

from ..config import resolve
from .base import PhotoRef, PhotoSource

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIME = "application/vnd.google-apps.folder"
FIELDS = "nextPageToken, files(id, name, mimeType, md5Checksum, modifiedTime, webViewLink)"


class GoogleDriveSource(PhotoSource):
    """Lee fotos directamente de Google Drive mediante la API (solo lectura)."""

    type = "gdrive"

    def __init__(self, name: str, credentials: str = "credentials.json", token: str = "data/token.json",
                 folder_id: str = ""):
        super().__init__(name)
        self.credentials_path = resolve(credentials)
        self.token_path = resolve(token)
        self.folder_id = folder_id or ""
        self._creds = None
        self._local = threading.local()
        self._auth_lock = threading.Lock()

    # --- autenticación -------------------------------------------------
    def authenticate(self):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow

        with self._auth_lock:
            creds = self._creds
            if creds is None and self.token_path.exists():
                creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
            if creds and creds.valid:
                self._creds = creds
                return creds
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.credentials_path.exists():
                    raise FileNotFoundError(
                        f"[{self.name}] Falta {self.credentials_path}. Descárgalo de Google Cloud Console "
                        "(credenciales OAuth tipo 'App de escritorio')."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_path), SCOPES)
                creds = flow.run_local_server(port=0)  # abre el navegador para autorizar
            Path(self.token_path).parent.mkdir(parents=True, exist_ok=True)
            self.token_path.write_text(creds.to_json(), encoding="utf-8")
            self._creds = creds
            return creds

    def _service(self):
        # Los objetos de googleapiclient no son thread-safe: uno por hilo
        svc = getattr(self._local, "svc", None)
        if svc is None:
            from googleapiclient.discovery import build

            svc = build("drive", "v3", credentials=self.authenticate(), cache_discovery=False)
            self._local.svc = svc
        return svc

    # --- PhotoSource ---------------------------------------------------
    def _iter_query(self, q: str):
        svc = self._service()
        token = None
        while True:
            resp = svc.files().list(
                q=q, fields=FIELDS, pageSize=1000, pageToken=token,
                supportsAllDrives=True, includeItemsFromAllDrives=True,
            ).execute(num_retries=5)
            yield from resp.get("files", [])
            token = resp.get("nextPageToken")
            if not token:
                return

    def _to_ref(self, f: dict, prefix: str = "") -> PhotoRef:
        return PhotoRef(
            source_id=f["id"],
            name=f"{prefix}{f['name']}",
            checksum=f.get("md5Checksum") or f.get("modifiedTime", ""),
            modified_at=f.get("modifiedTime", ""),
            link=f.get("webViewLink"),
        )

    def list_photos(self) -> Iterator[PhotoRef]:
        if not self.folder_id:
            for f in self._iter_query("mimeType contains 'image/' and trashed = false"):
                yield self._to_ref(f)
            return
        # Recorrido recursivo de la carpeta indicada
        pending = deque([(self.folder_id, "")])
        seen = set()
        while pending:
            folder, prefix = pending.popleft()
            if folder in seen:
                continue
            seen.add(folder)
            q = (f"'{folder}' in parents and trashed = false and "
                 f"(mimeType contains 'image/' or mimeType = '{FOLDER_MIME}')")
            for f in self._iter_query(q):
                if f["mimeType"] == FOLDER_MIME:
                    pending.append((f["id"], f"{prefix}{f['name']}/"))
                else:
                    yield self._to_ref(f, prefix)

    def read_bytes(self, source_id: str) -> bytes:
        from googleapiclient.http import MediaIoBaseDownload

        request = self._service().files().get_media(fileId=source_id, supportsAllDrives=True)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk(num_retries=5)
        return buf.getvalue()
