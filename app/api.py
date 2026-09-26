import base64
import logging
import mimetypes
import os
import re
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, RedirectResponse, Response
from pydantic import BaseModel

from .core import Services

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

STATIC = Path(__file__).parent / "static"

# Despliegue público: APP_PASSWORD activa el login HTTP Basic y FACESCAN_READONLY desactiva
# indexar y exportar (escriben en el servidor). En local, sin estas variables, todo sigue igual.
APP_USER = os.environ.get("APP_USER", "admin")
APP_PASSWORD = os.environ.get("APP_PASSWORD")
READ_ONLY = os.environ.get("FACESCAN_READONLY", "").lower() in ("1", "true", "yes")

app = FastAPI(title="Face Scan Search Photos")
svc = Services()


def _authorized(header: str | None) -> bool:
    if not header or not header.startswith("Basic "):
        return False
    try:
        user, _, password = base64.b64decode(header[6:]).decode("utf-8").partition(":")
    except ValueError:
        return False
    return secrets.compare_digest(user.encode(), APP_USER.encode()) and secrets.compare_digest(
        password.encode(), APP_PASSWORD.encode()
    )


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if APP_PASSWORD and not _authorized(request.headers.get("authorization")):
        return Response("Autenticación requerida", status_code=401,
                        headers={"WWW-Authenticate": 'Basic realm="Face Scan", charset="UTF-8"'})
    return await call_next(request)


def writable():
    if READ_ONLY:
        raise HTTPException(403, "Desactivado en modo solo lectura")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    return {
        "threshold": svc.cfg["threshold"],
        "read_only": READ_ONLY,
        "sources": [{"name": s.name, "type": s.type} for s in svc.sources.values()],
        "stats": svc.db.stats(),
    }


@app.post("/api/search")
async def search(files: list[UploadFile] = File(...), threshold: float = Form(None), limit: int = Form(500)):
    images = [await f.read() for f in files]
    try:
        results = await run_in_threadpool(svc.search, images, threshold, limit)
    except ValueError as e:
        raise HTTPException(400, str(e))
    out = []
    for r in results:
        photo = svc.db.get_photo(r["photo_id"])
        face = svc.db.get_face(r["face_id"])
        out.append({
            **r,
            "name": photo["name"],
            "source": photo["source"],
            "bbox": [face["x1"], face["y1"], face["x2"], face["y2"]] if face else None,
            "thumb_url": f"/api/thumb/{r['photo_id']}",
            "open_url": photo["link"] or f"/api/photo/{r['photo_id']}",
        })
    return {"results": out}


@app.post("/api/index", dependencies=[Depends(writable)])
def start_index(source: str | None = None):
    if source and source not in svc.sources:
        raise HTTPException(404, f"Fuente desconocida: {source}")
    started = svc.indexer.start([source] if source else None)
    return {"started": started, "status": svc.indexer.status}


@app.post("/api/index/stop", dependencies=[Depends(writable)])
def stop_index():
    svc.indexer.stop()
    return {"status": svc.indexer.status}


@app.get("/api/index/status")
def index_status():
    # No se fuerza la carga del modelo solo para consultar el estado
    idx = svc.__dict__.get("indexer")
    return idx.status if idx else {"running": False, "phase": "idle", "message": ""}


@app.get("/api/errors")
def errors():
    rows = svc.db.query("SELECT source, name, error, indexed_at FROM photos WHERE status = 'error' ORDER BY indexed_at DESC")
    return [dict(r) for r in rows]


@app.get("/api/thumb/{photo_id}")
def thumb(photo_id: int):
    path = svc.thumbs.path(photo_id)
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


def _read_original(photo_id: int):
    photo = svc.db.get_photo(photo_id)
    if not photo or photo["source"] not in svc.sources:
        raise HTTPException(404)
    return photo, svc.sources[photo["source"]].read_bytes(photo["source_id"])


@app.get("/api/photo/{photo_id}")
def photo(photo_id: int):
    record = svc.db.get_photo(photo_id)
    if record and record["link"]:
        return RedirectResponse(record["link"])
    record, data = _read_original(photo_id)
    media = mimetypes.guess_type(record["name"])[0] or "application/octet-stream"
    return Response(data, media_type=media)


class ExportRequest(BaseModel):
    photo_ids: list[int]
    destination: str


@app.post("/api/export", dependencies=[Depends(writable)])
def export(req: ExportRequest):
    """Copia los originales de las fotos seleccionadas a una carpeta local."""
    dest = Path(req.destination).expanduser()
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(400, f"No se puede usar la carpeta destino: {e}")
    copied, errors = 0, []
    for pid in req.photo_ids:
        try:
            record, data = _read_original(pid)
            name = re.sub(r'[\\/:*?"<>|]', "_", record["name"])
            target = dest / name
            if target.exists():
                target = dest / f"{target.stem}_{pid}{target.suffix}"
            target.write_bytes(data)
            copied += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"{pid}: {e}")
    return {"copied": copied, "errors": errors, "destination": str(dest)}
