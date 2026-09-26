"""Línea de comandos del buscador de fotos por cara.

    python cli.py auth                      # autoriza Google Drive (abre el navegador)
    python cli.py index [--source drive]    # indexa (incremental)
    python cli.py search yo1.jpg yo2.jpg    # busca coincidencias
    python cli.py stats
    python cli.py serve                     # web en http://localhost:8000
"""
import argparse
import logging
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Buscador de fotos por reconocimiento facial")
    p.add_argument("--config", help="ruta al config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("auth", help="autoriza el acceso a las fuentes de Google Drive")

    pi = sub.add_parser("index", help="indexa las fuentes")
    pi.add_argument("--source", action="append", help="solo esta fuente (repetible)")

    ps = sub.add_parser("search", help="busca fotos donde aparece la persona")
    ps.add_argument("images", nargs="+", type=Path, help="una o varias fotos de referencia")
    ps.add_argument("--threshold", type=float)
    ps.add_argument("--limit", type=int, default=100)

    sub.add_parser("stats", help="resumen del índice")

    pv = sub.add_parser("serve", help="arranca la web local")
    pv.add_argument("--host", default="127.0.0.1")
    pv.add_argument("--port", type=int, default=8000)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "serve":
        import os

        import uvicorn

        if args.config:
            os.environ["FACESCAN_CONFIG"] = args.config
        uvicorn.run("app.api:app", host=args.host, port=args.port)
        return

    from app.core import Services

    svc = Services(args.config)

    if args.cmd == "auth":
        for s in svc.sources.values():
            if s.type == "gdrive":
                s.authenticate()
                print(f"[{s.name}] autorizado")

    elif args.cmd == "index":
        def progress(st):
            print(f"\r[{st['source']}] {st['done']}/{st['total']}  caras: {st['faces']}  "
                  f"errores: {st['errors']}", end="", flush=True)

        try:
            st = svc.indexer.run(args.source, progress)
        except KeyboardInterrupt:
            print("\nInterrumpido. El progreso está guardado; vuelve a lanzar 'index' para continuar.")
            return
        print(f"\n{st['message']}. Sin cambios: {st['skipped']}, eliminadas: {st['removed']}")

    elif args.cmd == "search":
        try:
            results = svc.search([img.read_bytes() for img in args.images], args.threshold, args.limit)
        except ValueError as e:
            print(f"Error: {e}")
            return 1
        if not results:
            print("Sin coincidencias.")
        for r in results:
            photo = svc.db.get_photo(r["photo_id"])
            print(f"{r['score']:.3f}  [{photo['source']}] {photo['name']}  {photo['link'] or ''}")

    elif args.cmd == "stats":
        stats = svc.db.stats()
        if not stats:
            print("Índice vacío. Ejecuta: python cli.py index")
        for name, s in stats.items():
            print(f"{name}: {s['photos']} fotos ({s['ok']} ok, {s['errors']} errores), {s['faces']} caras")


if __name__ == "__main__":
    sys.exit(main())
