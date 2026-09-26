# Face Scan Search Photos

Busca en miles de fotos (Google Drive, carpeta local o NAS) aquellas en las que apareces, a partir de una o varias fotos tuyas. El reconocimiento facial se ejecuta **100% en local** (modelos InsightFace `buffalo_l`: detector SCRFD + ArcFace, vía `onnxruntime`); ninguna cara sale de tu equipo.

## Cómo funciona

1. **Indexado** (una vez, luego incremental): cada foto se descarga/lee, se detectan las caras y se guarda un *embedding* de 512 números por cara en SQLite (`data/faces.db`) junto con una miniatura.
2. **Búsqueda**: se calcula el embedding de tu foto y se compara con todos los guardados (similitud coseno). Las fotos por encima del umbral son coincidencias.

## Instalación (Windows)

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy config.example.yaml config.yaml
```

La primera ejecución descarga los modelos (~280 MB) a `data/models`.

## Carpeta pública de Google Drive (sin credenciales)

Si la carpeta está compartida como "cualquiera con el enlace", basta con poner su URL o ID en una fuente `gdrive_public` (ver `config.example.yaml`). No hace falta nada en Google Cloud: se lee la vista pública de la carpeta (recursiva) y se descargan versiones reducidas de las fotos (`download_size`). Si algún día Google cambia esa vista, crea una **API key** (Google Cloud → Credenciales → Crear API key, con Drive API habilitada) y ponla en `api_key`.

## Tus fotos de referencia

No se indexan: se suben en la web (arrastrándolas) o se pasan a `cli.py search`. Puedes guardarlas en `mis-fotos/` (ignorada por git). Para fotos de eventos, usa referencias parecidas a tu aspecto de ese día (gorra, gafas, de perfil...).

## Configurar Google Drive privado (OAuth)

1. En [Google Cloud Console](https://console.cloud.google.com/) crea un proyecto y **habilita "Google Drive API"**.
2. *Pantalla de consentimiento OAuth*: tipo "Externo", añádete como **usuario de prueba**.
3. *Credenciales → Crear credenciales → ID de cliente OAuth → App de escritorio*. Descarga el JSON y guárdalo como `credentials.json` en la raíz del proyecto.
4. En `config.yaml` deja `folder_id: ""` para todo el Drive, o pon el ID de una carpeta (lo que aparece en la URL `drive.google.com/drive/folders/<ID>`).
5. `.venv\Scripts\python cli.py auth` → se abre el navegador para autorizar (acceso de **solo lectura**). El token queda en `data/token.json`.

## Carpeta local o NAS

Añade otra fuente en `config.yaml`; puedes tener varias a la vez:

```yaml
  - name: nas
    type: local
    path: '\\NAS\fotos'   # o 'D:/Fotos'
```

## Uso

```powershell
.venv\Scripts\python cli.py index                 # indexa todas las fuentes (se puede interrumpir y continuar)
.venv\Scripts\python cli.py index --source drive  # solo una fuente
.venv\Scripts\python cli.py search yo1.jpg yo2.jpg
.venv\Scripts\python cli.py stats
.venv\Scripts\python cli.py serve                 # web en http://localhost:8000
```

En la web: arrastra tus fotos de referencia, ajusta el umbral, pulsa **Buscar**, y selecciona resultados para **copiarlos** a una carpeta. El botón **Indexar / actualizar** procesa solo fotos nuevas o modificadas.

## Consejos de precisión

- Usa **varias fotos de referencia** (de frente, de perfil, con/sin gafas, de distintas épocas): se toma la mejor similitud contra cualquiera de ellas.
- Umbral: 0.35–0.50 suele ir bien. Baja si faltan fotos; sube si aparecen desconocidos.
- Caras muy pequeñas, borrosas o de perfil extremo pueden no detectarse. Si tus fotos son de alta resolución con caras pequeñas, sube `max_side` (más lento).
- Con GPU NVIDIA: instala `onnxruntime-gpu` y pon `providers: [CUDAExecutionProvider, CPUExecutionProvider]`.

## Privacidad

`data/` (embeddings = datos biométricos, miniaturas, token de Google) y `credentials.json` están en `.gitignore`: no los subas a ningún repositorio. Los modelos preentrenados de InsightFace son para uso **no comercial**.
