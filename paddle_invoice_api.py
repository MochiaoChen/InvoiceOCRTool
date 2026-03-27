"""
paddle_invoice_api.py — Reconstructed from bytecode disassembly.
Serves a local PaddleOCR API and optionally the invoice-tool.html frontend.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import webbrowser
from functools import lru_cache
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 8866
DEFAULT_FRONTEND_NAME: str = "invoice-tool.html"
RUNTIME_LOG_NAME: str = "api.log"

IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PDF_EXTS: set[str] = {".pdf"}
ALL_EXTS: set[str] = IMAGE_EXTS | PDF_EXTS

# PP-OCRv5 model directory names bundled with the application.
REQUIRED_MODEL_NAMES: list[str] = [
    "PP-OCRv5_mobile_det",
    "PP-OCRv5_mobile_rec",
    "ch_ppocr_mobile_v2.0_cls",
]

# Files that must exist inside each model directory to consider it complete.
REQUIRED_MODEL_FILES: list[str] = [
    "inference.pdmodel",
    "inference.pdiparams",
]

# ---------------------------------------------------------------------------
# Path helpers — resolved once at import time
# ---------------------------------------------------------------------------

def get_app_root() -> Path:
    """Return the application root directory."""
    env_root = os.environ.get("PADDLE_INVOICE_APP_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_resource_root() -> Path:
    """Return the resource root (bundled assets).

    Priority: sys._MEIPASS (PyInstaller onefile) → _internal/ sibling
    (PyInstaller onedir) → get_app_root().
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass).resolve()
    if getattr(sys, "frozen", False):
        internal_dir = Path(sys.executable).resolve().parent / "_internal"
        if internal_dir.exists():
            return internal_dir.resolve()
    return get_app_root()


def get_safe_runtime_root() -> Path:
    """Return a writable runtime directory under %PUBLIC%\\Documents\\InvoiceOCRTool."""
    public_root = Path(os.environ.get("PUBLIC", r"C:\Users\Public"))
    return public_root / "Documents" / "InvoiceOCRTool"


# Module-level singletons derived from the path helpers.
RESOURCE_ROOT: Path = get_resource_root()
_SAFE_RUNTIME_ROOT: Path = get_safe_runtime_root()
RUNTIME_LOG_DIR: Path = _SAFE_RUNTIME_ROOT
RUNTIME_LOG_FILE: Path = _SAFE_RUNTIME_ROOT / RUNTIME_LOG_NAME

# ---------------------------------------------------------------------------
# Model sync helpers
# ---------------------------------------------------------------------------

def _sync_tree(src: Path, dst: Path) -> None:
    """Recursively copy src → dst, skipping .cache directories."""
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name == ".cache":
            continue
        target = dst / item.name
        if item.is_dir():
            _sync_tree(item, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _model_dir_is_complete(model_dir: Path) -> bool:
    """Return True if model_dir exists and contains all required files."""
    return model_dir.is_dir() and all(
        (model_dir / name).is_file() for name in REQUIRED_MODEL_FILES
    )


def _sync_bundled_models(bundled_models_dir: Path, safe_models_dir: Path) -> None:
    """Copy any incomplete bundled models to the writable safe location."""
    if not bundled_models_dir.exists():
        return
    for model_name in REQUIRED_MODEL_NAMES:
        src_dir = bundled_models_dir / model_name
        if not src_dir.is_dir():
            continue
        dst_dir = safe_models_dir / model_name
        if _model_dir_is_complete(dst_dir):
            continue
        _sync_tree(src_dir, dst_dir)


# ---------------------------------------------------------------------------
# Runtime environment configuration
# ---------------------------------------------------------------------------

def configure_runtime_environment(app_root: Path | None = None) -> Path:
    """Set all cache/temp env-vars to writable locations and sync models.

    Returns the resolved app root.
    """
    root: Path = app_root if app_root is not None else get_app_root()

    bundled_cache_home = get_resource_root() / "runtime" / "paddlex_cache"
    bundled_models_dir = bundled_cache_home / "official_models"

    safe_runtime_root = get_safe_runtime_root()
    safe_runtime_root.mkdir(parents=True, exist_ok=True)

    safe_cache_home = safe_runtime_root / "runtime" / "paddlex_cache"
    safe_models_dir = safe_cache_home / "official_models"
    safe_temp_dir = safe_runtime_root / "runtime" / "temp"

    safe_cache_home.mkdir(parents=True, exist_ok=True)
    safe_models_dir.mkdir(parents=True, exist_ok=True)
    safe_temp_dir.mkdir(parents=True, exist_ok=True)

    _sync_bundled_models(bundled_models_dir, safe_models_dir)

    os.environ["PADDLE_PDX_CACHE_HOME"] = str(safe_cache_home)
    os.environ["TMP"] = str(safe_temp_dir)
    os.environ["TEMP"] = str(safe_temp_dir)
    os.environ["HF_HOME"] = str(safe_runtime_root / "runtime" / "hf_home")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(safe_runtime_root / "runtime" / "hf_cache")
    os.environ["MODELSCOPE_CACHE"] = str(safe_runtime_root / "runtime" / "modelscope")
    os.environ["XDG_CACHE_HOME"] = str(safe_runtime_root / "runtime" / "xdg_cache")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    return root


# ---------------------------------------------------------------------------
# Frontend path
# ---------------------------------------------------------------------------

def get_default_frontend_path() -> Path:
    """Return the default path to invoice-tool.html."""
    return RESOURCE_ROOT / DEFAULT_FRONTEND_NAME


# ---------------------------------------------------------------------------
# Runtime logging
# ---------------------------------------------------------------------------

def log_runtime_exception(context: str, exc: Exception) -> None:
    """Append a formatted exception to the runtime log file."""
    RUNTIME_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    with RUNTIME_LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {context}\n{formatted}\n")


def log_runtime_note(message: str) -> None:
    """Append a plain note to the runtime log file."""
    RUNTIME_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with RUNTIME_LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def get_numpy():  # type: ignore[return]
    import numpy as np
    return np


def get_cv2():  # type: ignore[return]
    import cv2
    return cv2


# ---------------------------------------------------------------------------
# OCR engine lifecycle
# ---------------------------------------------------------------------------

_OCR_READY: threading.Event = threading.Event()
_OCR_LAST_ERROR: str | None = None
_OCR_WARMUP_LOCK: threading.Lock = threading.Lock()
_OCR_WARMUP_STARTED: bool = False


@lru_cache(maxsize=1)
def get_ocr_engine():  # type: ignore[return]
    """Create and cache the PaddleOCR engine (PP-OCRv5, CPU)."""
    global _OCR_LAST_ERROR
    try:
        from paddleocr import PaddleOCR
        engine = PaddleOCR(
            ocr_version="PP-OCRv5",
            lang="ch",
            use_textline_orientation=True,
            enable_mkldnn=False,
            enable_hpi=False,
            device="cpu",
        )
        _OCR_LAST_ERROR = None
        _OCR_READY.set()
        return engine
    except Exception as exc:
        _OCR_LAST_ERROR = str(exc)
        raise


def warm_ocr_engine() -> None:
    """Force the OCR engine to initialise (blocks until done)."""
    get_ocr_engine()


def start_ocr_warmup_in_background() -> None:
    """Start OCR warm-up in a daemon thread (idempotent)."""
    global _OCR_WARMUP_STARTED
    with _OCR_WARMUP_LOCK:
        if _OCR_WARMUP_STARTED:
            return
        _OCR_WARMUP_STARTED = True

    def _runner() -> None:
        try:
            warm_ocr_engine()
        except Exception:
            pass

    thread = threading.Thread(target=_runner, name="ocr-warmup", daemon=True)
    thread.start()


def is_ocr_ready() -> bool:
    """Return True if the OCR engine has been successfully initialised."""
    return _OCR_READY.is_set()


def get_ocr_last_error() -> str | None:
    """Return the last OCR initialisation error string, or None."""
    return _OCR_LAST_ERROR


# ---------------------------------------------------------------------------
# Image / PDF processing
# ---------------------------------------------------------------------------

def _poly_to_xywh(poly: Any) -> tuple[int, int, int]:
    """Convert a polygon to (x_center, y_center, width)."""
    np = get_numpy()
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x_min = float(np.min(pts[:, 0]))
    x_max = float(np.max(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    y_max = float(np.max(pts[:, 1]))
    x_center = int(round((x_min + x_max) / 2))
    y_center = int(round((y_min + y_max) / 2))
    width = max(1, int(round(x_max - x_min)))
    return x_center, y_center, width


def result_to_tokens(
    result: dict[str, Any], fallback_height: int | None = None
) -> list[dict[str, Any]]:
    """Convert a PaddleOCR result dict to a list of token dicts."""
    np = get_numpy()
    texts = result.get("rec_texts", []) or []
    polys = result.get("rec_polys", []) or []
    scores = result.get("rec_scores", []) or []

    image_height = fallback_height
    doc_pre = result.get("doc_preprocessor_res")
    if isinstance(doc_pre, dict):
        out_img = doc_pre.get("output_img")
        if isinstance(out_img, np.ndarray) and out_img.ndim >= 2:
            image_height = int(out_img.shape[0])

    if not image_height:
        image_height = 10000

    tokens: list[dict[str, Any]] = []
    count = min(len(texts), len(polys))
    for idx in range(count):
        text = str(texts[idx] or "").strip()
        if not text:
            continue
        x_center, y_center, width = _poly_to_xywh(polys[idx])
        y_for_parser = int(round(image_height - y_center))
        score = float(scores[idx]) if idx < len(scores) else 0.0
        tokens.append(
            {
                "text": text,
                "x": x_center,
                "y": y_for_parser,
                "w": width,
                "score": round(score, 4),
            }
        )
    return tokens


def read_image_height(path: Path) -> int | None:
    """Return the pixel height of an image file, or None on failure."""
    cv2 = get_cv2()
    img = cv2.imread(str(path))
    if img is None:
        return None
    return int(img.shape[0])


def extract_pdf_tokens(path: Path) -> list[list[dict[str, Any]]]:
    """Extract word tokens from every page of a PDF using PyMuPDF."""
    pages: list[list[dict[str, Any]]] = []
    with fitz.open(path) as doc:
        for page in doc:
            page_height = float(page.rect.height)
            words = page.get_text("words", sort=True)
            tokens: list[dict[str, Any]] = []
            for word in words:
                x0, y0, x1, y1, text = word[:5]
                text = str(text or "").strip()
                if not text:
                    continue
                x_center = int(round((float(x0) + float(x1)) / 2.0))
                y_center = int(round((float(y0) + float(y1)) / 2.0))
                width = max(1, int(round(float(x1) - float(x0))))
                tokens.append(
                    {
                        "text": text,
                        "x": x_center,
                        "y": int(round(page_height - y_center)),
                        "w": width,
                        "score": 1.0,
                    }
                )
            pages.append(tokens)
    return pages


def run_predict_on_path(path: Path) -> list[dict[str, Any]]:
    """Run PaddleOCR on an image and return a list of page dicts."""
    engine = get_ocr_engine()
    result_list = list(engine.predict(str(path)))
    fallback_h = read_image_height(path) if path.suffix.lower() in IMAGE_EXTS else None
    pages: list[dict[str, Any]] = []
    for result in result_list:
        pages.append({"tokens": result_to_tokens(result, fallback_h)})
    return pages


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------

def create_app(frontend_path: Path | None = None) -> FastAPI:
    """Build and return the FastAPI application."""
    frontend_file: Path = frontend_path if frontend_path else get_default_frontend_path()

    app = FastAPI(title="Paddle Invoice OCR API", version="1.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def serve_index() -> FileResponse:
        if not frontend_file.exists():
            raise HTTPException(status_code=500, detail=f"Frontend file not found: {frontend_file}")
        return FileResponse(frontend_file)

    @app.get(f"/{DEFAULT_FRONTEND_NAME}")
    def serve_frontend_file() -> FileResponse:
        return serve_index()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "ocrReady": is_ocr_ready(), "ocrError": get_ocr_last_error()}

    @app.get("/api/paddle/status")
    def paddle_status() -> dict[str, Any]:
        return {"ok": True, "ready": is_ocr_ready(), "error": get_ocr_last_error()}

    @app.post("/api/paddle/warmup")
    def paddle_warmup() -> dict[str, Any]:
        try:
            warm_ocr_engine()
            return {"ok": True, "ready": is_ocr_ready(), "error": get_ocr_last_error()}
        except Exception as exc:
            log_runtime_exception("OCR warmup failed.", exc)
            return {"ok": False, "ready": is_ocr_ready(), "error": str(exc)}

    @app.post("/api/paddle/extract")
    async def paddle_extract(file: UploadFile = File(...)) -> dict[str, Any]:
        filename = file.filename or "upload.bin"
        suffix = Path(filename).suffix.lower()
        if suffix not in ALL_EXTS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty upload")

        content_hash = hashlib.sha256(data).hexdigest()[:16]
        temp_path: Path | None = None

        try:
            temp_dir = Path(os.environ.get("TEMP", tempfile.gettempdir()))
            temp_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir) as tmp:
                tmp.write(data)
                temp_path = Path(tmp.name)

            if suffix in PDF_EXTS:
                token_pages = extract_pdf_tokens(temp_path)
                if not any(token_pages):
                    raise HTTPException(status_code=422, detail="PDF text extraction returned no tokens")
                engine_name = "pdf-text-layer"
            else:
                pages = run_predict_on_path(temp_path)
                token_pages = [p["tokens"] for p in pages]
                if not token_pages:
                    raise HTTPException(status_code=422, detail="OCR returned no pages")
                engine_name = "paddleocr-ppocrv5-server"

            return {
                "ok": True,
                "engine": engine_name,
                "filename": filename,
                "contentHash": content_hash,
                "pages": token_pages,
                "tokenCount": sum(len(p) for p in token_pages),
            }
        except HTTPException:
            raise
        except Exception as exc:
            log_runtime_exception("OCR extract failed.", exc)
            raise HTTPException(status_code=500, detail=f"Paddle OCR failed: {exc}") from exc
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    return app


# ---------------------------------------------------------------------------
# Server readiness / browser helpers
# ---------------------------------------------------------------------------

def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Poll /api/health until it returns 200 or the timeout expires."""
    deadline = time.time() + timeout
    health_url = url.rstrip("/") + "/api/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.2)
        if time.time() < deadline:
            continue
    return False


def open_browser_when_ready(url: str, timeout: float = 30.0) -> None:
    """Open the browser once the server is ready."""
    if wait_for_server(url, timeout=timeout):
        webbrowser.open(url)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local PaddleOCR API and optionally serve invoice-tool.html."
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"API host (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"API port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--frontend",
        default=str(get_default_frontend_path()),
        help=f"Frontend HTML path (default: {DEFAULT_FRONTEND_NAME})",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the local web UI after the API is ready.",
    )
    parser.add_argument(
        "--preload-ocr",
        action="store_true",
        help="Warm up the OCR engine in background after startup.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level.",
    )
    args = parser.parse_args()

    frontend_path = Path(args.frontend).expanduser().resolve()

    # When binding to all interfaces, use DEFAULT_HOST for the browser URL.
    browser_host = DEFAULT_HOST if args.host in {"0.0.0.0", "::"} else args.host
    base_url = f"http://{browser_host}:{args.port}/"

    if args.preload_ocr:
        start_ocr_warmup_in_background()

    if args.open_browser:
        thread = threading.Thread(
            target=open_browser_when_ready,
            args=(base_url,),
            name="browser-opener",
            daemon=True,
        )
        thread.start()

    uvicorn.run(
        create_app(frontend_path=frontend_path),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
