# InvoiceOCRTool

A professional desktop application and API for extracting text from invoices and other documents using PaddleOCR.

## Features

- **Local Processing**: All OCR processing happens locally using the PP-OCRv5 engine, ensuring your data never leaves your machine.
- **Cross-Platform**: Supports Windows, macOS, and Linux.
- **Support for Images & PDFs**: Extracts text from popular image formats (JPEG, PNG, WebP) and PDF documents.
- **FastAPI Backend**: Exposes a clean, RESTful API for integrating OCR into other applications.
- **Browser UI**: Includes a lightweight web frontend for easy dragging, dropping, and analyzing documents.
- **Desktop Launcher**: A Tkinter-based launcher (on supported platforms) to manage the API server and automatically launch the UI.

## Requirements

- Python 3.12 (Recommended: Python 3.12.10)
- The required dependencies listed in `requirements.txt` or `requirements.lock.txt`

## Installation

1. **Verify your Python version:**
   ```bash
   python --version
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate the environment:**
   - **Windows:**
     ```bash
     .venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies:**
   For exact versions, use the lockfile:
   ```bash
   pip install -r requirements.lock.txt
   ```
   If that fails, fall back to the standard requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### GUI Mode (Recommended)

To start the server and automatically open the graphical web interface, run the launcher:
```bash
python invoice_tool_launcher.py
```
This will open a small status window and automatically launch your default web browser to the application page.

### Headless API Mode

If you just want to run the API without opening a browser, you can start the API directly:
```bash
python paddle_invoice_api.py --host 127.0.0.1 --port 8866
```

Once running, you can manually access the frontend by opening the URL shown in the terminal (e.g., `http://127.0.0.1:8866/`) in your browser.

## Notes

- The first time you analyze a document, the OCR engine will need to initialize ("warm up"), which may take a few seconds. Subsequent extractions will be much faster.
- Extracted models and runtime caches are safely stored in your system's user directory (e.g., `%PUBLIC%\Documents\InvoiceOCRTool` on Windows, or `~/.local/share/InvoiceOCRTool` on macOS/Linux).
