from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
import webbrowser

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except Exception:
    tk = None
    messagebox = None
    ttk = None

import uvicorn


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_app_root() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_resource_root() -> Path:
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        return Path(meipass).resolve()
    if getattr(sys, 'frozen', False):
        internal_dir = Path(sys.executable).resolve().parent / '_internal'
        if internal_dir.exists():
            return internal_dir.resolve()
    return get_app_root()


def contains_non_ascii(path: Path) -> bool:
    return any(ord(ch) > 127 for ch in str(path))


def get_safe_app_root() -> Path:
    public_root = Path(os.environ.get('PUBLIC', r'C:\Users\Public'))
    return public_root / 'Documents' / 'InvoiceOCRTool' / 'app'


def sync_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            sync_tree(item, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def relaunch_from_safe_ascii_path_if_needed() -> None:
    if not getattr(sys, 'frozen', False):
        return
    if os.environ.get('INVOICE_OCR_SAFE_LAUNCHED') == '1':
        return
    current_root = get_app_root()
    current_exe = Path(sys.executable).resolve()
    if not contains_non_ascii(current_root):
        return
    safe_root = get_safe_app_root()
    safe_exe = safe_root / current_exe.name
    sync_tree(current_root, safe_root)
    child_env = os.environ.copy()
    child_env['INVOICE_OCR_SAFE_LAUNCHED'] = '1'
    child_env['PADDLE_INVOICE_APP_ROOT'] = str(safe_root)
    subprocess.Popen([str(safe_exe)] + sys.argv[1:], env=child_env)
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

relaunch_from_safe_ascii_path_if_needed()

APP_ROOT = get_app_root()
RESOURCE_ROOT = get_resource_root()
BUNDLED_CACHE_HOME = RESOURCE_ROOT / 'runtime' / 'paddlex_cache'
LOG_DIR = APP_ROOT / 'runtime' / 'logs'
LOG_FILE = LOG_DIR / 'launcher.log'

os.environ.setdefault('PADDLE_INVOICE_APP_ROOT', str(APP_ROOT))
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')

LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def write_log(message: str) -> None:
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with LOG_FILE.open('a', encoding='utf-8') as handle:
        handle.write(f'[{timestamp}] {message}\n')


def install_exception_logger() -> None:
    def _hook(exc_type, exc_value, exc_tb):
        formatted = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        write_log('Unhandled exception:\n' + formatted)

    def _thread_hook(args: threading.ExceptHookArgs) -> None:
        formatted = ''.join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        write_log(f'Unhandled thread exception in {args.thread.name}:\n' + formatted)

    sys.excepthook = _hook
    threading.excepthook = _thread_hook


install_exception_logger()
write_log('Launcher process started.')
write_log(
    f'Launcher roots. app_root={APP_ROOT} resource_root={RESOURCE_ROOT}'
    f' public={os.environ.get("PUBLIC")} temp={os.environ.get("TEMP")}'
)

from paddle_invoice_api import (  # noqa: E402
    create_app,
    get_default_frontend_path,
    get_ocr_last_error,
    is_ocr_ready,
    start_ocr_warmup_in_background,
)

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8866


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

def is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) != 0


def find_available_port(host: str, start_port: int, max_tries: int = 20) -> int:
    for offset in range(max_tries):
        port = start_port + offset
        if is_port_available(host, port):
            return port
    raise RuntimeError(
        f'No available port found from {start_port} to {start_port + max_tries - 1}'
    )


def create_server(host: str, port: int) -> uvicorn.Server:
    config = uvicorn.Config(
        create_app(frontend_path=get_default_frontend_path()),
        host=host,
        port=port,
        log_level='warning',
        access_log=False,
        log_config=None,
        use_colors=False,
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    return server


def run_headless(start_port: int, host: str, open_browser: bool, preload_ocr: bool) -> None:
    port = find_available_port(host, start_port)
    url = f'http://{host}:{port}/'
    write_log(
        f'Headless mode start. host={host} port={port}'
        f' open_browser={open_browser} preload_ocr={preload_ocr}'
    )
    if preload_ocr:
        start_ocr_warmup_in_background()
    if open_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    uvicorn.run(
        create_app(frontend_path=get_default_frontend_path()),
        host=host,
        port=port,
        log_level='warning',
        access_log=False,
        log_config=None,
        use_colors=False,
    )


# ---------------------------------------------------------------------------
# GUI launcher window
# ---------------------------------------------------------------------------

class LauncherWindow:
    def __init__(
        self,
        host: str,
        start_port: int,
        open_browser: bool,
        preload_ocr: bool,
    ) -> None:
        if tk is None or ttk is None or messagebox is None:
            raise RuntimeError('Tkinter is unavailable in this build.')

        self.host = host
        self.start_port = start_port
        self.open_browser_on_ready = open_browser
        self.preload_ocr = preload_ocr
        self.frontend_path = get_default_frontend_path()

        self.server = None
        self.server_thread = None
        self.url = ''
        self.browser_opened = False
        self.ocr_polling_started = False
        self.shutdown_deadline = 0.0

        self.root = tk.Tk()
        self.root.title('发票 OCR 工具')
        self.root.resizable(False, False)
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

        self.status_var = tk.StringVar(value='正在启动服务，请稍候...')
        self.detail_var = tk.StringVar(value='启动后可自动打开浏览器，也可关闭此窗口后退出。')
        self.url_var = tk.StringVar(value='正在分配端口...')

        self._build_ui()
        self._center_window(460, 210)
        self.start()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=18)
        frame.grid(sticky='nsew')
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        title = ttk.Label(frame, text='发票 OCR 工具正在启动', font=('Microsoft YaHei UI', 13, 'bold'))
        title.grid(row=0, column=0, sticky='w')

        status = ttk.Label(frame, textvariable=self.status_var, wraplength=410, justify='left')
        status.grid(row=1, column=0, sticky='w', pady=(12, 6))

        detail = ttk.Label(frame, textvariable=self.detail_var, wraplength=410, justify='left')
        detail.grid(row=2, column=0, sticky='w')

        url_label = ttk.Label(frame, textvariable=self.url_var, foreground='#1b5e20')
        url_label.grid(row=3, column=0, sticky='w', pady=(14, 0))

        note = ttk.Label(
            frame,
            text='如果浏览器没有自动打开，请手动点击下方打开工具。',
            foreground='#666666',
        )
        note.grid(row=4, column=0, sticky='w', pady=(10, 16))

        button_row = ttk.Frame(frame)
        button_row.grid(row=5, column=0, sticky='ew')
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)

        self.open_button = ttk.Button(
            button_row, text='打开工具', command=self.open_browser, state='disabled'
        )
        self.open_button.grid(row=0, column=0, sticky='ew', padx=(0, 8))

        quit_button = ttk.Button(button_row, text='退出', command=self.on_close)
        quit_button.grid(row=0, column=1, sticky='ew')

    def _center_window(self, width: int, height: int) -> None:
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = max(0, int((screen_width - width) / 2))
        y = max(0, int((screen_height - height) / 2))
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def start(self) -> None:
        if not self.frontend_path.exists():
            write_log(f'Frontend file missing: {self.frontend_path}')
            messagebox.showerror('启动失败', f'未找到前端文件：\n{self.frontend_path}')
            self.root.destroy()
            return

        try:
            port = find_available_port(self.host, self.start_port)
            self.url = f'http://{self.host}:{port}/'
            write_log(
                f'GUI mode start. host={self.host} port={port}'
                f' open_browser={self.open_browser_on_ready} preload_ocr={self.preload_ocr}'
            )
            self.url_var.set(self.url)
            self.server = create_server(self.host, port)
            self.server_thread = threading.Thread(
                target=self.server.run, name='invoice-server', daemon=True
            )
            self.server_thread.start()
            self.root.after(150, self._poll_startup)
        except Exception as exc:
            messagebox.showerror('启动失败', str(exc))
            self.root.destroy()

    def _poll_startup(self) -> None:
        if self.server is not None and self.server.started:
            self.status_var.set('服务已就绪，可关闭此窗口或一键关闭服务。')
            self.detail_var.set('打开页面后，首次连接 OCR 引擎时将在后台预热模型。')
            self.open_button.config(state='normal')
            if self.open_browser_on_ready and not self.browser_opened:
                self.browser_opened = True
                self.open_browser()
            if self.preload_ocr and not self.ocr_polling_started:
                self.ocr_polling_started = True
                start_ocr_warmup_in_background()
                self.root.after(500, self._poll_ocr_state)
            return

        if self.server_thread is not None and not self.server_thread.is_alive():
            self.status_var.set('服务启动失败。')
            self.detail_var.set('请检查目录中的模型文件或 Python 环境是否完整。')
            self.open_button.config(state='disabled')
            return

        self.root.after(150, self._poll_startup)

    def _poll_ocr_state(self) -> None:
        if is_ocr_ready():
            self.detail_var.set('OCR 引擎已就绪，可以开始识别。')
            return
        last_error = get_ocr_last_error()
        if last_error:
            self.detail_var.set(f'OCR 预热失败，首次识别时将自动重试：{last_error}')
            return
        self.detail_var.set('OCR 引擎正在后台预热，首次识别稍慢。')
        self.root.after(800, self._poll_ocr_state)

    def open_browser(self) -> None:
        if self.url:
            webbrowser.open(self.url)

    def on_close(self) -> None:
        self.open_button.config(state='disabled')
        self.status_var.set('正在退出，关闭服务...')
        self.detail_var.set('请稍候...')
        if self.server is None:
            self.root.destroy()
            return
        self.server.should_exit = True
        self.shutdown_deadline = time.time() + 3.0
        self.root.after(100, self._poll_shutdown)

    def _poll_shutdown(self) -> None:
        if (
            self.server_thread is not None
            and self.server_thread.is_alive()
            and time.time() < self.shutdown_deadline
        ):
            self.root.after(100, self._poll_shutdown)
            return
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Launch the packaged invoice OCR desktop app.')
    parser.add_argument('--host', default=DEFAULT_HOST, help=f'Bind host (default: {DEFAULT_HOST})')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'Preferred port (default: {DEFAULT_PORT})')
    parser.add_argument('--headless', action='store_true', help='Run without the launcher window.')
    parser.add_argument('--no-browser', action='store_true', help='Do not open the browser automatically.')
    parser.add_argument('--no-preload', action='store_true', help='Do not warm up the OCR engine in background.')
    args = parser.parse_args()

    if args.headless:
        run_headless(
            start_port=args.port,
            host=args.host,
            open_browser=not args.no_browser,
            preload_ocr=not args.no_preload,
        )
        return

    if tk is None or ttk is None or messagebox is None:
        write_log('Tkinter unavailable; falling back to headless mode.')
        run_headless(
            start_port=args.port,
            host=args.host,
            open_browser=not args.no_browser,
            preload_ocr=not args.no_preload,
        )
        return

    launcher = LauncherWindow(
        host=args.host,
        start_port=args.port,
        open_browser=not args.no_browser,
        preload_ocr=not args.no_preload,
    )
    launcher.run()


if __name__ == '__main__':
    main()
