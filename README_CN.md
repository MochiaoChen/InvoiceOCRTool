# 发票 OCR 工具 (InvoiceOCRTool)

一个专业的桌面应用程序和 API，用于使用 PaddleOCR 从发票和其他文档中提取文本。

## 功能特性

- **本地处理**: 所有 OCR 处理都在本地使用 PP-OCRv5 引擎完成，确保您的数据绝不离开您的设备。
- **跨平台**: 支持 Windows、macOS 和 Linux。
- **支持图像和 PDF**: 从流行的图像格式（JPEG、PNG、WebP）和 PDF 文档中提取文本。
- **FastAPI 后端**: 提供清晰的 RESTful API，用于将 OCR 集成到其他应用程序中。
- **浏览器用户界面**: 包含一个轻量级的 Web 前端，方便拖放和分析文档。
- **桌面启动器**: 基于 Tkinter 的启动器（在支持的平台上），用于管理 API 服务器并自动启动用户界面。

## 环境要求

- Python 3.12 (推荐使用 Python 3.12.10)
- `requirements.txt` 或 `requirements.lock.txt` 中列出的必需依赖项

## 安装

1. **验证您的 Python 版本:**
   ```bash
   python --version
   ```

2. **创建虚拟环境:**
   ```bash
   python -m venv .venv
   ```

3. **激活环境:**
   - **Windows:**
     ```bash
     .venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. **安装依赖项:**
   为了精确匹配版本，请使用锁文件：
   ```bash
   pip install -r requirements.lock.txt
   ```
   如果失败，请回退到标准依赖项：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### GUI 模式 (推荐)

要启动服务器并自动打开图形化的 Web 界面，请运行启动器：
```bash
python invoice_tool_launcher.py
```
这将打开一个小状态窗口，并自动启动您的默认 Web 浏览器进入应用页面。

### 无头 API 模式 (Headless)

如果您只想运行 API 而不打开浏览器，可以直接启动 API：
```bash
python paddle_invoice_api.py --host 127.0.0.1 --port 8866
```

运行后，您可以通过在浏览器中打开终端中显示的 URL（例如 `http://127.0.0.1:8866/`）来手动访问前端。

## 注意事项

- 第一次分析文档时，OCR 引擎需要初始化（“预热”），这可能需要几秒钟。随后的提取将快得多。
- 提取的模型和运行时缓存安全地存储在系统的用户目录中（例如，Windows 上的 `%PUBLIC%\Documents\InvoiceOCRTool`，或 macOS/Linux 上的 `~/.local/share/InvoiceOCRTool`）。
