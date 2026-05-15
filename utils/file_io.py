"""文件读写工具。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any


def read_text_file(file_path: str | Path, encoding: str = "utf-8") -> str:
    """读取文本文件内容。

    Args:
        file_path: 文件路径。
        encoding: 文件编码，默认 utf-8。

    Returns:
        文件文本内容。
    """
    return Path(file_path).read_text(encoding=encoding)


def write_text_file(
    file_path: str | Path,
    content: str,
    encoding: str = "utf-8",
) -> None:
    """写入文本文件内容。

    Args:
        file_path: 文件路径。
        content: 待写入的文本内容。
        encoding: 文件编码，默认 utf-8。
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding)


def extract_pdf_text(file_bytes: bytes) -> str:
    """从 PDF 字节内容中提取文本。

    Args:
        file_bytes: PDF 文件二进制内容。

    Returns:
        合并后的 PDF 文本内容。

    Raises:
        ValueError: 文件为空、无法解析，或未提取到文本。
        RuntimeError: 缺少 PDF 解析依赖。
    """
    if not file_bytes:
        raise ValueError("PDF 文件为空。")

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("缺少 PDF 解析依赖 pypdf，请先安装 requirements.txt。") from exc

    try:
        reader = PdfReader(BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"无法读取 PDF 文件：{exc}") from exc

    page_texts: list[str] = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            page_texts.append(text)

    extracted_text = "\n\n".join(page_texts).strip()
    if extracted_text:
        return extracted_text

    ocr_text = _extract_pdf_image_text_with_ocr(reader)
    if not ocr_text:
        raise ValueError("未能从 PDF 中提取到文本，可能是扫描件或图片型简历。")

    return ocr_text


def _extract_pdf_image_text_with_ocr(reader: Any) -> str:
    """对图片型 PDF 做 OCR 文本提取。"""
    try:
        from PIL import Image
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "该 PDF 是图片型简历，需要 OCR 依赖；请先安装 requirements.txt。"
        ) from exc

    ocr = RapidOCR()
    text_parts: list[str] = []

    for page in reader.pages:
        for image in getattr(page, "images", []):
            try:
                pil_image = Image.open(BytesIO(image.data)).convert("RGB")
                result, _ = ocr(pil_image)
            except Exception:
                continue

            if not result:
                continue

            page_lines = [
                str(item[1]).strip()
                for item in result
                if len(item) >= 2 and str(item[1]).strip()
            ]
            if page_lines:
                text_parts.append("\n".join(page_lines))

    return "\n\n".join(text_parts).strip()
