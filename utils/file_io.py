"""文件读写工具。"""

from __future__ import annotations

from pathlib import Path


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
