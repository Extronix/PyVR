"""
Кодек JPEG для тайлов. Тонкая обёртка над cv2.imencode/imdecode
с фиксированным quality и обработкой ошибок.
"""
from __future__ import annotations
import numpy as np
import cv2

from common.config import CFG


def encode_tile(tile_bgr: np.ndarray, quality: int | None = None) -> bytes:
    """
    Кодирует BGR-тайл в JPEG.

    :param tile_bgr: ndarray (H, W, 3), dtype=uint8
    :param quality: 0..100, по умолчанию из CFG
    :return: JPEG-байты
    """
    if quality is None:
        quality = CFG.tiles.jpeg_quality

    encode_params = [
        int(cv2.IMWRITE_JPEG_QUALITY), int(quality),
        int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,
    ]
    ok, buf = cv2.imencode(".jpg", tile_bgr, encode_params)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def decode_tile(jpeg_bytes: bytes) -> np.ndarray:
    """
    Декодирует JPEG-байты в BGR-тайл.
    
    :param jpeg_bytes: байты JPEG
    :return: ndarray (H, W, 3), dtype=uint8
    :raises RuntimeError: если декод не удался
    """
    if not jpeg_bytes:
        raise RuntimeError("decode_tile: empty buffer")
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cv2.imdecode failed ({len(jpeg_bytes)} bytes)")
    return img