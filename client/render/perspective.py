"""
Перспективная проекция viewport из ERP-канваса.

Использует common.geometry.build_view_map + cv2.remap.
Карты кэшируются по округлённым (yaw, pitch, fov).
"""
from __future__ import annotations
import logging
from collections import OrderedDict

import cv2
import numpy as np

from common.config import CFG
from common.geometry import build_view_map

logger = logging.getLogger(__name__)


class PerspectiveProjector:
    """
    Проектирует ERP-кадр в перспективный viewport.

    Кэширует карты cv2.remap по ключу (yaw_int, pitch_int, fov_int),
    что радикально ускоряет рендер при медленном/дискретном движении.
    """

    def __init__(self, view_w: int, view_h: int, erp_w: int, erp_h: int,
                 cache_size: int | None = None):
        self.view_w = view_w
        self.view_h = view_h
        self.erp_w = erp_w
        self.erp_h = erp_h
        self.cache_size = cache_size or CFG.view.sample_cache_max

        self._cache: OrderedDict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = OrderedDict()

        self.cache_hits = 0
        self.cache_misses = 0

    def _key(self, yaw: float, pitch: float, fov: float) -> tuple[int, int, int]:
        # округление до 1° достаточно для плавности
        return (int(round(yaw)) % 360, int(round(pitch)), int(round(fov)))

    def _get_map(self, yaw: float, pitch: float, fov: float):
        k = self._key(yaw, pitch, fov)
        cached = self._cache.get(k)
        if cached is not None:
            self.cache_hits += 1
            # подвинуть в "хвост" как most-recently-used
            self._cache.move_to_end(k)
            return cached

        self.cache_misses += 1
        mx, my = build_view_map(
            yaw_deg=yaw, pitch_deg=pitch, fov_deg=fov,
            view_w=self.view_w, view_h=self.view_h,
            erp_w=self.erp_w, erp_h=self.erp_h,
        )
        self._cache[k] = (mx, my)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return mx, my

    def project(self, erp_frame: np.ndarray,
                yaw_deg: float, pitch_deg: float, fov_deg: float) -> np.ndarray:
        """ERP → viewport (BGR uint8)."""
        mx, my = self._get_map(yaw_deg, pitch_deg, fov_deg)
        return cv2.remap(
            erp_frame, mx, my,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,    # горизонтально замыкаем
        )

    def stats(self) -> dict:
        total = self.cache_hits + self.cache_misses
        hit_pct = 100.0 * self.cache_hits / total if total else 0.0
        return {
            "cache_size": len(self._cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_pct": hit_pct,
        }