"""
Сборка ERP-канваса из тайлов.

Оптимизированная версия:
- Плейсхолдер генерируется ОДИН РАЗ векторизованно и используется как
  фон. Для отсутствующих тайлов мы НЕ перерисовываем их каждый кадр —
  фон уже корректный.
- last-known рисуется поверх только если он есть.
- Когда приходит новый реальный тайл — он перетирает старое содержимое.
"""
from __future__ import annotations
import logging
import time
from typing import Optional

import cv2
import numpy as np

from common.config import CFG
from common.tiles import TileGrid, TileSpec
from client.network.tile_buffer import TileBuffer, TileEntry

logger = logging.getLogger(__name__)


class EquirectCanvas:
    """ERP-буфер (frame_h × frame_w × 3, uint8)."""

    def __init__(self, grid: TileGrid):
        self.grid = grid

        t0 = time.perf_counter()
        # Фон-плейсхолдер (рисуем ОДИН РАЗ)
        self._background = self._make_checker_bg(grid.frame_w, grid.frame_h)
        # Рабочий канвас стартует копией фона
        self.canvas = self._background.copy()
        logger.info(
            f"EquirectCanvas init: {grid.frame_w}x{grid.frame_h}, "
            f"placeholder built in {(time.perf_counter()-t0)*1000:.1f}ms"
        )

        # last-known: декодированный тайл + frame_id обновления
        self._last_known: dict[int, np.ndarray] = {}
        self._last_known_age: dict[int, int] = {}

        # какие тайлы сейчас "помечены как плейсхолдер" на канвасе
        # (если это сменится — придётся перерисовать фон в эту область)
        self._tile_is_placeholder: dict[int, bool] = {
            spec.tile_id: True for spec in grid.all_specs()
        }

        # статистика
        self.last_frame_id: Optional[int] = None
        self.tiles_present_last = 0
        self.tiles_missing_last = 0
        self.tiles_from_lastknown_last = 0

        # хронометраж последнего update_from_buffer
        self.last_update_ms = 0.0

    # ---------- placeholder ----------

    @staticmethod
    def _make_checker_bg(W: int, H: int) -> np.ndarray:
        """Векторизованная шахматная доска. ~30мс для 5760x2880."""
        cs = CFG.tiles.placeholder_checker_size
        ca = np.array(CFG.tiles.placeholder_color_a, dtype=np.uint8)
        cb = np.array(CFG.tiles.placeholder_color_b, dtype=np.uint8)

        # Маска ячеек
        yy = (np.arange(H, dtype=np.int32) // cs)[:, None]
        xx = (np.arange(W, dtype=np.int32) // cs)[None, :]
        mask = ((yy + xx) & 1).astype(bool)   # (H, W)

        out = np.empty((H, W, 3), dtype=np.uint8)
        out[:] = cb
        out[mask] = ca
        return out

    # ---------- update ----------

    def update_from_buffer(self, buffer: TileBuffer,
                           current_frame_id: Optional[int] = None) -> Optional[int]:
        t0 = time.perf_counter()

        if current_frame_id is None:
            current_frame_id = buffer.latest_frame_id()
        if current_frame_id is None and not self._last_known:
            self.last_update_ms = (time.perf_counter() - t0) * 1000.0
            return None

        tiles_now: dict[int, TileEntry] = {}
        if current_frame_id is not None:
            tiles_now = buffer.frame_tiles(current_frame_id)

        present = 0
        missing = 0
        from_last = 0
        ttl = CFG.tiles.tile_cache_ttl_frames

        for spec in self.grid.all_specs():
            tid = spec.tile_id
            entry = tiles_now.get(tid)

            if entry is not None:
                # 1) новый реальный тайл
                self._paste(spec, entry.tile)
                self._last_known[tid] = entry.tile
                self._last_known_age[tid] = current_frame_id or 0
                self._tile_is_placeholder[tid] = False
                present += 1
                continue

            # тайла в этом кадре нет — пробуем last-known
            lk = self._last_known.get(tid)
            if lk is not None:
                age = (current_frame_id or 0) - self._last_known_age.get(tid, 0)
                if age <= ttl:
                    # last-known ещё актуален. Перерисовываем его только если
                    # сейчас в этой области плейсхолдер.
                    if self._tile_is_placeholder[tid]:
                        self._paste(spec, lk)
                        self._tile_is_placeholder[tid] = False
                    # иначе на канвасе уже стоит наш предыдущий last-known —
                    # лень и оптимизация, ничего не делаем.
                    from_last += 1
                    continue
                else:
                    # last-known протух — выкидываем
                    del self._last_known[tid]
                    self._last_known_age.pop(tid, None)

            # реально дыра → плейсхолдер. Восстанавливаем фон только если
            # сейчас там что-то другое.
            if not self._tile_is_placeholder[tid]:
                self._restore_bg_in(spec)
                self._tile_is_placeholder[tid] = True
            missing += 1

        self.last_frame_id = current_frame_id
        self.tiles_present_last = present
        self.tiles_missing_last = missing
        self.tiles_from_lastknown_last = from_last
        self.last_update_ms = (time.perf_counter() - t0) * 1000.0

        return current_frame_id

    # ---------- helpers ----------

    def _paste(self, spec: TileSpec, tile_img: np.ndarray) -> None:
        th, tw = tile_img.shape[:2]
        if th == spec.h and tw == spec.w:
            self.canvas[spec.y:spec.y2, spec.x:spec.x2] = tile_img
        else:
            self.canvas[spec.y:spec.y2, spec.x:spec.x2] = cv2.resize(
                tile_img, (spec.w, spec.h), interpolation=cv2.INTER_LINEAR
            )

    def _restore_bg_in(self, spec: TileSpec) -> None:
        """Копирует кусок шахматного фона в область тайла (быстро)."""
        self.canvas[spec.y:spec.y2, spec.x:spec.x2] = \
            self._background[spec.y:spec.y2, spec.x:spec.x2]

    def stats(self) -> dict:
        total = self.grid.total_tiles
        return {
            "frame_id": self.last_frame_id,
            "present": self.tiles_present_last,
            "from_lastknown": self.tiles_from_lastknown_last,
            "missing": self.tiles_missing_last,
            "total": total,
            "coverage_pct": 100.0 * (self.tiles_present_last + self.tiles_from_lastknown_last) / total,
            "update_ms": round(self.last_update_ms, 2),
        }