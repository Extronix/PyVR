"""
Логика тайлинга ERP-кадра.

ERP-кадр размером (W × H) делится на сетку cols × rows тайлов.
Каждый тайл покрывает участок:
    - yaw  ∈ [-180° + col * (360/cols),  -180° + (col+1) * (360/cols))
    - pitch ∈ [+90°  - row * (180/rows),  +90°  - (row+1) * (180/rows))

Tile IDs нумеруются построчно слева-направо, сверху-вниз:
    row 0: 0  1  2  3  4  5  6  7
    row 1: 8  9  10 11 12 13 14 15
    row 2: 16 17 ...
    row 3: 24 25 ...                (для 8×4)

ID = row * cols + col
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import math

import numpy as np

from common.config import CFG


@dataclass(frozen=True)
class TileSpec:
    """Описание одного тайла: его ID и пиксельные границы в ERP-кадре."""
    tile_id: int
    col: int
    row: int
    x: int          # левая граница в пикселях
    y: int          # верхняя граница в пикселях
    w: int          # ширина
    h: int          # высота
    
    @property
    def x2(self) -> int:
        return self.x + self.w
    
    @property
    def y2(self) -> int:
        return self.y + self.h


class TileGrid:
    """
    Сетка тайлов для ERP-кадра фиксированного разрешения.
    
    Тайлы могут иметь "хвостовой" разный размер если W не делится на cols
    (последний столбец/ряд получает остаток). Это не проблема — на клиенте
    мы знаем точные размеры каждого тайла по его spec.
    """
    
    def __init__(self, frame_width: int, frame_height: int,
                 cols: int | None = None, rows: int | None = None):
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.cols = cols if cols is not None else CFG.tiles.grid_cols
        self.rows = rows if rows is not None else CFG.tiles.grid_rows
        
        # Базовый размер тайла
        self.tile_w_base = frame_width // self.cols
        self.tile_h_base = frame_height // self.rows
        
        # Предвычислим все спеки (быстрый доступ по ID)
        self._specs: list[TileSpec] = []
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * self.tile_w_base
                y = r * self.tile_h_base
                # Последний столбец/строка забирают остаток
                w = frame_width - x if c == self.cols - 1 else self.tile_w_base
                h = frame_height - y if r == self.rows - 1 else self.tile_h_base
                tile_id = r * self.cols + c
                self._specs.append(TileSpec(tile_id, c, r, x, y, w, h))
    
    @property
    def total_tiles(self) -> int:
        return self.cols * self.rows
    
    def spec(self, tile_id: int) -> TileSpec:
        return self._specs[tile_id]
    
    def all_specs(self) -> list[TileSpec]:
        return list(self._specs)
    
    def specs_for_ids(self, ids: Iterable[int]) -> list[TileSpec]:
        return [self._specs[i] for i in ids]
    
    # --- Преобразование сферических углов → tile_id ---
    
    def tile_for_angle(self, yaw_deg: float, pitch_deg: float) -> int:
        """
        Возвращает tile_id, который покрывает точку (yaw, pitch) на сфере.
        yaw_deg: [-180..+180]
        pitch_deg: [-90..+90]
        """
        # Нормализуем yaw в [0..360)
        yaw_norm = (yaw_deg + 180.0) % 360.0
        col = int(yaw_norm / (360.0 / self.cols))
        col = max(0, min(self.cols - 1, col))
        
        # pitch: +90 = верх кадра (row=0), -90 = низ (row=rows-1)
        pitch_norm = 90.0 - pitch_deg     # [0..180]
        row = int(pitch_norm / (180.0 / self.rows))
        row = max(0, min(self.rows - 1, row))
        
        return row * self.cols + col
    
    # --- Выбор видимых тайлов для viewport ---
    
    def visible_tiles(self, yaw_deg: float, pitch_deg: float,
                      fov_deg: float, aspect: float = 16/9,
                      margin_deg: float = 0.0) -> list[int]:
        """
        Возвращает список tile_id, попадающих в viewport с заданными углами.
        
        Алгоритм: сэмплируем точки внутри viewport (равномерная сетка sample_n×sample_n),
        для каждой точки определяем tile_id, собираем уникальные.
        Этого достаточно для практических задач (FOV до 120°).
        
        margin_deg: расширение области выбора на каждую сторону (для предсказания).
        """
        # Полу-FOV по горизонтали и вертикали
        h_fov = fov_deg + margin_deg
        v_fov = (fov_deg / aspect) + margin_deg
        
        sample_n = 9   # 9×9 = 81 сэмпл, с запасом
        result: set[int] = set()
        
        for iy in range(sample_n):
            for ix in range(sample_n):
                # Локальные углы в frustum [-h_fov/2 .. +h_fov/2] × [-v_fov/2 .. +v_fov/2]
                local_yaw = (ix / (sample_n - 1) - 0.5) * h_fov
                local_pitch = (iy / (sample_n - 1) - 0.5) * v_fov
                
                # Простое сложение углов (без полной сферической математики).
                # Для FOV ≤ 120° и небольших pitch это даёт корректную оценку покрытия.
                # На полюсах сэмплы автоматически "схлопываются" в соседние столбцы,
                # что только увеличивает выбор тайлов на полюсах — это безопасно.
                world_yaw = yaw_deg + local_yaw
                world_pitch = pitch_deg + local_pitch
                
                # Зажимаем pitch
                world_pitch = max(-89.9, min(89.9, world_pitch))
                
                tile_id = self.tile_for_angle(world_yaw, world_pitch)
                result.add(tile_id)
        
        return sorted(result)
    
    # --- Нарезка кадра ---
    
    def cut_tile(self, frame: np.ndarray, tile_id: int) -> np.ndarray:
        """Возвращает срез кадра, соответствующий тайлу (без копирования)."""
        s = self._specs[tile_id]
        return frame[s.y:s.y2, s.x:s.x2]
    
    def cut_tile_copy(self, frame: np.ndarray, tile_id: int) -> np.ndarray:
        """То же, но с копированием (если нужно работать независимо)."""
        return self.cut_tile(frame, tile_id).copy()
    
    def __repr__(self) -> str:
        return (f"TileGrid({self.frame_w}x{self.frame_h}, "
                f"{self.cols}x{self.rows}={self.total_tiles} tiles, "
                f"base={self.tile_w_base}x{self.tile_h_base})")
    def visible_tiles_with_importance(self,
                                      yaw_deg: float,
                                      pitch_deg: float,
                                      fov_deg: float,
                                      aspect: float = 16 / 9,
                                      margin_deg: float = 0.0,
                                      ) -> tuple[list[int], dict[int, float]]:
        """
        Возвращает (tile_ids, importance), где importance[tile_id] ∈ [0..1]:
          1.0 — центр viewport,
          ~0 — край viewport,
          0.0 ровно — halo (вне эллипса FOV).

        Использует ту же логику покрытия, что visible_tiles(),
        плюс вычисляет importance для foveated quality.
        """
        # 1) Базовый список тайлов — как обычно
        visible = self.visible_tiles(yaw_deg, pitch_deg, fov_deg,
                                     aspect=aspect, margin_deg=margin_deg)

        # 2) FOV по горизонтали и вертикали (как в visible_tiles)
        h_fov = fov_deg
        v_fov = fov_deg / aspect

        # 3) Центр viewport в нормированных ERP-координатах [0..1]
        #    yaw  ∈ [-180..+180] → нормируем через (+180)/360
        #    pitch ∈ [-90..+90]  → нормируем через (90-pitch)/180  (как в tile_for_angle)
        cx = ((yaw_deg + 180.0) % 360.0) / 360.0
        cy = (90.0 - pitch_deg) / 180.0

        # 4) Полу-FOV в нормированных единицах
        hx = (h_fov / 2.0) / 360.0
        hy = (v_fov / 2.0) / 180.0

        # «Радиус» = полу-диагональ FOV
        norm_radius = math.hypot(hx, hy)
        if norm_radius < 1e-6:
            norm_radius = 1e-6

        tx_w = 1.0 / self.cols
        ty_h = 1.0 / self.rows

        # 5) Важность каждого тайла
        importance: dict[int, float] = {}
        for tid in visible:
            spec = self._specs[tid]
            # Центр тайла в нормированных ERP-координатах
            txc = (spec.col + 0.5) * tx_w
            tyc = (spec.row + 0.5) * ty_h

            # Дистанция по X с учётом wrap (360°)
            dx = abs(txc - cx)
            if dx > 0.5:
                dx = 1.0 - dx
            dy = tyc - cy
            d = math.hypot(dx, dy)

            if d >= norm_radius:
                importance[tid] = 0.0  # halo
            else:
                t = d / norm_radius
                importance[tid] = max(0.0, min(1.0, 1.0 - t))

        return visible, importance