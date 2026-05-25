"""
Построение viewport из ERP-кадра. Кэширует map_x/map_y, пересчитывает
только когда yaw/pitch/fov изменились заметно.
"""
import cv2
import numpy as np

from common.geometry import build_view_map
from common.config import CFG


class ViewBuilder:
    """Строит перспективный viewport из ERP-кадра с кэшем remap-карт."""
    
    # Порог: на сколько должна измениться поза, чтобы пересчитать карту
    EPS_DEG = 0.05
    
    def __init__(self, erp_w: int, erp_h: int,
                 view_w: int | None = None, view_h: int | None = None):
        self.erp_w = erp_w
        self.erp_h = erp_h
        self.view_w = view_w or CFG.video.view_w
        self.view_h = view_h or CFG.video.view_h
        
        # Кэш
        self._map_x: np.ndarray | None = None
        self._map_y: np.ndarray | None = None
        self._cached_yaw: float = float("nan")
        self._cached_pitch: float = float("nan")
        self._cached_fov: float = float("nan")
        
        # Статистика
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _needs_rebuild(self, yaw: float, pitch: float, fov: float) -> bool:
        if self._map_x is None:
            return True
        return (
            abs(yaw - self._cached_yaw) > self.EPS_DEG
            or abs(pitch - self._cached_pitch) > self.EPS_DEG
            or abs(fov - self._cached_fov) > self.EPS_DEG
        )
    
    def render(self, erp_frame: np.ndarray,
               yaw_deg: float, pitch_deg: float, fov_deg: float) -> np.ndarray:
        """Возвращает viewport (view_h, view_w, 3) uint8."""
        if self._needs_rebuild(yaw_deg, pitch_deg, fov_deg):
            self._map_x, self._map_y = build_view_map(
                yaw_deg, pitch_deg, fov_deg,
                self.view_w, self.view_h,
                self.erp_w, self.erp_h,
            )
            self._cached_yaw = yaw_deg
            self._cached_pitch = pitch_deg
            self._cached_fov = fov_deg
            self.cache_misses += 1
        else:
            self.cache_hits += 1
        
        return cv2.remap(
            erp_frame,
            self._map_x, self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )