"""
Высокоуровневый рендерер: ERP-канвас → viewport → HUD → cv2.imshow.
"""
from __future__ import annotations
import logging
import time

import cv2
import numpy as np

from common.config import CFG
from common.tiles import TileGrid

from client.network.tile_buffer import TileBuffer
from client.render.equirect_canvas import EquirectCanvas
from client.render.perspective import PerspectiveProjector

logger = logging.getLogger(__name__)


class Renderer:
    def __init__(self,
                 grid: TileGrid,
                 view_w: int | None = None,
                 view_h: int | None = None):
        self.grid = grid
        self.view_w = view_w or CFG.view.window_width
        self.view_h = view_h or CFG.view.window_height

        self.canvas = EquirectCanvas(grid)
        self.projector = PerspectiveProjector(
            view_w=self.view_w, view_h=self.view_h,
            erp_w=grid.frame_w, erp_h=grid.frame_h,
        )

        self.frame_count = 0
        self.fps_t0 = time.perf_counter()
        self.fps_frames = 0
        self.last_fps = 0.0

        # пер-этапные тайминги последнего кадра (мс)
        self.t_canvas_ms = 0.0
        self.t_project_ms = 0.0
        self.t_hud_ms = 0.0

    def render(self,
               buffer: TileBuffer,
               yaw_deg: float, pitch_deg: float, fov_deg: float,
               hud_lines: list[str] | None = None) -> np.ndarray:
        t0 = time.perf_counter()
        self.canvas.update_from_buffer(buffer)
        t1 = time.perf_counter()
        view = self.projector.project(self.canvas.canvas, yaw_deg, pitch_deg, fov_deg)
        t2 = time.perf_counter()

        if CFG.view.hud_enabled and hud_lines:
            self._draw_hud(view, hud_lines)
        t3 = time.perf_counter()

        self.t_canvas_ms = (t1 - t0) * 1000.0
        self.t_project_ms = (t2 - t1) * 1000.0
        self.t_hud_ms = (t3 - t2) * 1000.0

        self.frame_count += 1
        self.fps_frames += 1
        now = time.perf_counter()
        if now - self.fps_t0 >= 1.0:
            self.last_fps = self.fps_frames / (now - self.fps_t0)
            self.fps_t0 = now
            self.fps_frames = 0

        return view

    def _draw_hud(self, img: np.ndarray, lines: list[str]) -> None:
        scale = CFG.view.hud_font_scale
        color = CFG.view.hud_color_bgr
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_h = int(22 * scale * 2)

        y = 24
        for ln in lines:
            cv2.putText(img, ln, (11, y + 1), font, scale, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, ln, (10, y), font, scale, color, 1, cv2.LINE_AA)
            y += line_h

    def stats(self) -> dict:
        return {
            "fps": round(self.last_fps, 1),
            "frame": self.frame_count,
            "t_canvas_ms": round(self.t_canvas_ms, 1),
            "t_project_ms": round(self.t_project_ms, 1),
            "t_hud_ms": round(self.t_hud_ms, 1),
            **{f"canvas_{k}": v for k, v in self.canvas.stats().items()},
            **{f"proj_{k}": v for k, v in self.projector.stats().items()},
        }