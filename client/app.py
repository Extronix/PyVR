"""
ClientApp — основной цикл клиента.

Запускает:
  - TileReceiver  (поток приёма тайлов)
  - ROISender     (поток отправки ROI)
  - HeadInput     (мышь + клавиатура)
  - Renderer      (ERP → viewport)
"""
from __future__ import annotations
import logging
import time

import cv2

from common.config import CFG
from common.tiles import TileGrid
from common.protocol import ROIPacket, now_ms

from client.network.tile_buffer import TileBuffer
from client.network.tile_receiver import TileReceiver
from client.network.roi_sender import ROISender

from client.input.head_input import HeadInput
from client.render.renderer import Renderer

logger = logging.getLogger(__name__)


class ClientApp:
    """
    Основное приложение. Запускается через .run().
    """

    def __init__(self,
                 erp_w: int = 4096,
                 erp_h: int = 2048,
                 server_host: str | None = None,
                 roi_port: int | None = None,
                 tile_port: int | None = None,
                 view_w: int | None = None,
                 view_h: int | None = None,
                 roi_queue_size: int = 16):
        self.server_host = server_host or CFG.network.host
        self.roi_port = roi_port or CFG.network.roi_port
        self.tile_port = tile_port or CFG.network.tile_port
        self.view_w = view_w or CFG.view.window_width
        self.view_h = view_h or CFG.view.window_height

        # Сетка тайлов (под ERP-разрешение видео на сервере)
        self.grid = TileGrid(erp_w, erp_h)
        logger.info(f"Client grid: {self.grid}")

        # Буфер тайлов
        self.tile_buffer = TileBuffer(max_frames=4)

        # Поток приёма тайлов (UDP)
        self.tile_receiver = TileReceiver(
            buffer=self.tile_buffer,
            host="0.0.0.0",       # слушаем на всех интерфейсах
            port=self.tile_port,
        )

        # Поток отправки ROI (UDP)
        self.roi_sender = ROISender(
            host=self.server_host,
            port=self.roi_port,
            queue_size=roi_queue_size,
        )

        # Управление
        self.head = HeadInput()

        # Рендерер
        self.renderer = Renderer(
            grid=self.grid,
            view_w=self.view_w,
            view_h=self.view_h,
        )

        # Счётчики
        self._roi_seq = 0
        self._frame_id = 0
        self._last_roi_send_t = 0.0
        self._stop = False

    # ---------- main loop ----------

    def run(self) -> None:
        # Запуск фоновых потоков
        self.tile_receiver.start()
        # roi_sender уже запущен в __init__

        window = CFG.view.window_title
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, self.head.on_mouse)

        logger.info("ClientApp running. ESC / Q to quit, R to reset view.")

        try:
            roi_period = 1.0 / max(1, CFG.network.roi_send_rate_hz)

            while not self._stop:
                t0 = time.perf_counter()

                # 1. Обновить состояние головы (EMA скорости)
                hs = self.head.update()

                # 2. Отправить ROI (с rate-limit)
                if t0 - self._last_roi_send_t >= roi_period:
                    self._send_roi(hs)
                    self._last_roi_send_t = t0

                # 3. Рендер
                hud = self._build_hud(hs)
                frame = self.renderer.render(
                    buffer=self.tile_buffer,
                    yaw_deg=hs.yaw_deg,
                    pitch_deg=hs.pitch_deg,
                    fov_deg=hs.fov_deg,
                    hud_lines=hud,
                )

                cv2.imshow(window, frame)

                # 4. Клавиатура
                k = cv2.waitKey(1) & 0xFF
                if k == 27 or k == ord('q') or k == ord('Q'):
                    self._stop = True
                    break
                if k != 255:
                    self.head.on_key(k)

                # 5. Периодический лог
                self._frame_id += 1
                if self._frame_id % CFG.log.client_log_every_n_frames == 0:
                    self._log_stats()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.shutdown()

    # ---------- helpers ----------

    def _send_roi(self, hs) -> None:
        self._roi_seq += 1
        pkt = ROIPacket(
            seq=self._roi_seq,
            timestamp_ms=now_ms(),
            frame_id=self.tile_buffer.latest_frame_id() or 0,
            yaw_deg=hs.yaw_deg,
            pitch_deg=hs.pitch_deg,
            fov_deg=hs.fov_deg,
            yaw_vel_dps=hs.yaw_vel_dps,
            pitch_vel_dps=hs.pitch_vel_dps,
        )
        self.roi_sender.send(pkt)

    def _build_hud(self, hs) -> list[str]:
        if not CFG.view.hud_enabled:
            return []
        rs = self.renderer.stats()
        bs = self.tile_buffer.stats()
        rcv = self.tile_receiver

        lines = [
            f"FPS {rs['fps']:.1f}   frame#{self._frame_id}",
            f"t: canvas {rs['t_canvas_ms']:.1f}  proj {rs['t_project_ms']:.1f}  hud {rs['t_hud_ms']:.1f} ms",
            f"yaw {hs.yaw_deg:+6.1f}  pitch {hs.pitch_deg:+5.1f}  fov {hs.fov_deg:4.1f}",
            f"tiles {rs['canvas_present']}/{rs['canvas_total']}  "
            f"lk {rs['canvas_from_lastknown']}  miss {rs['canvas_missing']}  "
            f"cov {rs['canvas_coverage_pct']:.0f}%",
            f"net: tiles {rcv.tiles_completed}  lat avg {rcv.avg_latency_ms():.1f}ms  max {rcv.latency_max_ms:.0f}ms",
            f"proj_hit {rs['proj_hit_pct']:.0f}%   roi sent {self.roi_sender.sent_count}  dropped {self.roi_sender.dropped_count}",
        ]
        return lines

    def _log_stats(self) -> None:
        rs = self.renderer.stats()
        rcv = self.tile_receiver
        logger.info(
            f"frame={self._frame_id} fps={rs['fps']:.1f} "
            f"t[canv={rs['t_canvas_ms']:.1f} proj={rs['t_project_ms']:.1f}] "
            f"tiles_present={rs['canvas_present']}/{rs['canvas_total']} "
            f"net_tiles={rcv.tiles_completed} lat={rcv.avg_latency_ms():.1f}ms "
            f"roi_sent={self.roi_sender.sent_count}"
        )

    def shutdown(self) -> None:
        logger.info("Shutting down ClientApp...")
        try:
            self.tile_receiver.stop()
        except Exception as e:
            logger.warning(f"tile_receiver.stop: {e}")
        try:
            self.roi_sender.stop()
        except Exception as e:
            logger.warning(f"roi_sender.stop: {e}")
        cv2.destroyAllWindows()
        logger.info("ClientApp stopped.")