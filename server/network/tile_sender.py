"""
Отправитель JPEG-тайлов клиенту по UDP с фрагментацией.

Поддерживает foveated quality: JPEG quality зависит от важности тайла
(центр viewport — выше, периферия и halo — ниже).
"""
from __future__ import annotations
import logging
import socket
import time
from typing import Iterable

import numpy as np

from common.config import CFG
from common.tiles import TileGrid
from common.jpeg_codec import encode_tile
from common.protocol import TileChunkPacket, now_ms

logger = logging.getLogger(__name__)


class TileSender:
    """Кодирует и шлёт тайлы клиенту чанками по UDP."""

    def __init__(self,
                 grid: TileGrid,
                 host: str | None = None,
                 port: int | None = None):
        self.grid = grid
        self.host = host or CFG.network.host
        self.port = port or CFG.network.tile_port

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF,
                              CFG.network.socket_buffer_size)

        self.chunk_payload_size = CFG.tiles.chunk_payload_size

        # --- Foveation ---
        fov_cfg = getattr(CFG, "foveation", None)
        self.foveated = bool(fov_cfg and getattr(fov_cfg, "enabled", False))
        if self.foveated:
            self.q_center = int(fov_cfg.quality_center)
            self.q_edge = int(fov_cfg.quality_edge)
            self.q_outside = int(fov_cfg.quality_outside)
            self.q_gamma = float(getattr(fov_cfg, "falloff_gamma", 1.0))
        else:
            # uniform
            q = int(CFG.tiles.jpeg_quality)
            self.q_center = self.q_edge = self.q_outside = q
            self.q_gamma = 1.0
        self.jpeg_quality = self.q_center  # для совместимости с логом

        # Накопительные счётчики
        self.frames_sent = 0
        self.tiles_sent = 0
        self.chunks_sent = 0
        self.bytes_sent = 0
        self.encode_time_total = 0.0
        self.send_time_total = 0.0

        logger.info(f"TileSender → udp://{self.host}:{self.port} "
                    f"(foveated={self.foveated}, "
                    f"q[center/edge/outside]="
                    f"{self.q_center}/{self.q_edge}/{self.q_outside}, "
                    f"chunk_size={self.chunk_payload_size})")

    def _quality_for(self, importance: float) -> int:
        """
        importance ∈ [0..1]:
          1.0  → q_center
          0..1 → линейно между q_edge и q_center
          0.0 ровно → q_outside (halo)
        """
        if not self.foveated:
            return self.q_center
        if importance <= 0.0:
            return self.q_outside
        # Гамма-кривая делает падение более резким (>1) или мягким (<1)
        imp = importance ** self.q_gamma
        q = self.q_edge + (self.q_center - self.q_edge) * imp
        # Зажмём в безопасный диапазон JPEG
        return int(max(5, min(95, round(q))))

    def send_frame_tiles(self,
                         frame: np.ndarray,
                         tile_ids: Iterable[int],
                         frame_id: int,
                         importance: dict[int, float] | None = None,
                         ) -> dict:
        """
        Нарезает тайлы, кодирует и шлёт чанки.
        Если передан importance — quality каждого тайла зависит от его importance.
        Иначе используется единое quality_center.

        Возвращает per-frame статистику.
        """
        ts = now_ms()
        n_tiles = 0
        n_chunks = 0
        n_bytes = 0
        t_encode = 0.0
        t_send = 0.0

        # Для статистики качества/распределения
        q_sum = 0
        q_min = 999
        q_max = 0
        # Распределение тайлов: center (>0.66), edge (>0), outside (==0)
        n_center = n_edge = n_outside = 0

        for tid in tile_ids:
            tile = self.grid.cut_tile(frame, tid)

            imp = 1.0 if importance is None else importance.get(tid, 0.0)
            q = self._quality_for(imp)

            t0 = time.perf_counter()
            jpeg = encode_tile(tile, quality=q)
            t_encode += time.perf_counter() - t0

            total = (len(jpeg) + self.chunk_payload_size - 1) // self.chunk_payload_size

            t0 = time.perf_counter()
            for idx in range(total):
                start = idx * self.chunk_payload_size
                end = min(start + self.chunk_payload_size, len(jpeg))
                pkt = TileChunkPacket(
                    tile_id=tid,
                    frame_id=frame_id,
                    chunk_idx=idx,
                    total_chunks=total,
                    timestamp_ms=ts,
                    payload=jpeg[start:end],
                )
                try:
                    self._sock.sendto(pkt.to_bytes(), (self.host, self.port))
                except OSError as e:
                    logger.warning(f"sendto failed: {e}")
                    continue
                n_chunks += 1
                n_bytes += end - start
            t_send += time.perf_counter() - t0

            n_tiles += 1
            q_sum += q
            if q < q_min:
                q_min = q
            if q > q_max:
                q_max = q
            if imp <= 0.0:
                n_outside += 1
            elif imp > 0.66:
                n_center += 1
            else:
                n_edge += 1

        self.frames_sent += 1
        self.tiles_sent += n_tiles
        self.chunks_sent += n_chunks
        self.bytes_sent += n_bytes
        self.encode_time_total += t_encode
        self.send_time_total += t_send

        return {
            "tiles": n_tiles,
            "chunks": n_chunks,
            "bytes": n_bytes,
            "encode_ms": t_encode * 1000,
            "send_ms": t_send * 1000,
            "q_avg": (q_sum / n_tiles) if n_tiles else 0.0,
            "q_min": q_min if n_tiles else 0,
            "q_max": q_max if n_tiles else 0,
            "n_center": n_center,
            "n_edge": n_edge,
            "n_outside": n_outside,
        }

    def close(self) -> None:
        self._sock.close()
        logger.info(
            f"TileSender closed: "
            f"frames={self.frames_sent} tiles={self.tiles_sent} "
            f"chunks={self.chunks_sent} bytes={self.bytes_sent} "
            f"({self.bytes_sent / 1024 / 1024:.1f} MB)"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()