"""
Потокобезопасный кэш декодированных тайлов.

Хранит последние N кадров. По каждому кадру — словарь
tile_id → ndarray. Старые кадры выталкиваются.

Чтение из главного потока (рендер), запись из receiver-потока.
"""
from __future__ import annotations
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TileEntry:
    """Одна запись в буфере."""
    tile: np.ndarray         # BGR ndarray
    frame_id: int
    tile_id: int
    server_ts_ms: int        # timestamp отправки с сервера
    recv_ts_ms: int          # момент готовности тайла на клиенте (ms epoch)


class TileBuffer:
    """
    Кольцевой буфер кадров. Внутри — OrderedDict frame_id → dict[tile_id, TileEntry].
    """
    
    def __init__(self, max_frames: int = 4):
        self._frames: OrderedDict[int, dict[int, TileEntry]] = OrderedDict()
        self._max_frames = max_frames
        self._lock = threading.Lock()
        
        # статистика
        self.tiles_put = 0
        self.tiles_evicted = 0
        self.frames_evicted = 0
    
    def put(self, entry: TileEntry) -> None:
        """Кладёт декодированный тайл."""
        with self._lock:
            fid = entry.frame_id
            if fid not in self._frames:
                self._frames[fid] = {}
                # вытесняем старые кадры
                while len(self._frames) > self._max_frames:
                    old_fid, old_tiles = self._frames.popitem(last=False)
                    self.tiles_evicted += len(old_tiles)
                    self.frames_evicted += 1
            self._frames[fid][entry.tile_id] = entry
            self.tiles_put += 1
    
    def get_tile(self, frame_id: int, tile_id: int) -> Optional[TileEntry]:
        with self._lock:
            f = self._frames.get(frame_id)
            if f is None:
                return None
            return f.get(tile_id)
    
    def latest_frame_id(self) -> Optional[int]:
        """ID самого свежего кадра в буфере, или None."""
        with self._lock:
            if not self._frames:
                return None
            return next(reversed(self._frames))
    
    def frame_tiles(self, frame_id: int) -> dict[int, TileEntry]:
        """Копия словаря тайлов для кадра (или пустой dict)."""
        with self._lock:
            f = self._frames.get(frame_id)
            return dict(f) if f else {}
    
    def stats(self) -> dict:
        with self._lock:
            return {
                "frames_in_buffer": len(self._frames),
                "tiles_put": self.tiles_put,
                "tiles_evicted": self.tiles_evicted,
                "frames_evicted": self.frames_evicted,
                "tiles_total_now": sum(len(v) for v in self._frames.values()),
            }