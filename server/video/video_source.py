"""
Серверный источник ERP-видео.

В отличие от клиентского (который работал на шагах 1-7), серверный
VideoSource:
- читает кадр СИНХРОННО (без фонового потока — главный цикл сервера
  и так driven by video FPS)
- умеет зацикливаться
- логирует FPS источника

Если в будущем понадобится async-чтение (декодер не успевает) — можно
обернуть в Thread аналогично клиентскому варианту.
"""
from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from common.config import CFG

logger = logging.getLogger(__name__)


class VideoSource:
    """Синхронный читатель ERP-видеофайла для сервера."""
    
    def __init__(self, path: str | None = None, loop: bool | None = None):
        self.path = path or CFG.video.source_path
        self.loop = CFG.video.loop if loop is None else loop
        
        if not Path(self.path).exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")
        
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_period = 1.0 / self.fps
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.frame_id = 0      # сквозной счётчик прочитанных кадров
        self.loop_count = 0    # сколько раз перешли через конец файла
        
        # Для троттлинга
        self._next_frame_time: float | None = None
        
        logger.info(
            f"VideoSource opened: {self.path} "
            f"({self.width}x{self.height} @ {self.fps:.2f} fps, "
            f"{self.total_frames} frames)"
        )
    
    def read(self) -> Optional[np.ndarray]:
        """Читает один кадр. Возвращает None в конце (если loop=False)."""
        ok, frame = self._cap.read()
        if not ok:
            if self.loop:
                self.loop_count += 1
                logger.info(f"Video loop #{self.loop_count}, rewinding")
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if not ok:
                    logger.error("Failed to read after rewind")
                    return None
            else:
                logger.info("Video ended (loop=False)")
                return None
        
        self.frame_id += 1
        return frame
    
    def read_throttled(self) -> Optional[np.ndarray]:
        """
        Читает кадр, ограничивая темп по FPS источника.
        Если предыдущий вызов был раньше чем frame_period назад — спим.
        """
        now = time.perf_counter()
        if self._next_frame_time is not None and now < self._next_frame_time:
            time.sleep(self._next_frame_time - now)
            now = time.perf_counter()
        
        frame = self.read()
        
        # Планируем следующий кадр
        if self._next_frame_time is None:
            self._next_frame_time = now + self.frame_period
        else:
            self._next_frame_time += self.frame_period
            # Если сильно отстали — догоняем (не копим долг)
            if now - self._next_frame_time > self.frame_period:
                self._next_frame_time = now + self.frame_period
        return frame
    
    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            logger.info(f"VideoSource closed (read {self.frame_id} frames, "
                        f"{self.loop_count} loops)")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()