"""
Фоновое чтение видеофайла. Декодирование идёт в отдельном потоке,
кадры складываются в очередь — главный поток не блокируется.
"""
import threading
import queue
import time
import cv2
import numpy as np

from common.config import CFG


class ThreadedVideoSource:
    """Читает видеофайл в фоновом потоке, отдаёт кадры через .read()."""
    
    def __init__(self, src_path: str | None = None, loop: bool = True,
                 queue_size: int = 4):
        self.src_path = src_path or CFG.video.src_path
        self.loop = loop
        
        self._cap = cv2.VideoCapture(self.src_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.src_path}")
        
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._frame_id = 0
        
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
    
    def _worker(self) -> None:
        """Фоновый поток: читает кадры → складывает в очередь."""
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._frame_id = 0
                    continue
                else:
                    self._stop_event.set()
                    break
            
            self._frame_id += 1
            
            # Если очередь полна — ждём, не дропаем
            # (можно поменять на drop-old по желанию)
            try:
                self._queue.put((self._frame_id, frame), timeout=0.5)
            except queue.Full:
                # Главный поток завис — выходим
                if self._stop_event.is_set():
                    break
    
    def read(self, timeout: float = 1.0) -> tuple[int, np.ndarray] | None:
        """Возвращает (frame_id, frame) или None если поток остановлен."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self) -> None:
        self._stop_event.set()
        # Опустошим очередь, чтобы worker не висел на put
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=2.0)
        self._cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()