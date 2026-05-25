"""
UDP-отправка ROI-пакетов на сервер. Работает в отдельном потоке,
чтобы не блокировать рендер.
"""
import socket
import threading
import queue
import logging

from common.config import CFG
from common.protocol import ROIPacket

logger = logging.getLogger(__name__)


class ROISender:
    """Асинхронная отправка ROIPacket по UDP."""
    
    def __init__(self, host: str | None = None, port: int | None = None,
                 queue_size: int | None = None):
        self.host = host or CFG.network.host
        self.port = port or CFG.network.roi_port
        qsize = queue_size or CFG.network.sender_queue_size
        
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (self.host, self.port)
        
        self._queue: queue.Queue[ROIPacket] = queue.Queue(maxsize=qsize)
        self._stop_event = threading.Event()
        
        # Статистика
        self.sent_count = 0
        self.dropped_count = 0
        
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        
        logger.info(f"ROISender → udp://{self.host}:{self.port}")
    
    def send(self, pkt: ROIPacket) -> bool:
        """
        Поставить пакет в очередь. Если очередь полна — дропаем САМЫЙ
        СТАРЫЙ пакет (актуальность важнее доставки всех).
        Возвращает True если поставили, False если дропнули.
        """
        try:
            self._queue.put_nowait(pkt)
            return True
        except queue.Full:
            # Очередь полна: выкидываем старый, ставим новый
            try:
                self._queue.get_nowait()
                self.dropped_count += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(pkt)
                return True
            except queue.Full:
                self.dropped_count += 1
                return False
    
    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                pkt = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            
            try:
                blob = pkt.to_json_bytes()
                self._sock.sendto(blob, self._addr)
                self.sent_count += 1
            except OSError as e:
                logger.warning(f"sendto failed: {e}")
    
    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._sock.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()