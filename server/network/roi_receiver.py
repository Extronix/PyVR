"""
UDP-приёмник ROI-пакетов от клиента. Работает в фоновом потоке.
Главный поток сервера читает пакеты через .recv() из очереди.
"""
import socket
import threading
import queue
import logging
from typing import Optional

from common.config import CFG
from common.protocol import ROIPacket

logger = logging.getLogger(__name__)


class ROIReceiver:
    """Асинхронный приёмник UDP-пакетов с ROI от клиента."""
    
    def __init__(self, host: str | None = None, port: int | None = None,
                 queue_size: int = 64):
        self.host = host or CFG.network.host
        self.port = port or CFG.network.roi_port
        
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                              CFG.network.socket_buffer_size)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.2)  # чтобы поток мог проверить stop_event
        
        self._queue: queue.Queue[ROIPacket] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        
        # Статистика
        self.received_count = 0
        self.parse_errors = 0
        self.dropped_count = 0      # сброшено из-за переполнения очереди
        self.last_seq: int | None = None
        self.lost_packets = 0       # детекция дыр в seq
        
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        
        logger.info(f"ROIReceiver listening on udp://{self.host}:{self.port}")
    
    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                data, addr = self._sock.recvfrom(8192)
            except socket.timeout:
                continue
            except OSError as e:
                if not self._stop_event.is_set():
                    logger.warning(f"recvfrom failed: {e}")
                break
            
            try:
                pkt = ROIPacket.from_json_bytes(data)
            except Exception as e:
                self.parse_errors += 1
                if self.parse_errors <= 5:
                    logger.warning(f"Parse error: {e}, raw={data[:80]!r}")
                continue
            
            self.received_count += 1
            
            # Детекция потерянных пакетов по seq
            if self.last_seq is not None:
                gap = pkt.seq - self.last_seq - 1
                if gap > 0:
                    self.lost_packets += gap
            self.last_seq = pkt.seq
            
            # Положить в очередь (drop-old, актуальность важнее)
            try:
                self._queue.put_nowait(pkt)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self.dropped_count += 1
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(pkt)
                except queue.Full:
                    self.dropped_count += 1
    
    def recv(self, timeout: float = 0.1) -> Optional[ROIPacket]:
        """Получить следующий пакет из очереди (или None по таймауту)."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def recv_latest(self) -> Optional[ROIPacket]:
        """
        Опустошить очередь и вернуть самый последний пакет.
        Полезно когда сервер не успевает за клиентом — берём самое свежее.
        """
        latest = None
        try:
            while True:
                latest = self._queue.get_nowait()
        except queue.Empty:
            pass
        return latest
    
    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._sock.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()