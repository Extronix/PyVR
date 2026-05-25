"""
UDP-слушатель тайл-чанков. Работает в отдельном потоке.

Поток выполняет:
  recv → parse → TileAssembler.add_chunk → если тайл собран → decode JPEG → TileBuffer.put

Не блокирует главный поток клиента.
"""
from __future__ import annotations
import logging
import socket
import threading
import time
from typing import Optional

from common.config import CFG
from common.protocol import TileChunkPacket, now_ms
from common.jpeg_codec import decode_tile

from client.network.tile_assembler import TileAssembler
from client.network.tile_buffer import TileBuffer, TileEntry

logger = logging.getLogger(__name__)


class TileReceiver:
    """
    Принимает UDP-чанки, собирает, декодирует, кладёт в TileBuffer.
    """
    
    def __init__(self,
                 buffer: TileBuffer,
                 host: str | None = None,
                 port: int | None = None,
                 recv_buf_size: int | None = None):
        self.buffer = buffer
        self.host = host or CFG.network.host
        self.port = port or CFG.network.tile_port
        self.recv_buf_size = recv_buf_size or CFG.network.socket_buffer_size
        
        self.assembler = TileAssembler(
            slot_ttl_sec=0.5,
            max_slots=512,
        )
        
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._last_gc = 0.0
        
        # статистика
        self.chunks_recv = 0
        self.bytes_recv = 0
        self.parse_errors = 0
        self.decode_errors = 0
        self.tiles_completed = 0
        self.latency_sum_ms = 0.0
        self.latency_max_ms = 0.0
    
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("TileReceiver already started")
        
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.recv_buf_size)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.2)
        
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="TileReceiver",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"TileReceiver listening on udp://{self.host}:{self.port}")
    
    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_evt.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        logger.info(
            f"TileReceiver stopped: chunks={self.chunks_recv} "
            f"tiles={self.tiles_completed} bytes={self.bytes_recv} "
            f"parse_err={self.parse_errors} decode_err={self.decode_errors}"
        )
    
    def _loop(self) -> None:
        assert self._sock is not None
        sock = self._sock
        # читаем максимально возможный UDP-пакет (header + payload)
        bufsize = 2048
        
        while not self._stop_evt.is_set():
            try:
                data, _addr = sock.recvfrom(bufsize)
            except socket.timeout:
                self._gc_maybe()
                continue
            except OSError as e:
                if not self._stop_evt.is_set():
                    logger.warning(f"recvfrom error: {e}")
                break
            
            self.chunks_recv += 1
            self.bytes_recv += len(data)
            
            try:
                pkt = TileChunkPacket.from_bytes(data)
            except Exception:
                self.parse_errors += 1
                continue
            
            result = self.assembler.add_chunk(
                frame_id=pkt.frame_id,
                tile_id=pkt.tile_id,
                chunk_idx=pkt.chunk_idx,
                total_chunks=pkt.total_chunks,
                timestamp_ms=pkt.timestamp_ms,
                payload=pkt.payload,
            )
            
            if result is not None:
                jpeg_bytes, server_ts_ms = result
                try:
                    tile_img = decode_tile(jpeg_bytes)
                except RuntimeError as e:
                    self.decode_errors += 1
                    logger.warning(f"decode failed: {e}")
                    continue
                
                recv_ts = now_ms()
                lat = recv_ts - server_ts_ms
                self.latency_sum_ms += lat
                if lat > self.latency_max_ms:
                    self.latency_max_ms = lat
                
                self.buffer.put(TileEntry(
                    tile=tile_img,
                    frame_id=pkt.frame_id,
                    tile_id=pkt.tile_id,
                    server_ts_ms=server_ts_ms,
                    recv_ts_ms=recv_ts,
                ))
                self.tiles_completed += 1
            
            self._gc_maybe()
    
    def _gc_maybe(self) -> None:
        """Раз в ~200 мс чистим протухшие слоты."""
        now = time.perf_counter()
        if now - self._last_gc > 0.2:
            self.assembler.gc()
            self._last_gc = now
    
    def avg_latency_ms(self) -> float:
        if self.tiles_completed == 0:
            return 0.0
        return self.latency_sum_ms / self.tiles_completed