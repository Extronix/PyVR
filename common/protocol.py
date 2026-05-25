"""
Сетевой протокол ROI-пакетов (клиент → сервер).
"""
import json
import time
from dataclasses import dataclass, asdict
import struct
from dataclasses import dataclass


# Формат бинарного заголовка чанка тайла:
#   magic:        4s   ("PVRT")
#   version:      B    (1)
#   tile_id:      H    (0..255, помещается в byte, но H для запаса)
#   frame_id:     I    (uint32)
#   chunk_idx:    H    (uint16, 0..65535)
#   total_chunks: H    (uint16)
#   payload_len:  H    (uint16)
#   timestamp_ms: Q    (uint64, server time)
# Итого: 4+1+2+4+2+2+2+8 = 25 байт (выровняем до 25, не 24)
#
# Структура: !4sBHIHHHQ  → big-endian (сетевой порядок), без паддинга
_CHUNK_HEADER_FMT = "!4sBHIHHHQ"
_CHUNK_HEADER_SIZE = struct.calcsize(_CHUNK_HEADER_FMT)
_CHUNK_MAGIC = b"PVRT"
_CHUNK_VERSION = 1

@dataclass
class ROIPacket:
    """
    Пакет с состоянием головы клиента.
    Отправляется по UDP с частотой ~60 Hz.
    """
    seq: int                  # порядковый номер (для детекции потерь)
    timestamp_ms: float       # клиентское время отправки (для замера latency)
    frame_id: int             # текущий frame на клиенте
    
    yaw_deg: float            # head yaw
    pitch_deg: float          # head pitch
    fov_deg: float            # текущий FOV
    
    yaw_vel_dps: float        # угловая скорость yaw (deg/sec)
    pitch_vel_dps: float      # угловая скорость pitch (deg/sec)
    
    def to_json_bytes(self) -> bytes:
        """Сериализация в JSON-байты (для отправки в socket)."""
        return json.dumps(asdict(self)).encode('utf-8')
    
    @classmethod
    def from_json_bytes(cls, data: bytes) -> "ROIPacket":
        """Десериализация из JSON-байтов (на стороне сервера)."""
        d = json.loads(data.decode('utf-8'))
        return cls(**d)


def now_ms() -> float:
    """Текущее время в миллисекундах (для timestamp)."""
    return time.time() * 1000.0


# ============================================================
# Самопроверка
# ============================================================

if __name__ == "__main__":
    print("=== protocol.py self-test ===")
    
    pkt = ROIPacket(
        seq=42,
        timestamp_ms=now_ms(),
        frame_id=1270,
        yaw_deg=45.3,
        pitch_deg=-12.1,
        fov_deg=90.0,
        yaw_vel_dps=25.4,
        pitch_vel_dps=-3.1,
    )
    
    # Сериализация → десериализация
    blob = pkt.to_json_bytes()
    print(f"Serialized size: {len(blob)} bytes")
    print(f"Content: {blob.decode()}")
    
    pkt2 = ROIPacket.from_json_bytes(blob)
    assert pkt2 == pkt, "Round-trip failed!"
    print("✅ Round-trip OK")


@dataclass
class TileChunkPacket:
    """
    Один UDP-чанк тайла. Тайл фрагментирован на total_chunks чанков,
    каждый со своим chunk_idx (0..total_chunks-1).
    """
    tile_id: int
    frame_id: int
    chunk_idx: int
    total_chunks: int
    timestamp_ms: int
    payload: bytes              # часть JPEG-байтов тайла
    
    def to_bytes(self) -> bytes:
        header = struct.pack(
            _CHUNK_HEADER_FMT,
            _CHUNK_MAGIC,
            _CHUNK_VERSION,
            int(self.tile_id),
            int(self.frame_id),
            int(self.chunk_idx),
            int(self.total_chunks),
            len(self.payload),
            int(self.timestamp_ms),
        )
        return header + self.payload
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "TileChunkPacket":
        if len(data) < _CHUNK_HEADER_SIZE:
            raise ValueError(f"Packet too short: {len(data)} bytes")
        
        header = data[:_CHUNK_HEADER_SIZE]
        magic, version, tile_id, frame_id, chunk_idx, total_chunks, payload_len, timestamp_ms = \
            struct.unpack(_CHUNK_HEADER_FMT, header)
        
        if magic != _CHUNK_MAGIC:
            raise ValueError(f"Bad magic: {magic!r}")
        if version != _CHUNK_VERSION:
            raise ValueError(f"Unsupported version: {version}")
        
        payload = data[_CHUNK_HEADER_SIZE:_CHUNK_HEADER_SIZE + payload_len]
        if len(payload) != payload_len:
            raise ValueError(f"Payload truncated: expected {payload_len}, got {len(payload)}")
        
        return cls(
            tile_id=tile_id,
            frame_id=frame_id,
            chunk_idx=chunk_idx,
            total_chunks=total_chunks,
            timestamp_ms=timestamp_ms,
            payload=payload,
        )
    
    @staticmethod
    def header_size() -> int:
        return _CHUNK_HEADER_SIZE