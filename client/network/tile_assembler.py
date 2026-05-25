"""
Сборщик JPEG-тайла из UDP-чанков.

Ключ слота: (frame_id, tile_id).
Слот удаляется когда:
  - все чанки получены → отдаём готовый JPEG наружу
  - устарел по таймауту (для защиты от утечки при потерях)
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class _Slot:
    """Слот для сборки одного тайла."""
    total: int
    chunks: dict[int, bytes] = field(default_factory=dict)
    first_seen: float = field(default_factory=time.perf_counter)
    timestamp_ms: int = 0   # из заголовка пакета — момент отправки на сервере
    
    def add(self, idx: int, payload: bytes) -> None:
        self.chunks[idx] = payload
    
    def is_complete(self) -> bool:
        return len(self.chunks) == self.total
    
    def join(self) -> bytes:
        """Собирает чанки по возрастанию индекса."""
        return b"".join(self.chunks[i] for i in range(self.total))


class TileAssembler:
    """
    Накопитель чанков. НЕ thread-safe — должен использоваться
    только из одного потока (receiver loop).
    """
    
    def __init__(self,
                 slot_ttl_sec: float = 0.5,
                 max_slots: int = 256):
        """
        :param slot_ttl_sec: через сколько неполный слот считается мёртвым
        :param max_slots: жёсткий лимит количества активных слотов
        """
        self._slots: dict[tuple[int, int], _Slot] = {}
        self.slot_ttl_sec = slot_ttl_sec
        self.max_slots = max_slots
        
        # статистика
        self.completed = 0
        self.dropped_stale = 0
        self.dropped_overflow = 0
        self.duplicate_chunks = 0
    
    def add_chunk(self,
                  frame_id: int,
                  tile_id: int,
                  chunk_idx: int,
                  total_chunks: int,
                  timestamp_ms: int,
                  payload: bytes) -> Optional[tuple[bytes, int]]:
        """
        Добавляет чанк. Если тайл собран — возвращает (jpeg_bytes, timestamp_ms),
        иначе None.
        """
        key = (frame_id, tile_id)
        slot = self._slots.get(key)
        
        if slot is None:
            if len(self._slots) >= self.max_slots:
                self._gc_force()
            slot = _Slot(total=total_chunks, timestamp_ms=timestamp_ms)
            self._slots[key] = slot
        
        if chunk_idx in slot.chunks:
            self.duplicate_chunks += 1
            return None
        
        slot.add(chunk_idx, payload)
        
        if slot.is_complete():
            jpeg = slot.join()
            ts = slot.timestamp_ms
            del self._slots[key]
            self.completed += 1
            return jpeg, ts
        
        return None
    
    def gc(self) -> int:
        """Удаляет протухшие слоты. Возвращает кол-во удалённых."""
        now = time.perf_counter()
        ttl = self.slot_ttl_sec
        stale = [k for k, s in self._slots.items()
                 if now - s.first_seen > ttl]
        for k in stale:
            del self._slots[k]
        self.dropped_stale += len(stale)
        return len(stale)
    
    def _gc_force(self) -> None:
        """Аварийный сброс при переполнении: удаляем 25% самых старых."""
        n = max(1, len(self._slots) // 4)
        oldest = sorted(self._slots.items(),
                        key=lambda kv: kv[1].first_seen)[:n]
        for k, _ in oldest:
            del self._slots[k]
        self.dropped_overflow += n
        logger.warning(f"TileAssembler: overflow, dropped {n} oldest slots")
    
    def active_slots(self) -> int:
        return len(self._slots)