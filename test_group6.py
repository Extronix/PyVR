"""
Group 6 microtest.

Запускает на клиенте:
  - TileReceiver на 5002
  - фоновый ROI-sender (имитация HMD, шлёт меняющийся yaw)
  
Сервер должен быть запущен отдельно (python -m server.main).

Через 5 секунд печатает статистику и пинает критерии.
"""
from __future__ import annotations
import logging
import socket
import struct
import threading
import time

from common.config import CFG
from common.protocol import ROIPacket, now_ms
from client.network.tile_buffer import TileBuffer
from client.network.tile_receiver import TileReceiver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_group6")


# --- ROI sender (фоновый поток) ----------------------------------------------
class FakeROISender:
    def __init__(self, host=None, port=None, hz=60):
        self.host = host or CFG.network.host
        self.port = port or CFG.network.roi_port
        self.period = 1.0 / hz
        self._stop = threading.Event()
        self._thread = None
        self.sent = 0
    
    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        seq = 0
        t0 = time.perf_counter()
        try:
            while not self._stop.is_set():
                elapsed = time.perf_counter() - t0
                # вращаемся: -180..+180 за 6 секунд
                yaw = ((elapsed / 6.0) * 360.0) % 360.0 - 180.0
                pkt = ROIPacket(
                    seq=seq,
                    timestamp_ms=now_ms(),
                    frame_id=0,
                    yaw_deg=yaw,
                    pitch_deg=0.0,
                    fov_deg=90.0,
                    yaw_vel_dps=0.0,
                    pitch_vel_dps=0.0,
                )
                sock.sendto(pkt.to_json_bytes(), (self.host, self.port))
                seq += 1
                self.sent += 1
                time.sleep(self.period)
        finally:
            sock.close()


# --- main ---------------------------------------------------------------------
def main():
    print("=" * 60)
    print("GROUP 6 MICROTEST")
    print("=" * 60)
    print()
    print("ПЕРЕД ЗАПУСКОМ:")
    print("  1) В отдельном окне запусти сервер:  python -m server.main")
    print("  2) Затем здесь Enter для старта")
    print()
    input("Press Enter when server is running...")
    
    duration = 5.0
    
    buffer = TileBuffer(max_frames=4)
    receiver = TileReceiver(buffer=buffer)
    roi_sender = FakeROISender(hz=60)
    
    receiver.start()
    roi_sender.start()
    
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < duration:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    
    roi_sender.stop()
    time.sleep(0.2)  # дать receiver дособрать
    receiver.stop()
    
    # --- статистика ---
    asm = receiver.assembler
    buf_stats = buffer.stats()
    avg_lat = receiver.avg_latency_ms()
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Duration:              {duration:.2f} s")
    print()
    print(f"  Chunks received:       {receiver.chunks_recv}")
    print(f"  Bytes received:        {receiver.bytes_recv} "
          f"({receiver.bytes_recv / 1024 / 1024:.2f} MB)")
    print(f"  Parse errors:          {receiver.parse_errors}")
    print(f"  Decode errors:         {receiver.decode_errors}")
    print()
    print(f"  Tiles completed:       {receiver.tiles_completed}")
    print(f"  Duplicate chunks:      {asm.duplicate_chunks}")
    print(f"  Slots dropped (stale): {asm.dropped_stale}")
    print(f"  Slots dropped (over):  {asm.dropped_overflow}")
    print(f"  Active slots left:     {asm.active_slots()}")
    print()
    print(f"  Avg latency:           {avg_lat:.2f} ms "
          f"(max {receiver.latency_max_ms:.1f})")
    print()
    print(f"  Buffer frames now:     {buf_stats['frames_in_buffer']}")
    print(f"  Buffer tiles now:      {buf_stats['tiles_total_now']}")
    print(f"  Tiles put total:       {buf_stats['tiles_put']}")
    print(f"  Tiles evicted:         {buf_stats['tiles_evicted']}")
    print(f"  Frames evicted:        {buf_stats['frames_evicted']}")
    print()
    print(f"  ROI packets sent:      {roi_sender.sent}")
    print("=" * 60)
    
    # --- критерии ---
    print()
    fails = []
    
    if receiver.parse_errors > 0:
        fails.append(f"parse_errors={receiver.parse_errors}")
    if receiver.decode_errors > 0:
        fails.append(f"decode_errors={receiver.decode_errors}")
    if receiver.tiles_completed < 200:
        fails.append(f"tiles_completed={receiver.tiles_completed} < 200")
    if avg_lat > 50:
        fails.append(f"avg_latency={avg_lat:.1f} > 50ms")
    if buf_stats['frames_in_buffer'] == 0:
        fails.append("buffer empty at end")
    
    if fails:
        print("FAILED:")
        for f in fails:
            print(f"  - {f}")
    else:
        print("ALL CHECKS PASSED ✓")


if __name__ == "__main__":
    main()