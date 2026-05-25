"""
Микротест группы 5:
1) Слушает входящие UDP-тайлы на порту tile_port
2) Шлёт 30 фиктивных ROI-пакетов клиентом-эмулятором
3) Запускаем сервер вручную в другом окне — а здесь только эмулятор клиента
   и счётчик принятых тайлов.

ИСПОЛЬЗОВАНИЕ:
  Терминал 1:  python -m server.main
  Терминал 2:  python test_group5.py

Скрипт делает:
  - открывает UDP-сокет на tile_port (как будто это клиент)
  - параллельно шлёт фейковые ROIPacket'ы по roi_port
  - 5 секунд собирает входящие чанки
  - выводит статистику
"""
import socket
import threading
import time
from collections import defaultdict

from common.config import CFG
from common.protocol import ROIPacket, TileChunkPacket, now_ms


def roi_sender(stop_event: threading.Event):
    """Шлёт ROIPacket по 60 Hz, имитируя клиента."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 0
    yaw = 0.0
    t_start = time.perf_counter()
    
    while not stop_event.is_set():
        # Медленно вращаем yaw
        yaw = ((time.perf_counter() - t_start) * 30.0) % 360.0 - 180.0
        
        pkt = ROIPacket(
            seq=seq,
            frame_id=seq,
            timestamp_ms=now_ms(),
            yaw_deg=yaw,
            pitch_deg=0.0,
            fov_deg=90.0,
            yaw_vel_dps=30.0,
            pitch_vel_dps=0.0,
        )
        data = pkt.to_json_bytes()
        try:
            sock.sendto(data, (CFG.network.host, CFG.network.roi_port))
        except OSError:
            pass
        
        seq += 1
        time.sleep(1.0 / CFG.network.roi_send_rate_hz)
    
    sock.close()
    print(f"[ROI sender] stopped, sent {seq} packets")


def tile_listener(duration_s: float = 5.0):
    """Слушает входящие TileChunkPacket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                    CFG.network.socket_buffer_size)
    sock.bind((CFG.network.host, CFG.network.tile_port))
    sock.settimeout(0.5)
    
    print(f"[Tile listener] listening udp://{CFG.network.host}:"
          f"{CFG.network.tile_port} for {duration_s}s")
    
    t_start = time.perf_counter()
    
    chunks_received = 0
    bytes_received = 0
    parse_errors = 0
    
    # frame_id -> tile_id -> set(chunk_idx)
    frames: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    # frame_id -> tile_id -> total_chunks
    totals: dict[int, dict[int, int]] = defaultdict(dict)
    
    latencies = []
    
    while time.perf_counter() - t_start < duration_s:
        try:
            data, _ = sock.recvfrom(65536)
        except socket.timeout:
            continue
        
        try:
            pkt = TileChunkPacket.from_bytes(data)
        except Exception as e:
            parse_errors += 1
            if parse_errors <= 3:
                print(f"  Parse error: {e}")
            continue
        
        chunks_received += 1
        bytes_received += len(data)
        frames[pkt.frame_id][pkt.tile_id].add(pkt.chunk_idx)
        totals[pkt.frame_id][pkt.tile_id] = pkt.total_chunks
        
        # Latency по timestamp пакета
        lat = now_ms() - pkt.timestamp_ms
        latencies.append(lat)
    
    sock.close()
    
    # Анализ
    complete_tiles = 0
    incomplete_tiles = 0
    for fid, tiles in frames.items():
        for tid, chunks in tiles.items():
            total = totals[fid][tid]
            if len(chunks) == total:
                complete_tiles += 1
            else:
                incomplete_tiles += 1
    
    elapsed = time.perf_counter() - t_start
    
    print("\n" + "=" * 60)
    print("[Tile listener] results:")
    print("=" * 60)
    print(f"  Duration:         {elapsed:.2f} s")
    print(f"  Chunks received:  {chunks_received}")
    print(f"  Bytes received:   {bytes_received} "
          f"({bytes_received / 1024 / 1024:.2f} MB)")
    print(f"  Parse errors:     {parse_errors}")
    print(f"  Frames seen:      {len(frames)}")
    print(f"  Complete tiles:   {complete_tiles}")
    print(f"  Incomplete tiles: {incomplete_tiles}")
    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"  Avg chunk lat:    {avg:.2f} ms (max {max(latencies):.1f})")
    if elapsed > 0:
        print(f"  Throughput:       {bytes_received / elapsed / 1024 / 1024:.2f} MB/s")
        print(f"  Chunk rate:       {chunks_received / elapsed:.0f} chunks/s")
    print("=" * 60)
    
    # Проверки
    assert chunks_received > 0, "Сервер не прислал ни одного чанка!"
    assert complete_tiles > 0, "Ни один тайл не собрался целиком!"
    
    # Доля собранных целиком (для последнего кадра может быть incomplete — нормально)
    if complete_tiles + incomplete_tiles > 0:
        ratio = complete_tiles / (complete_tiles + incomplete_tiles)
        print(f"  Complete ratio:   {ratio*100:.1f}%")
        assert ratio > 0.5, f"Слишком много неполных тайлов: {ratio*100:.1f}%"
    
    print("\nALL CHECKS PASSED ✓")


def main():
    print("=" * 60)
    print("GROUP 5 MICROTEST")
    print("=" * 60)
    print("\nПЕРЕД ЗАПУСКОМ:")
    print("  1) В отдельном окне запусти сервер:  python -m server.main")
    print("  2) Затем здесь Enter для старта")
    input("\nPress Enter when server is running...")
    
    stop_event = threading.Event()
    
    # ROI sender в фоне
    roi_thread = threading.Thread(target=roi_sender, args=(stop_event,), daemon=True)
    roi_thread.start()
    
    # Даём серверу пару секунд получить первый ROI
    time.sleep(1.0)
    
    try:
        tile_listener(duration_s=5.0)
    finally:
        stop_event.set()
        roi_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()