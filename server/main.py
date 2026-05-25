"""
Точка входа сервера PyVR — шаг 9.

Сервер:
  1) читает ERP-видео data/RoSh.mp4 по FPS источника
  2) принимает ROI-пакеты от клиента (фоновый поток)
  3) для каждого кадра определяет видимые тайлы по последнему ROI
  4) режет тайлы, кодирует JPEG, шлёт чанки по UDP

Запуск:
    python -m server.main
"""
import logging
import time
from collections import deque

from common.config import CFG
from common.protocol import now_ms
from common.tiles import TileGrid

from server.network.roi_receiver import ROIReceiver
from server.network.tile_sender import TileSender
from server.video.video_source import VideoSource


logging.basicConfig(
    level=CFG.log.level,
    format=CFG.log.fmt,
    datefmt=CFG.log.datefmt,
)
log = logging.getLogger("server")


class LatencyStats:
    """Скользящее окно для latency ROI-пакетов."""
    
    def __init__(self, window_size: int = 120):
        self.latencies_ms: deque[float] = deque(maxlen=window_size)
    
    def add(self, latency_ms: float) -> None:
        self.latencies_ms.append(latency_ms)
    
    @property
    def avg(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0
    
    @property
    def max(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else 0.0


def main() -> None:
    log.info("=" * 60)
    log.info("PyVR Server — step 9 (tile streaming)")
    log.info("=" * 60)
    
    # Видео
    video = VideoSource()
    
    # Тайловая сетка по фактическим размерам кадра
    grid = TileGrid(video.width, video.height)
    log.info(f"Tile grid: {grid}")
    
    # Сеть
    receiver = ROIReceiver()
    sender = TileSender(grid)
    
    # Состояние
    last_roi = None             # последний полученный ROIPacket
    last_roi_age_frames = 0     # сколько кадров с момента получения ROI
    
    stats = LatencyStats(120)
    
    # Метрики кадрового цикла
    frame_times: deque[float] = deque(maxlen=120)
    encode_times: deque[float] = deque(maxlen=120)
    send_times: deque[float] = deque(maxlen=120)
    chunks_per_frame: deque[int] = deque(maxlen=120)
    bytes_per_frame: deque[int] = deque(maxlen=120)
    
    log_every = CFG.log.server_log_every_n_packets  # шкала "раз в N кадров"
    
    log.info("Server loop started")
    
    try:
        while True:
            loop_t0 = time.perf_counter()
            
            # 1) Читаем очередной кадр (с троттлингом по FPS видео)
            frame = video.read_throttled()
            if frame is None:
                log.info("Video stream ended")
                break
            
            # 2) Берём свежайший ROI (опустошаем очередь)
            new_roi = receiver.recv_latest()
            if new_roi is not None:
                last_roi = new_roi
                last_roi_age_frames = 0
                latency_ms = now_ms() - new_roi.timestamp_ms
                stats.add(latency_ms)
            else:
                last_roi_age_frames += 1
            
            if last_roi is None:
                visible, importance = grid.visible_tiles_with_importance(
                    yaw_deg=0.0,
                    pitch_deg=0.0,
                    fov_deg=CFG.view.fov_deg_default,
                )
            else:
                visible, importance = grid.visible_tiles_with_importance(
                    yaw_deg=last_roi.yaw_deg,
                    pitch_deg=last_roi.pitch_deg,
                    fov_deg=last_roi.fov_deg,
                )

            # 4) Кодируем и шлём (foveated quality)
            send_stats = sender.send_frame_tiles(
                frame=frame,
                tile_ids=visible,
                frame_id=video.frame_id,
                importance=importance,
            )
            
            loop_elapsed = time.perf_counter() - loop_t0
            frame_times.append(loop_elapsed * 1000)
            encode_times.append(send_stats["encode_ms"])
            send_times.append(send_stats["send_ms"])
            chunks_per_frame.append(send_stats["chunks"])
            bytes_per_frame.append(send_stats["bytes"])
            
            # 5) Периодический лог
            if video.frame_id % log_every == 0:
                avg_frame = sum(frame_times) / len(frame_times)
                avg_enc = sum(encode_times) / len(encode_times)
                avg_snd = sum(send_times) / len(send_times)
                avg_chunks = sum(chunks_per_frame) / len(chunks_per_frame)
                avg_bytes = sum(bytes_per_frame) / len(bytes_per_frame)
                
                roi_info = (f"yaw={last_roi.yaw_deg:+.0f} "
                            f"pitch={last_roi.pitch_deg:+.0f} "
                            f"fov={last_roi.fov_deg:.0f}"
                            ) if last_roi else "no-roi"
                
                log.info(
                    f"frame={video.frame_id} | {roi_info} | "
                    f"tiles={send_stats['tiles']} "
                    f"(c={send_stats['n_center']} e={send_stats['n_edge']} o={send_stats['n_outside']}) "
                    f"q={send_stats['q_avg']:.1f} [{send_stats['q_min']}..{send_stats['q_max']}] | "
                    f"chunks={send_stats['chunks']} | "
                    f"loop={avg_frame:.1f}ms (enc={avg_enc:.1f} snd={avg_snd:.1f}) | "
                    f"avg_bytes={avg_bytes:.0f} | "
                    f"roi_lat={stats.avg:.1f}ms (max={stats.max:.1f}) "
                    f"recv={receiver.received_count} lost={receiver.lost_packets} "
                    f"drop={receiver.dropped_count}"
                )
    
    except KeyboardInterrupt:
        log.info("Interrupted by Ctrl+C")
    
    finally:
        log.info("=" * 60)
        log.info("Shutting down...")
        sender.close()
        receiver.stop()
        video.release()
        
        if frame_times:
            log.info(f"Avg frame loop:    {sum(frame_times)/len(frame_times):.2f} ms")
        if encode_times:
            log.info(f"Avg encode time:   {sum(encode_times)/len(encode_times):.2f} ms")
        if send_times:
            log.info(f"Avg send time:     {sum(send_times)/len(send_times):.2f} ms")
        log.info(f"ROI received:      {receiver.received_count}")
        log.info(f"ROI lost (gaps):   {receiver.lost_packets}")
        log.info(f"ROI dropped:       {receiver.dropped_count}")
        log.info(f"Avg ROI latency:   {stats.avg:.2f} ms")
        log.info(f"Tiles sent:        {sender.tiles_sent}")
        log.info(f"Chunks sent:       {sender.chunks_sent}")
        log.info(f"Bytes sent:        {sender.bytes_sent} "
                 f"({sender.bytes_sent/1024/1024:.1f} MB)")
        log.info("=" * 60)
    
    log.info("Server shut down cleanly")


if __name__ == "__main__":
    main()