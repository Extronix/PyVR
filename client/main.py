"""
Точка входа клиента PyVR.

Запуск:
    python -m client.main
"""
import logging
import time

from common.config import CFG
from common.protocol import ROIPacket, now_ms

from client.sensors.mock_imu import MockIMU
from client.renderer.video_source import ThreadedVideoSource
from client.renderer.view_builder import ViewBuilder
from client.renderer.window import ClientWindow
from client.network.roi_sender import ROISender


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("client")


def main() -> None:
    log.info("=" * 60)
    log.info("PyVR Client — step 8")
    log.info("=" * 60)
    
    imu = MockIMU()
    
    with ThreadedVideoSource() as video, ROISender() as sender:
        log.info(f"Video: {video.width}x{video.height} @ {video.fps:.1f} fps, "
                 f"{video.total_frames} frames")
        
        view_builder = ViewBuilder(
            erp_w=video.width,
            erp_h=video.height,
            view_w=CFG.video.view_w,
            view_h=CFG.video.view_h,
        )
        window = ClientWindow(imu, CFG.video.view_w, CFG.video.view_h)
        
        # Контроль частоты ROI
        roi_period = 1.0 / CFG.network.roi_send_rate_hz
        last_roi_send = 0.0
        seq = 0
        
        # FPS-метрика
        fps_t0 = time.perf_counter()
        fps_frames = 0
        fps_value = 0.0
        
        try:
            while True:
                item = video.read(timeout=2.0)
                if item is None:
                    log.warning("Video stream ended")
                    break
                frame_id, erp_frame = item
                
                # IMU тик (обновляет скорости)
                imu.tick()
                
                # Рендер
                view = view_builder.render(
                    erp_frame,
                    imu.yaw_deg, imu.pitch_deg, imu.fov_deg,
                )
                
                # ROI отправка (rate-limited)
                now = time.perf_counter()
                if now - last_roi_send >= roi_period:
                    pkt = ROIPacket(
                        seq=seq,
                        timestamp_ms=now_ms(),
                        frame_id=frame_id,
                        yaw_deg=imu.yaw_deg,
                        pitch_deg=imu.pitch_deg,
                        fov_deg=imu.fov_deg,
                        yaw_vel_dps=imu.yaw_vel_dps,
                        pitch_vel_dps=imu.pitch_vel_dps,
                    )
                    sender.send(pkt)
                    seq += 1
                    last_roi_send = now
                
                # FPS
                fps_frames += 1
                if fps_frames >= 30:
                    elapsed = time.perf_counter() - fps_t0
                    fps_value = fps_frames / elapsed
                    fps_t0 = time.perf_counter()
                    fps_frames = 0
                
                # HUD
                hud = [
                    f"Frame {frame_id} | FPS: {fps_value:5.1f}",
                    f"Yaw: {imu.yaw_deg:+7.2f}  Pitch: {imu.pitch_deg:+6.2f}  "
                    f"FOV: {imu.fov_deg:5.1f}",
                    f"Vel: ({imu.yaw_vel_dps:+6.1f}, {imu.pitch_vel_dps:+6.1f}) dps",
                    f"ROI sent: {sender.sent_count}  dropped: {sender.dropped_count}",
                    f"Cache hits: {view_builder.cache_hits}  miss: {view_builder.cache_misses}",
                    "[LMB drag] look | [WASD] step | [Wheel] FOV | [Q/ESC] quit",
                ]
                
                key = window.show(view, hud)
                
                # Логи периодически
                if frame_id % CFG.log.client_log_every_n_frames == 0:
                    log.info(
                        f"frame={frame_id} fps={fps_value:.1f} "
                        f"yaw={imu.yaw_deg:+.1f} pitch={imu.pitch_deg:+.1f} "
                        f"sent={sender.sent_count} drop={sender.dropped_count}"
                    )
                # Выход
                if key in (ord('q'), ord('Q'), 27):  # Q или ESC
                    log.info("Quit requested by user")
                    break
                
                if window.is_window_closed():
                    log.info("Window closed by user")
                    break
        
        except KeyboardInterrupt:
            log.info("Interrupted by Ctrl+C")
        
        finally:
            log.info(
                f"Stats: ROI sent={sender.sent_count}, dropped={sender.dropped_count}, "
                f"cache hits={view_builder.cache_hits}, miss={view_builder.cache_misses}"
            )
            window.close()
    
    log.info("Client shut down cleanly")


if __name__ == "__main__":
    main()