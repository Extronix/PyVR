"""
Шаг 6: FOV-miss penalty + downscale рендера.

Симулирует сетевую задержку: стратегия "знает" позицию головы N кадров назад
и передаёт тайлы под ту позицию. Рендер делается по актуальной голове, но
"чёрные" пиксели — это те, что не попали в переданное стратегией.

Стратегии:
  - FULL-FRAME : передаём всё (miss=0)
  - NAIVE-FOV  : передаём ровно FOV из задержанной позиции
  - TILED-FOV  : передаём тайлы 6x3, пересекающие FOV из задержанной позиции

Управление:
  стрелки/мышь — вращение
  +/-          — FOV
  пробел       — пауза
  1            — переключить визуализацию miss-пикселей
  2/3          — задержка +1 / -1 кадр
  Enter        — сброс ориентации
  ESC          — выход
"""

import cv2
import numpy as np
import os
import time
import csv
from collections import deque
from datetime import datetime

# ============ НАСТРОЙКИ ============
VIDEO_PATH = "data/RoSh.mp4"
VIEW_W, VIEW_H = 1280, 720
INITIAL_FOV = 100.0
ROTATE_STEP_DEG = 5.0
MOUSE_SENSITIVITY = 0.2

RENDER_W, RENDER_H = 1920, 960
TILE_COLS = 6
TILE_ROWS = 3
INITIAL_LATENCY_FRAMES = 5

LOG_DIR = "logs"
# ===================================


def make_rotation(yaw_deg, pitch_deg):
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]], dtype=np.float32)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]], dtype=np.float32)
    return Ry @ Rx


def view_rays_equirect_coords(yaw_deg, pitch_deg, fov_deg, eq_w, eq_h,
                               out_w, out_h):
    R = make_rotation(yaw_deg, pitch_deg)
    fov = np.deg2rad(fov_deg)
    f = (out_w / 2) / np.tan(fov / 2)
    u, v = np.meshgrid(np.arange(out_w), np.arange(out_h))
    x = (u - out_w / 2).astype(np.float32)
    y = (v - out_h / 2).astype(np.float32)
    z = np.full_like(x, f)
    vec = np.stack([x, y, z], axis=-1) @ R.T
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
    lon = np.arctan2(vec[..., 0], vec[..., 2])
    lat = np.arcsin(np.clip(vec[..., 1], -1, 1))
    map_x = ((lon / (2 * np.pi) + 0.5) * eq_w).astype(np.float32)
    map_y = ((lat / np.pi + 0.5) * eq_h).astype(np.float32)
    return map_x, map_y


def render_view(eq_img, map_x, map_y):
    return cv2.remap(eq_img, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)


def fov_mask_on_equirect(yaw_deg, pitch_deg, fov_deg, eq_w, eq_h,
                          samples_w=200, samples_h=110):
    R = make_rotation(yaw_deg, pitch_deg)
    fov = np.deg2rad(fov_deg)
    f = (samples_w / 2) / np.tan(fov / 2)
    u, v = np.meshgrid(np.arange(samples_w), np.arange(samples_h))
    x = (u - samples_w / 2).astype(np.float32)
    y = ((v - samples_h / 2).astype(np.float32)
         * (samples_w / samples_h) / (VIEW_W / VIEW_H))
    z = np.full_like(x, f)
    vec = np.stack([x, y, z], axis=-1) @ R.T
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
    lon = np.arctan2(vec[..., 0], vec[..., 2])
    lat = np.arcsin(np.clip(vec[..., 1], -1, 1))
    px = ((lon / (2 * np.pi) + 0.5) * eq_w).astype(np.int32) % eq_w
    py = ((lat / np.pi + 0.5) * eq_h).astype(np.int32).clip(0, eq_h - 1)
    mask = np.zeros((eq_h, eq_w), dtype=np.uint8)
    mask[py.flatten(), px.flatten()] = 1
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    return mask.astype(bool)


def tiled_mask_from_fov(fov_mask, tile_cols, tile_rows):
    eq_h, eq_w = fov_mask.shape
    tile_w = eq_w // tile_cols
    tile_h = eq_h // tile_rows
    visible_grid = np.zeros((tile_rows, tile_cols), dtype=bool)
    tiled_mask = np.zeros_like(fov_mask)
    for r in range(tile_rows):
        for c in range(tile_cols):
            y0, y1 = r * tile_h, (r + 1) * tile_h
            x0, x1 = c * tile_w, (c + 1) * tile_w
            if fov_mask[y0:y1, x0:x1].any():
                visible_grid[r, c] = True
                tiled_mask[y0:y1, x0:x1] = True
    return tiled_mask, visible_grid, int(visible_grid.sum())


def draw_minimap(view, visible_grid, yaw, pitch, d_yaw, d_pitch):
    mm_w, mm_h, margin = 360, 180, 20
    x0 = view.shape[1] - mm_w - margin
    y0 = view.shape[0] - mm_h - margin
    overlay = view.copy()
    cv2.rectangle(overlay, (x0 - 4, y0 - 4),
                  (x0 + mm_w + 4, y0 + mm_h + 4), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, view, 0.4, 0, view)
    tw = mm_w // TILE_COLS
    th = mm_h // TILE_ROWS
    for r in range(TILE_ROWS):
        for c in range(TILE_COLS):
            tx = x0 + c * tw
            ty = y0 + r * th
            color = (0, 200, 0) if visible_grid[r, c] else (40, 40, 200)
            cv2.rectangle(view, (tx, ty), (tx + tw - 1, ty + th - 1), color, -1)
            cv2.rectangle(view, (tx, ty), (tx + tw - 1, ty + th - 1),
                          (255, 255, 255), 1)
    dx = x0 + int(((d_yaw + 180) / 360.0) * mm_w)
    dy = y0 + mm_h - int(((d_pitch + 90) / 180.0) * mm_h)
    cv2.drawMarker(view, (dx, dy), (0, 165, 255), cv2.MARKER_CROSS, 14, 2)
    gx = x0 + int(((yaw + 180) / 360.0) * mm_w)
    gy = y0 + mm_h - int(((pitch + 90) / 180.0) * mm_h)
    cv2.drawMarker(view, (gx, gy), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
    cv2.putText(view, "yellow=now  orange=delayed",
                (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


class MouseState:
    def __init__(self):
        self.dragging = False
        self.last_x = 0
        self.last_y = 0
        self.dyaw = 0.0
        self.dpitch = 0.0

mouse = MouseState()

def mouse_callback(event, x, y, flags, _):
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse.dragging = True
        mouse.last_x, mouse.last_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        mouse.dragging = False
    elif event == cv2.EVENT_MOUSEMOVE and mouse.dragging:
        mouse.dyaw += (x - mouse.last_x) * MOUSE_SENSITIVITY
        mouse.dpitch -= (y - mouse.last_y) * MOUSE_SENSITIVITY
        mouse.last_x, mouse.last_y = x, y


def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Не найден: {VIDEO_PATH}")
        return
    os.makedirs(LOG_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть {VIDEO_PATH}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📹 {VIDEO_PATH}")
    print(f"   Оригинал: {orig_w}×{orig_h} @ {src_fps:.1f} fps, {total_frames} кадров")
    print(f"   Рендер из: {RENDER_W}×{RENDER_H} (downscale)")
    print(f"   Трафик считаем по оригиналу: {orig_w*orig_h/1e6:.2f} MPx/кадр")
    print(f"   Тайлы: {TILE_COLS}×{TILE_ROWS}")
    print(f"   Задержка: {INITIAL_LATENCY_FRAMES} кадров "
          f"(≈{1000*INITIAL_LATENCY_FRAMES/src_fps:.0f} мс)\n")

    log_path = os.path.join(LOG_DIR,
                            f"miss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow([
        "t_sec", "frame", "yaw", "pitch", "fov", "latency_frames",
        "full_traffic", "naive_traffic", "tiled_traffic",
        "full_miss", "naive_miss", "tiled_miss",
        "n_visible_tiles",
    ])
    print(f"📝 Лог: {log_path}\n")

    window = "Step 6: FOV-miss penalty"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window, np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8))
    cv2.waitKey(1)
    cv2.setMouseCallback(window, mouse_callback)

    yaw, pitch = 0.0, 0.0
    fov = INITIAL_FOV
    latency = INITIAL_LATENCY_FRAMES
    paused = False
    show_miss = True
    frame_idx = 0
    current_frame_small = None
    t_start = time.time()
    last_time = time.time()
    fps_smooth = 0.0

    head_history = deque(maxlen=120)

    sum_naive_tr = 0.0; sum_tiled_tr = 0.0
    sum_naive_ms = 0.0; sum_tiled_ms = 0.0
    n_samples = 0

    print("🎮 стрелки/мышь | +/- FOV | пробел пауза | 1 toggle miss | 2/3 latency ±1 | ESC\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                head_history.clear()
                continue
            current_frame_small = cv2.resize(frame, (RENDER_W, RENDER_H),
                                              interpolation=cv2.INTER_AREA)
            frame_idx += 1

        if current_frame_small is None:
            cv2.waitKey(10)
            continue

        yaw += mouse.dyaw; pitch += mouse.dpitch
        mouse.dyaw = 0.0; mouse.dpitch = 0.0
        pitch = max(-89.0, min(89.0, pitch))
        yaw = ((yaw + 180) % 360) - 180

        head_history.append((yaw, pitch, fov))
        if len(head_history) > latency:
            d_yaw, d_pitch, d_fov = head_history[-1 - latency]
        else:
            d_yaw, d_pitch, d_fov = head_history[0]

        # Рендер по актуальной голове
        map_x, map_y = view_rays_equirect_coords(
            yaw, pitch, fov, RENDER_W, RENDER_H, VIEW_W, VIEW_H)
        view = render_view(current_frame_small, map_x, map_y)

        # Маски стратегий по задержанной голове
        delayed_fov_mask = fov_mask_on_equirect(
            d_yaw, d_pitch, d_fov, RENDER_W, RENDER_H)
        delayed_tiled_mask, visible_grid, n_visible = tiled_mask_from_fov(
            delayed_fov_mask, TILE_COLS, TILE_ROWS)

        # Трафик (по оригиналу)
        full_pix = orig_w * orig_h
        naive_pix = VIEW_W * VIEW_H
        orig_tile_w = orig_w // TILE_COLS
        orig_tile_h = orig_h // TILE_ROWS
        tiled_pix = n_visible * orig_tile_w * orig_tile_h

        naive_tr = naive_pix / full_pix
        tiled_tr = tiled_pix / full_pix

        # FOV-miss
        sample_x = np.clip(map_x.astype(np.int32), 0, RENDER_W - 1)
        sample_y = np.clip(map_y.astype(np.int32), 0, RENDER_H - 1)

        naive_delivered = delayed_fov_mask[sample_y, sample_x]
        tiled_delivered = delayed_tiled_mask[sample_y, sample_x]
        naive_miss_pixels = ~naive_delivered
        tiled_miss_pixels = ~tiled_delivered
        naive_ms = float(naive_miss_pixels.mean())
        tiled_ms = float(tiled_miss_pixels.mean())

        sum_naive_tr += naive_tr; sum_tiled_tr += tiled_tr
        sum_naive_ms += naive_ms; sum_tiled_ms += tiled_ms
        n_samples += 1

        # Визуализация: красным закрашиваем miss-пиксели по TILED
        if show_miss:
            view_vis = view.copy()
            red = view_vis.copy()
            red[tiled_miss_pixels] = (0, 0, 255)
            cv2.addWeighted(red, 0.6, view_vis, 0.4, 0, view_vis)
        else:
            view_vis = view.copy()

        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            inst = 1.0 / dt
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst if fps_smooth > 0 else inst

        log.writerow([
            f"{now - t_start:.3f}", frame_idx,
            f"{yaw:.2f}", f"{pitch:.2f}", f"{fov:.1f}", latency,
            "1.0000", f"{naive_tr:.5f}", f"{tiled_tr:.5f}",
            "0.0000", f"{naive_ms:.5f}", f"{tiled_ms:.5f}",
            n_visible,
        ])

        draw_minimap(view_vis, visible_grid, yaw, pitch, d_yaw, d_pitch)

        avg_n_tr = 100 * sum_naive_tr / n_samples
        avg_t_tr = 100 * sum_tiled_tr / n_samples
        avg_n_ms = 100 * sum_naive_ms / n_samples
        avg_t_ms = 100 * sum_tiled_ms / n_samples

        hud = [
            f"yaw={yaw:+6.1f}  pitch={pitch:+5.1f}  FOV={fov:.0f}   frame {frame_idx}/{total_frames}   {fps_smooth:.1f} fps",
            f"latency = {latency} frames (~{1000*latency/src_fps:.0f} ms)   miss-vis: {'ON' if show_miss else 'off'}",
            "",
            f"                  TRAFFIC          MISS",
            f"  FULL-FRAME :   100.0 %         0.00 %",
            f"  NAIVE-FOV  :   {100*naive_tr:5.1f} %       {100*naive_ms:5.2f} %",
            f"  TILED-FOV  :   {100*tiled_tr:5.1f} %       {100*tiled_ms:5.2f} %    tiles {n_visible}/{TILE_COLS*TILE_ROWS}",
            "",
            f"  AVG so far  NAIVE: traffic={avg_n_tr:5.1f}%  miss={avg_n_ms:5.2f}%",
            f"              TILED: traffic={avg_t_tr:5.1f}%  miss={avg_t_ms:5.2f}%   samples={n_samples}",
        ]
        for i, line in enumerate(hud):
            cv2.putText(view_vis, line, (12, 24 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(view_vis, line, (12, 24 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.drawMarker(view_vis, (VIEW_W // 2, VIEW_H // 2),
                       (255, 255, 255), cv2.MARKER_CROSS, 18, 1)

        if paused:
            cv2.putText(view_vis, "PAUSED", (VIEW_W // 2 - 70, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow(window, view_vis)

        key = cv2.waitKeyEx(1)
        if key == 27:
            break
        elif key == 32:
            paused = not paused
        elif key in (13, 10):
            yaw, pitch = 0.0, 0.0
        elif key in (43, 61):
            fov = max(30, fov - 5)
        elif key in (45, 95):
            fov = min(140, fov + 5)
        elif key == ord('1'):
            show_miss = not show_miss
        elif key == ord('2'):
            latency = min(60, latency + 1)
        elif key == ord('3'):
            latency = max(0, latency - 1)
        elif key == 2424832:
            yaw -= ROTATE_STEP_DEG
        elif key == 2555904:
            yaw += ROTATE_STEP_DEG
        elif key == 2490368:
            pitch += ROTATE_STEP_DEG
        elif key == 2621440:
            pitch -= ROTATE_STEP_DEG

        try:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cap.release()
    cv2.destroyAllWindows()
    log_file.close()

    if n_samples > 0:
        print("\n" + "=" * 60)
        print("📊 ИТОГОВАЯ СТАТИСТИКА")
        print("=" * 60)
        print(f"Кадров: {n_samples}   Задержка: {latency} кадров (≈{1000*latency/src_fps:.0f} мс)")
        print(f"")
        print(f"{'Стратегия':<14} {'Трафик':>10} {'Miss':>10}")
        print(f"{'FULL-FRAME':<14} {'100.0%':>10} {'0.00%':>10}")
        print(f"{'NAIVE-FOV':<14} {100*sum_naive_tr/n_samples:>9.1f}% "
              f"{100*sum_naive_ms/n_samples:>9.2f}%")
        print(f"{'TILED-FOV':<14} {100*sum_tiled_tr/n_samples:>9.1f}% "
              f"{100*sum_tiled_ms/n_samples:>9.2f}%")
        print(f"\n📝 Лог: {log_path}")
    print("👋 Готово")


if __name__ == "__main__":
    main()