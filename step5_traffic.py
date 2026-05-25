"""
Шаг 5: Измерение экономии трафика для 360° видео.

Сравнивает три стратегии доставки:
  1. FULL-FRAME — передаём весь equirectangular кадр (baseline = 100%)
  2. NAIVE-FOV  — передаём только то, что прямо сейчас в поле зрения (нижняя граница)
  3. TILED-FOV  — сетка 6x3 тайлов; передаём только тайлы, пересекающие FOV

Трафик измеряется в пикселях (приближение реальных байт после H.264).
Логирует всё в CSV для построения графиков.

Управление: стрелки/мышь = вращение, +/- = FOV, пробел = пауза, Enter = сброс, ESC = выход
"""

import cv2
import numpy as np
import os
import time
import csv
from datetime import datetime

# ============ НАСТРОЙКИ ============
VIDEO_PATH = "data/RoSh.mp4"
VIEW_W, VIEW_H = 1280, 720
INITIAL_FOV = 100.0
ROTATE_STEP_DEG = 5.0
MOUSE_SENSITIVITY = 0.2

# Тайловая сетка (по горизонтали x по вертикали)
TILE_COLS = 6      # 360° / 6 = 60° на тайл по долготе
TILE_ROWS = 3      # 180° / 3 = 60° на тайл по широте

# Лог
LOG_DIR = "logs"
LOG_EVERY_N_FRAMES = 1
# ===================================


# ---------------- Математика ----------------
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


def equirect_to_perspective(equirect_img, fov_deg, R, out_w, out_h):
    H, W = equirect_img.shape[:2]
    fov = np.deg2rad(fov_deg)
    f = (out_w / 2) / np.tan(fov / 2)
    u, v = np.meshgrid(np.arange(out_w), np.arange(out_h))
    x = (u - out_w / 2).astype(np.float32)
    y = (v - out_h / 2).astype(np.float32)
    z = np.full_like(x, f)
    vec = np.stack([x, y, z], axis=-1) @ R.T
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    vec_n = vec / norm
    lon = np.arctan2(vec_n[..., 0], vec_n[..., 2])
    lat = np.arcsin(np.clip(vec_n[..., 1], -1, 1))
    map_x = ((lon / (2 * np.pi) + 0.5) * W).astype(np.float32)
    map_y = ((lat / np.pi + 0.5) * H).astype(np.float32)
    return cv2.remap(equirect_img, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)


# ---------------- Расчёт трафика ----------------
def compute_fov_mask_on_equirect(yaw_deg, pitch_deg, fov_deg, eq_w, eq_h,
                                  samples_w=160, samples_h=90):
    """
    Возвращает булеву маску [eq_h × eq_w]: True там, где equirect-пиксель
    виден в текущем FOV. Использует обратную проекцию через сэмплирование.
    """
    # Генерим направления взгляда внутри FOV (на view-плоскости)
    R = make_rotation(yaw_deg, pitch_deg)
    fov = np.deg2rad(fov_deg)
    aspect = VIEW_W / VIEW_H
    f = (samples_w / 2) / np.tan(fov / 2)
    u, v = np.meshgrid(np.arange(samples_w), np.arange(samples_h))
    x = (u - samples_w / 2).astype(np.float32)
    y = (v - samples_h / 2).astype(np.float32) * (samples_w / samples_h) / aspect
    z = np.full_like(x, f)
    vec = np.stack([x, y, z], axis=-1) @ R.T
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
    lon = np.arctan2(vec[..., 0], vec[..., 2])
    lat = np.arcsin(np.clip(vec[..., 1], -1, 1))
    px = ((lon / (2 * np.pi) + 0.5) * eq_w).astype(np.int32) % eq_w
    py = ((lat / np.pi + 0.5) * eq_h).astype(np.int32).clip(0, eq_h - 1)
    mask = np.zeros((eq_h, eq_w), dtype=bool)
    mask[py.flatten(), px.flatten()] = True
    return mask


def compute_visible_tiles(yaw_deg, pitch_deg, fov_deg, eq_w, eq_h,
                          tile_cols, tile_rows):
    """
    Возвращает булев массив [tile_rows × tile_cols]: True если тайл пересекается с FOV.
    Метод: считаем FOV-маску на equirect, потом для каждого тайла проверяем,
    есть ли в его прямоугольнике хоть один True.
    """
    mask = compute_fov_mask_on_equirect(yaw_deg, pitch_deg, fov_deg, eq_w, eq_h)
    visible = np.zeros((tile_rows, tile_cols), dtype=bool)
    tile_w = eq_w // tile_cols
    tile_h = eq_h // tile_rows
    for r in range(tile_rows):
        for c in range(tile_cols):
            y0, y1 = r * tile_h, (r + 1) * tile_h
            x0, x1 = c * tile_w, (c + 1) * tile_w
            if mask[y0:y1, x0:x1].any():
                visible[r, c] = True
    return visible, mask


def compute_traffic_stats(yaw_deg, pitch_deg, fov_deg, eq_w, eq_h,
                          tile_cols, tile_rows):
    """
    Считает трафик в пикселях для трёх стратегий.
    """
    full_px = eq_w * eq_h
    
    # NAIVE-FOV: размер выходного view-окна (минимум того, что нужно нарисовать)
    naive_px = VIEW_W * VIEW_H
    
    # TILED-FOV: сумма площадей видимых тайлов
    visible_tiles, fov_mask = compute_visible_tiles(
        yaw_deg, pitch_deg, fov_deg, eq_w, eq_h, tile_cols, tile_rows
    )
    tile_w = eq_w // tile_cols
    tile_h = eq_h // tile_rows
    tile_area = tile_w * tile_h
    n_visible = int(visible_tiles.sum())
    tiled_px = n_visible * tile_area
    
    return {
        'full_px': full_px,
        'naive_px': naive_px,
        'tiled_px': tiled_px,
        'visible_tiles': visible_tiles,
        'n_visible': n_visible,
        'n_total_tiles': tile_cols * tile_rows,
    }


# ---------------- Визуализация ----------------
def draw_minimap(view, visible_tiles, yaw_deg, pitch_deg, fov_deg,
                  tile_cols, tile_rows):
    """
    Рисует мини-карту тайловой сетки в правом нижнем углу.
    Зелёные тайлы = передаются (в FOV). Красные = не передаются.
    Жёлтый маркер = центр взгляда.
    """
    mm_w = 360
    mm_h = 180
    margin = 20
    x0 = view.shape[1] - mm_w - margin
    y0 = view.shape[0] - mm_h - margin
    
    # Фон
    overlay = view.copy()
    cv2.rectangle(overlay, (x0 - 4, y0 - 4),
                  (x0 + mm_w + 4, y0 + mm_h + 4), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, view, 0.4, 0, view)
    
    tw = mm_w // tile_cols
    th = mm_h // tile_rows
    for r in range(tile_rows):
        for c in range(tile_cols):
            tx = x0 + c * tw
            ty = y0 + r * th
            color = (0, 200, 0) if visible_tiles[r, c] else (40, 40, 200)
            cv2.rectangle(view, (tx, ty), (tx + tw - 1, ty + th - 1),
                          color, -1)
            cv2.rectangle(view, (tx, ty), (tx + tw - 1, ty + th - 1),
                          (255, 255, 255), 1)
    
    # Маркер центра взгляда
    gx = x0 + int(((yaw_deg + 180) / 360.0) * mm_w)
    gy = y0 + int(((pitch_deg + 90) / 180.0) * mm_h)
    gy = mm_h - (gy - y0) + y0  # инверсия Y, чтобы верх был "небом"
    cv2.drawMarker(view, (gx, gy), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
    
    cv2.putText(view, "Tile map (green=sent, red=skipped)",
                (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ---------------- Мышь ----------------
class MouseState:
    def __init__(self):
        self.dragging = False
        self.last_x = 0
        self.last_y = 0
        self.dyaw = 0.0
        self.dpitch = 0.0

mouse = MouseState()

def mouse_callback(event, x, y, flags, _param):
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse.dragging = True
        mouse.last_x = x
        mouse.last_y = y
    elif event == cv2.EVENT_LBUTTONUP:
        mouse.dragging = False
    elif event == cv2.EVENT_MOUSEMOVE and mouse.dragging:
        mouse.dyaw += (x - mouse.last_x) * MOUSE_SENSITIVITY
        mouse.dpitch -= (y - mouse.last_y) * MOUSE_SENSITIVITY
        mouse.last_x = x
        mouse.last_y = y


# ---------------- Главное ----------------
def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Не найден: {VIDEO_PATH}")
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print(f"📹 Открываем {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    eq_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    eq_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   {total_frames} кадров @ {src_fps:.1f} fps, размер {eq_w}×{eq_h}")
    print(f"   Тайловая сетка: {TILE_COLS}×{TILE_ROWS} = {TILE_COLS*TILE_ROWS} тайлов")
    print(f"   Размер одного тайла: {eq_w//TILE_COLS}×{eq_h//TILE_ROWS} px")
    
    # CSV-лог
    log_path = os.path.join(LOG_DIR, f"traffic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "t_sec", "frame", "yaw_deg", "pitch_deg", "fov_deg",
        "full_px", "naive_px", "tiled_px",
        "naive_ratio", "tiled_ratio",
        "n_visible_tiles", "n_total_tiles",
    ])
    print(f"📝 Лог трафика: {log_path}\n")
    
    window = "Traffic measurement (step 5)"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window, np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8))
    cv2.waitKey(1)
    cv2.setMouseCallback(window, mouse_callback)
    
    yaw, pitch = 0.0, 0.0
    fov = INITIAL_FOV
    paused = False
    frame_idx = 0
    current_frame = None
    t_start = time.time()
    last_time = time.time()
    fps_smooth = 0.0
    
    # Накопленная статистика
    sum_full = 0
    sum_naive = 0
    sum_tiled = 0
    n_samples = 0
    
    print("🎮 Стрелки/мышь = вращение | +/- FOV | пробел пауза | Enter сброс | ESC выход\n")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue
            current_frame = frame
            frame_idx += 1
        
        if current_frame is None:
            cv2.waitKey(10)
            continue
        
        # Применяем движение мыши
        yaw += mouse.dyaw
        pitch += mouse.dpitch
        mouse.dyaw = 0.0
        mouse.dpitch = 0.0
        pitch = max(-89.0, min(89.0, pitch))
        yaw = ((yaw + 180) % 360) - 180
        
        # Рендер view
        R_view = make_rotation(yaw, pitch)
        view = equirect_to_perspective(current_frame, fov, R_view, VIEW_W, VIEW_H)
        
        # Считаем трафик
        stats = compute_traffic_stats(yaw, pitch, fov, eq_w, eq_h,
                                       TILE_COLS, TILE_ROWS)
        
        # Накапливаем статистику
        sum_full += stats['full_px']
        sum_naive += stats['naive_px']
        sum_tiled += stats['tiled_px']
        n_samples += 1
        
        # FPS
        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            inst = 1.0 / dt
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst if fps_smooth > 0 else inst
        
        # Лог
        if frame_idx % LOG_EVERY_N_FRAMES == 0:
            naive_r = stats['naive_px'] / stats['full_px']
            tiled_r = stats['tiled_px'] / stats['full_px']
            log_writer.writerow([
                f"{now - t_start:.3f}", frame_idx,
                f"{yaw:.2f}", f"{pitch:.2f}", f"{fov:.1f}",
                stats['full_px'], stats['naive_px'], stats['tiled_px'],
                f"{naive_r:.4f}", f"{tiled_r:.4f}",
                stats['n_visible'], stats['n_total_tiles'],
            ])
        
        # Мини-карта тайлов
        draw_minimap(view, stats['visible_tiles'], yaw, pitch, fov,
                      TILE_COLS, TILE_ROWS)
        
        # HUD сверху
        full_mpx = stats['full_px'] / 1e6
        naive_mpx = stats['naive_px'] / 1e6
        tiled_mpx = stats['tiled_px'] / 1e6
        naive_pct = 100 * stats['naive_px'] / stats['full_px']
        tiled_pct = 100 * stats['tiled_px'] / stats['full_px']
        
        avg_naive_pct = 100 * sum_naive / sum_full if sum_full > 0 else 0
        avg_tiled_pct = 100 * sum_tiled / sum_full if sum_full > 0 else 0
        
        hud = [
            f"yaw={yaw:+6.1f}  pitch={pitch:+5.1f}  FOV={fov:.0f}   frame {frame_idx}/{total_frames}   {fps_smooth:.1f} fps",
            "",
            f"FULL-FRAME : {full_mpx:6.2f} MPx  (100.0%)              <- baseline",
            f"NAIVE-FOV  : {naive_mpx:6.2f} MPx  ({naive_pct:5.1f}%)   saving {100-naive_pct:5.1f}%",
            f"TILED-FOV  : {tiled_mpx:6.2f} MPx  ({tiled_pct:5.1f}%)   saving {100-tiled_pct:5.1f}%   tiles {stats['n_visible']}/{stats['n_total_tiles']}",
            "",
            f"AVG so far  NAIVE={avg_naive_pct:5.1f}%   TILED={avg_tiled_pct:5.1f}%   samples={n_samples}",
        ]
        for i, line in enumerate(hud):
            cv2.putText(view, line, (12, 24 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(view, line, (12, 24 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.drawMarker(view, (VIEW_W // 2, VIEW_H // 2),
                       (255, 255, 255), cv2.MARKER_CROSS, 18, 1)
        
        if paused:
            cv2.putText(view, "PAUSED", (VIEW_W // 2 - 70, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        cv2.imshow(window, view)
        
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
    
    # Итоговая сводка в консоль
    if n_samples > 0:
        print("\n" + "=" * 60)
        print("📊 ИТОГОВАЯ СТАТИСТИКА")
        print("=" * 60)
        print(f"Кадров проанализировано: {n_samples}")
        print(f"FULL-FRAME : 100.0%  (baseline)")
        print(f"NAIVE-FOV  : {100*sum_naive/sum_full:5.1f}%   "
              f"saving {100-100*sum_naive/sum_full:5.1f}%")
        print(f"TILED-FOV  : {100*sum_tiled/sum_full:5.1f}%   "
              f"saving {100-100*sum_tiled/sum_full:5.1f}%")
        print(f"\n📝 Лог: {log_path}")
    print("👋 Готово")


if __name__ == "__main__":
    main()