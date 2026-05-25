"""
step7_buffer.py — VR-стриминг, финальная оптимизация.
cap.read неблокирующий + numba для metrics + кэш дилатаций.
"""

import os
import csv
import math
import time
import threading
from collections import deque
from queue import Queue, Empty
from datetime import datetime
from multiprocessing import cpu_count

import numpy as np
import cv2
os.environ["NUMBA_NUM_THREADS"] = str(min(8, os.cpu_count() or 4))
os.environ["NUMBA_CACHE_DIR"]   = ".numba_cache"
try:
    from numba import njit, prange
    NUMBA_OK = True
except ImportError:
    NUMBA_OK = False
    print("⚠️  numba не найдена, pip install numba")

# =========================================================================
# КОНФИГ
# =========================================================================
VIDEO_PATH = "data/RoSh.mp4"

SRC_W, SRC_H = 5760, 2880
VIEW_W, VIEW_H = 1280, 720

TILE_COLS, TILE_ROWS = 8, 4
N_TILES = TILE_COLS * TILE_ROWS

N_STRATEGIES = 5
STRATEGY_NAMES = [
    "TILED-tight",
    "TILED-1ring",
    "TILED-2ring",
    "TILED-predict",
    "TILED-pred+1r",
]
STRATEGY_KEYS = ["tight", "ring1", "ring2", "predict", "pred1r"]

INITIAL_LATENCY_FRAMES = 2
MAX_LATENCY_FRAMES     = 60

INITIAL_YAW   = 0.0
INITIAL_PITCH = 0.0
INITIAL_FOV   = 90.0
MIN_FOV, MAX_FOV = 50.0, 110.0

KEY_STEP_DEG = 2.5
MOUSE_SENS   = 0.18

COLOR_BUFFER = (255, 170, 50)
COLOR_MISS   = (0, 0, 255)

LOG_DIR  = "logs"
MINI_W, MINI_H = 160, 80
PROF_WINDOW    = 60


# =========================================================================
# NUMBA — build_view_map
# =========================================================================
if NUMBA_OK:
    @njit(parallel=True, cache=True, fastmath=True)
    def _build_map_numba(yaw_deg, pitch_deg, fov_deg,
                         view_w, view_h, src_w, src_h,
                         tile_cols, tile_rows):
        yaw   = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        f     = 0.5 * view_w / math.tan(math.radians(fov_deg) * 0.5)

        map_x       = np.empty((view_h, view_w), dtype=np.float32)
        map_y       = np.empty((view_h, view_w), dtype=np.float32)
        tile_id_map = np.empty((view_h, view_w), dtype=np.int32)

        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw),   math.sin(yaw)
        tile_w = src_w / tile_cols
        tile_h = src_h / tile_rows

        for i in prange(view_h):
            for j in range(view_w):
                xx = j - view_w * 0.5
                yy = i - view_h * 0.5
                zz = f
                y1 =  cp * yy - sp * zz
                z1 =  sp * yy + cp * zz
                x2 =  cy * xx + sy * z1
                z2 = -sy * xx + cy * z1
                y2 = y1
                norm = math.sqrt(x2*x2 + y2*y2 + z2*z2)
                lon  = math.atan2(x2, z2)
                lat  = math.asin(min(1.0, max(-1.0, y2 / norm)))
                mx = (lon / (2.0 * math.pi) + 0.5) * src_w
                my = (lat / math.pi         + 0.5) * src_h
                map_x[i, j] = mx
                map_y[i, j] = my
                tx = int(mx / tile_w)
                ty = int(my / tile_h)
                if tx < 0:          tx = 0
                if tx >= tile_cols: tx = tile_cols - 1
                if ty < 0:          ty = 0
                if ty >= tile_rows: ty = tile_rows - 1
                tile_id_map[i, j] = ty * tile_cols + tx
        return map_x, map_y, tile_id_map

    def build_view_map(yaw, pitch, fov,
                       vw=VIEW_W, vh=VIEW_H,
                       sw=SRC_W,  sh=SRC_H,
                       tc=TILE_COLS, tr=TILE_ROWS):
        return _build_map_numba(yaw, pitch, fov,
                                vw, vh, sw, sh, tc, tr)

    # ------------------------------------------------------------------
    # NUMBA — visible_tiles + metrics за ОДИН проход
    # ------------------------------------------------------------------
    @njit(parallel=True, cache=True, fastmath=True)
    def _visible_and_metrics_numba(flat_tile_ids,
                                   grids_flat,   # (N_STRAT, N_TILES) bool
                                   n_tiles, n_strat, n_pixels):
        """
        Возвращает:
          visible[n_tiles]  bool  — тайлы видимые клиенту
          traffic[n_strat]  float
          miss[n_strat]     float
        """
        visible  = np.zeros(n_tiles, dtype=np.bool_)
        miss_cnt = np.zeros(n_strat, dtype=np.int64)

        for p in prange(n_pixels):
            tid = flat_tile_ids[p]
            visible[tid] = True
            for s in range(n_strat):
                if not grids_flat[s, tid]:
                    miss_cnt[s] += 1

        traffic = np.empty(n_strat, dtype=np.float64)
        miss    = np.empty(n_strat, dtype=np.float64)
        for s in range(n_strat):
            cnt = 0
            for t in range(n_tiles):
                if grids_flat[s, t]:
                    cnt += 1
            traffic[s] = cnt / n_tiles
            miss[s]    = miss_cnt[s] / n_pixels

        return visible, traffic, miss

else:
    def build_view_map(yaw_deg, pitch_deg, fov_deg,
                       vw=VIEW_W, vh=VIEW_H,
                       sw=SRC_W,  sh=SRC_H,
                       tc=TILE_COLS, tr=TILE_ROWS):
        yaw   = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        f = 0.5 * vw / math.tan(math.radians(fov_deg) * 0.5)
        jj = np.arange(vw, dtype=np.float32) - vw * 0.5
        ii = np.arange(vh, dtype=np.float32) - vh * 0.5
        xx, yy = np.meshgrid(jj, ii)
        zz = np.full_like(xx, f)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw),   math.sin(yaw)
        y1 =  cp * yy - sp * zz
        z1 =  sp * yy + cp * zz
        y2 = y1
        x2 =  cy * xx + sy * z1
        z2 = -sy * xx + cy * z1
        norm = np.sqrt(x2*x2 + y2*y2 + z2*z2)
        lon  = np.arctan2(x2, z2)
        lat  = np.arcsin(np.clip(y2 / norm, -1.0, 1.0))
        map_x = ((lon / (2.0*math.pi) + 0.5) * sw).astype(np.float32)
        map_y = ((lat / math.pi       + 0.5) * sh).astype(np.float32)
        tile_w = sw / tc;  tile_h = sh / tr
        tx = np.clip((map_x / tile_w).astype(np.int32), 0, tc - 1)
        ty = np.clip((map_y / tile_h).astype(np.int32), 0, tr - 1)
        return map_x, map_y, (ty * tc + tx).astype(np.int32)


# =========================================================================
# ЕДИНАЯ ФУНКЦИЯ: visible + все метрики
# =========================================================================
def compute_visible_and_metrics(tile_id_map, grids):
    """
    grids: list of (TILE_ROWS, TILE_COLS) bool arrays
    Возвращает:
        visible_mask  (TILE_ROWS, TILE_COLS) bool
        tr            list[float]  трафик
        ms            list[float]  miss rate
    """
    flat      = tile_id_map.ravel()
    n_pix     = flat.size
    n_strat   = len(grids)

    if NUMBA_OK:
        grids_flat = np.array(
            [g.ravel() for g in grids], dtype=np.bool_)
        vis_flat, tr_arr, ms_arr = _visible_and_metrics_numba(
            flat, grids_flat,
            np.int64(N_TILES),
            np.int64(n_strat),
            np.int64(n_pix))
        visible = vis_flat.reshape(TILE_ROWS, TILE_COLS)
        return visible, list(tr_arr), list(ms_arr)
    else:
        # numpy fallback
        visible = np.zeros(N_TILES, dtype=bool)
        np.put(visible, flat, True)
        tr, ms = [], []
        for g in grids:
            sent    = g.ravel()[flat]
            tr.append(float(g.sum()) / N_TILES)
            ms.append(float((~sent).sum()) / n_pix)
        return visible.reshape(TILE_ROWS, TILE_COLS), tr, ms


# =========================================================================
# КЭШ КАРТ
# =========================================================================
class MapCache:
    def __init__(self, size=32):
        self._cache = {}
        self._order = deque()
        self._size  = size
        self.hits   = 0
        self.misses = 0

    def get(self, yaw, pitch, fov):
        key = (round(yaw, 1), round(pitch, 1), round(fov, 1))
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        val = build_view_map(yaw, pitch, fov)
        self._cache[key] = val
        self._order.append(key)
        if len(self._order) > self._size:
            self._cache.pop(self._order.popleft(), None)
        return val

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# =========================================================================
# КЭШ ДИЛАТАЦИЙ
# =========================================================================
class DilateCache:
    """Кэшируем dilate_ring — одинаковые маски не дилатируем повторно."""
    def __init__(self, size=64):
        self._cache = {}
        self._order = deque()
        self._size  = size

    def get(self, mask, rings):
        if rings <= 0:
            return mask.copy()
        key = (mask.tobytes(), rings)
        if key in self._cache:
            return self._cache[key]
        result = _dilate_ring_impl(mask, rings)
        self._cache[key] = result
        self._order.append(key)
        if len(self._order) > self._size:
            self._cache.pop(self._order.popleft(), None)
        return result


def _dilate_ring_impl(mask, rings):
    g = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(rings):
        padded = np.hstack([g[:, -1:], g, g[:, :1]])
        g = cv2.dilate(padded, kernel, iterations=1)[:, 1:-1]
    return g.astype(bool)


# =========================================================================
# ДЕКОДЕР — неблокирующий
# =========================================================================
class ThreadedCapture:
    """
    Декодирует в фоне.
    read()         — блокирующий (старое поведение)
    read_latest()  — берёт самый свежий кадр, не ждёт
    """
    def __init__(self, path, dst_size, queue_size=16):
        self.path  = path
        self.dst_w, self.dst_h = dst_size
        self._stop  = False
        self.q      = Queue(maxsize=queue_size)
        self._last  = None          # последний декодированный кадр
        self._lock  = threading.Lock()

        _c = cv2.VideoCapture(path)
        if not _c.isOpened():
            raise RuntimeError(f"Не открылся: {path}")
        self.fps   = _c.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(_c.get(cv2.CAP_PROP_FRAME_COUNT))
        self.src_w = int(_c.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.src_h = int(_c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _c.release()

        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop:
            cap = None
            try:
                cap = cv2.VideoCapture(self.path)
                if not cap.isOpened():
                    time.sleep(0.5)
                    continue
                while not self._stop:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if (frame.shape[1] != self.dst_w or
                            frame.shape[0] != self.dst_h):
                        frame = cv2.resize(
                            frame, (self.dst_w, self.dst_h),
                            interpolation=cv2.INTER_AREA)
                    # обновляем «последний кадр» всегда
                    with self._lock:
                        self._last = frame
                    # в очередь — только если есть место
                    try:
                        self.q.put_nowait(frame)
                    except Exception:
                        pass   # очередь полна — не страшно
            except Exception as e:
                print(f"[декодер] {e}")
            finally:
                if cap is not None:
                    cap.release()

    def read(self):
        """Блокирующий — ждём следующий кадр."""
        return True, self.q.get(timeout=10.0)

    def read_latest(self):
        """
        Неблокирующий — берём самый свежий кадр из очереди.
        Если очередь пуста — возвращаем последний известный.
        """
        frame = None
        # вычищаем очередь, берём самый свежий
        while True:
            try:
                frame = self.q.get_nowait()
            except Empty:
                break
        if frame is None:
            with self._lock:
                frame = self._last
        return (frame is not None), frame

    def release(self):
        self._stop = True


# =========================================================================
# ВИЗУАЛИЗАЦИЯ
# =========================================================================
def visualize_strategy(view_bgr, tile_id_map, visible_mask, sent_mask):
    out = view_bgr.copy()

    flat       = tile_id_map.ravel()
    sent_flat  = sent_mask.ravel()
    vis_flat   = visible_mask.ravel()

    pixel_sent = sent_flat[flat]
    pixel_vis  = vis_flat[flat]

    pix = out.reshape(-1, 3)

    # miss — красный
    miss_mask = ~pixel_sent
    if miss_mask.any():
        pix[miss_mask] = COLOR_MISS

    # буфер — оранжевый полупрозрачный (только sent, но не visible)
    buf_mask = pixel_sent & ~pixel_vis
    if buf_mask.any():
        pix[buf_mask] = (pix[buf_mask].astype(np.int32) // 2 +
                         np.array(COLOR_BUFFER, dtype=np.int32) // 2
                        ).astype(np.uint8)

    return out


def draw_minimap(sent_mask, visible_mask, w=MINI_W, h=MINI_H):
    cell_w = w // TILE_COLS
    cell_h = h // TILE_ROWS
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(TILE_ROWS):
        for col in range(TILE_COLS):
            x0, y0 = col * cell_w, row * cell_h
            x1, y1 = x0 + cell_w - 1, y0 + cell_h - 1
            if visible_mask[row, col]:
                color = (60, 220, 60)
            elif sent_mask[row, col]:
                color = (255, 170, 50)
            else:
                color = (40, 40, 40)
            cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (80, 80, 80), 1)
    cv2.putText(img, "tilemap", (2, h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
    return img


def paste_minimap(canvas, minimap):
    mh, mw = minimap.shape[:2]
    ch, cw = canvas.shape[:2]
    y0, x0 = ch - mh - 8, cw - mw - 8
    if y0 < 0 or x0 < 0:
        return
    roi = canvas[y0:y0+mh, x0:x0+mw]
    canvas[y0:y0+mh, x0:x0+mw] = cv2.addWeighted(
        minimap, 0.85, roi, 0.15, 0)


_HUD_FONT     = cv2.FONT_HERSHEY_SIMPLEX
_HUD_SCALE    = 0.42
_HUD_THICK    = 1
_HUD_LINE_H   = 16
_HUD_FG       = (220, 220, 220)
_HUD_BG_ALPHA = 0.35


def draw_hud(canvas, lines, x=10, y0=18):
    if not lines:
        return
    n  = len(lines)
    h  = _HUD_LINE_H * n + 8
    y1 = max(0, y0 - _HUD_LINE_H)
    y2 = min(canvas.shape[0], y1 + h)
    sub = canvas[y1:y2, x:x+500]
    if sub.size > 0:
        np.multiply(sub, _HUD_BG_ALPHA, out=sub, casting="unsafe")
        canvas[y1:y2, x:x+500] = sub
    for k, s in enumerate(lines):
        cv2.putText(canvas, s, (x, y0 + k * _HUD_LINE_H),
                    _HUD_FONT, _HUD_SCALE, _HUD_FG,
                    _HUD_THICK, cv2.LINE_AA)


# =========================================================================
# МЫШЬ
# =========================================================================
class MouseState:
    def __init__(self):
        self.dx = self.dy = 0.0
        self.last_x = self.last_y = None
        self.dragging = False


def mouse_cb(event, x, y, flags, st):
    if event == cv2.EVENT_LBUTTONDOWN:
        st.dragging = True
        st.last_x, st.last_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        st.dragging = False
    elif event == cv2.EVENT_MOUSEMOVE and st.dragging:
        if st.last_x is not None:
            st.dx += (x - st.last_x)
            st.dy -= (y - st.last_y)
        st.last_x, st.last_y = x, y


# =========================================================================
# PER-LOOP ПРОФАЙЛЕР
# =========================================================================
class LoopProfiler:
    def __init__(self, window=PROF_WINDOW, print_every=120):
        self.window      = window
        self.print_every = print_every
        self._keys  = []
        self._bufs  = {}
        self._frame = 0
        self._t     = None

    def sections(self, *keys):
        self._keys = list(keys)
        self._bufs = {k: deque(maxlen=self.window) for k in keys}

    def tick(self):
        self._t = time.perf_counter()

    def mark(self, key):
        now = time.perf_counter()
        if self._t is not None and key in self._bufs:
            self._bufs[key].append((now - self._t) * 1000.0)
        self._t = now

    def end_frame(self):
        self._frame += 1
        if self._frame % self.print_every == 0:
            self._print()

    def _print(self):
        print(f"\n⏱  Per-loop профайлер "
              f"(кадр {self._frame}, avg {self.window}):")
        rows  = []
        total = 0.0
        for k in self._keys:
            buf = self._bufs[k]
            if not buf:
                continue
            avg = sum(buf) / len(buf)
            total += avg
            rows.append((k, avg))
        for k, avg in rows:
            bar = "█" * max(1, int(avg / max(total, 1e-9) * 24))
            print(f"  {k:<26} {avg:6.2f} ms  {bar}")
        print(f"  {'ИТОГО':<26} {total:6.2f} ms  → "
              f"{1000/max(total,1e-9):.0f} fps теор.")


# =========================================================================
# ПРОФИЛИРОВЩИК ЗАПУСКА
# =========================================================================
def run_profiler(pano):
    print("\n⏱  Профилировка (30 итераций):")
    steps = []

    build_view_map(0.0, 0.0, 90.0)
    t0 = time.perf_counter()
    for _ in range(30):
        r = build_view_map(0.0, 0.0, 90.0)
    steps.append(("build_view_map",
                  (time.perf_counter() - t0) / 30 * 1000))

    map_x, map_y, tim = r

    t0 = time.perf_counter()
    for _ in range(30):
        cv2.remap(pano, map_x, map_y,
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    steps.append(("remap LINEAR",
                  (time.perf_counter() - t0) / 30 * 1000))

    vis = np.zeros(N_TILES, dtype=bool)
    np.put(vis, tim.ravel(), True)
    vis = vis.reshape(TILE_ROWS, TILE_COLS)
    grids = [vis,
             _dilate_ring_impl(vis, 1),
             _dilate_ring_impl(vis, 2),
             vis.copy(),
             _dilate_ring_impl(vis, 1)]

    # прогрев numba metrics
    if NUMBA_OK:
        compute_visible_and_metrics(tim, grids)

    t0 = time.perf_counter()
    for _ in range(30):
        compute_visible_and_metrics(tim, grids)
    steps.append(("visible+metrics×5",
                  (time.perf_counter() - t0) / 30 * 1000))

    view = cv2.remap(pano, map_x, map_y,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    t0 = time.perf_counter()
    for _ in range(30):
        visualize_strategy(view, tim, vis, grids[1])
    steps.append(("visualize",
                  (time.perf_counter() - t0) / 30 * 1000))

    total = sum(v for _, v in steps)
    print(f"  {'Шаг':<26} {'мс/вызов':>10}")
    print(f"  {'-'*38}")
    for name, ms in steps:
        bar = "█" * max(1, int(ms / max(total, 1e-9) * 20))
        print(f"  {name:<26} {ms:8.2f} ms  {bar}")
    print(f"  {'-'*38}")
    print(f"  {'ИТОГО':26} {total:8.2f} ms → "
          f"теор. макс {1000/max(total,1e-9):.0f} fps\n")


# =========================================================================
# MAIN
# =========================================================================
def main():
    print(f"📹 {VIDEO_PATH}")
    print(f"   CPU потоков: {cpu_count()}")
    print(f"   Numba: {'✅' if NUMBA_OK else '❌'}")

    cap = ThreadedCapture(VIDEO_PATH, dst_size=(SRC_W, SRC_H))
    print(f"   {cap.src_w}×{cap.src_h} @ {cap.fps:.0f} fps, "
          f"{cap.total} кадров")
    print(f"   Рендер: {SRC_W}×{SRC_H} → вид {VIEW_W}×{VIEW_H}")
    print(f"   Тайлы: {TILE_COLS}×{TILE_ROWS} = {N_TILES}")

    print("\n⏳ Прогрев numba JIT...")
    _, pano_first = cap.read()
    build_view_map(0.0, 0.0, 90.0)
    # прогрев metrics numba
    if NUMBA_OK:
        _tim = build_view_map(0.0, 0.0, 90.0)[2]
        _vis = np.zeros(N_TILES, dtype=bool)
        np.put(_vis, _tim.ravel(), True)
        _vis = _vis.reshape(TILE_ROWS, TILE_COLS)
        compute_visible_and_metrics(
            _tim, [_vis] * N_STRATEGIES)
    print("✅ JIT готов")

    run_profiler(pano_first)

    os.makedirs(LOG_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"buffer_{ts}.csv")
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    writer   = csv.writer(log_file)
    header   = ["frame", "t_sec", "latency", "yaw", "pitch", "fov"]
    for k in STRATEGY_KEYS:
        header += [f"{k}_traffic", f"{k}_miss"]
    writer.writerow(header)

    print(f"📝 Лог: {log_path}")
    print("🎮 стрелки/мышь | +/- FOV | space | "
          "1-5 viz | 9/0 latency | p=профайлер | ESC\n")

    win = "step7 — VR streaming buffer"
    # стало (попытка OpenGL, при ошибке — fallback):
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_OPENGL)
    except cv2.error:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(win, VIEW_W, VIEW_H)
    mouse = MouseState()
    cv2.setMouseCallback(win, mouse_cb, mouse)

    map_cache    = MapCache(size=32)
    dilate_cache = DilateCache(size=64)

    lp = LoopProfiler(window=PROF_WINDOW, print_every=120)
    lp.sections("cap.read_latest", "cache.get×3",
                 "remap", "strategies+metrics",
                 "visualize", "minimap+hud+show")
    show_loop_prof = False

    yaw, pitch, fov = INITIAL_YAW, INITIAL_PITCH, INITIAL_FOV
    yaw_hist   = [yaw,   yaw]
    pitch_hist = [pitch, pitch]

    latency      = INITIAL_LATENCY_FRAMES
    pose_buffer  = []
    viz_strategy = 0
    paused       = False

    sums_tr  = [0.0] * N_STRATEGIES
    sums_ms  = [0.0] * N_STRATEGIES
    n_frames = 0

    t_prev   = time.time()
    fps_disp = 0.0
    t_start  = time.time()
    frame_idx = 0

    last_view        = None
    last_tile_id_map = None
    pano = pano_first

    while True:

        # ── 1. неблокирующий захват
                # ── 1. неблокирующий захват кадра ───────────────────────────────
        lp.tick()
        ok, new_pano = cap.read_latest()
        if ok and new_pano is not None and not paused:
            pano = new_pano
            frame_idx += 1
        elif pano is None:
            pano = pano_first
        lp.mark("cap.read_latest")

        # ── 2. клавиши ──────────────────────────────────────────────────
        key = cv2.waitKeyEx(1)
        if key != -1:
            if   key == 27:                                    break
            elif key == ord(' '):                              paused = not paused
            elif key == ord('p'):                              show_loop_prof = not show_loop_prof
            elif key in (2424832, ord('a'), ord('A')):         yaw   -= KEY_STEP_DEG
            elif key in (2555904, ord('d'), ord('D')):         yaw   += KEY_STEP_DEG
            elif key in (2490368, ord('w'), ord('W')):         pitch += KEY_STEP_DEG
            elif key in (2621440, ord('s'), ord('S')):         pitch -= KEY_STEP_DEG
            elif key in (ord('+'), ord('=')):                  fov    = max(MIN_FOV, fov - 2.0)
            elif key in (ord('-'), ord('_')):                  fov    = min(MAX_FOV, fov + 2.0)
            elif key == ord('1'):  viz_strategy = 0
            elif key == ord('2'):  viz_strategy = 1
            elif key == ord('3'):  viz_strategy = 2
            elif key == ord('4'):  viz_strategy = 3
            elif key == ord('5'):  viz_strategy = 4
            elif key == ord('9'):  latency = max(0, latency - 1)
            elif key == ord('0'):  latency = min(MAX_LATENCY_FRAMES, latency + 1)

        # мышь
        if mouse.dx or mouse.dy:
            yaw   += mouse.dx * MOUSE_SENS
            pitch += mouse.dy * MOUSE_SENS
            mouse.dx = mouse.dy = 0.0

        pitch = max(-89.0, min(89.0, pitch))
        if yaw >  180.0: yaw -= 360.0
        if yaw < -180.0: yaw += 360.0

        if not paused:
            yaw_hist.append(yaw)
            pitch_hist.append(pitch)
            if len(yaw_hist) > 10:
                yaw_hist.pop(0)
                pitch_hist.pop(0)

        # буфер поз
        pose_buffer.append((yaw, pitch, fov))
        if len(pose_buffer) > latency + 1:
            pose_buffer.pop(0)
        yaw_srv, pitch_srv, fov_srv = pose_buffer[0]

        # предсказание позы
        if len(yaw_hist) >= 2:
            dyaw   = max(-15.0, min(15.0,
                         yaw_hist[-1]   - yaw_hist[-2]))
            dpitch = max(-10.0, min(10.0,
                         pitch_hist[-1] - pitch_hist[-2]))
        else:
            dyaw = dpitch = 0.0
        yaw_pred   = yaw_hist[-1]  + dyaw   * latency
        pitch_pred = max(-89.0, min(89.0,
                         pitch_hist[-1] + dpitch * latency))

        # ── 3. карты проекций (с кэшем) ─────────────────────────────────
        lp.tick()
        map_x, map_y, tile_id_map     = map_cache.get(yaw,      pitch,      fov)
        _,     _,     tile_id_map_srv = map_cache.get(yaw_srv,  pitch_srv,  fov_srv)
        _,     _,     tile_id_map_prd = map_cache.get(yaw_pred, pitch_pred, fov_srv)
        lp.mark("cache.get×3")

        # ── 4. remap ────────────────────────────────────────────────────
        lp.tick()
        if not paused or last_view is None:
            view = cv2.remap(pano, map_x, map_y,
                             cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_WRAP)
            last_view        = view
            last_tile_id_map = tile_id_map
        else:
            view        = last_view
            tile_id_map = last_tile_id_map
        lp.mark("remap")

        # ── 5. стратегии + метрики одним проходом ───────────────────────
        lp.tick()

        # базовые маски (из серверной и предсказанной точек зрения)
        vis_srv_flat  = np.zeros(N_TILES, dtype=bool)
        vis_prd_flat  = np.zeros(N_TILES, dtype=bool)
        np.put(vis_srv_flat, tile_id_map_srv.ravel(), True)
        np.put(vis_prd_flat, tile_id_map_prd.ravel(), True)
        vis_srv = vis_srv_flat.reshape(TILE_ROWS, TILE_COLS)
        vis_prd = vis_prd_flat.reshape(TILE_ROWS, TILE_COLS)

        # дилатации с кэшем
        grids = [
            vis_srv,
            dilate_cache.get(vis_srv, rings=1),
            dilate_cache.get(vis_srv, rings=2),
            vis_prd,
            dilate_cache.get(vis_prd, rings=1),
        ]

        # единый проход: visible(клиент) + traffic + miss
        visible_now, tr, ms = compute_visible_and_metrics(
            tile_id_map, grids)

        n_frames += 1
        for i in range(N_STRATEGIES):
            sums_tr[i] += tr[i]
            sums_ms[i] += ms[i]
        avg_tr = [s / n_frames for s in sums_tr]
        avg_ms = [s / n_frames for s in sums_ms]
        lp.mark("strategies+metrics")

        # ── 6. визуализация + минимап + HUD + imshow ────────────────────
        lp.tick()
        viz  = visualize_strategy(view, tile_id_map,
                                  visible_now, grids[viz_strategy])
        mini = draw_minimap(grids[viz_strategy], visible_now)
        paste_minimap(viz, mini)

        # fps
        t_now    = time.time()
        dt       = max(t_now - t_prev, 1e-6)
        t_prev   = t_now
        fps_disp = 0.9 * fps_disp + 0.1 / dt

        def marker(i):
            return ">>" if i == viz_strategy else "  "

        hud = [
            f"yaw={yaw:+6.1f}  pitch={pitch:+5.1f}  "
            f"FOV={fov:4.1f}   "
            f"frame={frame_idx}/{cap.total}   "
            f"{fps_disp:5.1f} fps",

            f"latency={latency} fr "
            f"(~{latency*1000.0/cap.fps:.0f} ms)   "
            f"cache={map_cache.hit_rate*100:.0f}%hit"
            f"  dcache={dilate_cache._cache.__len__()}ent"
            f"{'   [PAUSED]' if paused else ''}",

            "",
            "  STRATEGY         TRAFFIC   MISS    "
            "TILES   AVG_TR  AVG_MS",
        ]
        for i in range(N_STRATEGIES):
            hud.append(
                f"{marker(i)}{i+1} {STRATEGY_NAMES[i]:<14}  "
                f"{100*tr[i]:5.1f}%  {100*ms[i]:5.2f}%   "
                f"{int(grids[i].sum()):2d}/{N_TILES:2d}    "
                f"{100*avg_tr[i]:5.1f}%  {100*avg_ms[i]:5.2f}%"
            )

        if show_loop_prof:
            hud += ["", "  [p] per-loop ON  (см. консоль)"]

        draw_hud(viz, hud)
        display = cv2.resize(viz, (VIEW_W, VIEW_H), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(win, display)
        lp.mark("minimap+hud+show")

        # ── per-loop профайлер ───────────────────────────────────────────
        if show_loop_prof:
            lp.end_frame()
        else:
            lp._frame += 1

        # ── лог ─────────────────────────────────────────────────────────
        if not paused:
            t_sec = time.time() - t_start
            row   = [frame_idx, f"{t_sec:.3f}", latency,
                     f"{yaw:.3f}", f"{pitch:.3f}", f"{fov:.2f}"]
            for i in range(N_STRATEGIES):
                row += [f"{tr[i]:.6f}", f"{ms[i]:.6f}"]
            writer.writerow(row)

    # ── завершение ──────────────────────────────────────────────────────
    cap.release()
    log_file.close()
    cv2.destroyAllWindows()

    print("\n" + "=" * 72)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 72)
    print(f"Кадров: {n_frames}   "
          f"Задержка: {latency} fr "
          f"(~{latency*1000.0/cap.fps:.0f} мс)")
    print(f"Кэш карт:     {map_cache.hits} hits / "
          f"{map_cache.misses} misses  "
          f"({map_cache.hit_rate*100:.1f}%)")
    print(f"Кэш дилатац.: {len(dilate_cache._cache)} записей")
    print()
    print(f"{'Стратегия':<20} {'Трафик ср.':>12}  {'Miss ср.':>10}")
    print("-" * 46)
    if n_frames > 0:
        for i in range(N_STRATEGIES):
            print(f"{STRATEGY_NAMES[i]:<20} "
                  f"{100*sums_tr[i]/n_frames:11.2f}%  "
                  f"{100*sums_ms[i]/n_frames:9.3f}%")

    print()
    lp._print()
    print(f"\n📝 {log_path}")
    print("👋 Готово")


# =========================================================================
if __name__ == "__main__":
    main()