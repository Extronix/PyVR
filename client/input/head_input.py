"""
HeadInput — управление позой просмотра.

Два источника событий:
  1. cv2.setMouseCallback (события в окне OpenCV)  — wheel, клик внутри окна
  2. Win32 polling (GetCursorPos + GetAsyncKeyState) — drag даже когда
     курсор вне окна (Windows only).

Polling включается опционально. По умолчанию on для win32.
"""
from __future__ import annotations
import logging
import math
import sys
import time
from dataclasses import dataclass

import cv2

from common.config import CFG

logger = logging.getLogger(__name__)

# --- Win32 helpers (только под Windows) ---
_IS_WIN = sys.platform.startswith("win")
_win32 = None
if _IS_WIN:
    try:
        import ctypes
        from ctypes import wintypes

        _user32 = ctypes.windll.user32

        class _POINT(ctypes.Structure):
            _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

        _GetCursorPos = _user32.GetCursorPos
        _GetCursorPos.argtypes = [ctypes.POINTER(_POINT)]
        _GetCursorPos.restype = wintypes.BOOL

        _GetAsyncKeyState = _user32.GetAsyncKeyState
        _GetAsyncKeyState.argtypes = [ctypes.c_int]
        _GetAsyncKeyState.restype = ctypes.c_short

        _VK_LBUTTON = 0x01
        _POINT_INST = _POINT()

        def _cursor_pos() -> tuple[int, int]:
            _GetCursorPos(ctypes.byref(_POINT_INST))
            return _POINT_INST.x, _POINT_INST.y

        def _lmb_down() -> bool:
            # Старший бит = "сейчас нажата"
            return (_GetAsyncKeyState(_VK_LBUTTON) & 0x8000) != 0

        _win32 = True
        logger.info("HeadInput: Win32 mouse polling available")
    except Exception as e:
        logger.warning(f"HeadInput: Win32 init failed, polling disabled: {e}")
        _win32 = False


@dataclass
class HeadState:
    yaw_deg: float
    pitch_deg: float
    fov_deg: float
    yaw_vel_dps: float
    pitch_vel_dps: float


class HeadInput:
    """
    Поза просмотра (yaw, pitch, fov).
    Поддерживает мышь (drag + wheel) и клавиатуру (+/- для FOV, R для сброса).
    """

    def __init__(self,
                 yaw_init: float = 0.0,
                 pitch_init: float = 0.0,
                 fov_init: float | None = None,
                 use_win32_polling: bool | None = None):
        self.yaw_deg = float(yaw_init)
        self.pitch_deg = float(pitch_init)
        self.fov_deg = float(fov_init if fov_init is not None else CFG.view.fov_deg_default)

        # EMA-фильтр скорости
        self._ema_alpha = 0.25
        self.yaw_vel_dps = 0.0
        self.pitch_vel_dps = 0.0
        self._last_t = time.perf_counter()
        self._last_yaw = self.yaw_deg
        self._last_pitch = self.pitch_deg

        # OpenCV callback state
        self._dragging = False
        self._cv_last_x: int | None = None
        self._cv_last_y: int | None = None
        # Таймштамп последнего cv-движения — для эвристик «курсор покинул окно»
        self._cv_last_move_t = 0.0

        # Win32 polling state
        if use_win32_polling is None:
            self.use_polling = bool(_win32)
        else:
            self.use_polling = bool(use_win32_polling) and bool(_win32)
        self._poll_last_pos: tuple[int, int] | None = None
        self._poll_was_down = False

        # Чувствительности
        # Берём из CFG, если есть; иначе — дефолты (как в OpenCV-VR-просмотрщиках).
        cfg_input = getattr(CFG, "input", None)
        if cfg_input is not None:
            self.mouse_sens_deg_per_px = getattr(
                cfg_input, "mouse_sensitivity_deg_per_px", 0.15)
            self.wheel_sens_fov_per_notch = getattr(
                cfg_input, "wheel_sensitivity_fov_per_notch", 3.0)
        else:
            # Может быть, настройки лежат в CFG.view — пробуем оттуда
            self.mouse_sens_deg_per_px = getattr(
                CFG.view, "mouse_sensitivity_deg_per_px", 0.15)
            self.wheel_sens_fov_per_notch = getattr(
                CFG.view, "wheel_sensitivity_fov_per_notch", 3.0)

        v = CFG.view
        self.fov_min   = getattr(v, "fov_min",   getattr(v, "fov_deg_min",   30.0))
        self.fov_max   = getattr(v, "fov_max",   getattr(v, "fov_deg_max",   110.0))
        self.pitch_clip = getattr(v, "pitch_clip_deg",getattr(v, "pitch_limit_deg", 89.0))

        logger.info(
            f"HeadInput: sens={self.mouse_sens_deg_per_px}°/px, "
            f"wheel={self.wheel_sens_fov_per_notch}°/notch, "
            f"polling={self.use_polling}"
        )

    # ---------- OpenCV mouse callback ----------

    def on_mouse(self, event: int, x: int, y: int, flags: int, _userdata) -> None:
        """
        Используется ТОЛЬКО для wheel и для отслеживания клика, когда курсор в окне.
        Движение мыши при polling=True обрабатывается через polling.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._cv_last_x = x
            self._cv_last_y = y
            self._cv_last_move_t = time.perf_counter()
            return

        if event == cv2.EVENT_LBUTTONUP:
            self._dragging = False
            self._cv_last_x = None
            self._cv_last_y = None
            return

        if event == cv2.EVENT_MOUSEMOVE:
            self._cv_last_move_t = time.perf_counter()
            # Если polling включён — движение обрабатывает polling, тут только трекаем
            if self.use_polling:
                self._cv_last_x = x
                self._cv_last_y = y
                return
            # Иначе — классическая обработка drag через callback
            if self._dragging and self._cv_last_x is not None:
                # Сбрасываем dragging, если флаг ЛКМ ушёл (мышь могла отпуститься вне окна)
                if not (flags & cv2.EVENT_FLAG_LBUTTON):
                    self._dragging = False
                    self._cv_last_x = None
                    self._cv_last_y = None
                    return
                dx = x - self._cv_last_x
                dy = y - self._cv_last_y
                self._apply_delta(dx, dy)
            self._cv_last_x = x
            self._cv_last_y = y
            return

        if event == cv2.EVENT_MOUSEWHEEL:
            # flags > 0 → вверх (zoom in → FOV меньше), < 0 → вниз
            notch = 1 if flags > 0 else -1
            self._apply_wheel(notch)
            return

    # ---------- Win32 polling (вызывать раз в кадр) ----------

    def _poll_win32(self) -> None:
        if not self.use_polling:
            return
        pos = _cursor_pos()
        lmb = _lmb_down()

        if lmb and not self._poll_was_down:
            # Только что нажали — фиксируем начальную точку
            self._poll_last_pos = pos
            self._dragging = True
        elif lmb and self._poll_was_down and self._poll_last_pos is not None:
            dx = pos[0] - self._poll_last_pos[0]
            dy = pos[1] - self._poll_last_pos[1]
            if dx != 0 or dy != 0:
                self._apply_delta(dx, dy)
            self._poll_last_pos = pos
        elif not lmb and self._poll_was_down:
            # Кнопку отпустили (возможно вне окна) — финализируем
            self._dragging = False
            self._poll_last_pos = None

        self._poll_was_down = lmb

    # ---------- Apply movement ----------

    def _apply_delta(self, dx: int, dy: int) -> None:
        # Чем уже FOV — тем точнее реагирует на мышь
        fov_factor = self.fov_deg / CFG.view.fov_deg_default
        self.yaw_deg = (self.yaw_deg + dx * self.mouse_sens_deg_per_px * fov_factor) % 360.0
        self.pitch_deg = max(-self.pitch_clip,
                             min(self.pitch_clip,
                                 self.pitch_deg - dy * self.mouse_sens_deg_per_px * fov_factor))

    def _apply_wheel(self, notch: int) -> None:
        self.fov_deg = max(self.fov_min,
                           min(self.fov_max,
                               self.fov_deg - notch * self.wheel_sens_fov_per_notch))

    # ---------- Keyboard ----------

    def on_key(self, k: int) -> None:
        if k == ord('+') or k == ord('='):
            self._apply_wheel(+1)
        elif k == ord('-') or k == ord('_'):
            self._apply_wheel(-1)
        elif k == ord('r') or k == ord('R'):
            self.yaw_deg = 0.0
            self.pitch_deg = 0.0
            self.fov_deg = CFG.view.fov_deg_default
            logger.info("HeadInput: view reset")

    # ---------- Update (вызывать раз в кадр) ----------

    def update(self) -> HeadState:
        # 1. опрос Win32 (если включён)
        self._poll_win32()

        # 2. посчитать угловые скорости (для предсказания на сервере)
        now = time.perf_counter()
        dt = now - self._last_t
        if dt > 1e-4:
            dyaw = self._wrap_deg(self.yaw_deg - self._last_yaw)
            dpitch = self.pitch_deg - self._last_pitch
            inst_yaw_v = dyaw / dt
            inst_pitch_v = dpitch / dt
            a = self._ema_alpha
            self.yaw_vel_dps = a * inst_yaw_v + (1 - a) * self.yaw_vel_dps
            self.pitch_vel_dps = a * inst_pitch_v + (1 - a) * self.pitch_vel_dps
        self._last_t = now
        self._last_yaw = self.yaw_deg
        self._last_pitch = self.pitch_deg

        return HeadState(
            yaw_deg=self.yaw_deg,
            pitch_deg=self.pitch_deg,
            fov_deg=self.fov_deg,
            yaw_vel_dps=self.yaw_vel_dps,
            pitch_vel_dps=self.pitch_vel_dps,
        )

    @staticmethod
    def _wrap_deg(a: float) -> float:
        """Приводит градусную разницу в диапазон (-180, 180]."""
        a = (a + 180.0) % 360.0 - 180.0
        return a