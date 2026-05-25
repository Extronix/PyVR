"""
Имитация IMU-датчиков VR-гарнитуры через мышь и клавиатуру.
Отслеживает yaw, pitch, FOV и вычисляет угловые скорости.
"""
import time
import numpy as np

from common.config import CFG


class MockIMU:
    """
    Псевдо-IMU: накапливает yaw/pitch от пользовательского ввода,
    считает угловые скорости (EMA-сглаженные).
    """
    
    def __init__(self):
        # Текущая поза
        self.yaw_deg: float = 0.0
        self.pitch_deg: float = 0.0
        self.fov_deg: float = CFG.video.fov_deg_default
        
        # Угловые скорости (deg/sec), EMA-сглаженные
        self.yaw_vel_dps: float = 0.0
        self.pitch_vel_dps: float = 0.0
        
        # Для расчёта скорости
        self._last_yaw = 0.0
        self._last_pitch = 0.0
        self._last_time = time.perf_counter()
        
        self._alpha = CFG.input.velocity_ema_alpha
    
    # --- Применение ввода ---
    
    def apply_mouse_delta(self, dx: int, dy: int) -> None:
        """Сдвиг мыши → изменение yaw/pitch."""
        sens = CFG.input.mouse_sensitivity
        self.yaw_deg += dx * sens
        self.pitch_deg -= dy * sens  # мышь вниз = смотрим вниз
        
        # Нормализация yaw в [-180, 180]
        self.yaw_deg = ((self.yaw_deg + 180.0) % 360.0) - 180.0
        
        # Ограничение pitch
        self.pitch_deg = float(np.clip(
            self.pitch_deg,
            CFG.video.pitch_deg_min,
            CFG.video.pitch_deg_max,
        ))
    
    def apply_keyboard(self, key: int) -> None:
        """
        Клавиши стрелок: yaw/pitch step.
        cv2.waitKey коды: 0/1/2/3 = up/down/left/right в некоторых билдах,
        но надёжнее ловить через arrow scan codes (Windows: 2424832/...).
        Поэтому используем wasd как дублёр.
        """
        step = CFG.input.keyboard_step_deg
        
        # ASCII WASD
        if key in (ord('a'), ord('A')):
            self.yaw_deg -= step
        elif key in (ord('d'), ord('D')):
            self.yaw_deg += step
        elif key in (ord('w'), ord('W')):
            self.pitch_deg += step
        elif key in (ord('s'), ord('S')):
            self.pitch_deg -= step
        else:
            return
        
        self.yaw_deg = ((self.yaw_deg + 180.0) % 360.0) - 180.0
        self.pitch_deg = float(np.clip(
            self.pitch_deg,
            CFG.video.pitch_deg_min,
            CFG.video.pitch_deg_max,
        ))
    
    def apply_wheel(self, delta: int) -> None:
        """Колесо мыши → FOV (zoom)."""
        step = CFG.input.fov_wheel_step
        # delta > 0 → zoom in (FOV меньше)
        self.fov_deg -= np.sign(delta) * step
        self.fov_deg = float(np.clip(
            self.fov_deg,
            CFG.video.fov_deg_min,
            CFG.video.fov_deg_max,
        ))
    
    # --- Обновление скоростей ---
    
    def tick(self) -> None:
        """
        Вызывать каждый кадр. Пересчитывает yaw_vel / pitch_vel.
        """
        now = time.perf_counter()
        dt = now - self._last_time
        if dt < 1e-6:
            return
        
        # Дельты с учётом wrap-around yaw
        d_yaw = self.yaw_deg - self._last_yaw
        if d_yaw > 180:
            d_yaw -= 360
        elif d_yaw < -180:
            d_yaw += 360
        d_pitch = self.pitch_deg - self._last_pitch
        
        # Мгновенная скорость
        inst_yaw_vel = d_yaw / dt
        inst_pitch_vel = d_pitch / dt
        
        # EMA-сглаживание
        a = self._alpha
        self.yaw_vel_dps = a * inst_yaw_vel + (1 - a) * self.yaw_vel_dps
        self.pitch_vel_dps = a * inst_pitch_vel + (1 - a) * self.pitch_vel_dps
        
        # Запоминаем для следующего тика
        self._last_yaw = self.yaw_deg
        self._last_pitch = self.pitch_deg
        self._last_time = now
    
    def snapshot(self) -> dict:
        """Текущее состояние для логов/отладки."""
        return {
            "yaw": self.yaw_deg,
            "pitch": self.pitch_deg,
            "fov": self.fov_deg,
            "yaw_vel": self.yaw_vel_dps,
            "pitch_vel": self.pitch_vel_dps,
        }