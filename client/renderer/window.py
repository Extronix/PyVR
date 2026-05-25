"""
OpenCV-окно клиента. Захватывает ввод, отрисовывает viewport,
рисует HUD с состоянием.
"""
import cv2
import numpy as np

from client.sensors.mock_imu import MockIMU


WINDOW_NAME = "PyVR Client"


class ClientWindow:
    """Менеджмент окна + ввода. Передаёт ввод в MockIMU."""
    
    def __init__(self, imu: MockIMU, view_w: int, view_h: int):
        self.imu = imu
        self.view_w = view_w
        self.view_h = view_h
        
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, view_w, view_h)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        
        # Состояние мыши
        self._dragging = False
        self._last_mx = 0
        self._last_my = 0
    
    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._last_mx = x
            self._last_my = y
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            dx = x - self._last_mx
            dy = y - self._last_my
            self._last_mx = x
            self._last_my = y
            self.imu.apply_mouse_delta(dx, dy)
        elif event == cv2.EVENT_MOUSEWHEEL:
            # На Windows flags содержит знак прокрутки в старших битах
            delta = 1 if flags > 0 else -1
            self.imu.apply_wheel(delta)
    
    def show(self, frame: np.ndarray, hud_lines: list[str]) -> int:
        """Рисует frame + HUD, возвращает код нажатой клавиши."""
        # HUD: чёрная плашка слева сверху
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (430, 20 + 22 * len(hud_lines)),
                      (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
        
        for i, line in enumerate(hud_lines):
            cv2.putText(frame, line, (10, 28 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Передаём клавиши в IMU (WASD)
        if key != 255:  # 255 = нет нажатия
            self.imu.apply_keyboard(key)
        
        return key
    
    def is_window_closed(self) -> bool:
        try:
            return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True
    
    def close(self) -> None:
        cv2.destroyAllWindows()