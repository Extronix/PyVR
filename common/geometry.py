"""
Геометрия ERP-проекции и построение viewport-карты.
"""
import numpy as np


# ============================================================
# Координатные преобразования
# ============================================================

def pixel_to_angle(u: float, v: float, W: int, H: int) -> tuple[float, float]:
    """
    ERP pixel → (yaw, pitch) в градусах.
    
    u ∈ [0, W-1] → yaw ∈ [-180, 180]
    v ∈ [0, H-1] → pitch ∈ [90, -90] (сверху вниз)
    """
    yaw = (u / W) * 360.0 - 180.0
    pitch = 90.0 - (v / H) * 180.0
    return yaw, pitch


def angle_to_pixel(yaw: float, pitch: float, W: int, H: int) -> tuple[float, float]:
    """(yaw, pitch) в градусах → ERP pixel (u, v)."""
    u = (yaw + 180.0) / 360.0 * W
    v = (90.0 - pitch) / 180.0 * H
    return u, v


def angular_distance(yaw1: float, pitch1: float,
                     yaw2: float, pitch2: float) -> float:
    """
    Угловое расстояние между двумя точками на сфере (в градусах).
    Haversine formula.
    """
    lam1, phi1 = np.radians(yaw1), np.radians(pitch1)
    lam2, phi2 = np.radians(yaw2), np.radians(pitch2)
    
    dlam = lam2 - lam1
    dphi = phi2 - phi1
    
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    
    return float(np.degrees(c))


# ============================================================
# Построение viewport-карты для cv2.remap
# ============================================================

def build_view_map(yaw_deg: float, pitch_deg: float, fov_deg: float,
                   view_w: int, view_h: int,
                   erp_w: int, erp_h: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Строит map_x, map_y для cv2.remap: ERP → перспективный viewport.
    Оптимизированная версия: всё в float32, минимум аллокаций.
    """
    yaw = np.float32(np.radians(yaw_deg))
    pitch = np.float32(np.radians(pitch_deg))
    fov = np.float32(np.radians(fov_deg))

    f = np.float32((view_w * 0.5) / np.tan(fov * 0.5))

    # Координаты пикселей viewport (центрированные), сразу float32
    x_axis = (np.arange(view_w, dtype=np.float32) - view_w * 0.5)
    y_axis = (np.arange(view_h, dtype=np.float32) - view_h * 0.5)
    x, y = np.meshgrid(x_axis, y_axis, indexing='xy')  # (H, W) float32
    # z — скаляр f (broadcast)

    cos_p = np.float32(np.cos(pitch)); sin_p = np.float32(np.sin(pitch))
    cos_y = np.float32(np.cos(yaw));   sin_y = np.float32(np.sin(yaw))

    # Pitch вокруг X: y' = y*cos_p - f*sin_p,  z' = y*sin_p + f*cos_p
    y2 = y * cos_p - f * sin_p          # (H,W)
    z2 = y * sin_p + f * cos_p          # (H,W)
    # x2 == x

    # Yaw вокруг Y: x'' = x*cos_y + z2*sin_y,  z'' = -x*sin_y + z2*cos_y
    x3 = x * cos_y + z2 * sin_y
    z3 = -x * sin_y + z2 * cos_y
    # y3 == y2

    # Сферические координаты — без полной нормы вектора:
    # lon = atan2(x3, z3), lat = atan2(y3, sqrt(x3^2+z3^2))
    # (атан2 для широты избавляет от деления y/r и asin)
    hyp = np.sqrt(x3 * x3 + z3 * z3, dtype=np.float32)
    lon = np.arctan2(x3, z3)            # float32
    lat = np.arctan2(y2, hyp)           # float32

    # Сферические → ERP-пиксели (всё ещё float32)
    inv_2pi = np.float32(1.0 / (2.0 * np.pi))
    inv_pi  = np.float32(1.0 / np.pi)
    map_x = (lon * inv_2pi + np.float32(0.5)) * np.float32(erp_w)
    map_y = (lat * inv_pi  + np.float32(0.5)) * np.float32(erp_h)

    return map_x, map_y


# ============================================================
# Самопроверка
# ============================================================

if __name__ == "__main__":
    # Тесты sanity
    print("=== geometry.py self-test ===")
    
    # Центр ERP должен быть (0, 0)
    yaw, pitch = pixel_to_angle(2048, 1024, 4096, 2048)
    print(f"Center ERP: yaw={yaw:.2f}, pitch={pitch:.2f}  (expect 0, 0)")
    
    # Расстояние от точки до самой себя = 0
    d = angular_distance(45, 30, 45, 30)
    print(f"Self-distance: {d:.4f}  (expect 0)")
    
    # Расстояние между полюсами = 180°
    d = angular_distance(0, 90, 0, -90)
    print(f"Pole-to-pole: {d:.2f}  (expect 180)")
    
    # 90° по экватору
    d = angular_distance(0, 0, 90, 0)
    print(f"Equator 90°: {d:.2f}  (expect 90)")
    
    # Карта viewport
    mx, my = build_view_map(0, 0, 90, 1280, 720, 4096, 2048)
    print(f"View map shape: {mx.shape}, dtype: {mx.dtype}")
    print(f"  map_x range: [{mx.min():.1f}, {mx.max():.1f}]")
    print(f"  map_y range: [{my.min():.1f}, {my.max():.1f}]")
    print("✅ OK")