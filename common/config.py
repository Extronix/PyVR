"""
Глобальная конфигурация проекта PyVR.

Все настройки в одном месте, frozen dataclasses (иммутабельны).
Доступ через CFG.<section>.<param>.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================
# Пути
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# Видео
# ============================================================

@dataclass(frozen=True)
class VideoConfig:
    # Путь к ERP-видеофайлу (используется СЕРВЕРОМ начиная с шага 9)
    source_path: str = str(DATA_DIR / "RoSh.mp4")
    
    # Зацикливать видео при достижении конца
    loop: bool = True
    
    # Размер очереди декодированных кадров (для фонового декодера)
    decode_queue_size: int = 4
    
    # Если True — VideoSource ждёт чтения, чтобы не "убегать" вперёд
    block_on_full_queue: bool = True


# ============================================================
# View / Renderer
# ============================================================

@dataclass(frozen=True)
class ViewConfig:
    # Размер окна клиента
    window_width: int = 1280
    window_height: int = 720
    window_title: str = "PyVR Client"
    
    # FOV: горизонтальный угол обзора в градусах
    fov_deg_default: float = 90.0
    fov_deg_min: float = 30.0
    fov_deg_max: float = 120.0
    fov_step_deg: float = 5.0     # шаг изменения FOV колесом / клавишами
    
    # Чувствительность мыши: градусов на пиксель
    mouse_sensitivity: float = 0.15
    
    # Сглаживание скорости поворота (EMA alpha)
    velocity_ema_alpha: float = 0.2
    
    # Сэмплирование viewport: точек по горизонтали (вертикаль — пропорционально)
    sample_grid_w: int = 128       # → 128 × 72 ≈ 9k точек
    
    # Кэш карт сэмплирования (по углам округлённым до целых градусов)
    sample_cache_max: int = 256
    
    # HUD
    hud_enabled: bool = True
    hud_font_scale: float = 0.5
    hud_color_bgr: tuple[int, int, int] = (255, 255, 0)   # cyan-ish


# ============================================================
# Сеть
# ============================================================

@dataclass(frozen=True)
class NetworkConfig:
    host: str = "127.0.0.1"
    
    # Порты
    roi_port: int = 5001            # клиент -> сервер: ROI / поза головы
    tile_port: int = 5002           # сервер -> клиент: чанки тайлов
    
    # Размер сокетного буфера ОС (увеличили для тайлов — много трафика)
    socket_buffer_size: int = 2 * 1024 * 1024     # 2 MB
    
    # Частота отправки ROI клиентом
    roi_send_rate_hz: int = 60


# ============================================================
# Тайлинг
# ============================================================

@dataclass(frozen=True)
class TilesConfig:
    # Сетка тайлов в ERP-кадре
    grid_cols: int = 8        # тайлов по горизонтали (yaw)
    grid_rows: int = 4        # тайлов по вертикали (pitch)
    
    # JPEG-кодек
    jpeg_quality: int = 65    # 0..100
    
    # UDP-фрагментация
    chunk_payload_size: int = 1200      # байт полезной нагрузки на чанк
    chunk_header_size: int = 25         # размер бинарного заголовка (см. protocol.py)
    
    # Сборка тайлов на клиенте
    reassembly_timeout_ms: int = 500    # сколько ждать недостающие чанки
    tile_cache_ttl_frames: int = 30     # сколько кадров хранить last-known
    
    # Сетевая нагрузка
    max_inflight_tiles: int = 64        # макс. одновременно собираемых тайлов
    
    # Визуализация дыр (тайлов нет в кэше)
    placeholder_checker_size: int = 32  # размер клеток "шахматной доски" в px
    placeholder_color_a: tuple[int, int, int] = (96, 96, 96)
    placeholder_color_b: tuple[int, int, int] = (160, 160, 160)


# ============================================================
# Логирование
# ============================================================

@dataclass(frozen=True)
class LogConfig:
    level: str = "INFO"
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt: str = "%H:%M:%S"
    
    # Как часто логировать статистику клиента (раз в N кадров)
    client_log_every_n_frames: int = 60
    
    # Как часто логировать статистику сервера (раз в N принятых ROI)
    server_log_every_n_packets: int = 60

from dataclasses import dataclass, field

@dataclass
class FoveationConfig:
    enabled: bool = True
    # Качество JPEG в центре viewport (на оси взгляда)
    quality_center: int = 90
    # Качество JPEG на самом краю (≥ периферии viewport)
    quality_edge: int = 35
    # Качество тайлов вне viewport (halo / prefetch)
    quality_outside: int = 25
    # Гамма для кривой важности (1.0 = линейная, >1 — резче падение к краям)
    falloff_gamma: float = 1.5
    # Если True — пишет в лог распределение качеств
    log_distribution: bool = False
# ============================================================
# Корневой конфиг
# ============================================================
@dataclass
class FoveationConfig:
    enabled: bool = True
    quality_center: int = 90        # JPEG quality в центре viewport
    quality_edge: int = 40          # на краю viewport
    quality_outside: int = 25       # вне viewport (halo)
    falloff_gamma: float = 1.5      # >1 — резче падение к краям
    log_distribution: bool = False  # печатать ли распределение в лог
    
@dataclass(frozen=True)
class _Config:
    foveation: FoveationConfig = field(default_factory=FoveationConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    view: ViewConfig = field(default_factory=ViewConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    tiles: TilesConfig = field(default_factory=TilesConfig)
    log: LogConfig = field(default_factory=LogConfig)
    foveation: FoveationConfig = field(default_factory=FoveationConfig)
from dataclasses import dataclass, field



CFG = _Config()