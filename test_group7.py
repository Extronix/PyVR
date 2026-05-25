"""
Запуск клиента (Group 7).

Перед запуском убедись, что сервер уже работает:
    python -m server.main
"""
import logging
import sys

from common.config import CFG
from client.app import ClientApp


def main():
    logging.basicConfig(
        level=CFG.log.level,
        format=CFG.log.fmt,
        datefmt=CFG.log.datefmt,
    )
    log = logging.getLogger("test_group7")
    log.info("=== Group 7: ClientApp test ===")
    log.info(f"server: {CFG.network.host}  roi:{CFG.network.roi_port}  tile:{CFG.network.tile_port}")
    log.info(f"window: {CFG.view.window_width}x{CFG.view.window_height}")
    log.info("Mouse drag = look around. Wheel / + - = FOV. R = reset. ESC / Q = quit.")

    # ВАЖНО: erp_w / erp_h должны совпадать с разрешением видео на сервере.
    # Стандарт для ERP-роликов: 4096×2048. Если у тебя другое — поменяй здесь.
    app = ClientApp(
        erp_w=5760,
        erp_h=2880,
    )
    try:
        app.run()
    except Exception as e:
        log.exception(f"ClientApp crashed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())