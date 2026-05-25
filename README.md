# PyVR

Прототип VR-стриминга 360°-видео через UDP с тайловой нарезкой
и foveated encoding.

## Архитектура

- **Server** читает ERP-видео, режет на тайлы 8×4, кодирует JPEG
  с разным quality в зависимости от importance тайла относительно
  viewport клиента, шлёт по UDP.
- **Client** принимает тайлы, собирает equirect canvas, проецирует
  viewport по углам мыши.

## Запуск

```bash
# Терминал 1 — сервер
python -m server.main

# Терминал 2 — клиент
python test_group7.py