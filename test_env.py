import sys
print("Python path:", sys.executable)
print("Python version:", sys.version)

try:
    from ultralytics import YOLO
    print("✅ ultralytics OK")
except ImportError as e:
    print("❌ ultralytics:", e)

try:
    import cv2
    print("✅ opencv version:", cv2.__version__)
except ImportError as e:
    print("❌ opencv:", e)

try:
    import py360convert
    print("✅ py360convert OK")
except ImportError as e:
    print("❌ py360convert:", e)

try:
    from flask import Flask
    print("✅ flask OK")
except ImportError as e:
    print("❌ flask:", e)