import sys
import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

model = YOLO("yolo11m.pt")
CONF = 0.5  # seuil de certitude
TOP = 82  # Mac : hauteur (pt) de la barre de titre + bandeau de l'app, à ignorer
CAMERA = 0  # Windows : index de la caméra virtuelle de l'app (essayer 1, 2... si besoin)
MAC = sys.platform == "darwin"
DEVICE = "mps" if MAC else (0 if torch.cuda.is_available() else "cpu")


if MAC:
    import Quartz

    def window_id():
        for w in Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID):
            if w["kCGWindowOwnerName"] == "Goggles3 USB Live" and w.get("kCGWindowName") == "Goggles3 USB Live":
                return w["kCGWindowNumber"]
        raise SystemExit("Fenêtre 'Goggles3 USB Live' introuvable : lancez l'app et cliquez sur Connect")


    def grab(wid):
        img = Quartz.CGWindowListCreateImage(
            Quartz.CGRectNull, Quartz.kCGWindowListOptionIncludingWindow, wid,
            Quartz.kCGWindowImageBoundsIgnoreFraming | Quartz.kCGWindowImageNominalResolution,
        )
        w, h = Quartz.CGImageGetWidth(img), Quartz.CGImageGetHeight(img)
        row = Quartz.CGImageGetBytesPerRow(img)
        data = np.frombuffer(Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img)), np.uint8)
        return data.reshape(h, row // 4, 4)[:, :w, :3].copy()  # BGRA -> BGR

    wid = window_id()

    def read():
        frame = grab(wid)
        return frame[TOP * frame.shape[0] // 732 :]

else:
    cap = cv2.VideoCapture(CAMERA, cv2.CAP_DSHOW)

    def read():
        return cap.read()[1]


latest = None   # dernière image à analyser
boxes = []      # dernières détections (x1, y1, x2, y2, label)


def detect():
    global boxes
    while True:
        if latest is None:
            time.sleep(0.01)
            continue
        r = model(latest, device=DEVICE, conf=CONF, classes=[0], imgsz=640, verbose=False)[0]
        boxes = [(*map(int, b.xyxy[0]), f"{r.names[int(b.cls)]} {float(b.conf):.0%}") for b in r.boxes]
        time.sleep(0.02)  # laisse du temps à la boucle d'affichage


threading.Thread(target=detect, daemon=True).start()

while True:
    frame = read()
    if frame is None:
        continue
    latest = frame.copy()
    for x1, y1, x2, y2, label in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Drone", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
