"""Fixed-tick screen-grab + YOLO + OCR capture loop.

Grabs the emulator window at a fixed tick (~15 Hz), runs a YOLO detector,
maps detections onto minimap coordinates with a homography, and OCRs gold
and timer crops. Always works on the newest frame; stale frames are dropped
and states older than STALE_MS are not published.

This is the perception prototype. It does not yet emit mlbb.state/v1
snapshots or feed the decision layer.

    pip install -r requirements.txt
    python perception/mlbb_capture.py --weights yolov8n.pt
"""
import time
import json
import argparse

import cv2
import mss
import numpy as np
from ultralytics import YOLO

try:
    from paddleocr import PaddleOCR
except ImportError:  # OCR is optional
    PaddleOCR = None

# Emulator window region on screen (edit to match your setup)
REGION = {"left": 0, "top": 0, "width": 1920, "height": 1080}

# Fixed crops, relative to REGION (x, y, w, h) - tune for your layout
CROPS = {
    "gold":    (60, 20, 140, 40),
    "timer":   (900, 10, 120, 40),
    "minimap": (0, 0, 300, 300),
}

TICK_HZ = 15          # decisions per second
INFER_SIZE = 640      # YOLO input size
STALE_MS = 150        # drop actions older than this


def crop(img, box):
    x, y, w, h = box
    return img[y:y + h, x:x + w]


def screen_to_minimap(cx, cy, H):
    """Project a screen point to minimap coords using a homography H."""
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, H)[0][0].tolist()


def read_text(ocr, img):
    if ocr is None:
        return None
    res = ocr.ocr(img, cls=False)
    if not res or not res[0]:
        return None
    return " ".join(line[1][0] for line in res[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt",
                    help="YOLO weights (fine-tune on MLBB labels)")
    ap.add_argument("--homography", default=None,
                    help="JSON 3x3 screen->minimap homography")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.weights)
    ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False) if PaddleOCR else None
    H = np.array(json.load(open(args.homography)), dtype=np.float32) if args.homography else np.eye(3, dtype=np.float32)

    period = 1.0 / TICK_HZ
    with mss.mss() as sct:
        while True:
            t0 = time.perf_counter()
            # always grab the newest frame; nothing is queued
            frame = cv2.cvtColor(np.array(sct.grab(REGION)), cv2.COLOR_BGRA2BGR)
            ts = time.time()

            det = model.predict(frame, imgsz=INFER_SIZE, verbose=False)[0]
            objects = []
            for b in det.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                objects.append({
                    "cls": model.names[int(b.cls)],
                    "conf": round(float(b.conf), 3),
                    "screen": [round(cx), round(cy)],
                    "minimap": [round(v, 1) for v in screen_to_minimap(cx, cy, H)],
                })

            state = {
                "ts": ts,
                "gold": read_text(ocr, crop(frame, CROPS["gold"])),
                "timer": read_text(ocr, crop(frame, CROPS["timer"])),
                "objects": objects,
            }

            # downstream decision layer should ignore state older than STALE_MS
            age_ms = (time.time() - ts) * 1000
            if age_ms <= STALE_MS:
                print(json.dumps(state), flush=True)

            if args.show:
                cv2.imshow("mlbb", det.plot())
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # fixed tick: sleep the remainder, skip if we're behind
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)


if __name__ == "__main__":
    main()
