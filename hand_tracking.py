"""
Hand Tracking AI - Stable version with temporal smoothing
Works with: mediapipe 0.10.13+ | Python 3.12 | opencv-python 4.x

Install:
    pip install opencv-python mediapipe

Controls:
  Q / ESC → Quit
  S       → Save screenshot
  F       → Toggle FPS
  D       → Toggle landmark IDs
"""

import cv2
import mediapipe as mp
import time
import os
import math
import urllib.request
from collections import deque

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_opts

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print("⬇️  Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅  Model downloaded.")

# ── Constants ────────────────────────────────────────────────────────────────
FINGERTIP_IDS = [4, 8, 12, 16, 20]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# (tip, dip, pip, mcp)
FINGER_JOINTS = [
    (8,  7,  6,  5),
    (12, 11, 10, 9),
    (16, 15, 14, 13),
    (20, 19, 18, 17),
]

SMOOTHING_FRAMES = 5   # nombre de frames pour la moyenne
CHANGE_THRESHOLD = 1   # changement minimum pour mettre à jour l'affichage


# ── Temporal smoother ─────────────────────────────────────────────────────────
class FingerSmoother:
    """Stabilise le comptage en faisant une moyenne sur N frames."""
    def __init__(self, n=SMOOTHING_FRAMES):
        self.history = deque(maxlen=n)
        self.last_stable = 0

    def update(self, count):
        self.history.append(count)
        avg = sum(self.history) / len(self.history)
        rounded = round(avg)
        # N'affiche un changement que si la différence est > seuil
        if abs(rounded - self.last_stable) >= CHANGE_THRESHOLD:
            self.last_stable = rounded
        return self.last_stable


# ── Geometry ──────────────────────────────────────────────────────────────────
def dist3d(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

def angle3(a, b, c):
    """Angle en degrés au point b."""
    v1 = (a.x-b.x, a.y-b.y, a.z-b.z)
    v2 = (c.x-b.x, c.y-b.y, c.z-b.z)
    n1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    n2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
    if n1 == 0 or n2 == 0:
        return 0.0
    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))

def is_finger_extended(lm, tip, dip, pip, mcp):
    """
    Triple vérification pour chaque doigt :
    1. Angle au PIP > 155° (presque droit)
    2. Angle au DIP > 150°
    3. Distance tip→poignet > distance mcp→poignet
    Les 3 doivent être vrais.
    """
    pip_angle = angle3(lm[mcp], lm[pip], lm[dip])
    dip_angle = angle3(lm[pip], lm[dip], lm[tip])
    tip_far   = dist3d(lm[tip], lm[0]) > dist3d(lm[mcp], lm[0])
    return pip_angle > 155 and dip_angle > 150 and tip_far

def count_fingers(lm, label="Right"):
    if len(lm) < 21:
        return 0

    fingers = 0
    palm_size = dist3d(lm[0], lm[9])
    if palm_size == 0:
        return 0

    # Pouce : angle à l'IP + distance par rapport à l'index MCP
    thumb_ip_angle = angle3(lm[2], lm[3], lm[4])
    thumb_spread   = dist3d(lm[4], lm[5]) / palm_size
    if thumb_ip_angle > 150 and thumb_spread > 0.45:
        fingers += 1

    # 4 doigts
    for tip, dip, pip, mcp in FINGER_JOINTS:
        if is_finger_extended(lm, tip, dip, pip, mcp):
            fingers += 1

    return fingers


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_landmarks(frame, hand_landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 255), 2)
    for i, (x, y) in enumerate(pts):
        is_tip = i in FINGERTIP_IDS
        cv2.circle(frame, (x, y), 6 if is_tip else 4,
                   (0, 0, 255) if is_tip else (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 6 if is_tip else 4, (0, 0, 0), 1)

def draw_landmark_ids(frame, hand_landmarks):
    h, w = frame.shape[:2]
    for idx, lm in enumerate(hand_landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.putText(frame, str(idx), (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)

def get_positions(frame, hand_landmarks):
    h, w = frame.shape[:2]
    return [(i, int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(hand_landmarks)]

def draw_hand_box(frame, positions, label, fingers):
    if not positions:
        return
    xs = [p[1] for p in positions]
    ys = [p[2] for p in positions]
    x1 = max(0, min(xs) - 20)
    y1 = max(0, min(ys) - 20)
    x2 = min(frame.shape[1], max(xs) + 20)
    y2 = min(frame.shape[0], max(ys) + 20)
    color = (0, 200, 255) if label == "Right" else (255, 100, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} | Fingers: {fingers}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, text, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    options = vision.HandLandmarkerOptions(
        base_options=base_opts.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Un smoother par main (max 2)
    smoothers = [FingerSmoother(), FingerSmoother()]

    prev_time      = time.time()
    show_fps       = True
    show_ids       = False
    screenshot_dir = "screenshots"

    print("✅  Hand Tracking running.")
    print("   Q/ESC → Quit | S → Screenshot | F → FPS | D → IDs")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame    = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result   = detector.detect(mp_image)

        hand_count = len(result.hand_landmarks) if result.hand_landmarks else 0

        for i, hand_lms in enumerate(result.hand_landmarks or []):
            label     = result.handedness[i][0].category_name if result.handedness else "Right"
            draw_landmarks(frame, hand_lms)
            if show_ids:
                draw_landmark_ids(frame, hand_lms)
            positions = get_positions(frame, hand_lms)
            raw_count = count_fingers(hand_lms, label)
            fingers   = smoothers[i].update(raw_count)   # lissage temporel
            draw_hand_box(frame, positions, label, fingers)

        # Réinitialise les smoothers des mains non détectées
        for i in range(hand_count, 2):
            smoothers[i].history.clear()

        now       = time.time()
        fps       = 1 / max(now - prev_time, 1e-6)
        prev_time = now
        if show_fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.putText(frame, f"Hands: {hand_count}", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Hand Tracking AI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            os.makedirs(screenshot_dir, exist_ok=True)
            path = os.path.join(screenshot_dir,
                                f"hand_{time.strftime('%Y%m%d_%H%M%S')}.png")
            cv2.imwrite(path, frame)
            print(f"📸  Saved → {path}")
        elif key == ord('f'):
            show_fps = not show_fps
        elif key == ord('d'):
            show_ids = not show_ids

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("  Stopped.")


if __name__ == "__main__":
    main()