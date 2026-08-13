#!/usr/bin/env python3
"""
MVP: CUP/BOTTLE-TO-MOUTH detection for medication consumption.

Flow:
  1. Dispenser trigger (D key or trigger file) opens a monitoring window.
  2. YOLO detects cup / bottle / wine glass.
  3. MediaPipe Pose provides nose + face scale for normalization.
  4. Cup stays near mouth for several detections -> cup_to_mouth.
  5. cup_to_mouth inside the window -> likely_taken.
  6. Window expires with nothing -> dispense_window_missed.
"""

import argparse
import json
import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp

try:
    import requests
except ImportError:
    requests = None

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("Cup-to-mouth needs YOLO. Run: pip install ultralytics")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pose_landmarker_lite.task",
)

NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

DRINK_CLASSES = {"cup", "bottle", "wine glass"}


def distance(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def send_alert(alert, telegram_token, telegram_chat_id):
    alert["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(alert), flush=True)

    NOTIFY_EVENTS = {
        "cup_to_mouth",
        "likely_taken",
        "dispense_window_missed",
        "dispenser_triggered",
        "dispenser_trigger_simulated",
    }

    if requests is None or not telegram_token or not telegram_chat_id:
        return
    if alert["event"] not in NOTIFY_EVENTS:
        return

    ev = alert["event"]
    ts = alert["timestamp"]
    conf = alert.get("confidence")

    if ev == "likely_taken":
        msg = f"✅ Med likely TAKEN at {ts} (confidence {conf})"
    elif ev == "cup_to_mouth":
        msg = f"🥤 Cup/bottle at mouth at {ts} (confidence {conf})"
    elif ev == "dispense_window_missed":
        msg = f"⚠️ No cup-to-mouth after dispensing ({ts}). Please verify."
    elif ev in ("dispenser_triggered", "dispenser_trigger_simulated"):
        msg = f"💊 Dose dispensed at {ts}. Monitoring {alert.get('window_sec')}s."
    else:
        msg = f"ℹ️ {ev} at {ts}"

    try:
        requests.post(
            f"https://api.telegram.org/bot{telegram_token}/sendMessage",
            json={"chat_id": telegram_chat_id, "text": msg},
            timeout=3,
        )
    except Exception as exc:
        print(f"[WARN] Telegram send failed: {exc}")


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Downloading pose model -> {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def create_landmarker():
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def main():
    parser = argparse.ArgumentParser(description="Cup-to-mouth medication MVP")

    parser.add_argument("--source", default="0")
    parser.add_argument("--headless", action="store_true")

    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--object-conf", type=float, default=0.35)
    parser.add_argument("--detect-every", type=int, default=2)

    parser.add_argument("--cup-threshold", type=float, default=0.6,
                        help="Normalized cup-mouth distance. Higher = more sensitive.")
    parser.add_argument("--confirm-frames", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=5.0)

    parser.add_argument("--dispense-window", type=float, default=90.0)
    parser.add_argument("--trigger-file", default="")

    parser.add_argument("--telegram-token", default="")
    parser.add_argument("--telegram-chat-id", default="")

    args = parser.parse_args()

    ensure_model()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("ERROR: Cannot open camera/stream source.")

    landmarker = create_landmarker()
    yolo = YOLO(args.yolo_model)

    frame_idx = 0
    near_count = 0
    last_cup_alert = 0.0
    object_near_now = False

    dispense_until = 0.0
    dispense_confirmed = False

    print("[INFO] Cup-to-mouth MVP running.")
    print("[INFO] Click the video window: D = dispenser trigger, Q = quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        now = time.time()
        height, width, _ = frame.shape

        # ---------- Real dispenser trigger (file-based) ----------
        if args.trigger_file and os.path.exists(args.trigger_file):
            try:
                os.remove(args.trigger_file)
            except OSError:
                pass
            dispense_until = now + args.dispense_window
            dispense_confirmed = False
            send_alert({"event": "dispenser_triggered",
                        "window_sec": args.dispense_window},
                       args.telegram_token, args.telegram_chat_id)

        # ---------- Pose: nose + face scale ----------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )

        nose = None
        face_scale = None
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            nose_lm = lm[NOSE]
            lsh = lm[LEFT_SHOULDER]
            rsh = lm[RIGHT_SHOULDER]

            if not args.headless:
                for p in (nose_lm, lsh, rsh):
                    if p.visibility > 0.4:
                        cv2.circle(frame, (int(p.x * width), int(p.y * height)),
                                   5, (0, 255, 0), -1)

            if nose_lm.visibility > 0.5 and max(lsh.visibility, rsh.visibility) > 0.5:
                nose = (nose_lm.x, nose_lm.y)
                scx = (lsh.x + rsh.x) / 2.0
                scy = (lsh.y + rsh.y) / 2.0
                face_scale = distance(nose[0], nose[1], scx, scy)
                if face_scale <= 0.02:
                    face_scale = None

        # ---------- Cup / bottle detection ----------
        if (frame_idx % max(1, args.detect_every) == 0
                and nose is not None and face_scale is not None):

            object_near_now = False
            preds = yolo.predict(frame, verbose=False, conf=args.object_conf)

            for pred in preds:
                if pred.boxes is None:
                    continue
                names = pred.names

                for box in pred.boxes:
                    label = names.get(int(box.cls[0].item()), "").lower()
                    if label not in DRINK_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())

                    # Closest point of the box to the nose (works for tall bottles)
                    cx = min(max(nose[0] * width, x1), x2) / width
                    cy = min(max(nose[1] * height, y1), y2) / height
                    d = distance(cx, cy, nose[0], nose[1]) / face_scale

                    if not args.headless:
                        cv2.rectangle(frame, (int(x1), int(y1)),
                                      (int(x2), int(y2)), (0, 255, 255), 2)
                        cv2.putText(frame, f"{label} {d:.2f}",
                                    (int(x1), max(20, int(y1) - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 255), 2)

                    if d < args.cup_threshold:
                        object_near_now = True

            if object_near_now:
                near_count += 1
            else:
                near_count = 0

        # ---------- Cup-to-mouth event ----------
        cup_to_mouth = False
        if near_count >= args.confirm_frames and (now - last_cup_alert) > args.cooldown:
            cup_to_mouth = True
            last_cup_alert = now
            near_count = 0

        if cup_to_mouth:
            confidence = 0.8
            if now < dispense_until:
                confidence = min(0.98, confidence + 0.1)

            send_alert({"event": "cup_to_mouth",
                        "confidence": round(confidence, 2)},
                       args.telegram_token, args.telegram_chat_id)

            if now < dispense_until:
                dispense_confirmed = True
                send_alert({"event": "likely_taken",
                            "reason": "cup_to_mouth_after_dispense",
                            "confidence": round(min(0.99, confidence + 0.05), 2)},
                           args.telegram_token, args.telegram_chat_id)

        # ---------- Window timeout ----------
        if dispense_until > 0 and now > dispense_until:
            if not dispense_confirmed:
                send_alert({"event": "dispense_window_missed",
                            "reason": "no_cup_to_mouth_detected",
                            "confidence": 0.65},
                           args.telegram_token, args.telegram_chat_id)
            dispense_until = 0.0
            dispense_confirmed = False

        # ---------- Display ----------
        if not args.headless:
            status = []
            if now < dispense_until:
                status.append(f"window: {dispense_until - now:03.0f}s")
            if object_near_now:
                status.append("cup at mouth")
            status.append(f"confirm: {near_count}/{args.confirm_frames}")

            cv2.putText(frame, " | ".join(status), (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "D = dispenser trigger, Q = quit",
                        (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

            cv2.imshow("Cup-to-mouth MVP", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                dispense_until = now + args.dispense_window
                dispense_confirmed = False
                send_alert({"event": "dispenser_trigger_simulated",
                            "window_sec": args.dispense_window},
                           args.telegram_token, args.telegram_chat_id)
        else:
            time.sleep(0.001)

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()