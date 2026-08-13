#!/usr/bin/env python3
"""
MVP: hand-to-mouth + optional cup/bottle detection.
Uses MediaPipe Tasks API + Telegram notifications.
"""

import argparse
import json
import math
import os
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    import sys
    sys.exit(1)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pose_landmarker_lite.task",
)

# BlazePose landmark indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16

def distance(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)

def send_alert(alert: dict, telegram_token: str, telegram_chat_id: str):
    """Logs to console and sends Telegram notification for important events."""
    alert["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(alert), flush=True)

    # Only send Telegram notifications for important events (to avoid spam)
    NOTIFY_EVENTS = {
        "likely_taken",
        "dispense_window_missed",
        "dispenser_trigger_simulated",
    }
    
    if not telegram_token or not telegram_chat_id or alert["event"] not in NOTIFY_EVENTS:
        return

    ev = alert.get("event")
    ts = alert.get("timestamp")
    conf = alert.get("confidence")

    if ev == "likely_taken":
        msg = f"✅ Med likely TAKEN at {ts} (confidence {conf})"
    elif ev == "dispense_window_missed":
        msg = f"⚠️ No medication event after dispensing ({ts}). Please verify."
    elif ev == "dispenser_trigger_simulated":
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
    parser = argparse.ArgumentParser(description="Hand-to-mouth MVP (Tasks API + Telegram)")

    parser.add_argument("--source", default="0",
                        help="0 for webcam, or RTSP URL / video file path")
    parser.add_argument("--headless", action="store_true")

    parser.add_argument("--dist-threshold", type=float, default=0.45)
    parser.add_argument("--confirm-frames", type=int, default=5)
    parser.add_argument("--cooldown", type=float, default=5.0)

    parser.add_argument("--require-motion", dest="require_motion",
                        action="store_true", default=True)
    parser.add_argument("--no-require-motion", dest="require_motion",
                        action="store_false")
    parser.add_argument("--motion-delta", type=float, default=0.03)

    parser.add_argument("--enable-object", action="store_true")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--object-conf", type=float, default=0.45)
    parser.add_argument("--object-every", type=int, default=3)
    parser.add_argument("--object-mouth-threshold", type=float, default=0.45)
    parser.add_argument("--object-memory-sec", type=float, default=2.0)

    parser.add_argument("--dispense-window", type=float, default=90.0)

    # Telegram Args
    parser.add_argument("--telegram-token", default="", help="Telegram Bot Token")
    parser.add_argument("--telegram-chat-id", default="", help="Your Telegram Chat ID")

    args = parser.parse_args()

    ensure_model()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("ERROR: Cannot open camera/stream source.")

    landmarker = create_landmarker()

    yolo = None
    if args.enable_object:
        try:
            from ultralytics import YOLO
            yolo = YOLO(args.yolo_model)
            print(f"[INFO] YOLO enabled: {args.yolo_model}")
        except Exception as exc:
            print(f"[WARN] YOLO disabled: {exc}")

    frame_idx = 0
    ema_dist = None
    dist_history = deque(maxlen=8)

    near_count = 0
    last_hand_alert = 0.0
    last_object_near = 0.0

    dispense_until = 0.0
    dispense_confirmed = False

    print("[INFO] Running. Click the video window and press 'D' to simulate dispenser trigger. Press 'Q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        now = time.time()
        height, width, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        landmarks = None
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

        hand_to_mouth = False
        event_side = None
        event_dist = None
        nose = None
        face_scale = None

        # -----------------------------
        # Hand-to-mouth detection
        # -----------------------------
        if landmarks:
            nose_lm = landmarks[NOSE]
            lsh = landmarks[LEFT_SHOULDER]
            rsh = landmarks[RIGHT_SHOULDER]
            lw = landmarks[LEFT_WRIST]
            rw = landmarks[RIGHT_WRIST]

            if not args.headless:
                for p in (nose_lm, lsh, rsh, lw, rw):
                    if p.visibility > 0.4:
                        cv2.circle(frame, (int(p.x * width), int(p.y * height)),
                                   5, (0, 255, 0), -1)

            if nose_lm.visibility > 0.5 and max(lsh.visibility, rsh.visibility) > 0.5:
                nose = (nose_lm.x, nose_lm.y)

                shoulder_center = (
                    (lsh.x + rsh.x) / 2.0,
                    (lsh.y + rsh.y) / 2.0,
                )

                face_scale = distance(nose[0], nose[1],
                                      shoulder_center[0], shoulder_center[1])

                if face_scale > 0.02:
                    candidates = []

                    if lw.visibility > 0.4:
                        candidates.append((
                            distance(lw.x, lw.y, nose[0], nose[1]) / face_scale,
                            "left"
                        ))

                    if rw.visibility > 0.4:
                        candidates.append((
                            distance(rw.x, rw.y, nose[0], nose[1]) / face_scale,
                            "right"
                        ))

                    if candidates:
                        wrist_dist, side = min(candidates)

                        if ema_dist is None:
                            ema_dist = wrist_dist
                        else:
                            ema_dist = 0.65 * ema_dist + 0.35 * wrist_dist

                        dist_history.append(ema_dist)

                        near = ema_dist < args.dist_threshold

                        moving_toward = False
                        if len(dist_history) >= 5:
                            moving_toward = (dist_history[0] - dist_history[-1]) > args.motion_delta

                        if near:
                            near_count += 1
                        else:
                            near_count = max(0, near_count - 1)

                        motion_ok = moving_toward if args.require_motion else True

                        if near_count >= args.confirm_frames and motion_ok:
                            if now - last_hand_alert > args.cooldown:
                                hand_to_mouth = True
                                event_side = side
                                event_dist = ema_dist
                                last_hand_alert = now

                                near_count = 0
                                dist_history.clear()
                                ema_dist = None

                        if not args.headless:
                            # FIX: safe formatting when ema_dist is None after a reset
                            dist_val = ema_dist if ema_dist is not None else 0.0
                            cv2.putText(frame, f"wrist-mouth: {dist_val:.2f}",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (0, 255, 0), 2)
                            cv2.putText(frame, f"near_count: {near_count}",
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (0, 255, 0), 2)

        # -----------------------------
        # Optional cup / bottle detection
        # -----------------------------
        if (yolo is not None
                and frame_idx % max(1, args.object_every) == 0
                and nose is not None
                and face_scale is not None):

            preds = yolo.predict(frame, verbose=False, conf=args.object_conf)

            for pred in preds:
                if pred.boxes is None:
                    continue

                names = pred.names

                for box in pred.boxes:
                    cls_id = int(box.cls[0].item())
                    label = names.get(cls_id, "").lower()

                    if label not in {"cup", "bottle"}:
                        continue

                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())

                    obj_cx = ((x1 + x2) / 2.0) / width
                    obj_cy = ((y1 + y2) / 2.0) / height

                    obj_mouth_dist = distance(obj_cx, obj_cy,
                                              nose[0], nose[1]) / face_scale

                    if obj_mouth_dist < args.object_mouth_threshold:
                        last_object_near = now

                        if not args.headless:
                            cv2.rectangle(frame, (int(x1), int(y1)),
                                          (int(x2), int(y2)), (0, 255, 255), 2)
                            cv2.putText(frame, f"{label} {obj_mouth_dist:.2f}",
                                        (int(x1), max(20, int(y1) - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 255), 2)

        # -----------------------------
        # Alerts
        # -----------------------------
        if hand_to_mouth:
            drinking_recently = (now - last_object_near) <= args.object_memory_sec

            confidence = 0.75
            if drinking_recently:
                confidence += 0.10
            if now < dispense_until:
                confidence += 0.10
            confidence = min(0.98, confidence)

            send_alert({
                "event": "hand_to_mouth",
                "side": event_side,
                "normalized_distance": round(float(event_dist or 0.0), 3),
                "drinking_object_near": drinking_recently,
                "confidence": round(confidence, 2),
            }, args.telegram_token, args.telegram_chat_id)

            if now < dispense_until:
                dispense_confirmed = True
                send_alert({
                    "event": "likely_taken",
                    "reason": "hand_to_mouth_after_dispense",
                    "drinking_object_near": drinking_recently,
                    "confidence": round(min(0.99, confidence + 0.05), 2),
                }, args.telegram_token, args.telegram_chat_id)

        # Dispenser window timeout
        if dispense_until > 0 and now > dispense_until:
            if not dispense_confirmed:
                send_alert({
                    "event": "dispense_window_missed",
                    "reason": "no_hand_to_mouth_detected",
                    "confidence": 0.65,
                }, args.telegram_token, args.telegram_chat_id)
            dispense_until = 0.0
            dispense_confirmed = False

        # -----------------------------
        # Display
        # -----------------------------
        if not args.headless:
            status = []

            if now < dispense_until:
                status.append(f"dispense window: {dispense_until - now:03.0f}s")

            if (now - last_object_near) <= args.object_memory_sec:
                status.append("cup/bottle near mouth")

            cv2.putText(frame, " | ".join(status), (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "Press D in THIS WINDOW for trigger, Q to quit",
                        (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

            cv2.imshow("Hand-to-mouth MVP", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                dispense_until = now + args.dispense_window
                dispense_confirmed = False
                send_alert({
                    "event": "dispenser_trigger_simulated",
                    "window_sec": args.dispense_window,
                }, args.telegram_token, args.telegram_chat_id)
        else:
            time.sleep(0.001)

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()