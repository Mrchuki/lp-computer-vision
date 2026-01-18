import os
import re
import cv2
from ultralytics import YOLO

model_lp = YOLO("runs/detect/lp_detector/weights/best.pt")
model_ocr = YOLO("runs/detect/ocr_detector/weights/best.pt")

video_path = "assets/video.MOV"
conf_lp = 0.3
conf_ocr = 0.5
process_every_n = 1
bar_width = 30

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Could not open video: {video_path}")

folder = os.path.dirname(video_path)
base = os.path.basename(video_path)
name, ext = os.path.splitext(base)
out_path = os.path.join(folder, f"{name}_compiled{ext}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
if not writer.isOpened():
    raise IOError(f"Could not create output video: {out_path}")


def render_bar(pct: int) -> str:
    pct = max(0, min(100, pct))
    filled = int(bar_width * pct / 100)
    return "█" * filled + "░" * (bar_width - filled)


font = cv2.FONT_HERSHEY_SIMPLEX
ui_scale = 1.5
ui_th = 4

green = (0, 255, 0)
red = (0, 0, 255)
white = (255, 255, 255)
blue = (255, 0, 0)
black = (0, 0, 0)

plate_re = re.compile(r"^\d{4}[A-Z]{3}$")

frame_idx = 0
last_progress = -1

last_lp_detected = False
last_boxes = []
last_lp_text = ""

recognized = False
recognized_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = frame.copy()

    if process_every_n <= 1 or frame_idx % process_every_n == 0:
        results_lp = model_lp(frame, conf=conf_lp, verbose=False, save=False)

        lp_detected = len(results_lp[0].boxes) > 0
        boxes_data = []
        best_text = ""

        if lp_detected:
            for box in results_lp[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes_data.append((x1, y1, x2, y2))

                lp_crop = frame[y1:y2, x1:x2]
                chars = []

                if lp_crop.size > 0:
                    results_ocr = model_ocr(lp_crop, conf=conf_ocr, verbose=False, save=False)
                    for char_box in results_ocr[0].boxes:
                        cx1, cy1, cx2, cy2 = map(int, char_box.xyxy[0])
                        char = results_ocr[0].names[int(char_box.cls[0])]
                        chars.append(((cx1 + cx2) / 2, char))

                chars.sort(key=lambda x: x[0])
                cand = "".join(c[1] for c in chars).upper()

                if len(cand) > len(best_text):
                    best_text = cand

        last_lp_detected = lp_detected
        last_boxes = boxes_data
        last_lp_text = best_text

        if not lp_detected:
            recognized = False
            recognized_text = ""

        if not recognized:
            if plate_re.match(last_lp_text):
                recognized = True
                recognized_text = last_lp_text

    for x1, y1, x2, y2 in last_boxes:
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), green, 2)
        cv2.putText(annotated_frame, "LP", (x1, y1 - 10), font, 0.9, green, 2)

        lp_crop = frame[y1:y2, x1:x2]
        if lp_crop.size > 0 and (process_every_n <= 1 or frame_idx % process_every_n == 0):
            results_ocr = model_ocr(lp_crop, conf=conf_ocr, verbose=False, save=False)
            for char_box in results_ocr[0].boxes:
                cx1, cy1, cx2, cy2 = map(int, char_box.xyxy[0])
                char = results_ocr[0].names[int(char_box.cls[0])]
                cv2.rectangle(
                    annotated_frame,
                    (x1 + cx1, y1 + cy1),
                    (x1 + cx2, y1 + cy2),
                    blue,
                    1,
                )
                cv2.putText(
                    annotated_frame,
                    char,
                    (x1 + cx1, y1 + cy1 - 5),
                    font,
                    0.5,
                    blue,
                    1,
                )

    x0, y0 = 10, 50

    cv2.rectangle(annotated_frame, (x0 - 5, y0 - 35), (x0 + 500, y0 + 75), black, -1)
    cv2.rectangle(annotated_frame, (x0 - 5, y0 - 35), (x0 + 500, y0 + 75), white, 2)

    cv2.putText(annotated_frame, "LP detected:", (x0, y0), font, ui_scale, white, ui_th)
    det_val = "True" if last_lp_detected else "False"
    det_color = green if last_lp_detected else red
    w_det, _ = cv2.getTextSize("LP detected:", font, ui_scale, ui_th)[0]
    cv2.putText(annotated_frame, det_val, (x0 + w_det + 15, y0), font, ui_scale, det_color, ui_th)

    y1 = y0 + 50
    cv2.putText(annotated_frame, "LP:", (x0, y1), font, ui_scale, white, ui_th)
    show_text = recognized_text if recognized else (last_lp_text or "")
    lp_color = green if recognized else white
    w_lp, _ = cv2.getTextSize("LP:", font, ui_scale, ui_th)[0]
    cv2.putText(annotated_frame, show_text, (x0 + w_lp + 15, y1), font, ui_scale, lp_color, ui_th)

    writer.write(annotated_frame)

    if total_frames > 0:
        progress = int((frame_idx + 1) / total_frames * 100)
        if progress != last_progress:
            print(f"\r[{render_bar(progress)}] {progress:3d}% ", end="", flush=True)
            last_progress = progress

    frame_idx += 1

cap.release()
writer.release()

if total_frames > 0 and last_progress < 100:
    print(f"\r[{render_bar(100)}] 100% ", end="", flush=True)

print(f"\nCompilation finished: {out_path}")
