# SpaceTrace preprocessing — public research version
# Generated from spacetrace_preprocess.ipynb.
# Interactive identity validation is intended to run in Jupyter/Colab.


# SpaceTrace — CCTV Person Detection, Tracking, Re-identification, and Anonymised Export

# 1. Install dependencies
# Install dependencies separately: pip install -r requirements.txt

# 2. Configuration
from pathlib import Path
import os

# -------------------------------------------------------------------
# Required paths
# -------------------------------------------------------------------
VIDEO_PATH = "/path/to/input_video.mp4"
OUTPUT_DIR = "/path/to/output_folder"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Session metadata
# -------------------------------------------------------------------
# DATE and START_TIME_MANUAL define the wall-clock time of frame 0.
# If START_TIME_MANUAL is left empty, the notebook can parse a 14-digit
# timestamp (YYYYMMDDHHMMSS) from the filename as a fallback.
DATE = "2026-02-01"          # YYYY-MM-DD
DAY = 1                      # study day number
SESSION = "Morning"          # e.g. Morning / Midday / Evening
CAMERA_ID = "C1"             # e.g. C1–C7
START_TIME_MANUAL = "08:00:00"  # HH:MM:SS; use "" to enable filename fallback
USE_FILENAME_DATETIME_BACKUP = True

# -------------------------------------------------------------------
# Detection / tracking
# -------------------------------------------------------------------
YOLO_MODEL = "yolo11m.pt"
REID_MODEL = "yolo11m-cls.pt"
CONF_THRES = 0.25
VID_STRIDE = 300             # process one frame every N original frames
SAVE_ANNOTATED_VIDEO = True

# -------------------------------------------------------------------
# Identity labels
# -------------------------------------------------------------------
# Replace or extend this list for your own study.
PARTICIPANT_IDS = [
    "AA-01", "AA-02", "AA-03", "AA-04", "AA-05", "AA-06", "AA-07",
    "Visitor", "Noise", "Skip"
]
EXCLUDE_NOISE_FROM_FINAL = True

# -------------------------------------------------------------------
# Output files
# -------------------------------------------------------------------
STEM = Path(VIDEO_PATH).stem
ANNOTATED_VIDEO = os.path.join(OUTPUT_DIR, f"{STEM}_detection_tracking.mp4")
PRIVACY_VIDEO = os.path.join(OUTPUT_DIR, f"{STEM}_person_id_blur_overlay.mp4")
FINAL_CSV = os.path.join(OUTPUT_DIR, f"{STEM}_final_person_id.csv")
REF_FRAME = os.path.join(OUTPUT_DIR, f"{STEM}_reference_frame.jpg")
TRACKER_YAML = os.path.join(OUTPUT_DIR, "botsort_reid.yaml")

print("Video:", VIDEO_PATH)
print("Output:", OUTPUT_DIR)
print("Detector:", YOLO_MODEL)
print("ReID model:", REID_MODEL)


# 3. Configure BoT-SORT + ReID
TRACKER_CFG = f"""\
tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.35
track_buffer: 180
match_thresh: 0.85
fuse_score: True
gmc_method: sparseOptFlow
proximity_thresh: 0.4
appearance_thresh: 0.7
with_reid: True
model: {REID_MODEL}
"""

with open(TRACKER_YAML, "w") as f:
    f.write(TRACKER_CFG)

print("Tracker config written:", TRACKER_YAML)


# 4. Load video and establish recording time
import cv2
import re
from datetime import datetime, timedelta
from ultralytics import YOLO


def parse_filename_timestamp(video_path):
    """Return the first YYYYMMDDHHMMSS timestamp found in the filename."""
    stem = Path(video_path).stem
    candidates = re.findall(r"(20\d{12})", stem)
    for ts in candidates:
        try:
            return datetime.strptime(ts, "%Y%m%d%H%M%S"), f"filename timestamp {ts}"
        except ValueError:
            pass
    return None, "no filename timestamp"


def infer_recording_start(video_path, manual_date, manual_start_time):
    """
    Priority:
    1) DATE + START_TIME_MANUAL
    2) full timestamp parsed from filename
    3) DATE + 00:00:00
    """
    manual_date = str(manual_date).strip()
    manual_start_time = str(manual_start_time).strip()

    if manual_start_time:
        return (
            datetime.strptime(f"{manual_date} {manual_start_time}", "%Y-%m-%d %H:%M:%S"),
            "manual DATE + START_TIME_MANUAL",
        )

    if USE_FILENAME_DATETIME_BACKUP:
        file_dt, source = parse_filename_timestamp(video_path)
        if file_dt is not None:
            return file_dt, source

    return (
        datetime.strptime(f"{manual_date} 00:00:00", "%Y-%m-%d %H:%M:%S"),
        "manual DATE + default 00:00:00",
    )


model = YOLO(YOLO_MODEL)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open VIDEO_PATH: {VIDEO_PATH}")

FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

recording_start, TIME_SOURCE = infer_recording_start(
    VIDEO_PATH, DATE, START_TIME_MANUAL
)
DATE_EXPORT = recording_start.strftime("%Y-%m-%d")
START_TIME_USED = recording_start.strftime("%H:%M:%S")
ANALYSIS_FPS = FPS / VID_STRIDE
FRAME_INTERVAL = VID_STRIDE / FPS

print(f"Video: {TOTAL:,} frames | {FPS:.2f} fps | {TOTAL/FPS:.1f}s | {W}×{H}")
print(f"Stride: {VID_STRIDE} → effective sampling rate {ANALYSIS_FPS:.4f} fps")
print(f"Recording start: {recording_start} ({TIME_SOURCE})")


# 5. Run person detection and tracking
import math
import time
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

TOTAL_PROCESSED = math.ceil(TOTAL / VID_STRIDE)
writer = None
if SAVE_ANNOTATED_VIDEO:
    writer = cv2.VideoWriter(
        ANNOTATED_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        ANALYSIS_FPS,
        (W, H),
    )

raw_rows = []
raw_track_thumbs = {}
proc_idx = 0


def update_track_thumbnail(tid, conf, frame_bgr, frame_idx, x1, y1, x2, y2):
    pad = 10
    x1c = max(0, int(x1) - pad)
    y1c = max(0, int(y1) - pad)
    x2c = min(W, int(x2) + pad)
    y2c = min(H, int(y2) + pad)

    crop_bgr = frame_bgr[y1c:y2c, x1c:x2c]
    if crop_bgr.size == 0:
        return

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    th = 128
    tw = max(1, int(crop_rgb.shape[1] * th / max(1, crop_rgb.shape[0])))
    crop_thumb = cv2.resize(crop_rgb, (tw, th))

    if tid not in raw_track_thumbs:
        raw_track_thumbs[tid] = {
            "crop": crop_thumb,
            "frame": frame_idx,
            "count": 1,
            "conf": float(conf),
        }
    else:
        raw_track_thumbs[tid]["count"] += 1
        if float(conf) > raw_track_thumbs[tid]["conf"]:
            raw_track_thumbs[tid]["crop"] = crop_thumb
            raw_track_thumbs[tid]["frame"] = frame_idx
            raw_track_thumbs[tid]["conf"] = float(conf)


start_time_run = time.time()
pbar = tqdm(total=TOTAL_PROCESSED, desc="Detection + tracking", unit="sampled frame")

try:
    for result in model.track(
        source=VIDEO_PATH,
        tracker=TRACKER_YAML,
        persist=True,
        stream=True,
        classes=[0],
        conf=CONF_THRES,
        vid_stride=VID_STRIDE,
        verbose=False,
    ):
        orig_frame_idx = proc_idx * VID_STRIDE
        proc_idx += 1

        video_t = orig_frame_idx / FPS
        dt = recording_start + timedelta(seconds=video_t)
        ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        time_of_day = dt.strftime("%H:%M:%S")

        if SAVE_ANNOTATED_VIDEO and writer is not None:
            writer.write(result.plot())

        frame_bgr = result.orig_img
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                if box.id is None:
                    continue

                tid = int(box.id[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                update_track_thumbnail(
                    tid, conf, frame_bgr, orig_frame_idx, x1, y1, x2, y2
                )

                raw_rows.append({
                    "frame": orig_frame_idx,
                    "datetime": ts_str,
                    "time_of_day": time_of_day,
                    "time_sec": round(video_t, 2),
                    "date": DATE_EXPORT,
                    "day": DAY,
                    "session": SESSION,
                    "camera": CAMERA_ID,
                    "track_id": tid,
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "cx": round(cx, 2),
                    "cy": round(cy, 2),
                    "cx_norm": round(cx / W, 4),
                    "cy_norm": round(cy / H, 4),
                    "confidence": round(conf, 3),
                })

        pbar.update(1)
finally:
    pbar.close()
    if writer is not None:
        writer.release()

elapsed = time.time() - start_time_run
df_raw = pd.DataFrame(raw_rows)

print(f"Finished in {elapsed/60:.1f} min")
print(f"Sampled frames processed: {proc_idx:,}")
print(f"Raw detections: {len(df_raw):,}")
print(f"Raw track IDs: {df_raw['track_id'].nunique() if len(df_raw) else 0}")

df_raw.head()


# 6. Manually validate and merge identities
import base64
import io
import ipywidgets as widgets
from IPython.display import display
from PIL import Image as PILImage

if len(df_raw) == 0:
    raise RuntimeError("No detections found. Check VIDEO_PATH, model, threshold, and video visibility.")


def crop_to_b64(arr):
    img = PILImage.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


label_map = {}
track_ids = sorted(df_raw["track_id"].dropna().astype(int).unique())
rows = []

for tid in track_ids:
    info = raw_track_thumbs.get(int(tid))
    if info is None:
        continue

    b64 = crop_to_b64(info["crop"])
    html = widgets.HTML(
        value=f"""
        <div style="width:220px;border:1px solid #ddd;padding:8px;margin:4px">
          <img src="data:image/png;base64,{b64}" style="max-width:200px;max-height:150px"><br>
          <b>raw track_id: {tid}</b><br>
          detections: {info['count']}<br>
          best confidence: {info['conf']:.2f}<br>
          representative frame: {info['frame']}
        </div>
        """
    )
    dropdown = widgets.Dropdown(
        options=PARTICIPANT_IDS,
        value="Noise",
        description="person_id:",
        layout=widgets.Layout(width="220px"),
    )
    rows.append((tid, dropdown, widgets.VBox([html, dropdown])))

ui = widgets.GridBox(
    [box for _, _, box in rows],
    layout=widgets.Layout(grid_template_columns="repeat(3, 250px)", grid_gap="10px"),
)
confirm_btn = widgets.Button(description="Confirm labels", button_style="success")
out = widgets.Output()


def on_confirm(_):
    global label_map
    label_map = {int(tid): dd.value for tid, dd, _ in rows}
    with out:
        out.clear_output()
        print("Labels confirmed.")
        print(label_map)


confirm_btn.on_click(on_confirm)
display(ui, confirm_btn, out)


# 7. Export cleaned anonymised tracking CSV
if not label_map:
    raise RuntimeError("label_map is empty. Confirm identity labels in the previous cell first.")

df_final = df_raw.copy()
df_final["person_id"] = df_final["track_id"].map(label_map).fillna("UNLABELLED")

if EXCLUDE_NOISE_FROM_FINAL:
    df_final = df_final[
        ~df_final["person_id"].isin(["Noise", "Skip", "UNLABELLED"])
    ].copy()
else:
    df_final = df_final[df_final["person_id"] != "Skip"].copy()

COL_ORDER = [
    "frame",
    "date", "time_of_day", "datetime", "time_sec",
    "day", "session", "camera",
    "track_id", "person_id",
    "x1", "y1", "x2", "y2", "cx", "cy", "cx_norm", "cy_norm",
    "confidence",
]

df_final = df_final[[c for c in COL_ORDER if c in df_final.columns]]
df_final.to_csv(FINAL_CSV, index=False)

print("Cleaned CSV:", FINAL_CSV)
print("Rows:", len(df_final))
print("person_id values:", sorted(df_final["person_id"].unique()))
df_final.head()


# 8. Save a reference frame for floor-plan calibration
REF_FRAME_IDX = 0  # change if another frame is more suitable for calibration

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, REF_FRAME_IDX)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not read the requested reference frame.")

cv2.imwrite(REF_FRAME, frame)
print("Reference frame:", REF_FRAME)


# 9. Optional privacy-preserving video export
def clip_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width, int(round(x2))))
    y2 = max(0, min(height, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def blur_roi(frame, x1, y1, x2, y2, blur_strength=51):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame
    k = max(15, int(blur_strength))
    if k % 2 == 0:
        k += 1
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return frame


def draw_person_id_label(frame, label, x1, y1, x2, y2):
    label = str(label)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad = 5

    label_x1 = x1
    label_y1 = max(0, y1 - th - baseline - 2 * pad)
    label_x2 = min(frame.shape[1], x1 + tw + 2 * pad)
    label_y2 = min(frame.shape[0], label_y1 + th + baseline + 2 * pad)

    if y1 - th - baseline - 2 * pad < 0:
        label_y1 = y1
        label_y2 = min(frame.shape[0], y1 + th + baseline + 2 * pad)

    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (0, 0, 0), -1)
    cv2.putText(
        frame, label,
        (label_x1 + pad, label_y2 - baseline - pad),
        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )
    return frame


boxes_by_frame = {
    int(frame): group.to_dict("records")
    for frame, group in df_final.groupby("frame", sort=True)
}
frames_to_export = sorted(boxes_by_frame)

cap = cv2.VideoCapture(VIDEO_PATH)
writer = cv2.VideoWriter(
    PRIVACY_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    ANALYSIS_FPS,
    (W, H),
)

written = 0
try:
    for frame_idx in tqdm(frames_to_export, desc="Privacy video", unit="frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue

        clipped_boxes = []
        for row in boxes_by_frame[frame_idx]:
            clipped = clip_box(row["x1"], row["y1"], row["x2"], row["y2"], W, H)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            frame = blur_roi(frame, x1, y1, x2, y2)
            clipped_boxes.append((row["person_id"], x1, y1, x2, y2))

        for person_id, x1, y1, x2, y2 in clipped_boxes:
            frame = draw_person_id_label(frame, person_id, x1, y1, x2, y2)

        writer.write(frame)
        written += 1
finally:
    cap.release()
    writer.release()

print("Privacy-preserving video:", PRIVACY_VIDEO)
print("Frames written:", written)


# Output for downstream SpaceTrace analysis