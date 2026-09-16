# SpaceTrace preprocessing

This folder contains the CCTV preprocessing workflow used before floor-plan projection and spatial analysis in SpaceTrace.

## Workflow

`CCTV video → YOLO11m detection → BoT-SORT + ReID tracking → manual identity validation → anonymised CSV → homography calibration → floor-plan mapping → occupancy / co-presence analysis`

The notebook is designed for Jupyter or Google Colab because identity validation uses an interactive widget interface.

## What the notebook does

- detects the **person** class with Ultralytics YOLO11m;
- tracks detections with **BoT-SORT**;
- enables appearance-based ReID using `yolo11m-cls.pt`;
- samples video at a configurable frame stride;
- derives frame timestamps from researcher-provided session metadata, with optional filename timestamp fallback;
- presents representative crops for manual validation and merging of raw track IDs;
- exports a cleaned CSV with anonymised `person_id` labels;
- exports a reference frame for homography calibration;
- optionally creates a privacy-preserving video by blurring each validated person region and overlaying only the anonymised ID.

## Files

- `spacetrace_preprocess.ipynb` — preprocessing notebook.
- `requirements.txt` — Python dependencies.

## Setup

1. Open `spacetrace_preprocess.ipynb` in Jupyter or Google Colab.
2. Install the dependencies in the first code cell.
3. Set `VIDEO_PATH`, `OUTPUT_DIR`, session date/time, study day, session label, and camera ID in the configuration cell.
4. Adjust `VID_STRIDE` and `CONF_THRES` if needed.
5. Run detection and tracking.
6. Use the interactive validation interface to map raw `track_id` values to anonymised `person_id` values.
7. Export the cleaned CSV and reference frame.
8. Use those outputs in the SpaceTrace calibration / analysis interface.

## Main output schema

The cleaned CSV contains:

- `frame`
- `date`, `time_of_day`, `datetime`, `time_sec`
- `day`, `session`, `camera`
- `track_id`, `person_id`
- `x1`, `y1`, `x2`, `y2`
- `cx`, `cy`, `cx_norm`, `cy_norm`
- `confidence`

## Reproducibility notes

The default settings reproduce the methodological logic used in the associated SpaceTrace study, but paths, timestamps, participant IDs, and sampling stride should be adapted to each dataset. Model weights are obtained through Ultralytics and are not redistributed in this repository.

## Privacy and ethics

Do not commit raw identifiable CCTV footage or participant names to a public repository. Use anonymised participant IDs and publish only data and media permitted by the relevant ethics approval and participant consent. The optional privacy-video export blurs detected person regions and overlays anonymised IDs only.
