# SpaceTrace

**SpaceTrace is an open-source spatial-behaviour analysis pipeline for transforming CCTV video into anonymised occupant trajectories and mapping them onto architectural floor plans for movement, occupancy, proximity, co-presence, and behavioural analysis.**

The workflow combines video-based person detection and tracking, manual identity validation, homography-based floor-plan projection, Space Syntax integration, and browser-based spatial analysis.

---

## Workflow

**CCTV video**  
→ YOLO11m person detection  
→ BoT-SORT + appearance-based ReID  
→ Manual identity validation  
→ Anonymised tracking CSV  
→ CCTV-to-floor-plan calibration  
→ Floor-plan projection  
→ Movement, occupancy, proximity, co-presence, and behavioural analysis

Video preprocessing and identity validation are provided in [`preprocessing/`](./preprocessing/).

---

## Repository structure

- [`preprocessing/`](./preprocessing/) — YOLO11m person detection, BoT-SORT + ReID tracking, manual identity validation, anonymised CSV export, and privacy-preserving video export
- `spacetrace-calibrate.html` — CCTV and Space Syntax calibration to an architectural floor plan
- `spacetrace-analyse.html` — movement replay and spatial-behaviour analysis
- `example/` — example inputs and outputs

The calibration and analysis interfaces run locally in the browser with no server required.  
The optional video preprocessing pipeline runs in Python/Google Colab.

---

## Quick start

### Option 1 — Start from existing tracking data

1. Download or clone the repository
2. Open `spacetrace-calibrate.html` in Chrome or Safari
3. Upload your floor plan
4. Add Space Syntax output and/or CCTV tracking data
5. Complete the calibration workflow
6. Click **Open Analysis →** to explore the calibrated data

Keep `spacetrace-calibrate.html` and `spacetrace-analyse.html` in the same folder.

### Option 2 — Start from CCTV video

Use the workflow in [`preprocessing/`](./preprocessing/) to:

1. Detect people using YOLO11m
2. Track detections using BoT-SORT with appearance-based ReID
3. Manually validate and merge track identities
4. Assign anonymised participant IDs
5. Export a cleaned tracking CSV
6. Import the CSV into SpaceTrace for floor-plan calibration and analysis

---

## Preprocessing

The preprocessing workflow converts CCTV footage into anonymised tracking data suitable for SpaceTrace.

### Detection and tracking

Person detections are generated with **YOLO11m** and linked across frames using **BoT-SORT** with appearance-based re-identification.

Only the person class is retained.

### Identity validation

Automated track IDs are manually reviewed before spatial analysis.

Multiple raw track IDs can be merged into a single anonymised `person_id`, allowing fragmented tracks to be reconciled before export.

### Outputs

The preprocessing workflow can export:

- annotated detection/tracking video
- cleaned tracking CSV with anonymised `person_id`
- CCTV reference frame for spatial calibration
- privacy-preserving video with participant regions blurred and anonymised IDs overlaid

Raw CCTV footage is not required for downstream SpaceTrace analysis once the cleaned tracking data have been generated.

---

## Calibrate

### Step 1 — Floor plan

Upload a PNG or JPG of the architectural floor plan.

---

### Step 2 — Space Syntax *(optional)*

Upload a Space Syntax CSV containing spatial coordinates (`x`, `y`) and configurational metrics such as Integration, Connectivity, Mean Visual Depth, or Isovist Area.

Click corresponding control points between the spatial point cloud and the floor plan.

Use identifiable architectural features such as corners, openings, and columns, with points distributed across the analysed area.

Click **Apply** to inspect the alignment.

Calibration can be saved as a JSON preset for reuse.

---

### Step 3 — Tracking data *(optional)*

Upload:

- a CCTV reference frame
- a tracking CSV

Accepted coordinate formats:

| Format | Columns |
|--------|---------|
| Pre-calibrated | `floor_x`, `floor_y` |
| Normalised centre | `cx_norm`, `cy_norm` |
| Bounding box | `x1`, `y1`, `x2`, `y2` |

For bounding-box input, SpaceTrace uses the bottom-centre point

`((x1 + x2) / 2, y2)`

as the estimated floor-contact position.

Select at least four corresponding control points between the CCTV frame and architectural floor plan, then apply the homography transformation.

---

### Step 4 — Zones *(optional)*

Draw named polygons directly on the floor plan.

Zones can be assigned spatial categories such as:

- Public
- Shared
- Private

Zone labels are written into the calibrated tracking CSV.

---

### Step 5 — Export

Download the calibrated outputs:

- `space_syntax_calibrated.csv`
- `tracking_calibrated.csv`

Click **Open Analysis →** to transfer the calibrated data directly into the analysis interface.

---

## Analyse

### Loading data

Upload files through the **Data** panel:

- **Tracking CSV** — requires `floor_x`, `floor_y`, `person_id`, and `time_sec`
- **Floor plan** — the architectural plan used during calibration
- **Space Syntax CSV** *(optional)*

---

### Replay

| Control | Action |
|---------|--------|
| ▶ / ⏸ | Play / Pause |
| ‹ › | Step one frame |
| ↺ | Reset |
| Timeline | Scrub through time |
| Speed | 0.5× to 500× |

---

### Display layers

The **Display** panel supports:

- Heatmap
- Tracks
- Live positions
- Floor plan

---

### Spatial configuration

Space Syntax metrics can be overlaid directly on the architectural floor plan.

Select a metric and adjust the overlay opacity to compare configurational properties with observed occupancy and movement.

---

### Behavioural analysis

| Module | Output |
|--------|--------|
| Heatmap | Cumulative occupancy density |
| Raw Trajectory | Individual movement paths |
| Proximity | Interpersonal distance |
| Visual Encounter | Co-presence based on isovist overlap |
| Congregation | Spatial clustering |
| Approach & Avoid | Movement vectors toward or away from others |

Each module supports configurable temporal windows:

**Current · Past 10–500 frames · Custom**

---

### Video validation

**Video Validate** provides frame-level comparison between the original CCTV perspective and projected floor-plan positions.

Upload the corresponding MP4 to synchronise video and spatial replay through the `time_sec` field.

The interface displays bounding boxes and anonymised participant IDs, allowing tracking and floor-plan projection to be manually inspected frame by frame.

---

## Tracking CSV

Core fields used by SpaceTrace:

| Column | Required | Description |
|--------|----------|-------------|
| `person_id` | ✓ | Anonymised participant identifier |
| `time_sec` | ✓ | Time from recording start, used for replay and synchronisation |
| `floor_x`, `floor_y` | ✓ | Position on the calibrated floor plan |
| `day` | recommended | Mission/study day |
| `session` | recommended | Observation session |
| `camera` | optional | Camera identifier |

The preprocessing pipeline additionally retains frame number, timestamp, raw track ID, bounding-box coordinates, and detection confidence where required for validation.

---

## Privacy

SpaceTrace is designed so that spatial analysis can be conducted on anonymised tracking data rather than identifiable video.

The preprocessing workflow supports:

- anonymised participant IDs
- manual removal of noise and invalid tracks
- blurred video exports for validation
- separation of raw CCTV footage from downstream spatial-analysis files

Researchers remain responsible for obtaining appropriate participant consent and ethical approval for video-based data collection and analysis.

---

## Browser support

The calibration and analysis interfaces are designed for:

- Chrome
- Safari
- Edge

Firefox is supported but may be slower for large datasets.

---

## Research

SpaceTrace was developed as part of **SPACE4SPACE**, a research project examining relationships between habitat spatial configuration, observed co-presence, and team health in isolated, confined, and extreme (ICE) environments.

The toolkit supports the computational workflow described in:

**SPACE4SPACE: Linking Habitat Spatial Configuration to Team Health Through Observed Co-Presence in a 14-Day Space Analogue Mission**

Chenxi Wang, Simon Ladouce, and Michal Gath-Morad  
University of Cambridge

---

## Project

Developed at the **Cambridge Cognitive Architecture Lab** and **NeuroCivitas Lab for NeuroArchitecture**, University of Cambridge.

**Principal Investigator:** Dr. Michal Gath-Morad  
**Developer:** Chenxi Wang

---

## License

See [`LICENSE`](./LICENSE) for licensing information.
