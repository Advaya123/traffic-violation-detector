"""
solution.py  —  TrafficViolationDetector  (AID 728 Final Submission)

Models:
    • yolov8m.pt              — COCO-pretrained medium model (better than yolov8s)
                                Falls back to yolov8s.pt if yolov8m.pt not found.
    • model2_helmet_plate.pt  — custom YOLOv8n (helmet / no_helmet / license_plate)

Pipeline:
    1. CLAHE pre-processing     → normalise exposure before inference
    2. Model 1 + TTA @ 1280px  → detect bikes + persons (higher res, augmented)
    3. 5-signal 2D association  → link persons to bikes (depth, feet, torso, align, IoU)
    4. Model 2 + TTA            → detect helmet / no_helmet / license_plate
    5. Plate aspect-ratio guard → discard non-plate detections
    6. 3-variant OCR pipeline   → upscale → CLAHE → sharpen+thresh (fallback chain)
    7. Build JSON output

OCR Models (fully offline — must be present in ./models/):
    • en_PP-OCRv3_det_infer/   — text detection
    • en_PP-OCRv4_rec_infer/   — text recognition
    • ch_ppocr_mobile_v2.0_cls_infer/ — angle classifier

Download from:
    https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md
"""

import re
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
BIKE_CONF   = 0.35
PERSON_CONF = 0.40
LP_CONF     = 0.30
ASSOC_MIN   = 0.15

MAX_RIDERS_PER_BIKE        = 4
MIN_PERSON_BIKE_AREA_RATIO = 0.15
MIN_PLATE_SIDE             = 128
PLATE_AR_MIN, PLATE_AR_MAX = 1.5, 8.0

# COCO class IDs
COCO_MOTORCYCLE = 3
COCO_BICYCLE    = 1   # some two-wheelers are labelled bicycle by COCO models
COCO_PERSON     = 0

# Custom model class IDs
M2_HELMET   = 0
M2_NOHELMET = 1
M2_PLATE    = 2

PLATE_REGEX = re.compile(r'[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}')


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe(bgr: np.ndarray) -> np.ndarray:
    """CLAHE on LAB-L channel — improves dark/overexposed images for detection."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def box_area(box) -> float:
    x1, y1, x2, y2 = box[:4]
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0: return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0

def inter_area(a, b) -> float:
    return (max(0.0, min(a[2], b[2]) - max(a[0], b[0])) *
            max(0.0, min(a[3], b[3]) - max(a[1], b[1])))

def overlaps(a, b) -> bool:
    return inter_area(a, b) > 0

def point_near_box(px, py, box, margin=0.3) -> bool:
    x1, y1, x2, y2 = box[:4]
    w, h = x2 - x1, y2 - y1
    return (x1 - margin*w <= px <= x2 + margin*w and
            y1 - margin*h <= py <= y2 + margin*h)


# ─────────────────────────────────────────────────────────────────────────────
# Person NMS  —  removes duplicate overlapping person boxes
# ─────────────────────────────────────────────────────────────────────────────

def apply_person_nms(boxes, iou_thresh=0.45):
    """
    Custom NMS on person boxes.
    YOLOv8's internal NMS sometimes leaks 2 overlapping boxes for the same
    person (especially at imgsz=1280 with augment=True), inflating rider count.
    This pass removes any box whose IoU with a larger box exceeds iou_thresh.
    """
    if len(boxes) <= 1:
        return boxes
    arr = np.array(boxes, dtype=float)
    areas = (arr[:,2]-arr[:,0]) * (arr[:,3]-arr[:,1])
    order = areas.argsort()[::-1]   # process largest area first
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(arr[i,0], arr[rest,0])
        yy1 = np.maximum(arr[i,1], arr[rest,1])
        xx2 = np.minimum(arr[i,2], arr[rest,2])
        yy2 = np.minimum(arr[i,3], arr[rest,3])
        inter = np.maximum(0.0, xx2-xx1) * np.maximum(0.0, yy2-yy1)
        union = areas[i] + areas[rest] - inter
        iou_vals = inter / np.maximum(union, 1e-6)
        order = rest[iou_vals <= iou_thresh]
    return [boxes[k] for k in keep]


# ─────────────────────────────────────────────────────────────────────────────
# Person ↔️ Bike association  (5-signal)
# ─────────────────────────────────────────────────────────────────────────────

def is_plausible_rider(pbox, bbox) -> bool:
    """
    3-gate fast rejection before full scoring:
    ① Area ratio   — background peds appear smaller; reject if < 15 % of bike area
    ② Vertical feet — person's feet must land within bike's vertical extent ± 30 %
    ③ Torso-seat   — lower 60 % of person must intersect upper 70 % of bike
    """
    px1, py1, px2, py2 = pbox[:4]
    bx1, by1, bx2, by2 = bbox[:4]
    bike_h = by2 - by1

    # ① size ratio
    b_area = box_area(bbox)
    if b_area > 0 and box_area(pbox) / b_area < MIN_PERSON_BIKE_AREA_RATIO:
        return False

    # ② vertical feet
    if not (by1 - 0.30*bike_h <= py2 <= by2 + 0.30*bike_h):
        return False

    # ③ torso ∩ seat
    torso = [px1, py1 + int(0.40*(py2-py1)), px2, py2]
    seat  = [bx1, by1, bx2, by1 + int(0.70*bike_h)]
    if inter_area(torso, seat) == 0:
        return False

    return True

def association_score(pbox, bbox) -> float:
    """
    Additive 5-signal score:
    IoU + feet-near-bike (+0.40) + horiz-align (+0.15) + size-sim (+0.10) + vert-contain (+0.10)
    """
    px1, py1, px2, py2 = pbox[:4]
    bx1, by1, bx2, by2 = bbox[:4]
    bike_w = bx2 - bx1

    score = iou(pbox, bbox)

    if point_near_box((px1+px2)/2, py2, bbox, 0.3):
        score += 0.40
    if abs((px1+px2)/2 - (bx1+bx2)/2) < 0.70 * bike_w:
        score += 0.15
    ratio = box_area(pbox) / max(box_area(bbox), 1)
    if 0.40 <= ratio <= 2.00:
        score += 0.10
    if not (py1 + int(0.50*(py2-py1)) > by2 or py2 < by1):
        score += 0.10

    return score

def associate_persons_to_bikes(bike_boxes, person_boxes) -> dict:
    assignments = {i: [] for i in range(len(bike_boxes))}
    if not bike_boxes or not person_boxes:
        return assignments

    candidates = []
    for p_idx, pbox in enumerate(person_boxes):
        for b_idx, bbox in enumerate(bike_boxes):
            if not is_plausible_rider(pbox, bbox): continue
            score = association_score(pbox, bbox)
            if score > ASSOC_MIN:
                candidates.append((score, b_idx, p_idx))

    candidates.sort(reverse=True)
    assigned = set()
    count    = {i: 0 for i in range(len(bike_boxes))}
    for score, b_idx, p_idx in candidates:
        if p_idx in assigned or count[b_idx] >= MAX_RIDERS_PER_BIKE:
            continue
        assignments[b_idx].append(p_idx)
        assigned.add(p_idx)
        count[b_idx] += 1

    return assignments


# ─────────────────────────────────────────────────────────────────────────────
# Plate validation
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_plate_box(box) -> bool:
    w = max(1, box[2]-box[0]); h = max(1, box[3]-box[1])
    return PLATE_AR_MIN <= w/h <= PLATE_AR_MAX


# ─────────────────────────────────────────────────────────────────────────────
# OCR  —  3-variant preprocessing fallback chain
# ─────────────────────────────────────────────────────────────────────────────

def _upscale(crop, min_side=MIN_PLATE_SIDE):
    if crop is None or crop.size == 0: return crop
    h, w = crop.shape[:2]
    scale = max(min_side/max(h,1), min_side/max(w,1), 1.0)
    if scale <= 1.0: return crop
    return cv2.resize(crop, (max(1,int(round(w*scale))), max(1,int(round(h*scale)))),
                      interpolation=cv2.INTER_CUBIC)

def _plate_v1(crop):
    """Upscale only — works for clear plates."""
    return _upscale(crop)

def _plate_v2(crop):
    """Upscale + CLAHE — for low-contrast / dirty plates."""
    crop = _upscale(crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4)).apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def _plate_v3(crop):
    """Upscale + unsharp mask + adaptive threshold — for blurry / tiny plates."""
    crop  = _upscale(crop, 160)
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (0,0), 3)
    sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    thresh = cv2.adaptiveThreshold(sharp, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 8)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

def _sort_ocr_lines(lines: list) -> list:
    """
    Sort PaddleOCR result lines by their 2D bounding-box position:
    top-to-bottom first, then left-to-right within the same row.
    """
    def _center(line):
        box = line[0]
        ys  = [pt[1] for pt in box]
        xs  = [pt[0] for pt in box]
        return (sum(ys) / len(ys), sum(xs) / len(xs))
    try:
        return sorted(lines, key=_center)
    except Exception:
        return lines


def _parse_ocr(result) -> str:
    """
    Parse PaddleOCR output into a single string.
    Lines are geometry-sorted (top→bottom, left→right) before joining so that
    multi-row plate text is always assembled in the correct reading order.
    """
    if not result or not result[0]: return ""
    obj = result[0]; texts = []
    if isinstance(obj, dict):
        for t, s in zip(obj.get('rec_texts',[]), obj.get('rec_scores',[])):
            if s > 0.25: texts.append(t)
    elif isinstance(obj, list):
        sorted_lines = _sort_ocr_lines(
            [line for line in obj
             if len(line) >= 2 and isinstance(line[1], (list, tuple))]
        )
        for line in sorted_lines:
            t, s = line[1][0], line[1][1]
            if s > 0.25: texts.append(t)
    return " ".join(texts)

def _clean_plate(raw: str) -> str:
    text = re.sub(r'[^A-Z0-9]', '', raw.upper().replace(" ","").replace("-","").replace(".",""))
    m = PLATE_REGEX.search(text)
    return m.group(0) if m else (text if len(text) >= 4 else "UNKNOWN")

def run_ocr(ocr_engine, crop: np.ndarray) -> str:
    if crop is None or crop.size == 0: return "UNKNOWN"
    for fn in (_plate_v1, _plate_v2, _plate_v3):
        try:
            raw = _parse_ocr(ocr_engine.ocr(fn(crop)))
            if raw.strip():
                cleaned = _clean_plate(raw)
                if cleaned != "UNKNOWN": return cleaned
        except Exception: continue
    return "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Main class  (mandatory interface)
# ─────────────────────────────────────────────────────────────────────────────

class TrafficViolationDetector:

    def __init__(self, model_dir: str = "./models"):
        md = Path(model_dir)

        # ── Model 1: yolov8m for best accuracy; graceful fallback to yolov8s ──
        m1_path = md / "yolov8m.pt"
        if not m1_path.exists():
            m1_path = md / "yolov8s.pt"
            print("WARNING: yolov8m.pt not found — using yolov8s.pt")
        self.model1 = YOLO(str(m1_path))

        # ── Model 2: custom helmet + plate detector ───────────────────────────
        self.model2 = YOLO(str(md / "model2_helmet_plate.pt"))

        # ── OCR: fully offline — loads weights from ./models/ subfolders ──────
        # Required folder structure inside model_dir:
        #   en_PP-OCRv3_det_infer/        ← detection model
        #   en_PP-OCRv4_rec_infer/        ← recognition model
        #   ch_ppocr_mobile_v2.0_cls_infer/  ← angle classifier
        #
        # Download from:
        #   https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md
        from paddleocr import PaddleOCR

        det_dir = md / "en_PP-OCRv3_det_infer"
        rec_dir = md / "en_PP-OCRv4_rec_infer"
        cls_dir = md / "ch_ppocr_mobile_v2.0_cls_infer"

        # Warn loudly if any model folder is missing — don't silently fall back
        # to internet download, which would crash on an offline grading machine.
        for folder, name in [(det_dir, "det"), (rec_dir, "rec"), (cls_dir, "cls")]:
            if not folder.exists():
                raise FileNotFoundError(
                    f"PaddleOCR {name} model not found at: {folder}\n"
                    f"Download it from the PaddleOCR model zoo and place it in {md}/"
                )

        self.ocr = PaddleOCR(
            lang='en',
            use_angle_cls=True,
            det_model_dir=str(det_dir),
            rec_model_dir=str(rec_dir),
            cls_model_dir=str(cls_dir),
            show_log=False,         # suppress verbose PaddlePaddle logs
        )

    def predict(self, image_path: str) -> dict:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {"violations": []}
        h, w = img_bgr.shape[:2]

        # CLAHE normalisation
        img_proc = apply_clahe(img_bgr)

        # ── Stage 1: Detect bikes + persons ──────────────────────────────────
        res1 = self.model1(img_proc, verbose=False, imgsz=1280, augment=True)[0]
        bike_boxes, person_boxes_raw = [], []

        for box in res1.boxes:
            cls_id = int(box.cls[0]); conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if cls_id in (COCO_MOTORCYCLE, COCO_BICYCLE) and conf >= BIKE_CONF:
                bike_boxes.append([x1, y1, x2, y2])
            elif cls_id == COCO_PERSON and conf >= PERSON_CONF:
                person_boxes_raw.append([x1, y1, x2, y2])

        if not bike_boxes:
            return {"violations": []}

        # Deduplicate overlapping person boxes (TTA + high-res can produce duplicates)
        person_boxes = apply_person_nms(person_boxes_raw, iou_thresh=0.45)

        # ── Stage 2: Detect helmets + plates ─────────────────────────────────
        res2 = self.model2(img_proc, verbose=False, conf=0.25, augment=True)[0]
        helmet_boxes, nohelmet_boxes, plate_boxes = [], [], []

        for box in res2.boxes:
            cls_id = int(box.cls[0]); conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if cls_id == M2_HELMET:
                helmet_boxes.append([x1, y1, x2, y2])
            elif cls_id == M2_NOHELMET:
                nohelmet_boxes.append([x1, y1, x2, y2])
            elif cls_id == M2_PLATE and conf >= LP_CONF:
                if is_valid_plate_box([x1, y1, x2, y2]):
                    plate_boxes.append([x1, y1, x2, y2, conf])

        # ── Stage 3: Associate persons → bikes (5-signal) ────────────────────
        assignments = associate_persons_to_bikes(bike_boxes, person_boxes)

        # ── Stage 4–5: Per-bike helmet + plate logic ──────────────────────────
        violations = []

        for b_idx, rider_idxs in assignments.items():
            bx1, by1, bx2, by2 = bike_boxes[b_idx]
            num_riders        = len(rider_idxs)
            helmet_violations = 0

            for r_idx in rider_idxs:
                px1, py1, px2, py2 = person_boxes[r_idx]
                person_h = py2 - py1
                head_box = [px1, py1, px2, py1 + int(0.40 * person_h)]

                found_helmet = found_nohelmet = False
                for hb in helmet_boxes:
                    if overlaps(head_box, hb) or \
                       point_near_box((hb[0]+hb[2])/2, (hb[1]+hb[3])/2, head_box, 0.3):
                        found_helmet = True
                for nhb in nohelmet_boxes:
                    if overlaps(head_box, nhb) or \
                       point_near_box((nhb[0]+nhb[2])/2, (nhb[1]+nhb[3])/2, head_box, 0.3):
                        found_nohelmet = True

                if found_nohelmet or not found_helmet:
                    helmet_violations += 1

            is_violation = (num_riders > 2) or (helmet_violations > 0)
            if not is_violation:
                continue

            # Find best plate near this bike
            search_box = [bx1-20, by1-20, bx2+20, by2+20]
            plate_text = "UNKNOWN"; best_crop = None; best_conf = 0.0

            for pb in plate_boxes:
                lx1, ly1, lx2, ly2, c = pb
                if point_near_box((lx1+lx2)/2, (ly1+ly2)/2, search_box, 0.0) and c > best_conf:
                    best_conf = c
                    # Crop from ORIGINAL image (better colour fidelity for OCR)
                    best_crop = img_bgr[max(0,ly1):min(h,ly2), max(0,lx1):min(w,lx2)]

            if best_crop is not None and best_crop.size > 0:
                plate_text = run_ocr(self.ocr, best_crop)

            violations.append({
                "num_riders":        num_riders,
                "helmet_violations": helmet_violations,
                "license_plate":     plate_text,
            })

        return {"violations": violations}


# ─────────────────────────────────────────────────────────────────────────────
# CLI  (for local testing only — not part of graded interface)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, sys

    print("Initializing...", flush=True)
    try:
        detector = TrafficViolationDetector()
        print("Ready!\n", flush=True)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)

    test_dir = Path('./test_images')
    if not list(test_dir.glob('*.*')):
        fb = Path('/Users/advaya/Downloads/test_images')
        if fb.exists(): test_dir = fb

    image_files = sorted(
        list(test_dir.glob('*.jpg')) +
        list(test_dir.glob('*.jpeg')) +
        list(test_dir.glob('*.png'))
    )
    if not image_files:
        print(f"⚠️  No images found in {test_dir.absolute()}"); sys.exit(0)

    print(f"Processing {len(image_files)} image(s) from {test_dir.absolute()}",
          flush=True)
    final_output = {}
    for i, img_path in enumerate(image_files, 1):
        print(f"  [{i}/{len(image_files)}] {img_path.name}", flush=True)
        try:
            final_output[img_path.name] = detector.predict(str(img_path))
        except Exception as e:
            print(f"    WARNING: {e}", file=sys.stderr)
            final_output[img_path.name] = {"violations": [], "error": str(e)}

    out = Path('./output.json')
    out.write_text(json.dumps(final_output, indent=2))
    print(f"\nDone! -> {out.absolute()}", flush=True)
    print("\n========== OUTPUT ==========", flush=True)
    print(json.dumps(final_output, indent=2), flush=True)
    print("============================", flush=True)