# Course Project: Traffic Rule Violation Detection (AID 728)

**Team 24**
* **Shivansh Shah** (BT2024243)
* **Advaya Bhardwaj** (BT2024231)
* **Aman Lahoti** (BT2024123)

## 📌 Project Overview
This project implements a modular, highly efficient computer vision pipeline to detect traffic rule violations involving two-wheelers. The system processes RGB street images entirely **offline** and under strict resource constraints (Total Models < 250 MB).

It detects the following violations:
1. More than two riders on a single two-wheeler.
2. One or more riders not wearing a helmet.
3. Automatically extracts and reads the **license plate (registration number)** of any violating vehicle using OCR.

## 📁 Repository Structure
To evaluate, the directory must follow the exact structure outlined in the guidelines:
```text
<ROLL_NUMBER>/
│
├── solution.py                 # Core evaluation script containing TrafficViolationDetector
├── requirements.txt            # Python dependencies (Ultralytics, PaddleOCR, etc.)
├── README.md                   # Project documentation
├── download_ocr_models.sh      # Script to download offline PaddleOCR weights if missing
└── models/                     # Weights directory (Must be < 250 MB combined)
    ├── yolov8m.pt              # Base generalized feature extractor (Bikes & Persons)
    ├── model2_helmet_plate.pt  # Custom-trained YOLOv8n (Helmets & License Plates)
    ├── en_PP-OCRv3_det_infer/  # PaddleOCR text detection offline weights
    ├── en_PP-OCRv4_rec_infer/  # PaddleOCR text recognition offline weights
    └── ch_ppocr_mobile_v2.0_cls_infer/ # PaddleOCR angle classifier offline weights
```
*(Note: Evaluators may find fallback weights such as `yolov8s.pt` to guarantee execution if memory is severely constrained.)*

**📥 Model Weights Download (Backup):**
If the `models/` folder is not bundled or you need to re-download the `.pt` files, you can access the complete weights directory here:
🔗 [Google Drive: Model Weights Folder](https://drive.google.com/drive/folders/1cHAHG3iIyHeVe6fqIQnsLfuTWXam7Cyq?usp=sharing)


## 🧠 Model Pipeline Architecture
The system employs a 4-stage cascaded detection pipeline implemented in `solution.py`:

1. **Global Detection (YOLOv8m):** 
   - Runs at `imgsz=1280` with Test-Time Augmentation (TTA) to detect small and heavily occluded Motorcycles, Bicycles, and Persons.
   - Applies greedy Non-Maximum Suppression (NMS) to eliminate duplicate overlapping bounding boxes.
2. **Specialized Detection (Custom YOLOv8n):**
   - The custom model `model2_helmet_plate.pt` runs at `imgsz=640` to detect `helmet`, `no_helmet`, and `license_plate`.
3. **Geometric Association:**
   - A proprietary 5-signal geometric alignment algorithm maps Persons to Bikes (via intersection area rules).
   - Maps upper-body bounding boxes (heads) to detected Helmets/No-Helmets.
4. **Adaptive OCR (PaddleOCR):**
   - Violating vehicles trigger the OCR module.
   - The image undergoes CLAHE (Contrast Limited Adaptive Histogram Equalization) to recover details in low-lighting.
   - PaddleOCR reads the text, applying geometric row-sorting to correctly stitch double-row license plates into a single string.

## 🏋️ Training Configuration (`train_hf.ipynb` / `train.py`)
The custom YOLOv8n model was trained for **70 Epochs** on an NVIDIA A10G (Hugging Face Spaces) using an aggregated dataset of **34,574 images** compiled from 5 merged datasets (Anees Arom, Ronak Gohil, Roboflow). 

**Key Training Interventions:**
- **Class Imbalance:** The `license_plate` class dominated with 19,739 instances. A `cls_loss` weight of 1.5 was applied to compensate and boost helmet classification.
- **Augmentations:** Mosaic (1.0), Mixup (0.1), and robust HSV/Affine shifts were applied to simulate occlusion and extreme lighting.
- **Convergence:** The training run successfully completed the full 70 epochs, reaching optimal validation stabilization at the final epoch.

## 🚀 Execution & Usage
The inference pipeline is designed to be fully stateless and initialized exactly once per evaluation batch.

```python
from solution import TrafficViolationDetector

# 1. Initialize models (All models loaded into memory here)
detector = TrafficViolationDetector(model_dir="./models")

# 2. Predict on an image
output = detector.predict(image_path="test_images/sample.jpg")

# 3. Output Format:
print(output)
# {
#   "violations": [
#     {
#       "num_riders": 3,
#       "helmet_violations": 1,
#       "license_plate": "MH12AB1234"
#     }
#   ]
# }
```

## ⚠️ Dependencies & Offline Compliance
Install dependencies via:
```bash
pip install -r requirements.txt
```
All logic is contained within the `predict()` function without relying on internet access, strictly adhering to the offline execution protocol. PaddleOCR weights are explicitly loaded from local paths inside `./models/` to prevent grading timeouts.

**Note on PaddleOCR Weights:** 
If the three PaddleOCR model folders are missing from `./models/`, they can be easily re-downloaded for offline use by running the included helper script:
```bash
bash download_ocr_models.sh
```

Any runtime errors (e.g., missing images or missing weights) are gracefully handled to return an empty `{"violations": []}` response rather than halting the evaluator.
