from pathlib import Path

ROOT_SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
SEGMENTATION_DIR = ROOT_SRC_DIR / "segmentation"
MASKS_DIR = ROOT_SRC_DIR / "masks"
FACE_DETECTION_DIR = ROOT_SRC_DIR / "face_detection"
OUTPUT_DIR = ROOT_DIR / "output"
